//! Adjacency and switchover-time bookkeeping.
//!
//! `matrix[i*n + j]` is the *mean* switchover time from configuration `i`
//! to configuration `j` — entry `T_ij` of the paper's mean
//! reconfiguration-time matrix. Negative entries mean "no edge" and are
//! never selectable; callers enumerate the legal moves out of `i` with
//! `neighbors(i)`.
//!
//! Every family is mean-preserving: its parameters are derived from the
//! matrix mean `m` so that `E[τ] = m`. Policies and shortest-path
//! computations read the mean through `time(i, j)`; the simulator samples
//! the actual duration through `sample_time(i, j, rng)` when it schedules
//! a switchover. Swapping a family therefore changes the spread of a
//! reconfiguration time without moving any routing decision.
//!
//! A family can be set globally (one family for the whole graph) or per
//! edge (`from_matrix_with_overrides`), which is how a scenario expresses
//! "fast scripted setup outbound, heavy-tailed slow return" at a matched
//! mean.

use rand::Rng;
use rand_distr::{Distribution, Exp, LogNormal};

/// Switchover-time family for a single edge. Each variant is
/// mean-preserving: given the matrix mean `m`, draws have `E[τ] = m`.
///
/// `Eq` is intentionally not derived (the parameterized variants carry
/// `f64`); `PartialEq` is enough for the few equality checks we make.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SwitchoverDistribution {
    /// Each traversal takes exactly the mean. No randomness; CV = 0.
    Deterministic,
    /// `Exp(rate = 1/m)`. Fully determined by its mean (CV = 1).
    Exponential,
    /// Lognormal with coefficient of variation `cv`. With
    /// `σ = √ln(1+cv²)` and `μ = ln(m) − σ²/2` the draw `e^{μ+σZ}` has
    /// mean `m` and CV `cv`. All moments finite but subexponential, so
    /// large `cv` gives a heavy P99 at fixed mean.
    Lognormal { cv: f64 },
    /// Pareto Type II (Lomax), tail index `alpha > 1`, support `[0, ∞)`.
    /// Survival `S(x) = (1 + x/λ)^(−α)` with `λ = m·(α−1)` so the mean
    /// is `m`. Variance `m²·α/(α−2)` for `α>2` and **infinite** for
    /// `α ∈ (1, 2]` — the genuine regularly-varying heavy-tail regime
    /// (sojourn tail index `α−1`).
    ParetoII { alpha: f64 },
}

/// Standard-normal CDF inverse Φ⁻¹(p), Acklam's rational
/// approximation (abs error < 1.2e-9 over the open interval). Used for
/// lognormal quantiles. Clamps the boundaries to ±∞.
pub fn inv_norm_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Coefficients.
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

impl SwitchoverDistribution {
    /// Sample one draw with the given mean `m` (matrix entry).
    #[inline]
    pub fn sample<R: Rng + ?Sized>(&self, m: f64, rng: &mut R) -> f64 {
        match *self {
            SwitchoverDistribution::Deterministic => m,
            SwitchoverDistribution::Exponential => {
                debug_assert!(m > 0.0, "Exp distribution needs a positive mean");
                Exp::new(1.0 / m).expect("Exp parameter > 0").sample(rng)
            }
            SwitchoverDistribution::Lognormal { cv } => {
                debug_assert!(m > 0.0 && cv >= 0.0, "Lognormal needs m>0, cv>=0");
                let s2 = (1.0 + cv * cv).ln();
                let sigma = s2.sqrt();
                let mu = m.ln() - 0.5 * s2;
                // σ = 0 (cv = 0) collapses to the constant e^μ = m.
                LogNormal::new(mu, sigma)
                    .expect("LogNormal params valid")
                    .sample(rng)
            }
            SwitchoverDistribution::ParetoII { alpha } => {
                debug_assert!(m > 0.0 && alpha > 1.0, "Lomax needs m>0, alpha>1");
                let lambda = m * (alpha - 1.0);
                // U ~ (0,1] via 1 - rng.gen() so U^(-1/α) is never +inf.
                let u = 1.0 - rng.gen::<f64>();
                lambda * (u.powf(-1.0 / alpha) - 1.0)
            }
        }
    }

    /// Standard deviation of a draw with mean `m`. `+∞` for an
    /// infinite-variance Lomax (`α ≤ 2`).
    pub fn std(&self, m: f64) -> f64 {
        match *self {
            SwitchoverDistribution::Deterministic => 0.0,
            SwitchoverDistribution::Exponential => m,
            SwitchoverDistribution::Lognormal { cv } => m * cv,
            SwitchoverDistribution::ParetoII { alpha } => {
                if alpha > 2.0 {
                    m * (alpha / (alpha - 2.0)).sqrt()
                } else {
                    f64::INFINITY
                }
            }
        }
    }

    /// The `p`-quantile of a draw with mean `m` (`0 < p < 1`). Finite
    /// for every family, including infinite-variance Lomax — which is
    /// why a quantile spread is a safe tail surrogate for routing.
    pub fn quantile(&self, m: f64, p: f64) -> f64 {
        match *self {
            SwitchoverDistribution::Deterministic => m,
            // Exp quantile: -m·ln(1-p).
            SwitchoverDistribution::Exponential => -m * (1.0 - p).ln(),
            // Lognormal quantile: exp(μ + σ·Φ⁻¹(p)).
            SwitchoverDistribution::Lognormal { cv } => {
                let s2 = (1.0 + cv * cv).ln();
                let sigma = s2.sqrt();
                let mu = m.ln() - 0.5 * s2;
                (mu + sigma * inv_norm_cdf(p)).exp()
            }
            // Lomax quantile: λ·((1-p)^(-1/α) - 1), λ = m(α-1).
            SwitchoverDistribution::ParetoII { alpha } => {
                let lambda = m * (alpha - 1.0);
                lambda * ((1.0 - p).powf(-1.0 / alpha) - 1.0)
            }
        }
    }

    /// True for any family that draws a non-degenerate random duration
    /// (everything but `Deterministic`). Such edges require mean > 0.
    #[inline]
    pub fn is_stochastic(&self) -> bool {
        !matches!(self, SwitchoverDistribution::Deterministic)
    }
}

#[derive(Debug, Clone)]
pub struct Topology {
    pub n: usize,
    /// Global default family (introspection / back-compat). Per-edge
    /// dispatch always goes through `edge_dist`.
    pub distribution: SwitchoverDistribution,
    matrix: Vec<f64>,
    /// Per-edge family, length `n*n`, indexed `from*n + to`. Set to the
    /// global `distribution` everywhere unless an override applies.
    edge_dist: Vec<SwitchoverDistribution>,
    neighbors: Vec<Vec<usize>>,
}

impl Topology {
    /// Build a deterministic-switchover topology from the raw matrix.
    /// Equivalent to the previous behaviour and used by tests.
    pub fn from_matrix(rows: &[Vec<f64>]) -> Self {
        Self::from_matrix_with(rows, SwitchoverDistribution::Deterministic)
    }

    pub fn from_matrix_with(rows: &[Vec<f64>], distribution: SwitchoverDistribution) -> Self {
        Self::from_matrix_with_overrides(rows, distribution, &[])
    }

    /// Build a topology whose edges all use `distribution` except those
    /// named in `overrides`, each `(from, to, family)`. Out-of-range or
    /// non-edge override coordinates are ignored here — callers
    /// (`config.rs`) validate them up front.
    pub fn from_matrix_with_overrides(
        rows: &[Vec<f64>],
        distribution: SwitchoverDistribution,
        overrides: &[(usize, usize, SwitchoverDistribution)],
    ) -> Self {
        let n = rows.len();
        let mut matrix = vec![0.0; n * n];
        let mut edge_dist = vec![distribution; n * n];
        let mut neighbors = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                matrix[i * n + j] = rows[i][j];
                if i != j && rows[i][j] >= 0.0 {
                    neighbors[i].push(j);
                }
            }
        }
        for &(from, to, fam) in overrides {
            if from < n && to < n {
                edge_dist[from * n + to] = fam;
            }
        }
        Self {
            n,
            distribution,
            matrix,
            edge_dist,
            neighbors,
        }
    }

    /// Mean switchover time from `from` to `to`. Caller must ensure
    /// `(from, to)` is a real edge.
    #[inline]
    pub fn time(&self, from: usize, to: usize) -> f64 {
        debug_assert!(from < self.n && to < self.n);
        let v = self.matrix[from * self.n + to];
        debug_assert!(v >= 0.0, "topology.time called on a non-edge");
        v
    }

    /// Family used on edge `(from, to)`.
    #[inline]
    pub fn edge_distribution(&self, from: usize, to: usize) -> SwitchoverDistribution {
        self.edge_dist[from * self.n + to]
    }

    /// Sample the actual switchover duration for the edge `(from, to)`,
    /// dispatching on that edge's family. For `Deterministic` this
    /// returns the mean; the stochastic families draw mean-preservingly.
    /// The mean must be > 0 under any stochastic family (validated in
    /// `config.rs`).
    #[inline]
    pub fn sample_time<R: Rng + ?Sized>(&self, from: usize, to: usize, rng: &mut R) -> f64 {
        let mean = self.time(from, to);
        self.edge_distribution(from, to).sample(mean, rng)
    }

    #[inline]
    pub fn neighbors(&self, from: usize) -> &[usize] {
        &self.neighbors[from]
    }
}

#[cfg(test)]
mod tests {
    //! Statistical sampling tests for the switchover families. These
    //! live in the library (not `tests/`) because they need the crate's
    //! RNG dependencies (`rand`, `rand_xoshiro`) which integration tests
    //! do not link. Config-layer tests (parsing, validation, per-edge
    //! resolution) live in `tests/switchover_dist.rs`.
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    const N: usize = 300_000;

    fn draws(d: SwitchoverDistribution, m: f64, seed: u64) -> Vec<f64> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..N).map(|_| d.sample(m, &mut rng)).collect()
    }

    fn mean_of(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
    fn cv_of(xs: &[f64]) -> f64 {
        let m = mean_of(xs);
        let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
        var.sqrt() / m
    }
    fn quantile_of(xs: &[f64], p: f64) -> f64 {
        let mut v = xs.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (((v.len() - 1) as f64) * p).round() as usize;
        v[idx]
    }

    /// Empirical median and 0.9-quantile match the closed-form
    /// `quantile()` (robust even for heavy tails) — validates both
    /// mean-preservation and the quantile helper used by routing.
    fn check_quantiles(d: SwitchoverDistribution, m: f64, seed: u64, q90_tol: f64) {
        let xs = draws(d, m, seed);
        let q50 = quantile_of(&xs, 0.5);
        let q90 = quantile_of(&xs, 0.9);
        let q50_th = d.quantile(m, 0.5);
        let q90_th = d.quantile(m, 0.9);
        assert!(
            (q50 / q50_th - 1.0).abs() < 0.05,
            "{d:?}: median {q50:.4} vs theory {q50_th:.4}"
        );
        assert!(
            (q90 / q90_th - 1.0).abs() < q90_tol,
            "{d:?}: q90 {q90:.4} vs theory {q90_th:.4}"
        );
    }

    #[test]
    fn exponential_mean_cv_quantiles() {
        let d = SwitchoverDistribution::Exponential;
        let m = 8.0;
        let xs = draws(d, m, 1);
        assert!((mean_of(&xs) / m - 1.0).abs() < 0.03, "exp mean off");
        assert!((cv_of(&xs) - 1.0).abs() < 0.05, "exp CV should be ~1");
        // Closed-form q90 = m·ln(10).
        assert!((d.quantile(m, 0.9) - m * 10f64.ln()).abs() < 1e-9);
        check_quantiles(d, m, 7, 0.05);
        assert_eq!(d.std(m), m);
    }

    #[test]
    fn lognormal_mean_cv_quantiles() {
        let cv = 3.0;
        let d = SwitchoverDistribution::Lognormal { cv };
        let m = 6.0;
        let xs = draws(d, m, 2);
        // Sample mean converges well (all moments finite).
        assert!((mean_of(&xs) / m - 1.0).abs() < 0.03, "lognormal mean off");
        // CV estimator is noisy at cv=3; accept a wide band that still
        // rejects a wrong parameterization (cv=1 would be exp-like).
        let c = cv_of(&xs);
        assert!((2.0..4.2).contains(&c), "lognormal CV {c} not near 3");
        assert!((d.std(m) - m * cv).abs() < 1e-9);
        check_quantiles(d, m, 9, 0.06);
    }

    #[test]
    fn pareto2_finite_variance_mean_quantiles() {
        let alpha = 3.0;
        let d = SwitchoverDistribution::ParetoII { alpha };
        let m = 8.0;
        let xs = draws(d, m, 3);
        assert!((mean_of(&xs) / m - 1.0).abs() < 0.04, "lomax(3) mean off");
        // Closed-form std = m·sqrt(α/(α-2)).
        assert!((d.std(m) - m * (alpha / (alpha - 2.0)).sqrt()).abs() < 1e-9);
        check_quantiles(d, m, 11, 0.08);
    }

    #[test]
    fn pareto2_infinite_variance_median_and_spread() {
        let alpha = 1.5;
        let d = SwitchoverDistribution::ParetoII { alpha };
        let m = 8.0;
        // Variance is infinite — the sample mean is unreliable, so we
        // validate the robust median against the closed form instead.
        let xs = draws(d, m, 4);
        let med = quantile_of(&xs, 0.5);
        let med_th = d.quantile(m, 0.5);
        assert!(
            (med / med_th - 1.0).abs() < 0.05,
            "lomax(1.5) median {med:.4} vs theory {med_th:.4}"
        );
        assert!(d.std(m).is_infinite(), "lomax(1.5) variance must be +inf");
        // The upper quantile stays finite even though the variance does
        // not, and at a matched mean it exceeds an exponential edge's --
        // the heavy tail is in the quantile, not just in the moments.
        let q99 = d.quantile(m, 0.99);
        assert!(q99.is_finite() && q99 > m, "lomax(1.5) q99 = {q99}");
        let exp_q99 = SwitchoverDistribution::Exponential.quantile(m, 0.99);
        assert!(q99 > exp_q99, "lomax q99 {q99} should exceed exp {exp_q99}");
    }

    #[test]
    fn deterministic_is_exact_and_has_no_spread() {
        let d = SwitchoverDistribution::Deterministic;
        let m = 5.0;
        let xs = draws(d, m, 5);
        assert!(xs.iter().all(|&x| x == m), "deterministic must be exact");
        assert_eq!(d.std(m), 0.0);
        assert_eq!(d.quantile(m, 0.99), m);
        assert!(!d.is_stochastic());
    }

    #[test]
    fn sampling_is_seed_reproducible() {
        for d in [
            SwitchoverDistribution::Exponential,
            SwitchoverDistribution::Lognormal { cv: 2.5 },
            SwitchoverDistribution::ParetoII { alpha: 1.7 },
        ] {
            let a = draws(d, 4.0, 1234);
            let b = draws(d, 4.0, 1234);
            assert_eq!(a, b, "{d:?} not reproducible under a fixed seed");
        }
    }

    #[test]
    fn per_edge_override_dispatch() {
        // Edge (1,0) heavy Lomax, all others exponential.
        let topo = Topology::from_matrix_with_overrides(
            &[vec![0.0, 0.2], vec![8.0, 0.0]],
            SwitchoverDistribution::Exponential,
            &[(1, 0, SwitchoverDistribution::ParetoII { alpha: 1.5 })],
        );
        assert_eq!(
            topo.edge_distribution(0, 1),
            SwitchoverDistribution::Exponential
        );
        assert_eq!(
            topo.edge_distribution(1, 0),
            SwitchoverDistribution::ParetoII { alpha: 1.5 }
        );
        // Both edges keep their matrix mean; only the family differs.
        assert_eq!(topo.time(0, 1), 0.2);
        assert_eq!(topo.time(1, 0), 8.0);
    }
}
