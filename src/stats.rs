//! Per-replication accumulators and cross-replication aggregation.
//!
//! Statistics are collected only after the warm-up interval. Two kinds of
//! quantity are recorded:
//!
//!   * **Time-weighted** — queue-length area and server-state fractions,
//!     accumulated over every inter-event interval.
//!   * **Job-weighted** — sojourn, waiting and inter-visit observations,
//!     recorded once per completion or per visit.
//!
//! Job-weighted quantities are kept in fixed-capacity reservoirs (Vitter's
//! Algorithm R) so memory stays bounded on long horizons, with one
//! exception: the objective itself. The pooled weighted sojourn `w_i·S` is
//! kept in full for every completed job, because a reservoir caps each
//! configuration at `SOJOURN_SAMPLE_K` samples and would under-represent a
//! high-throughput configuration in the pooled quantile. Keeping the full
//! vector makes the per-replication 0.99 quantile exact.
//!
//! The simulator owns the RNG, so the reservoir routines take a `u64`
//! uniform drawn by the caller rather than sampling themselves.

use std::io::{self, Write};

use serde::Serialize;

/// Percentile points reported for every distribution.
pub const PERCENTILES: [f64; 5] = [0.50, 0.90, 0.95, 0.99, 0.999];

/// Largest queue length tracked exactly by the time-weighted histogram;
/// longer queues are lumped into the top bin.
pub const QLEN_HIST_MAX: usize = 500;

/// Reservoir capacity per configuration for sojourn, waiting and
/// inter-visit samples.
pub const SOJOURN_SAMPLE_K: usize = 5000;

/// Everything one replication observes.
#[derive(Debug, Clone)]
pub struct ReplicationStats {
    pub n: usize,
    /// Length of the observation window, `horizon − warmup`.
    pub observation_time: f64,

    // ---- time-weighted ------------------------------------------------
    /// `∫ Q_i(t) dt` over the observation window.
    pub queue_area: Vec<f64>,
    /// Time-weighted queue-length histogram: `qlen_hist[i][b]` is the time
    /// configuration `i` spent holding `b` jobs.
    pub qlen_hist: Vec<Vec<f64>>,
    /// Time the server spent serving, reconfiguring and idle.
    pub busy_time: f64,
    pub switch_time: f64,
    pub idle_time: f64,

    // ---- job-weighted -------------------------------------------------
    pub sojourn_sum: Vec<f64>,
    pub sojourn_count: Vec<u64>,
    pub wait_sum: Vec<f64>,
    pub wait_count: Vec<u64>,
    pub served: Vec<u64>,
    /// Reservoir of completed-job sojourn times per configuration, with the
    /// population size seen so far (needed for unbiased replacement).
    pub sojourn_samples: Vec<Vec<f64>>,
    pub sojourn_seen: Vec<u64>,
    /// Same for waiting times (sojourn minus service).
    pub wait_samples: Vec<Vec<f64>>,
    pub wait_seen: Vec<u64>,
    /// **The objective.** Pooled weighted sojourn `w_i·S` of every
    /// post-warm-up completion, in completion order and kept in full.
    pub weighted_sojourn_pooled: Vec<f64>,

    // ---- visit structure ----------------------------------------------
    /// Reconfigurations completed.
    pub switchovers: u64,
    /// Visits per configuration, counting only visits that *began* after
    /// warm-up so every visit is observed under the same window.
    pub visits: Vec<u64>,
    /// Services performed at one configuration while a *different* one was
    /// the committed target — the en-route service of Eq. (11). Always `0`
    /// for a policy that never serves in transit.
    pub transit_services: u64,
    /// Per-configuration inter-visit gaps: running sum, count, and a
    /// reservoir for percentiles.
    pub intervisit_sum: Vec<f64>,
    pub intervisit_count: Vec<u64>,
    pub intervisit_samples: Vec<Vec<f64>>,
    pub intervisit_seen: Vec<u64>,
}

impl ReplicationStats {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            observation_time: 0.0,
            queue_area: vec![0.0; n],
            qlen_hist: vec![vec![0.0; QLEN_HIST_MAX]; n],
            busy_time: 0.0,
            switch_time: 0.0,
            idle_time: 0.0,
            sojourn_sum: vec![0.0; n],
            sojourn_count: vec![0; n],
            wait_sum: vec![0.0; n],
            wait_count: vec![0; n],
            served: vec![0; n],
            sojourn_samples: vec![Vec::with_capacity(SOJOURN_SAMPLE_K); n],
            sojourn_seen: vec![0; n],
            wait_samples: vec![Vec::with_capacity(SOJOURN_SAMPLE_K); n],
            wait_seen: vec![0; n],
            weighted_sojourn_pooled: Vec::new(),
            switchovers: 0,
            visits: vec![0; n],
            transit_services: 0,
            intervisit_sum: vec![0.0; n],
            intervisit_count: vec![0; n],
            intervisit_samples: vec![Vec::with_capacity(SOJOURN_SAMPLE_K); n],
            intervisit_seen: vec![0; n],
        }
    }

    /// Time-weighted histogram update: configuration `i` held `l` jobs for
    /// `dt` time units. Saturates at the top bin.
    #[inline]
    pub fn record_qlen(&mut self, i: usize, l: usize, dt: f64) {
        let b = l.min(QLEN_HIST_MAX - 1);
        self.qlen_hist[i][b] += dt;
    }

    #[inline]
    pub fn record_sojourn(&mut self, i: usize, x: f64, rng_u64: u64) {
        reservoir_push(
            &mut self.sojourn_samples[i],
            &mut self.sojourn_seen[i],
            x,
            rng_u64,
        );
    }

    #[inline]
    pub fn record_wait(&mut self, i: usize, x: f64, rng_u64: u64) {
        reservoir_push(
            &mut self.wait_samples[i],
            &mut self.wait_seen[i],
            x,
            rng_u64,
        );
    }

    #[inline]
    pub fn record_intervisit(&mut self, i: usize, gap: f64, rng_u64: u64) {
        self.intervisit_sum[i] += gap;
        self.intervisit_count[i] += 1;
        reservoir_push(
            &mut self.intervisit_samples[i],
            &mut self.intervisit_seen[i],
            gap,
            rng_u64,
        );
    }
}

/// Vitter's Algorithm R: keep the reservoir until it is full, then replace
/// a uniformly chosen slot with probability `K / n`.
#[inline]
fn reservoir_push(reservoir: &mut Vec<f64>, seen: &mut u64, x: f64, rng_u64: u64) {
    *seen += 1;
    let n = *seen;
    if (n as usize) <= SOJOURN_SAMPLE_K {
        reservoir.push(x);
    } else {
        let r = (rng_u64 % n) as usize;
        if r < SOJOURN_SAMPLE_K {
            reservoir[r] = x;
        }
    }
}

/// Cross-replication summary. Vectors indexed by configuration are length
/// `n`; vectors indexed by percentile follow [`PERCENTILES`] order.
#[derive(Debug, Clone, Serialize)]
pub struct Aggregate {
    pub n: usize,
    pub n_reps: usize,
    pub weights: Vec<f64>,

    // ---- queue lengths -------------------------------------------------
    pub mean_qlen: Vec<f64>,
    pub std_qlen: Vec<f64>,
    /// Per-configuration queue-length percentiles from the time-weighted
    /// histograms pooled across replications.
    pub qlen_percentiles: Vec<Vec<f64>>,
    /// `Σ_i w_i E[L_i]`, its spread across replications, and its value
    /// normalized by `Σ_i w_i`.
    pub mean_weighted_qlen: f64,
    pub std_weighted_qlen: f64,
    pub mean_weighted_qlen_normalized: f64,

    // ---- delays --------------------------------------------------------
    pub mean_sojourn: Vec<f64>,
    pub mean_wait: Vec<f64>,
    pub throughput: Vec<f64>,
    pub sojourn_percentiles: Vec<Vec<f64>>,
    pub wait_percentiles: Vec<Vec<f64>>,
    /// Unweighted sojourn percentiles pooled over all configurations.
    pub sojourn_percentiles_overall: Vec<f64>,

    // ---- the objective -------------------------------------------------
    /// Per-replication weighted-sojourn percentiles: outer index is the
    /// replication, inner follows [`PERCENTILES`]. Each row is an exact
    /// quantile of that replication's job population.
    pub weighted_sojourn_percentiles_per_rep: Vec<Vec<f64>>,
    /// Mean across replications of the per-replication percentiles. The
    /// P99 entry is the headline metric.
    pub weighted_sojourn_pct_mean: Vec<f64>,
    /// Half-width of the 95% Student-t interval (df = `n_reps − 1`) over
    /// the per-replication percentiles.
    pub weighted_sojourn_pct_ci95: Vec<f64>,
    /// Weighted-sojourn percentiles pooled across every replication's
    /// samples. Reference value; the headline is the per-replication mean.
    pub weighted_sojourn_percentiles_overall: Vec<f64>,

    // ---- server and visit structure ------------------------------------
    pub utilization: f64,
    pub switch_frac: f64,
    pub idle_frac: f64,
    pub mean_switchovers: f64,
    pub switches_per_time: f64,
    pub mean_visits: Vec<f64>,
    pub mean_transit_services: f64,
    pub transit_services_per_time: f64,
    pub mean_intervisit: Vec<f64>,
    pub intervisit_percentiles: Vec<Vec<f64>>,
}

fn qlen_percentiles_from_hist(hist: &[f64]) -> Vec<f64> {
    let total: f64 = hist.iter().sum();
    if total <= 0.0 {
        return vec![0.0; PERCENTILES.len()];
    }
    PERCENTILES
        .iter()
        .map(|&p| {
            let target = p * total;
            let mut acc = 0.0;
            for (b, &t) in hist.iter().enumerate() {
                acc += t;
                if acc >= target {
                    return b as f64;
                }
            }
            (hist.len() - 1) as f64
        })
        .collect()
}

/// Two-sided 95% Student-t critical value `t_{0.975, df}`. Exact for the
/// replication counts used here, falling back to the normal approximation
/// beyond df = 30. `df = 0` gives NaN: one replication has no interval.
fn t_crit_95(df: usize) -> f64 {
    const T: [f64; 31] = [
        f64::NAN, // df = 0
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228, // 1..10
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086, // 11..20
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042, // 21..30
    ];
    if df == 0 {
        f64::NAN
    } else if df < T.len() {
        T[df]
    } else {
        1.96
    }
}

/// Percentiles by linear interpolation between order statistics. Sorts in
/// place.
fn percentiles_from_samples(samples: &mut [f64]) -> Vec<f64> {
    if samples.is_empty() {
        return vec![0.0; PERCENTILES.len()];
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = samples.len();
    PERCENTILES
        .iter()
        .map(|&p| {
            let idx = p * (m as f64 - 1.0);
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            let frac = idx - lo as f64;
            samples[lo] * (1.0 - frac) + samples[hi] * frac
        })
        .collect()
}

impl Aggregate {
    pub fn from_replications(reps: Vec<ReplicationStats>, weights: &[f64]) -> Self {
        assert!(!reps.is_empty(), "no replications to aggregate");
        let n = reps[0].n;
        assert_eq!(weights.len(), n, "weights length must equal n_queues");
        let nr = reps.len();
        let nrf = nr as f64;

        let mut sum_q = vec![0.0; n];
        let mut sum_q2 = vec![0.0; n];
        let mut sum_s = vec![0.0; n];
        let mut sum_w = vec![0.0; n];
        let mut sum_thr = vec![0.0; n];
        let mut sum_visits = vec![0.0_f64; n];
        let mut sum_intervisit_mean = vec![0.0_f64; n];
        let mut intervisit_count_reps = vec![0_usize; n];
        let mut sum_busy = 0.0;
        let mut sum_switch = 0.0;
        let mut sum_idle = 0.0;
        let mut sum_swovers = 0.0;
        let mut sum_obs_time = 0.0_f64;
        let mut sum_transit_services = 0.0_f64;

        let weight_sum: f64 = weights.iter().sum();
        let mut weighted_qlen_per_rep: Vec<f64> = Vec::with_capacity(nr);

        for r in &reps {
            let t_obs = r.observation_time;
            let mut wq_rep = 0.0;
            for i in 0..n {
                let qi = r.queue_area[i] / t_obs;
                sum_q[i] += qi;
                sum_q2[i] += qi * qi;
                wq_rep += weights[i] * qi;
                if r.sojourn_count[i] > 0 {
                    sum_s[i] += r.sojourn_sum[i] / r.sojourn_count[i] as f64;
                }
                if r.wait_count[i] > 0 {
                    sum_w[i] += r.wait_sum[i] / r.wait_count[i] as f64;
                }
                sum_thr[i] += r.served[i] as f64 / t_obs;
                sum_visits[i] += r.visits[i] as f64;
                if r.intervisit_count[i] > 0 {
                    sum_intervisit_mean[i] += r.intervisit_sum[i] / r.intervisit_count[i] as f64;
                    intervisit_count_reps[i] += 1;
                }
            }
            weighted_qlen_per_rep.push(wq_rep);
            sum_busy += r.busy_time / t_obs;
            sum_switch += r.switch_time / t_obs;
            sum_idle += r.idle_time / t_obs;
            sum_swovers += r.switchovers as f64;
            sum_obs_time += t_obs;
            sum_transit_services += r.transit_services as f64;
        }

        let mean_qlen: Vec<f64> = sum_q.iter().map(|x| x / nrf).collect();
        let std_qlen: Vec<f64> = (0..n)
            .map(|i| {
                let m = sum_q[i] / nrf;
                ((sum_q2[i] / nrf) - m * m).max(0.0).sqrt()
            })
            .collect();

        let mean_weighted_qlen: f64 = weighted_qlen_per_rep.iter().sum::<f64>() / nrf;
        let std_weighted_qlen = (weighted_qlen_per_rep
            .iter()
            .map(|x| (x - mean_weighted_qlen).powi(2))
            .sum::<f64>()
            / nrf)
            .max(0.0)
            .sqrt();
        let mean_weighted_qlen_normalized = if weight_sum > 0.0 {
            mean_weighted_qlen / weight_sum
        } else {
            0.0
        };

        // Queue-length percentiles from the histograms pooled across reps.
        let qlen_percentiles: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut combined = vec![0.0_f64; QLEN_HIST_MAX];
                for r in &reps {
                    for (b, c) in combined.iter_mut().enumerate() {
                        *c += r.qlen_hist[i][b];
                    }
                }
                qlen_percentiles_from_hist(&combined)
            })
            .collect();

        // Sojourn / wait / inter-visit percentiles: each replication's
        // reservoir is unbiased for its own run, so concatenating them
        // samples the union population.
        let mut sojourn_percentiles = Vec::with_capacity(n);
        let mut wait_percentiles = Vec::with_capacity(n);
        let mut intervisit_percentiles = Vec::with_capacity(n);
        let mut sojourn_overall: Vec<f64> = Vec::new();
        for i in 0..n {
            let mut s_all: Vec<f64> = Vec::new();
            let mut w_all: Vec<f64> = Vec::new();
            let mut iv_all: Vec<f64> = Vec::new();
            for r in &reps {
                s_all.extend_from_slice(&r.sojourn_samples[i]);
                w_all.extend_from_slice(&r.wait_samples[i]);
                iv_all.extend_from_slice(&r.intervisit_samples[i]);
            }
            sojourn_overall.extend_from_slice(&s_all);
            sojourn_percentiles.push(percentiles_from_samples(&mut s_all));
            wait_percentiles.push(percentiles_from_samples(&mut w_all));
            intervisit_percentiles.push(percentiles_from_samples(&mut iv_all));
        }
        let sojourn_percentiles_overall = percentiles_from_samples(&mut sojourn_overall);

        // The objective. Each replication's pooled `w_i·S` vector is the
        // exact job population, so its quantiles are unbiased; the reported
        // value is the mean over replications with a Student-t interval, so
        // one run backs both the point estimate and its error bar.
        let np = PERCENTILES.len();
        let mut weighted_sojourn_percentiles_per_rep: Vec<Vec<f64>> = Vec::with_capacity(nr);
        let mut weighted_overall: Vec<f64> = Vec::new();
        for r in &reps {
            let mut buf = r.weighted_sojourn_pooled.clone();
            weighted_sojourn_percentiles_per_rep.push(percentiles_from_samples(&mut buf));
            weighted_overall.extend_from_slice(&r.weighted_sojourn_pooled);
        }
        let weighted_sojourn_percentiles_overall = percentiles_from_samples(&mut weighted_overall);
        let mut weighted_sojourn_pct_mean = vec![0.0_f64; np];
        let mut weighted_sojourn_pct_ci95 = vec![0.0_f64; np];
        let t = t_crit_95(nr.saturating_sub(1));
        for p in 0..np {
            let mean: f64 = weighted_sojourn_percentiles_per_rep
                .iter()
                .map(|row| row[p])
                .sum::<f64>()
                / nrf;
            weighted_sojourn_pct_mean[p] = mean;
            if nr >= 2 {
                let var: f64 = weighted_sojourn_percentiles_per_rep
                    .iter()
                    .map(|row| (row[p] - mean).powi(2))
                    .sum::<f64>()
                    / (nrf - 1.0);
                weighted_sojourn_pct_ci95[p] = t * var.max(0.0).sqrt() / nrf.sqrt();
            }
        }

        let mean_intervisit: Vec<f64> = (0..n)
            .map(|i| {
                if intervisit_count_reps[i] > 0 {
                    sum_intervisit_mean[i] / intervisit_count_reps[i] as f64
                } else {
                    0.0
                }
            })
            .collect();
        let mean_obs_time = sum_obs_time / nrf;
        let per_time = |x: f64| {
            if mean_obs_time > 0.0 {
                x / mean_obs_time
            } else {
                0.0
            }
        };
        let mean_switchovers = sum_swovers / nrf;
        let mean_transit_services = sum_transit_services / nrf;

        Self {
            n,
            n_reps: nr,
            weights: weights.to_vec(),
            mean_qlen,
            std_qlen,
            qlen_percentiles,
            mean_weighted_qlen,
            std_weighted_qlen,
            mean_weighted_qlen_normalized,
            mean_sojourn: sum_s.iter().map(|x| x / nrf).collect(),
            mean_wait: sum_w.iter().map(|x| x / nrf).collect(),
            throughput: sum_thr.iter().map(|x| x / nrf).collect(),
            sojourn_percentiles,
            wait_percentiles,
            sojourn_percentiles_overall,
            weighted_sojourn_percentiles_per_rep,
            weighted_sojourn_pct_mean,
            weighted_sojourn_pct_ci95,
            weighted_sojourn_percentiles_overall,
            utilization: sum_busy / nrf,
            switch_frac: sum_switch / nrf,
            idle_frac: sum_idle / nrf,
            mean_switchovers,
            switches_per_time: per_time(mean_switchovers),
            mean_visits: sum_visits.iter().map(|x| x / nrf).collect(),
            mean_transit_services,
            transit_services_per_time: per_time(mean_transit_services),
            mean_intervisit,
            intervisit_percentiles,
        }
    }

    /// Index of percentile `p` in [`PERCENTILES`].
    pub fn pct_idx(p: f64) -> usize {
        PERCENTILES
            .iter()
            .position(|&x| (x - p).abs() < 1e-9)
            .expect("percentile not in PERCENTILES")
    }

    /// Write the human-readable summary to `out`.
    pub fn write_summary(&self, out: &mut impl Write) -> io::Result<()> {
        let p99 = Self::pct_idx(0.99);
        writeln!(
            out,
            "=== DARTsim results ({} replications) ===",
            self.n_reps
        )?;
        writeln!(
            out,
            "Server: serving={:.4}  reconfiguring={:.4}  idle={:.4}",
            self.utilization, self.switch_frac, self.idle_frac
        )?;
        writeln!(
            out,
            "Reconfigurations per replication: {:.2}",
            self.mean_switchovers
        )?;
        writeln!(
            out,
            "En-route services per replication: {:.2}",
            self.mean_transit_services
        )?;
        writeln!(
            out,
            "Sojourn P99 (unweighted, pooled): {:.4}",
            self.sojourn_percentiles_overall[p99]
        )?;
        writeln!(
            out,
            "Weighted sojourn P99 (objective): {:.2} ± {:.2}  (mean ± 95% t-CI over {} reps)",
            self.weighted_sojourn_pct_mean[p99], self.weighted_sojourn_pct_ci95[p99], self.n_reps
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "{:>3}  {:>8}  {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "i", "weight", "E[L]", "stdev[L]", "E[W]", "E[T]", "T_p99", "throughput"
        )?;
        for i in 0..self.n {
            writeln!(
                out,
                "{:>3}  {:>8.3}  {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                i,
                self.weights[i],
                self.mean_qlen[i],
                self.std_qlen[i],
                self.mean_wait[i],
                self.mean_sojourn[i],
                self.sojourn_percentiles[i][p99],
                self.throughput[i]
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Weighted queue length  Σ_i w_i E[L_i]  = {:.4}  (stdev across reps {:.4})",
            self.mean_weighted_qlen, self.std_weighted_qlen
        )?;
        writeln!(
            out,
            "Weighted queue length  / Σ_i w_i       = {:.4}",
            self.mean_weighted_qlen_normalized
        )
    }
}
