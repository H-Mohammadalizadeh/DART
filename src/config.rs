//! TOML configuration model and validation.
//!
//! A configuration fully determines a run: the queueing system, the
//! reconfiguration graph, the simulation protocol, and the policy. It is
//! split into two parts that are composed at run time:
//!
//!   * a **scenario** — `[meta]`, `[system]`, `[arrivals]`, `[service]`,
//!     `[priorities]`, `[topology]` — which describes the system under
//!     study and is shared by every policy, and
//!   * a **protocol + policy** — `[simulation]`, `[policy]` — which
//!     describes how the system is measured and by which rule it is
//!     controlled.
//!
//! The reproduction pipeline in `reproduce/` keeps the scenario files in
//! `scenarios/` verbatim and appends the two remaining sections, so every
//! policy provably sees the same system.

use serde::Deserialize;
use std::path::Path;

use crate::dist::all_pairs_shortest_paths;
use crate::topology::Topology;

/// Top-level configuration loaded from a TOML file.
///
/// Every section rejects unknown keys. A misspelled parameter is a silent
/// change of experiment otherwise: the run succeeds, quietly using the
/// default, and the result looks plausible.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub meta: Meta,
    pub system: System,
    pub arrivals: Arrivals,
    pub service: Service,
    pub priorities: Priorities,
    pub topology: TopologyConfig,
    pub simulation: Simulation,
    pub policy: PolicyConfig,
}

/// Human-readable identification of a scenario. Carried through so that
/// scenario files are self-describing; the simulator never reads it.
#[derive(Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct Meta {
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct System {
    /// Number of configurations `N`. Must be `>= 2`.
    pub n_queues: usize,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Arrivals {
    /// Poisson arrival rate `λ_i` at each configuration. Strictly positive.
    pub rates: Vec<f64>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Service {
    /// Exponential service rate `μ_i` at each configuration. Strictly
    /// positive, and `μ_i > λ_i` is required for stability.
    pub rates: Vec<f64>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Priorities {
    /// Holding-cost weight `w_i` of each configuration. Strictly positive:
    /// a zero weight makes a configuration invisible to every score-based
    /// policy and can strand its jobs forever.
    pub weights: Vec<f64>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct TopologyConfig {
    /// Mean reconfiguration-time matrix `T`. Entry `[i][j]` is the mean of
    /// `τ_ij`; a negative entry marks an absent edge; the diagonal is `0`.
    pub switchover_times: Vec<Vec<f64>>,
    /// Family from which actual reconfiguration durations are drawn. The
    /// matrix above is the *mean* under every family, so switching family
    /// changes only the spread, never the routing distances.
    #[serde(default)]
    pub distribution: SwitchoverDistribution,
    /// Coefficient of variation for `distribution = "lognormal"`.
    /// Required (finite, `>= 0`) under that family; ignored otherwise.
    #[serde(default)]
    pub distribution_cv: Option<f64>,
    /// Tail index `κ` for `distribution = "pareto2"` (Lomax). Required
    /// (finite, `> 1`) under that family; ignored otherwise. `κ <= 2` is
    /// the infinite-variance regime.
    #[serde(default)]
    pub distribution_alpha: Option<f64>,
    /// Per-edge, per-direction family overrides applied on top of the
    /// global `distribution`. This is how a scenario expresses "outbound
    /// deterministic, slow return heavy-tailed" at a matched mean.
    #[serde(default)]
    pub edge_overrides: Option<Vec<EdgeOverride>>,
}

/// One per-edge reconfiguration-family override. `kind` selects the family
/// and `cv` / `alpha` carry its parameter, validated like the global ones.
/// The mean is always the matrix entry `switchover_times[from][to]`.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct EdgeOverride {
    pub from: usize,
    pub to: usize,
    pub kind: SwitchoverDistribution,
    #[serde(default)]
    pub cv: Option<f64>,
    #[serde(default)]
    pub alpha: Option<f64>,
}

/// Mean-preserving reconfiguration-time families.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SwitchoverDistribution {
    #[default]
    Deterministic,
    Exponential,
    Lognormal,
    Pareto2,
}

impl SwitchoverDistribution {
    /// Resolve a config family plus its parameter into the runtime family,
    /// validating the parameter. `ctx` labels the source in error messages
    /// (e.g. `"topology"` or `"edge override (1,0)"`).
    pub fn resolve(
        self,
        cv: Option<f64>,
        alpha: Option<f64>,
        ctx: &str,
    ) -> Result<crate::topology::SwitchoverDistribution, String> {
        use crate::topology::SwitchoverDistribution as T;
        match self {
            SwitchoverDistribution::Deterministic => Ok(T::Deterministic),
            SwitchoverDistribution::Exponential => Ok(T::Exponential),
            SwitchoverDistribution::Lognormal => {
                let cv = cv.ok_or_else(|| {
                    format!("{ctx}: distribution = lognormal requires distribution_cv (cv)")
                })?;
                if !(cv.is_finite() && cv >= 0.0) {
                    return Err(format!(
                        "{ctx}: lognormal cv = {cv} must be finite and >= 0"
                    ));
                }
                Ok(T::Lognormal { cv })
            }
            SwitchoverDistribution::Pareto2 => {
                let alpha = alpha.ok_or_else(|| {
                    format!("{ctx}: distribution = pareto2 requires distribution_alpha (alpha)")
                })?;
                if !(alpha.is_finite() && alpha > 1.0) {
                    return Err(format!(
                        "{ctx}: pareto2 alpha = {alpha} must be finite and > 1"
                    ));
                }
                Ok(T::ParetoII { alpha })
            }
        }
    }
}

impl TopologyConfig {
    /// Resolve the global default family plus its parameter.
    pub fn resolve_global_dist(&self) -> Result<crate::topology::SwitchoverDistribution, String> {
        self.distribution
            .resolve(self.distribution_cv, self.distribution_alpha, "topology")
    }

    /// Resolve the per-edge overrides into `(from, to, family)` tuples.
    pub fn resolved_edge_overrides(
        &self,
    ) -> Result<Vec<(usize, usize, crate::topology::SwitchoverDistribution)>, String> {
        let mut out = Vec::new();
        if let Some(ovs) = &self.edge_overrides {
            for ov in ovs {
                let ctx = format!("edge override ({},{})", ov.from, ov.to);
                let fam = ov.kind.resolve(ov.cv, ov.alpha, &ctx)?;
                out.push((ov.from, ov.to, fam));
            }
        }
        Ok(out)
    }

    /// Build the `n*n` resolved per-edge family vector (global default with
    /// overrides applied), indexed `from * n + to`. Used by both validation
    /// and topology construction.
    pub fn resolved_edge_families(
        &self,
        n: usize,
    ) -> Result<Vec<crate::topology::SwitchoverDistribution>, String> {
        let global = self.resolve_global_dist()?;
        let mut fams = vec![global; n * n];
        for (from, to, fam) in self.resolved_edge_overrides()? {
            if from < n && to < n {
                fams[from * n + to] = fam;
            }
        }
        Ok(fams)
    }
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Simulation {
    /// Length of one replication in simulated time units.
    pub horizon: f64,
    /// Leading interval discarded before statistics are collected.
    pub warmup: f64,
    /// Number of independent replications.
    pub n_replications: usize,
    /// Base seed. Replication `r` runs on a stream derived from it, so all
    /// policies compared under the same seed see the same arrival and
    /// service draws.
    pub seed: u64,
}

/// Policy selection and parameters.
///
/// Every field other than `kind` is optional and applies to a subset of the
/// policies, as marked. Values are validated only when supplied; defaults
/// are filled in when the router is built.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
    pub kind: PolicyKind,
    /// Initial server location. Default `0`.
    #[serde(default)]
    pub start: Option<usize>,
    /// **DART** Return discount `α` on `d(j,i)` in the urgency denominator
    /// `1 + d(i,j) + α d(j,i)`. Default `0.5`. Finite and `>= 0`.
    #[serde(default)]
    pub alpha: Option<f64>,
    /// **DART** Delay-guard strength `β`: how heavily the round trip
    /// `d + r` a skipped configuration would wait counts against skipping
    /// it. Default `6.0`. Finite and `>= 0`.
    #[serde(default)]
    pub beta: Option<f64>,
    /// **Tian, Tian-T** Lookahead horizon `K`. Default `1`; `>= 1`.
    #[serde(default)]
    pub k: Option<usize>,
    /// **Tian-T** Transit-service reluctance `η`: the traversed
    /// configuration `c` is served only while `w_c Q_c μ_c >= η ψ`. Large
    /// `η` converges back to unmodified Tian. Default `1.0`; finite, `>= 0`.
    #[serde(default)]
    pub eta: Option<f64>,
    /// **Tian, Tian-T** Finite-difference epsilon for the `ψ`-derivative
    /// test. Default `1e-6`; finite and `> 0`.
    #[serde(default)]
    pub epsilon: Option<f64>,
    /// **Tian, Tian-T** Hard cap on candidate sequences enumerated per
    /// decision. Default `100_000`; `>= 1`.
    #[serde(default)]
    pub max_sequences: Option<usize>,
}

/// The policies this simulator implements: DART and the three baselines it
/// is compared against.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PolicyKind {
    /// DART: opportunistic commitment. Commits to the configuration of
    /// highest urgency and serves configurations crossed on the way when
    /// either the delay guard or the backlog guard opens.
    Dart,
    /// Tian & Shone's `K`-stop network index, with no transit service.
    Tian,
    /// Tian `K`-stop extended with the guarded transit service of DART, so
    /// the comparison turns on target selection rather than feature access.
    TianTransit,
    /// Duenyas & Van Oyen's reconfiguration-aware index, graph-adapted.
    Dvo,
}

impl PolicyKind {
    /// TOML key name, used in output and error messages.
    pub fn name(self) -> &'static str {
        match self {
            PolicyKind::Dart => "dart",
            PolicyKind::Tian => "tian",
            PolicyKind::TianTransit => "tian_transit",
            PolicyKind::Dvo => "dvo",
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let raw = std::fs::read_to_string(path)?;
        let cfg: Config = toml::from_str(&raw)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.validate_sizes()?;
        self.validate_rates_and_weights()?;
        self.validate_topology_shape()?;
        self.validate_simulation()?;
        self.validate_policy()?;
        Ok(())
    }

    /// Resolve the starting server location. Default `0`.
    pub fn resolved_start(&self) -> Result<usize, String> {
        let n = self.system.n_queues;
        let s = self.policy.start.unwrap_or(0);
        if s >= n {
            return Err(format!("policy.start = {s} must be < n_queues = {n}"));
        }
        Ok(s)
    }

    // ---- internal validators ------------------------------------------

    fn validate_sizes(&self) -> Result<(), String> {
        let n = self.system.n_queues;
        if n < 2 {
            return Err("system.n_queues must be >= 2".into());
        }
        for (field, len) in [
            ("arrivals.rates", self.arrivals.rates.len()),
            ("service.rates", self.service.rates.len()),
            ("priorities.weights", self.priorities.weights.len()),
        ] {
            if len != n {
                return Err(format!("{field} length {len} != n_queues {n}"));
            }
        }
        Ok(())
    }

    fn validate_rates_and_weights(&self) -> Result<(), String> {
        for (field, xs) in [
            ("arrivals.rates", &self.arrivals.rates),
            ("service.rates", &self.service.rates),
            ("priorities.weights", &self.priorities.weights),
        ] {
            if xs.iter().any(|&x| !(x.is_finite() && x > 0.0)) {
                return Err(format!("{field} must all be finite and > 0"));
            }
        }
        Ok(())
    }

    fn validate_topology_shape(&self) -> Result<(), String> {
        let n = self.system.n_queues;
        let rows = &self.topology.switchover_times;
        if rows.len() != n {
            return Err(format!(
                "topology.switchover_times rows {} != n_queues {n}",
                rows.len()
            ));
        }
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n {
                return Err(format!(
                    "topology.switchover_times row {i} has len {} != n_queues {n}",
                    row.len()
                ));
            }
            if row[i] > 0.0 {
                return Err(format!(
                    "topology.switchover_times[{i}][{i}] = {} > 0 is invalid (no self-loops)",
                    row[i]
                ));
            }
            if !row.iter().enumerate().any(|(j, &v)| j != i && v >= 0.0) {
                return Err(format!(
                    "topology row {i} has no outgoing edge; the server would get stranded there"
                ));
            }
        }

        if let Some(ovs) = &self.topology.edge_overrides {
            for ov in ovs {
                if ov.from >= n || ov.to >= n {
                    return Err(format!(
                        "topology.edge_overrides: ({},{}) out of range for n_queues = {n}",
                        ov.from, ov.to
                    ));
                }
                if ov.from == ov.to {
                    return Err(format!(
                        "topology.edge_overrides: ({},{}) is a self-loop, not an edge",
                        ov.from, ov.to
                    ));
                }
                if rows[ov.from][ov.to] < 0.0 {
                    return Err(format!(
                        "topology.edge_overrides: ({},{}) is not a real edge (mean < 0)",
                        ov.from, ov.to
                    ));
                }
            }
        }

        // A stochastic family needs a strictly positive mean to be
        // well-defined; a deterministic zero-cost edge is fine.
        let fams = self.topology.resolved_edge_families(n)?;
        for i in 0..n {
            for j in 0..n {
                if i != j && rows[i][j] == 0.0 && fams[i * n + j].is_stochastic() {
                    return Err(format!(
                        "edge {i}->{j} has mean 0 under a stochastic reconfiguration family (must be > 0)"
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_simulation(&self) -> Result<(), String> {
        let sim = &self.simulation;
        if !(sim.horizon.is_finite() && sim.warmup.is_finite()) {
            return Err("simulation.horizon and warmup must be finite".into());
        }
        if sim.warmup < 0.0 {
            return Err("simulation.warmup must be >= 0".into());
        }
        if sim.horizon <= sim.warmup {
            return Err("simulation.horizon must be > simulation.warmup".into());
        }
        if sim.n_replications == 0 {
            return Err("simulation.n_replications must be >= 1".into());
        }
        Ok(())
    }

    fn validate_policy(&self) -> Result<(), String> {
        let n = self.system.n_queues;
        let kind = self.policy.kind.name();

        // Every policy walks between any pair of configurations, so the
        // reconfiguration graph must be strongly connected.
        let paths =
            all_pairs_shortest_paths(&Topology::from_matrix(&self.topology.switchover_times));
        for i in 0..n {
            for j in 0..n {
                if i != j && !paths.dist[i][j].is_finite() {
                    return Err(format!(
                        "policy = {kind}: queue {j} is unreachable from queue {i}; \
                         the reconfiguration graph must be strongly connected"
                    ));
                }
            }
        }

        if let Some(s) = self.policy.start {
            if s >= n {
                return Err(format!(
                    "policy.start = {s} is out of range for n_queues = {n}"
                ));
            }
        }
        for (field, v) in [
            ("policy.alpha", self.policy.alpha),
            ("policy.beta", self.policy.beta),
            ("policy.eta", self.policy.eta),
        ] {
            if let Some(v) = v {
                if !(v.is_finite() && v >= 0.0) {
                    return Err(format!("{field} = {v} must be finite and >= 0"));
                }
            }
        }
        if let Some(eps) = self.policy.epsilon {
            if !(eps.is_finite() && eps > 0.0) {
                return Err(format!("policy.epsilon = {eps} must be finite and > 0"));
            }
        }
        if self.policy.k == Some(0) {
            return Err("policy.k must be >= 1".into());
        }
        if self.policy.max_sequences == Some(0) {
            return Err("policy.max_sequences must be >= 1".into());
        }

        // DART's anticipated delay divides by the net clearance rate
        // `μ_i − λ_i`; Tian's fluid scoring and Duenyas-Van Oyen's
        // reward-rate denominators need the same quantity. Per-queue
        // stability is therefore required by every policy here.
        for i in 0..n {
            let (lam, mu) = (self.arrivals.rates[i], self.service.rates[i]);
            if mu <= lam {
                return Err(format!(
                    "policy = {kind}: queue {i} violates the per-queue stability \
                     condition μ > λ (got μ={mu}, λ={lam})"
                ));
            }
        }
        Ok(())
    }
}
