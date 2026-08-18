//! DART: opportunistic commitment under a weighted tail objective.
//!
//! DART makes three coupled decisions — which configuration to target,
//! which path to take, and whether to serve the configurations crossed on
//! the way. It picks the target of highest *urgency*, commits to a shortest
//! path toward it, and serves a traversed configuration only while one of
//! two guards justifies the pause.
//!
//! # The rule
//!
//! At the current configuration `i`, the candidate targets are the other
//! nonempty configurations, `D_i = { j != i : Q_j > 0 }`  (Eq. 6). Each
//! candidate `j` is ranked by an *anticipated delay* that combines the
//! delay already suffered by its head-of-line job with the time to clear
//! the backlog behind it,
//!
//! ```text
//!   â_j = a_j + (Q_j − 1)^+ / (μ_j − λ_j)                         (Eq. 7)
//! ```
//!
//! and by the reconfiguration times to reach and to leave it. With
//! `d = d(i,j)` the time to reach `j` and `r = d(j,i)` a proxy for the time
//! to leave it again, the urgency is
//!
//! ```text
//!   U_ij = w_j (â_j + d) / (1 + d + α r)                          (Eq. 8)
//! ```
//!
//! The committed target is `argmax_j U_ij`  (Eq. 9), tie-broken by smaller
//! `d` and then by smaller index. It is held fixed until reached, so the
//! server cannot be pulled into a myopic detour part-way there.
//!
//! While travelling toward the committed target `j*`, the policy decides at
//! each traversed configuration `k` whether to serve or move on. It serves
//! `k` when either guard opens (Eq. 11):
//!
//! ```text
//!   delay guard:    w_k (a_k + β(d + r)) ≥ U_kj*
//!   backlog guard:  w_k Q_k μ_k          ≥ Ũ_kj*
//! ```
//!
//! where `d = d(k,j*)`, `r = d(j*,k)`, and the backlog-domain threshold is
//!
//! ```text
//!   Ũ_kj* = w_j* (Q_j* + λ_j* d) μ_j* / (1 + μ_j* d + α μ_j* r)   (Eq. 10)
//! ```
//!
//! The delay guard protects a configuration whose head-of-line job would
//! grow too old if skipped: a skipped `k` waits at least the round trip
//! `d + r` for the next visit, and `β` sets how heavily that wait counts
//! against skipping. The backlog guard protects against a build-up whose
//! head-of-line age is still moderate. Both thresholds grow with the
//! target's own backlog `Q_j*` and with the arrivals `λ_j* d` accumulating
//! there during the trip, so a burst at `k` cannot starve `j*`: the growing
//! target eventually fails both guards and the server moves on.
//!
//! # Parameters
//!
//! Exactly two, both fixed across every topology in the evaluation:
//!
//! * `α` (`alpha`, default `0.5`) discounts the return time `d(j,i)` in the
//!   urgency denominator. `d(j,i)` is a proxy for a future move rather than
//!   the move being taken now, so it counts for less than the reach time.
//! * `β` (`beta`, default `6.0`) is the delay guard's strength: how heavily
//!   the round trip a skipped configuration would wait counts against
//!   skipping it. Larger `β` protects intermediate configurations more, and
//!   is the knob that trades delay between the high- and low-priority
//!   classes.

use crate::dist::{DistMatrix, NextHopMatrix};
use crate::topology::Topology;

use super::{Action, DecisionContext, Router, SystemView};

/// Return discount `α` used throughout the evaluation.
pub const DEFAULT_ALPHA: f64 = 0.5;
/// Delay-guard strength `β` used throughout the evaluation.
pub const DEFAULT_BETA: f64 = 6.0;

pub struct DartRouter {
    n: usize,
    weights: Vec<f64>,
    service_rates: Vec<f64>,
    arrival_rates: Vec<f64>,
    /// Shortest-path reconfiguration distances `d(i,j)` (Eq. 2).
    dist: DistMatrix,
    /// First hop of the shortest path, used to walk a committed route.
    next_hop: NextHopMatrix,
    /// The graph itself, for the mean of the single edge being taken.
    direct: Topology,
    /// Return discount `α` in the urgency denominator (Eq. 8).
    alpha: f64,
    /// Delay-guard strength `β` in the serve rule (Eq. 11).
    beta: f64,
    /// Net clearance rate `μ_j − λ_j` of each configuration, precomputed
    /// for the anticipated delay (Eq. 7). Strictly positive: per-queue
    /// stability is validated before a run starts. Because it is the *net*
    /// rate, the backlog term grows without bound as `ρ_j → 1`, which is
    /// exactly when that configuration's tail becomes dangerous.
    clearance_rate: Vec<f64>,
}

impl DartRouter {
    pub fn new(system: SystemView, alpha: f64, beta: f64) -> Self {
        let clearance_rate: Vec<f64> = (0..system.n)
            .map(|i| system.service_rates[i] - system.arrival_rates[i])
            .collect();
        debug_assert!(
            clearance_rate.iter().all(|&r| r > 0.0),
            "per-queue stability μ > λ is validated before a run starts"
        );
        Self {
            n: system.n,
            weights: system.weights,
            service_rates: system.service_rates,
            arrival_rates: system.arrival_rates,
            dist: system.paths.dist,
            next_hop: system.paths.next_hop,
            direct: system.topology,
            alpha,
            beta,
            clearance_rate,
        }
    }

    /// Anticipated delay `â_j` of configuration `j` (Eq. 7): the current
    /// head-of-line age plus the time to clear the backlog queued behind
    /// it. The head-of-line job is already counted in `a_j`, so the backlog
    /// behind it holds `Q_j − 1` jobs.
    #[inline]
    fn anticipated_delay(&self, j: usize, ages: &[f64], q: &[usize]) -> f64 {
        let backlog_behind = (q[j] as f64 - 1.0).max(0.0);
        ages[j] + backlog_behind / self.clearance_rate[j]
    }

    /// Urgency `U_ij` of heading from `origin` to `j` (Eq. 8).
    /// `NEG_INFINITY` if `j` is unreachable in either direction.
    #[inline]
    fn urgency(
        &self,
        origin: usize,
        j: usize,
        ages: &[f64],
        q: &[usize],
        dist: &DistMatrix,
    ) -> f64 {
        let d = dist[origin][j];
        let r = dist[j][origin];
        if !d.is_finite() || !r.is_finite() {
            return f64::NEG_INFINITY;
        }
        let denom = 1.0 + d + self.alpha * r;
        self.weights[j] * (self.anticipated_delay(j, ages, q) + d) / denom
    }

    /// Backlog-domain threshold `Ũ_kj*` of the committed target (Eq. 10),
    /// used by the backlog guard. Returns `0` when the target is
    /// unreachable, so the guard never fires on an infeasible target.
    #[inline]
    fn backlog_threshold(&self, origin: usize, j: usize, q: &[usize], dist: &DistMatrix) -> f64 {
        let d = dist[origin][j];
        let r = dist[j][origin];
        if !d.is_finite() || !r.is_finite() {
            return 0.0;
        }
        let mu = self.service_rates[j];
        let lam = self.arrival_rates[j];
        let denom = 1.0 + mu * d + self.alpha * mu * r;
        self.weights[j] * (q[j] as f64 + lam * d) * mu / denom
    }

    /// Committed target `j*` (Eq. 9): the nonempty configuration other than
    /// `current` of highest urgency. Ties break on smaller `d(i,j)` and
    /// then on smaller index, so the choice is fully deterministic.
    fn choose_target(
        &self,
        current: usize,
        q: &[usize],
        ages: &[f64],
        dist: &DistMatrix,
    ) -> Option<usize> {
        let mut best: Option<(usize, f64, f64)> = None;
        for k in 0..self.n {
            if k == current || q[k] == 0 {
                continue;
            }
            let d = dist[current][k];
            if !d.is_finite() || !dist[k][current].is_finite() {
                continue;
            }
            let score = self.urgency(current, k, ages, q, dist);
            best = Some(match best {
                None => (k, score, d),
                Some((bi, bs, bd)) => {
                    let better =
                        score > bs || (score == bs && d < bd) || (score == bs && d == bd && k < bi);
                    if better {
                        (k, score, d)
                    } else {
                        (bi, bs, bd)
                    }
                }
            });
        }
        best.map(|(k, _, _)| k)
    }
}

impl Router for DartRouter {
    fn decide(&self, current: usize, queue_lens: &[usize], ctx: DecisionContext<'_>) -> Action {
        if !queue_lens.iter().any(|&l| l > 0) {
            return Action::Idle;
        }
        let dist = &self.dist;
        let next_hop = &self.next_hop;
        // Ages drive the delay rule. Fall back to all-zero (backlog-only
        // behaviour) rather than panicking if a caller omits them.
        let zero;
        let ages: &[f64] = match ctx.ages {
            Some(a) => a,
            None => {
                zero = vec![0.0; self.n];
                &zero
            }
        };

        // Commitment, part one: on physically arriving at the committed
        // target, serve at least once before re-deciding.
        if ctx.server_just_arrived {
            if let Some(t) = ctx.locked_target {
                if t == current && queue_lens[current] > 0 {
                    return Action::Serve;
                }
            }
        }

        // Commitment, part two: keep an existing commitment while it is
        // still meaningful — a different, reachable, still-nonempty target.
        // A lock pointing at `current`, or at a configuration that has
        // drained in the meantime, is released and the target re-chosen.
        let kept_target = ctx
            .locked_target
            .filter(|&t| t != current && queue_lens[t] > 0 && dist[current][t].is_finite());

        let target =
            match kept_target.or_else(|| self.choose_target(current, queue_lens, ages, dist)) {
                Some(t) => t,
                None => {
                    // No other configuration has work: stay and serve here.
                    return if queue_lens[current] > 0 {
                        Action::Serve
                    } else {
                        Action::Idle
                    };
                }
            };

        // Serve-or-move at the traversed configuration `current` (Eq. 11).
        if queue_lens[current] > 0 {
            let d = dist[current][target];
            let r = dist[target][current];
            let round_trip = if r.is_finite() { d + r } else { d };
            // Both guards are compared in multiplied-through form, which
            // avoids a division and keeps the sign unambiguous.
            let stay = self.weights[current]
                * (ages[current] + self.beta * round_trip)
                * (1.0 + d + self.alpha * r.max(0.0));
            let go = self.weights[target] * (self.anticipated_delay(target, ages, queue_lens) + d);
            let mut serve = stay >= go;
            if !serve {
                let backlog_here = self.weights[current]
                    * queue_lens[current] as f64
                    * self.service_rates[current];
                serve = backlog_here >= self.backlog_threshold(current, target, queue_lens, dist);
            }
            // Defensive: a non-finite `d` cannot occur after choose_target,
            // but serving here would be the only safe action if it did.
            if serve || !d.is_finite() {
                return Action::Serve;
            }
        }

        let nh = next_hop[current][target].expect("strong connectivity guarantees a next hop");
        Action::MoveTo {
            target: nh,
            duration: self.direct.time(current, nh),
            lock: Some(target),
        }
    }
}
