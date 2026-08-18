//! Tian & Shone (2025) baseline policies: K-stop and (K from L)-stop
//! index heuristics, adapted to this simulator's directed, weighted
//! switchover graph.
//!
//! Reference: Tian and Shone, "Stochastic dynamic job scheduling with
//! interruptible setup and processing times: An approach based on
//! queueing control", arXiv:2509.06578v1, Sep 2025.
//!
//! Faithfulness rule for `kind = "tian"`: every queue is a "demand point"
//! (D = V), there are no separate intermediate stages with their own
//! queues, and **no transit service is allowed**. When the chosen route
//! from `c` to a demand point `s_1` passes through some other queue `u`,
//! the server walks through `u` without serving it, mirroring Tian &
//! Shone's setup-time semantics.
//!
//! `kind = "tian_transit"` deliberately relaxes that rule. It is the
//! *fair variant* of the evaluation: original Tian serves only its
//! selected stops, whereas DART may serve a configuration crossed on the
//! way, so the comparison would otherwise turn on feature access rather
//! than on decision logic. When Tian-T crosses `c` on the way to its
//! committed target it serves `c` while `w_c Q_c μ_c >= η ψ`, where `ψ`
//! is Tian's own reward rate for continuing and `η` controls how
//! reluctant the policy is to pause. A large `η` converges back to
//! unmodified Tian.
//!
//! Quantities (all evaluated at the current state x = (c, Q)):
//!
//! For a candidate sequence s = (s_1, ..., s_m) of distinct demand
//! points with s_0 = c:
//!
//!   T_j(x, s, t) =
//!       [ Q_{s_j}
//!         + λ_{s_j} · ( t
//!             + Σ_{k=1}^{j-1} [ d(s_{k-1}, s_k) + T_k(x,s,t) ]
//!             + d(s_{j-1}, s_j)
//!         )
//!       ]
//!       / [ μ_{s_j} − λ_{s_j} ]
//!
//!   R_j(x,s,t) = w_{s_j} · μ_{s_j} · T_j(x,s,t)
//!
//!   ψ(x,s,t) = Σ R_k / [ t + Σ ( d(s_{k-1},s_k) + T_k ) ]
//!
//!   φ_j(x,s,t) = Σ_{k=1}^j R_k
//!                / [ t + Σ_{k=1}^j ( d(s_{k-1},s_k) + T_k ) + d(s_j,c) ]
//!
//!   β_j(x,s,t) = Σ R_k / Σ T_k · ρ + w_c·μ_c·(1−ρ)   if c ∉ {s_1..s_j}
//!              = 0                                    otherwise
//!
//!   γ(x,s,t) = Σ R_k / Σ T_k · ρ
//!
//! Decision rules per case (`c` = current node):
//!
//! Case A (c is a demand point, Q_c > 0): collect sequences for which
//!   ∂ψ/∂t|_{t=0} ≤ 0  AND  φ_j ≥ β_j for every prefix j. If empty,
//!   Serve. Else move one edge toward s*_1, the first stop of the best
//!   sequence by ψ.
//!
//! Case B/C (c is empty demand point or intermediate): require derivative
//!   condition for membership in σ_2, and additionally ψ ≥ γ at x and at
//!   y = state with server moved to s_1 for membership in σ_1. σ = σ_1
//!   if non-empty else σ_2. If σ empty, Idle. Otherwise move one edge
//!   toward s*_1.
//!
//! Replanning is performed at every decision epoch — sequences are
//! never "committed". This mirrors the receding-horizon interpretation
//! of K-stop.

// The numeric guards below are written `!(a > b)` rather than `a <= b`
// on purpose: the negated form also rejects NaN, which is exactly the
// behaviour a guard against a degenerate denominator or score needs.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::{Action, DecisionContext, Router, SystemView};
use crate::dist::{DistMatrix, NextHopMatrix};
use crate::topology::Topology;

/// Numerical tolerance used by the derivative check `ψ(ε) ≤ ψ(0)+tol`.
const DERIVATIVE_TOL: f64 = 1e-10;

/// Tian K-stop baseline router. With `with_transit_service`, also
/// implements Tian K-stop + guarded transit.
pub struct TianRouter {
    pub n: usize,
    pub weights: Vec<f64>,
    pub service_rates: Vec<f64>,
    pub arrival_rates: Vec<f64>,
    pub dist: DistMatrix,
    pub next_hop: NextHopMatrix,
    pub direct: Topology,
    /// Lookahead horizon. K = 1 reduces to the one-stop variant.
    pub k: usize,
    /// Finite-difference epsilon for the derivative test.
    pub epsilon: f64,
    /// Hard cap on candidate-sequence count per decision (defence in
    /// depth against accidental combinatorial explosions). When the
    /// generator would exceed this, the router still returns *some*
    /// safe action by truncating; a debug log line is recorded if
    /// detailed tracing is added later.
    pub max_sequences: usize,
    /// Cached system traffic intensity ρ = Σ_i λ_i / μ_i.
    rho: f64,
    /// If true, this router keeps the chosen first stop locked as the
    /// final target and may serve non-target queues encountered on the
    /// path when their local marginal value clears `transit_eta`.
    transit_service: bool,
    /// Transit-service guard. Larger values make intermediate service
    /// catches rarer. Used only when `transit_service` is true.
    transit_eta: f64,
}

/// Lookahead horizon `K` when the configuration does not set one.
pub const DEFAULT_K: usize = 1;
/// Finite-difference step for the `ψ`-derivative test.
pub const DEFAULT_EPSILON: f64 = 1e-6;
/// Cap on candidate sequences enumerated per decision.
pub const DEFAULT_MAX_SEQUENCES: usize = 100_000;
/// Transit reluctance `η` of the transit-serving variant.
pub const DEFAULT_ETA: f64 = 1.0;

impl TianRouter {
    pub fn new(system: SystemView, k: usize, epsilon: f64, max_sequences: usize) -> Self {
        assert!(k >= 1, "Tian K must be >= 1");
        let n = system.n;
        let rho = (0..n)
            .map(|i| system.arrival_rates[i] / system.service_rates[i])
            .sum::<f64>();
        Self {
            n,
            weights: system.weights,
            service_rates: system.service_rates,
            arrival_rates: system.arrival_rates,
            dist: system.paths.dist,
            next_hop: system.paths.next_hop,
            direct: system.topology,
            k,
            epsilon: if epsilon > 0.0 { epsilon } else { 1e-6 },
            max_sequences,
            rho,
            transit_service: false,
            transit_eta: f64::INFINITY,
        }
    }

    pub fn with_transit_service(mut self, eta: f64) -> Self {
        self.transit_service = true;
        self.transit_eta = if eta.is_finite() && eta > 0.0 {
            eta
        } else {
            1.0
        };
        self
    }

    /// Recursively evaluate T_j and the running cumulative arrival-time
    /// at stop j (the bracketed inner sum used in T_{j+1}'s numerator).
    ///
    /// Returns Some((T_vec, R_vec, sum_d_plus_T)) on success, or None if
    /// the sequence is infeasible (any λ ≥ μ along the way, or any
    /// distance is infinite).
    fn eval_sequence(
        &self,
        current: usize,
        q: &[usize],
        dist: &DistMatrix,
        s: &[usize],
        t: f64,
    ) -> Option<SeqEval> {
        let m = s.len();
        let mut t_vals = Vec::with_capacity(m);
        let mut r_vals = Vec::with_capacity(m);
        let mut d_vals = Vec::with_capacity(m);
        let mut prev = current;
        // running_sum tracks Σ_{k=1}^{j-1} ( d(s_{k-1},s_k) + T_k(x,s,t) )
        // at the moment we are about to compute stop j.
        let mut running_sum = 0.0_f64;
        for &sj in s {
            let dseg = dist[prev][sj];
            if !dseg.is_finite() {
                return None;
            }
            d_vals.push(dseg);
            let lam = self.arrival_rates[sj];
            let mu = self.service_rates[sj];
            if mu - lam <= 0.0 {
                return None;
            }
            let xj = q[sj] as f64;
            // Per the definition:
            // T_j = [ Q_{s_j} + λ ( t + running_sum + d(s_{j-1}, s_j) ) ]
            //       / (μ - λ)
            let tj = (xj + lam * (t + running_sum + dseg)) / (mu - lam);
            let rj = self.weights[sj] * mu * tj;
            t_vals.push(tj);
            r_vals.push(rj);
            running_sum += dseg + tj;
            prev = sj;
        }
        Some(SeqEval {
            t_vals,
            r_vals,
            d_vals,
        })
    }

    /// ψ(x, s, t) — average reward per unit time over the sequence.
    fn psi_from_eval(&self, eval: &SeqEval, t: f64) -> f64 {
        let r_sum: f64 = eval.r_vals.iter().sum();
        let dt_sum: f64 = eval
            .d_vals
            .iter()
            .zip(eval.t_vals.iter())
            .map(|(d, tj)| d + tj)
            .sum();
        let denom = t + dt_sum;
        if denom <= 0.0 {
            return f64::NEG_INFINITY;
        }
        r_sum / denom
    }

    /// φ_j(x, s, 0) — reward per unit time for the (s_1..s_j) prefix
    /// followed by a return to `current`.
    fn phi_prefix(
        &self,
        current: usize,
        eval: &SeqEval,
        s: &[usize],
        j: usize,
        dist: &DistMatrix,
    ) -> f64 {
        let r_sum: f64 = eval.r_vals[..j].iter().sum();
        let dt_sum: f64 = eval.d_vals[..j]
            .iter()
            .zip(eval.t_vals[..j].iter())
            .map(|(d, tj)| d + tj)
            .sum();
        let return_d = dist[s[j - 1]][current];
        if !return_d.is_finite() {
            return f64::NEG_INFINITY;
        }
        let denom = dt_sum + return_d;
        if denom <= 0.0 {
            return f64::NEG_INFINITY;
        }
        r_sum / denom
    }

    /// β_j(x, s, 0). Returns 0 if `current` ∈ {s_1..s_j} (the server
    /// is scheduled to revisit its starting point, so the term drops
    /// out of the eligibility test).
    fn beta_prefix(&self, current: usize, eval: &SeqEval, s: &[usize], j: usize) -> f64 {
        if s[..j].contains(&current) {
            return 0.0;
        }
        let r_sum: f64 = eval.r_vals[..j].iter().sum();
        let t_sum: f64 = eval.t_vals[..j].iter().sum();
        let stay_value = self.weights[current] * self.service_rates[current] * (1.0 - self.rho);
        if t_sum <= 0.0 {
            // No useful service planned — we should never leave on
            // these terms (return +∞ blocks eligibility).
            return f64::INFINITY;
        }
        (r_sum / t_sum) * self.rho + stay_value
    }

    /// γ(x, s, 0).
    fn gamma_full(&self, eval: &SeqEval) -> f64 {
        let r_sum: f64 = eval.r_vals.iter().sum();
        let t_sum: f64 = eval.t_vals.iter().sum();
        if t_sum <= 0.0 {
            return f64::INFINITY;
        }
        (r_sum / t_sum) * self.rho
    }
}

#[derive(Debug)]
struct SeqEval {
    t_vals: Vec<f64>,
    r_vals: Vec<f64>,
    /// d_vals[j] = d(s_{j-1}, s_j), with s_0 = current.
    d_vals: Vec<f64>,
}

impl Router for TianRouter {
    fn decide(&self, current: usize, queue_lens: &[usize], ctx: DecisionContext<'_>) -> Action {
        let dist: &DistMatrix = &self.dist;
        let next_hop: &NextHopMatrix = &self.next_hop;

        if self.transit_service {
            if let Some(target) = ctx.locked_target {
                if current == target && queue_lens[current] > 0 {
                    return Action::Serve;
                }
            }
        }

        // Build the candidate domain D' over which sequences are
        // enumerated. The current queue may appear later in a sequence;
        // only the first stop is forbidden to equal `current`.
        let domain = self.build_domain(current, queue_lens, dist);
        if domain.is_empty() {
            // No demand points reachable / non-trivial. Fall back to
            // serve-or-idle locally.
            return if queue_lens[current] > 0 {
                Action::Serve
            } else {
                Action::Idle
            };
        }

        // Enumerate sequences and run the case-A or case-B/C logic.
        let case_a = queue_lens[current] > 0;

        // Track best (action, (psi, length, lex tiebreak vec)) pairs
        // for σ_1 (high) and σ_2 (low) under case B/C, or simply the
        // single σ under case A.
        let mut best_low: Option<Best> = None;
        let mut best_high: Option<Best> = None;
        let mut count = 0usize;

        // Walk all distinct sequences of length 1..=K over `domain`.
        // We enumerate via permutations: first stop iterates over
        // domain; recursive nested loops manage prefix uniqueness.
        let k_eff = self.k.min(domain.len());
        let mut stack: Vec<usize> = Vec::with_capacity(k_eff);
        let mut used = vec![false; self.n];
        // The enumerator enforces Tian's s_1 != current rule while
        // still allowing current to appear at later stops.
        self.enumerate_sequences(
            current,
            queue_lens,
            dist,
            &domain,
            k_eff,
            &mut stack,
            &mut used,
            &mut count,
            self.max_sequences,
            case_a,
            &mut best_low,
            &mut best_high,
        );

        // Apply the case-A vs case-B/C decision rule.
        let chosen = if case_a {
            best_low // under case A there is only one σ
        } else if best_high.is_some() {
            best_high
        } else {
            best_low
        };

        let chosen = match chosen {
            Some(b) => b,
            None => {
                // No eligible sequence.
                return if case_a {
                    // Stay and serve current queue (Tian step 1(c)).
                    Action::Serve
                } else if queue_lens[current] > 0 {
                    // Empty/intermediate by case but a queue lives at
                    // `current` with work — defensive fallback rather
                    // than idling on backlog. Should be rare given
                    // sequence-of-length-one always satisfies
                    // ∂ψ/∂t|_0 ≤ 0, but we keep it safe.
                    Action::Serve
                } else {
                    Action::Idle
                };
            }
        };

        // Optional guarded transit catch: if we are on the way to a
        // different locked target and the local queue is valuable
        // enough, serve here; otherwise keep following the newly chosen
        // Tian first stop. Baseline Tian has this disabled.
        if self.transit_service {
            if let Some(target) = ctx.locked_target {
                if target != current && queue_lens[current] > 0 {
                    let local = self.weights[current]
                        * queue_lens[current] as f64
                        * self.service_rates[current];
                    if local >= self.transit_eta * chosen.psi {
                        return Action::Serve;
                    }
                }
            }
        }

        // Move one edge along the shortest path toward s*_1.
        let target = chosen.first_stop;
        if target == current {
            // Defensive: can only happen if numerical noise selected
            // an invalid sequence. Stay.
            return if queue_lens[current] > 0 {
                Action::Serve
            } else {
                Action::Idle
            };
        }
        let nh = match next_hop[current][target] {
            Some(h) => h,
            None => {
                // Should be unreachable on a strongly-connected graph;
                // fall back safely.
                return if queue_lens[current] > 0 {
                    Action::Serve
                } else {
                    Action::Idle
                };
            }
        };
        let duration = self.direct.time(current, nh);
        Action::MoveTo {
            target: nh,
            // Tian baseline carries no transit-service notion: we do
            // not lock the *final* target into ctx so the dispatcher
            // never enters its "arrived at lock" fast-path. Setting
            // `lock = Some(nh)` (the next hop) means the lock fires
            // exactly when we arrive at the immediate neighbour, at
            // which point we re-plan from scratch — which is the
            // intended receding-horizon behaviour.
            duration,
            lock: Some(if self.transit_service { target } else { nh }),
        }
    }
}

/// Per-decision aggregate for the best eligible sequence so far.
#[derive(Debug, Clone)]
struct Best {
    psi: f64,
    /// Lex-tiebreak vector: the sequence itself, smallest-first.
    sequence: Vec<usize>,
    first_stop: usize,
}

impl TianRouter {
    /// Build the candidate domain D' (queues over which sequences are
    /// enumerated): all queues reachable from `current`, including
    /// `current` itself so it can appear after the first stop.
    fn build_domain(&self, current: usize, _q: &[usize], dist: &DistMatrix) -> Vec<usize> {
        (0..self.n)
            .filter(|&j| j == current || dist[current][j].is_finite())
            .collect()
    }

    /// Recursively enumerate distinct sequences of length 1..=K over
    /// `domain`, scoring each one against the case-A or case-B/C
    /// eligibility rules and updating the best-so-far slots.
    #[allow(clippy::too_many_arguments)]
    fn enumerate_sequences(
        &self,
        current: usize,
        q: &[usize],
        dist: &DistMatrix,
        domain: &[usize],
        k_max: usize,
        stack: &mut Vec<usize>,
        used: &mut [bool],
        count: &mut usize,
        cap: usize,
        case_a: bool,
        best_low: &mut Option<Best>,
        best_high: &mut Option<Best>,
    ) {
        if !stack.is_empty() {
            // Score this sequence.
            self.try_sequence(current, q, dist, stack, case_a, best_low, best_high, count);
            if *count >= cap {
                return;
            }
        }
        if stack.len() == k_max {
            return;
        }
        for &j in domain {
            if used[j] {
                continue;
            }
            if stack.is_empty() && j == current {
                continue;
            }
            used[j] = true;
            stack.push(j);
            self.enumerate_sequences(
                current, q, dist, domain, k_max, stack, used, count, cap, case_a, best_low,
                best_high,
            );
            stack.pop();
            used[j] = false;
            if *count >= cap {
                return;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_sequence(
        &self,
        current: usize,
        q: &[usize],
        dist: &DistMatrix,
        s: &[usize],
        case_a: bool,
        best_low: &mut Option<Best>,
        best_high: &mut Option<Best>,
        count: &mut usize,
    ) {
        *count += 1;
        // First element ≠ current is required by both cases; current
        // may still appear later in the sequence.
        debug_assert!(!s.is_empty());
        debug_assert_ne!(s[0], current);

        let eval = match self.eval_sequence(current, q, dist, s, 0.0) {
            Some(e) => e,
            None => return, // infeasible (λ ≥ μ somewhere or unreachable)
        };
        let psi0 = self.psi_from_eval(&eval, 0.0);
        if !psi0.is_finite() {
            return;
        }

        // Derivative test: ψ(ε) ≤ ψ(0) + tol.
        let eval_eps = match self.eval_sequence(current, q, dist, s, self.epsilon) {
            Some(e) => e,
            None => return,
        };
        let psi_eps = self.psi_from_eval(&eval_eps, self.epsilon);
        if !(psi_eps <= psi0 + DERIVATIVE_TOL) {
            return;
        }

        if case_a {
            // φ_j ≥ β_j for every prefix.
            for j in 1..=s.len() {
                let phi_j = self.phi_prefix(current, &eval, s, j, dist);
                let beta_j = self.beta_prefix(current, &eval, s, j);
                if !(phi_j + DERIVATIVE_TOL >= beta_j) {
                    return;
                }
            }
            update_best(best_low, psi0, s);
        } else {
            // Case B/C. σ_2 always; σ_1 if ψ(x,s,0) ≥ γ(x,s,0) (and
            // additionally ψ(y,s,0) ≥ γ(y,s,0) for |s| ≥ 2 with y the
            // state after teleporting current to s_1).
            update_best(best_low, psi0, s);
            let psi_x = psi0;
            let gamma_x = self.gamma_full(&eval);
            if !(psi_x + DERIVATIVE_TOL >= gamma_x) {
                return;
            }
            if s.len() >= 2 {
                // Build state y: server at s_1, queues unchanged.
                let y_current = s[0];
                let eval_y = match self.eval_sequence(y_current, q, dist, s, 0.0) {
                    Some(e) => e,
                    None => return,
                };
                let psi_y = self.psi_from_eval(&eval_y, 0.0);
                let gamma_y = self.gamma_full(&eval_y);
                if !(psi_y + DERIVATIVE_TOL >= gamma_y) {
                    return;
                }
            }
            update_best(best_high, psi0, s);
        }
    }
}

fn update_best(slot: &mut Option<Best>, psi: f64, s: &[usize]) {
    let candidate = Best {
        psi,
        sequence: s.to_vec(),
        first_stop: s[0],
    };
    match slot {
        None => *slot = Some(candidate),
        Some(existing) => {
            // Replace iff candidate strictly beats existing on (psi,
            // lex-smaller sequence).
            if candidate.psi > existing.psi
                || (candidate.psi == existing.psi
                    && candidate.sequence.as_slice() < existing.sequence.as_slice())
            {
                *existing = candidate;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::all_pairs_shortest_paths;
    use crate::topology::{SwitchoverDistribution, Topology};

    fn two_node_router() -> TianRouter {
        let sw = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let topo = Topology::from_matrix_with(&sw, SwitchoverDistribution::Deterministic);
        let paths = all_pairs_shortest_paths(&topo);
        TianRouter::new(
            SystemView::new(vec![1.0, 1.0], vec![1.0, 1.0], vec![0.1, 0.1], paths, topo),
            2,
            1e-6,
            100,
        )
    }

    #[test]
    fn full_k_stop_allows_current_after_first_stop() {
        let router = two_node_router();
        let q = vec![3_usize, 3];
        let domain = router.build_domain(0, &q, &router.dist);
        assert!(
            domain.contains(&0),
            "current queue must remain in the domain for later sequence positions"
        );
        assert!(domain.contains(&1));

        let mut best_low = None;
        let mut best_high = None;
        let mut stack = Vec::new();
        let mut used = vec![false; router.n];
        let mut count = 0usize;
        router.enumerate_sequences(
            0,
            &q,
            &router.dist,
            &domain,
            router.k.min(domain.len()),
            &mut stack,
            &mut used,
            &mut count,
            router.max_sequences,
            true,
            &mut best_low,
            &mut best_high,
        );

        assert_eq!(
            count, 2,
            "with domain {{0,1}} and current 0, Tian should score [1] and [1,0]"
        );
    }
}
