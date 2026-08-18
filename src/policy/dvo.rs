//! Duenyas & Van Oyen (1996) reward-rate setup-time scheduling baseline,
//! adapted to this simulator's directed weighted switchover graph
//! (graph-adapted form only).
//!
//! Reference: Duenyas, I. and Van Oyen, M. P., "Heuristic scheduling of
//! parallel heterogeneous queues with set-ups," Management Science 42(6),
//! 1996, 814-829.
//!
//! Setup time from `i` to `j` is the directed shortest-path distance
//! `d(i, j)`, and the return setup uses `d(j, i)`. Movement is
//! non-interruptible; no transit service.
//!
//! Quantities (current location `i`, target `j`, queue lengths `Q`):
//!
//!   p_k     = w_k · μ_k                                         (priority)
//!   ρ       = Σ_k λ_k / μ_k                                     (load)
//!   S_{ij}  = d(i, j)                                           (setup mean)
//!   β_j(i)  = w_j · μ_j · (Q_j + λ_j · S_{ij})
//!             / (Q_j + μ_j · S_{ij} + (μ_j − λ_j) · S_{ji})     (switching index)
//!   η_j(i)  = w_j · μ_j · (Q_j + λ_j · S_{ij})
//!             / (Q_j + μ_j · S_{ij})                            (idling-rule index)
//!
//! Decision rules:
//!
//!   * `Q_i > 0`: for every `j` with `p_j > p_i`, the eligibility test is
//!     `β_j(i) ≥ p_j · ρ + p_i · (1 − ρ)`. The eligible candidate with the
//!     largest `β_j(i)` becomes the new target; if none is eligible, the
//!     server serves one more job at `i`.
//!   * `Q_i = 0` (idling rule): collect `σ = { j ≠ i : η_j(i) ≥ p_j · ρ }`.
//!     If `σ` is non-empty the candidate is `argmax_{j∈σ} η_j(i)`, and
//!     otherwise `argmax_{j≠i} η_j(i)`. Switch to that candidate `k` iff
//!     `Q_k > λ_k · S_{ki}`; otherwise idle.
//!
//! Numerical safety:
//!   * Targets `j` with `μ_j ≤ λ_j` or non-finite `S_{ij}` / `S_{ji}`
//!     are skipped. Config validation already requires `μ_i > λ_i` for
//!     every queue under DVO, so this is defence-in-depth.
//!   * Denominators are checked to be strictly positive.
//!
//! Tie-breaking: largest index, smallest queue id on ties. Deterministic.

// The numeric guards below are written `!(a > b)` rather than `a <= b`
// on purpose: the negated form also rejects NaN, which is exactly the
// behaviour a guard against a degenerate denominator or score needs.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::sync::Mutex;

use super::{Action, DecisionContext, Router, SystemView};
use crate::dist::{DistMatrix, NextHopMatrix};
use crate::topology::Topology;

/// Per-router mutable state. The `Router` trait demands `Send + Sync`
/// and shared `&self`, so we hide the mutation behind a `Mutex`.
#[derive(Debug, Clone, Copy, Default)]
struct DvoState {
    /// Effective lock the router believed was active on the previous
    /// `decide()` call. `None` until the very first call (see
    /// `first_call`). Used to detect new arrivals: when the simulator
    /// reports a different `locked_target`, we know setup just
    /// completed.
    last_effective_lock: Option<usize>,
    /// True iff at least one `Serve` action has been emitted at the
    /// queue identified by `last_effective_lock` since that lock was
    /// established. Reset whenever the lock changes.
    served_since_arrival: bool,
    /// `true` until the first `decide()` call returns. The first call
    /// is treated as "the server has just been configured at its
    /// starting location": the must-serve-on-arrival rule fires there
    /// too, mirroring the paper's assumption that node 1 is set up
    /// before t=0.
    first_call: bool,
}

/// Duenyas-Van Oyen reward-rate scheduling router (graph-adapted).
pub struct DvoRouter {
    pub n: usize,
    pub weights: Vec<f64>,
    pub service_rates: Vec<f64>,
    pub arrival_rates: Vec<f64>,
    pub dist: DistMatrix,
    pub next_hop: NextHopMatrix,
    pub direct: Topology,
    /// p_k = w_k · μ_k. Cached at construction.
    priorities: Vec<f64>,
    /// ρ = Σ_k λ_k / μ_k. Cached at construction.
    rho: f64,
    /// Mutable state. See `DvoState`.
    state: Mutex<DvoState>,
}

impl DvoRouter {
    pub fn new(system: SystemView) -> Self {
        let priorities: Vec<f64> = system
            .weights
            .iter()
            .zip(system.service_rates.iter())
            .map(|(&w, &mu)| w * mu)
            .collect();
        let rho: f64 = system
            .arrival_rates
            .iter()
            .zip(system.service_rates.iter())
            .map(|(&lam, &mu)| lam / mu)
            .sum();
        Self {
            n: system.n,
            weights: system.weights,
            service_rates: system.service_rates,
            arrival_rates: system.arrival_rates,
            dist: system.paths.dist,
            next_hop: system.paths.next_hop,
            direct: system.topology,
            priorities,
            rho,
            state: Mutex::new(DvoState {
                last_effective_lock: None,
                served_since_arrival: false,
                first_call: true,
            }),
        }
    }

    /// Setup-time mean from `i` to `j`: directed shortest-path distance.
    /// Self-loops are 0; non-edges or unreachable destinations are
    /// `f64::INFINITY`.
    pub fn setup_time(&self, dist: &DistMatrix, i: usize, j: usize) -> f64 {
        if i == j {
            return 0.0;
        }
        dist[i][j]
    }

    /// β_j(i): switching index when the current queue is non-empty.
    /// Returns `None` if `j` is infeasible (unstable, unreachable, or
    /// non-positive denominator).
    pub fn beta(&self, dist: &DistMatrix, q: &[usize], i: usize, j: usize) -> Option<f64> {
        if i == j {
            return None;
        }
        let s_ij = self.setup_time(dist, i, j);
        let s_ji = self.setup_time(dist, j, i);
        if !s_ij.is_finite() || !s_ji.is_finite() {
            return None;
        }
        let mu_j = self.service_rates[j];
        let lam_j = self.arrival_rates[j];
        if mu_j <= lam_j {
            return None;
        }
        let xj = q[j] as f64;
        let denom = xj + mu_j * s_ij + (mu_j - lam_j) * s_ji;
        if !(denom > 0.0) {
            return None;
        }
        Some(self.priorities[j] * (xj + lam_j * s_ij) / denom)
    }

    /// η_j(i): index for the empty-current-queue idling rule.
    pub fn eta(&self, dist: &DistMatrix, q: &[usize], i: usize, j: usize) -> Option<f64> {
        if i == j {
            return None;
        }
        let s_ij = self.setup_time(dist, i, j);
        if !s_ij.is_finite() {
            return None;
        }
        let mu_j = self.service_rates[j];
        let lam_j = self.arrival_rates[j];
        let xj = q[j] as f64;
        let denom = xj + mu_j * s_ij;
        if !(denom > 0.0) {
            return None;
        }
        Some(self.priorities[j] * (xj + lam_j * s_ij) / denom)
    }

    /// Eligibility threshold for the switching rule:
    /// `β_j(i) ≥ p_j · ρ + p_i · (1 − ρ)`.
    fn switch_threshold(&self, i: usize, j: usize) -> f64 {
        self.priorities[j] * self.rho + self.priorities[i] * (1.0 - self.rho)
    }

    /// Eligibility threshold for the idling rule's high-priority set:
    /// `η_j(i) ≥ p_j · ρ`.
    fn idle_high_threshold(&self, j: usize) -> f64 {
        self.priorities[j] * self.rho
    }

    /// Best higher-priority candidate `j` whose β_j passes the
    /// switching threshold from current `i`. Returns `None` when no
    /// candidate qualifies (the server should serve / stay).
    fn best_switch_candidate(&self, dist: &DistMatrix, q: &[usize], i: usize) -> Option<usize> {
        let p_i = self.priorities[i];
        let mut best: Option<(usize, f64)> = None;
        for j in 0..self.n {
            if j == i {
                continue;
            }
            if !(self.priorities[j] > p_i) {
                continue;
            }
            let beta_j = match self.beta(dist, q, i, j) {
                Some(b) => b,
                None => continue,
            };
            if beta_j < self.switch_threshold(i, j) {
                continue;
            }
            best = Some(match best {
                None => (j, beta_j),
                Some((bj, bb)) => {
                    if beta_j > bb || (beta_j == bb && j < bj) {
                        (j, beta_j)
                    } else {
                        (bj, bb)
                    }
                }
            });
        }
        best.map(|(j, _)| j)
    }

    /// Best candidate `j ≠ i` for the idling rule, applying the
    /// high-set / fallback logic. Returns `None` when no candidate is
    /// reachable / feasible at all.
    fn best_idle_candidate(&self, dist: &DistMatrix, q: &[usize], i: usize) -> Option<usize> {
        let mut high: Option<(usize, f64)> = None;
        let mut any: Option<(usize, f64)> = None;
        for j in 0..self.n {
            if j == i {
                continue;
            }
            let eta_j = match self.eta(dist, q, i, j) {
                Some(e) => e,
                None => continue,
            };
            any = Some(match any {
                None => (j, eta_j),
                Some((bj, bb)) => {
                    if eta_j > bb || (eta_j == bb && j < bj) {
                        (j, eta_j)
                    } else {
                        (bj, bb)
                    }
                }
            });
            if eta_j >= self.idle_high_threshold(j) {
                high = Some(match high {
                    None => (j, eta_j),
                    Some((bj, bb)) => {
                        if eta_j > bb || (eta_j == bb && j < bj) {
                            (j, eta_j)
                        } else {
                            (bj, bb)
                        }
                    }
                });
            }
        }
        high.or(any).map(|(j, _)| j)
    }

    /// Build a non-interruptible move toward `target`. The simulator
    /// walks the full shortest path in a single switchover, so DVO
    /// cannot re-decide en route.
    fn commit_move(&self, dist: &DistMatrix, current: usize, target: usize) -> Action {
        let d = dist[current][target];
        debug_assert!(
            d.is_finite(),
            "DVO: committed-move target {target} unreachable from {current}"
        );
        Action::MoveTo {
            target,
            duration: d,
            lock: Some(target),
        }
    }
}

impl Router for DvoRouter {
    fn decide(&self, current: usize, queue_lens: &[usize], ctx: DecisionContext<'_>) -> Action {
        let dist: &DistMatrix = &self.dist;

        // Stage 0. Non-interruptible transit: if the simulator reports a
        // lock that does not match our current location, continue moving
        // toward the lock target without re-decision.
        if let Some(t_lock) = ctx.locked_target {
            if t_lock != current {
                return self.commit_move(dist, current, t_lock);
            }
        }

        // Stage 1. Mutable-state bookkeeping.
        let mut state = self
            .state
            .lock()
            .expect("DVO state mutex poisoned by panic in another thread");

        // Bootstrap on the very first call: pretend the simulator
        // committed to the starting queue. This makes the must-serve
        // rule apply at startup, mirroring the paper's assumption that
        // queue 1 is set up before t=0.
        let effective_lock = match (ctx.locked_target, state.first_call) {
            (Some(t), _) => Some(t),
            (None, true) => Some(current),
            (None, false) => state.last_effective_lock,
        };
        state.first_call = false;

        // Detect a fresh commitment: any change in the effective lock
        // resets the "served since arrival" flag.
        if state.last_effective_lock != effective_lock {
            state.last_effective_lock = effective_lock;
            state.served_since_arrival = false;
        }

        // Stage 2. Must-serve-on-arrival.
        if let Some(t) = effective_lock {
            if t == current && queue_lens[current] > 0 && !state.served_since_arrival {
                state.served_since_arrival = true;
                return Action::Serve;
            }
        }

        // Stage 3. Main decision: switching (Q_i > 0) or idling (= 0).
        let q_i_empty = queue_lens[current] == 0;

        if !q_i_empty {
            // Switching rule.
            if let Some(target) = self.best_switch_candidate(dist, queue_lens, current) {
                state.last_effective_lock = Some(target);
                state.served_since_arrival = false;
                return self.commit_move(dist, current, target);
            }
            // No eligible target: stay and serve.
            state.served_since_arrival = true;
            return Action::Serve;
        }

        // Idling rule (Q_i == 0).
        let candidate = self.best_idle_candidate(dist, queue_lens, current);
        if let Some(k) = candidate {
            // Check the paper's strict threshold Q_k > λ_k · S_{ki}.
            let s_ki = self.setup_time(dist, k, current);
            let xk = queue_lens[k] as f64;
            let lam_k = self.arrival_rates[k];
            let switch_ok = s_ki.is_finite() && xk > lam_k * s_ki;
            if switch_ok {
                state.last_effective_lock = Some(k);
                state.served_since_arrival = false;
                return self.commit_move(dist, current, k);
            }
        }
        // Idle (or there is genuinely nothing reachable to serve).
        Action::Idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::all_pairs_shortest_paths;
    use crate::policy::DecisionContext;
    use crate::topology::{SwitchoverDistribution, Topology};

    fn unit_full_topology(n: usize, edge: f64) -> Topology {
        let mut sw = vec![vec![edge; n]; n];
        for (i, row) in sw.iter_mut().enumerate() {
            row[i] = 0.0;
        }
        Topology::from_matrix_with(&sw, SwitchoverDistribution::Deterministic)
    }

    fn router_for_test(
        weights: Vec<f64>,
        service_rates: Vec<f64>,
        arrival_rates: Vec<f64>,
        topo: Topology,
    ) -> DvoRouter {
        let paths = all_pairs_shortest_paths(&topo);
        DvoRouter::new(SystemView::new(
            weights,
            service_rates,
            arrival_rates,
            paths,
            topo,
        ))
    }

    /// Priority ordering (p_i = w_i · μ_i) is consistent — the router
    /// picks the highest-priority queue when distances are equal and
    /// several candidates pass the switching threshold.
    #[test]
    fn priority_breaks_ties_by_higher_p() {
        // Q1 has w·μ = 5, Q2 has w·μ = 3. Server at Q0 (priority 1).
        // From an empty Q0, both Q1 and Q2 are full; we should switch
        // to Q1 (higher priority) under the idling rule.
        let topo = unit_full_topology(3, 1.0);
        let r = router_for_test(
            vec![1.0, 5.0, 3.0],
            vec![1.0, 1.0, 1.0],
            vec![0.1, 0.1, 0.1],
            topo,
        );
        let q = [0usize, 4, 4];
        let action = r.decide(0, &q, DecisionContext::default());
        match action {
            Action::MoveTo { target, .. } => assert_eq!(target, 1),
            other => panic!("expected MoveTo to queue 1, got {other:?}"),
        }
    }

    /// Switching index β_j is computed exactly per the paper. Two queues
    /// with directed edges (no alternate routing), so the shortest-path
    /// distance equals the direct edge. Hand-evaluate β and check the
    /// implementation.
    #[test]
    fn beta_matches_hand_computation() {
        // Asymmetric topology with no third node to route through:
        // d(0,1) = 2 (direct), d(1,0) = 3 (direct).
        let sw = vec![vec![0.0, 2.0], vec![3.0, 0.0]];
        let topo = Topology::from_matrix_with(&sw, SwitchoverDistribution::Deterministic);
        let r = router_for_test(vec![1.0, 1.0], vec![2.0, 1.5], vec![0.5, 0.4], topo);
        let dist = &r.dist;
        let q = [0usize, 5];
        let beta_1 = r.beta(dist, &q, 0, 1).expect("β_1 should be finite");
        let expected = 1.0 * 1.5 * (5.0 + 0.4 * 2.0) / (5.0 + 1.5 * 2.0 + (1.5 - 0.4) * 3.0);
        assert!(
            (beta_1 - expected).abs() < 1e-12,
            "β_1 = {beta_1} ≠ expected {expected}"
        );
    }

    /// Switching index uses d(j, i) for the return-setup term.
    /// Construct a graph where d(0,1) ≠ d(1,0) and verify β changes
    /// accordingly.
    #[test]
    fn beta_uses_directed_return_distance() {
        let sw_a = vec![
            vec![0.0, 1.0, 1.0],
            vec![5.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let sw_b = vec![
            vec![0.0, 5.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let topo_a = Topology::from_matrix_with(&sw_a, SwitchoverDistribution::Deterministic);
        let topo_b = Topology::from_matrix_with(&sw_b, SwitchoverDistribution::Deterministic);
        let r_a = router_for_test(
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![0.1, 0.1, 0.1],
            topo_a,
        );
        let r_b = router_for_test(
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![0.1, 0.1, 0.1],
            topo_b,
        );
        let q = [0usize, 5, 0];
        let beta_a = r_a.beta(&r_a.dist, &q, 0, 1).expect("β");
        let beta_b = r_b.beta(&r_b.dist, &q, 0, 1).expect("β");
        assert!(
            (beta_a - beta_b).abs() > 1e-9,
            "β should differ between asymmetric topologies; got {beta_a} vs {beta_b}"
        );
    }

    /// Nonempty current queue with no eligible higher-priority candidate
    /// → action is Serve.
    #[test]
    fn nonempty_current_no_switch_serves() {
        let topo = unit_full_topology(2, 1.0);
        let r = router_for_test(vec![5.0, 2.0], vec![1.0, 1.0], vec![0.2, 0.2], topo);
        let q = [3usize, 7];
        let action = r.decide(0, &q, DecisionContext::default());
        assert!(
            matches!(action, Action::Serve),
            "expected Serve when no higher-priority queue exists; got {action:?}"
        );
    }

    /// Nonempty current queue with a higher-priority queue that has
    /// enough backlog to pass the switching threshold → action is
    /// MoveTo that queue.
    #[test]
    fn nonempty_current_eligible_switch_moves() {
        let topo = unit_full_topology(2, 0.1);
        let r = router_for_test(vec![5.0, 2.0], vec![1.0, 1.0], vec![0.1, 0.1], topo);
        // Bootstrap state so the first-call must-serve doesn't suppress
        // the switch.
        let q_warmup = [0usize, 5];
        let _ = r.decide(1, &q_warmup, DecisionContext::default());
        let q = [50usize, 5];
        let action = r.decide(1, &q, DecisionContext::default());
        match action {
            Action::MoveTo { target, lock, .. } => {
                assert_eq!(target, 0, "should switch to higher-priority Q0");
                assert_eq!(lock, Some(0), "lock should be the final target");
            }
            other => panic!("expected MoveTo to Q0; got {other:?}"),
        }
    }

    /// Must-serve-on-arrival. Right after the simulator commits a lock
    /// to queue 1 and the server arrives there, the router should Serve
    /// once even if a much higher-priority queue has plenty of work.
    #[test]
    fn must_serve_one_after_setup() {
        let topo = unit_full_topology(3, 0.1);
        let r = router_for_test(
            vec![1.0, 1.0, 5.0],
            vec![1.0, 1.0, 1.0],
            vec![0.1, 0.1, 0.1],
            topo,
        );
        let q = [0usize, 3, 100];
        let ctx = DecisionContext {
            locked_target: Some(1),
            ..Default::default()
        };
        let action = r.decide(1, &q, ctx);
        assert!(
            matches!(action, Action::Serve),
            "should serve at least once on arrival; got {action:?}"
        );
        let action2 = r.decide(1, &q, ctx);
        match action2 {
            Action::MoveTo { target, .. } => assert_eq!(target, 2),
            other => panic!(
                "after the must-serve fires once, the router should consider switching; got {other:?}"
            ),
        }
    }

    /// Empty current queue with no eligible high candidate passing
    /// Q_k > λ_k · S_{ki} → action is Idle.
    #[test]
    fn empty_current_below_threshold_idles() {
        let sw = vec![vec![0.0, 1.0], vec![10.0, 0.0]];
        let topo = Topology::from_matrix_with(&sw, SwitchoverDistribution::Deterministic);
        let r = router_for_test(vec![1.0, 1.0], vec![1.0, 1.0], vec![0.4, 0.4], topo);
        let q_warmup = [0usize, 0];
        let _ = r.decide(0, &q_warmup, DecisionContext::default());
        let q = [0usize, 1];
        let action = r.decide(0, &q, DecisionContext::default());
        assert!(
            matches!(action, Action::Idle),
            "expected Idle below threshold; got {action:?}"
        );
    }

    /// Empty current queue with enough backlog at the candidate → switch.
    #[test]
    fn empty_current_above_threshold_switches() {
        let sw = vec![vec![0.0, 1.0], vec![10.0, 0.0]];
        let topo = Topology::from_matrix_with(&sw, SwitchoverDistribution::Deterministic);
        let r = router_for_test(vec![1.0, 1.0], vec![1.0, 1.0], vec![0.4, 0.4], topo);
        let q_warmup = [0usize, 0];
        let _ = r.decide(0, &q_warmup, DecisionContext::default());
        let q = [0usize, 100];
        let action = r.decide(0, &q, DecisionContext::default());
        match action {
            Action::MoveTo { target, lock, .. } => {
                assert_eq!(target, 1);
                assert_eq!(lock, Some(1));
            }
            other => panic!("expected MoveTo Q1; got {other:?}"),
        }
    }

    /// The DVO paper uses a strict idling threshold, so exact equality
    /// must idle rather than switch.
    #[test]
    fn empty_current_at_threshold_idles() {
        let sw = vec![vec![0.0, 1.0], vec![10.0, 0.0]];
        let topo = Topology::from_matrix_with(&sw, SwitchoverDistribution::Deterministic);
        let r = router_for_test(vec![1.0, 1.0], vec![1.0, 1.0], vec![0.4, 0.4], topo);
        let q_warmup = [0usize, 0];
        let _ = r.decide(0, &q_warmup, DecisionContext::default());
        let q = [0usize, 4];
        let action = r.decide(0, &q, DecisionContext::default());
        assert!(
            matches!(action, Action::Idle),
            "expected Idle at the strict DVO threshold; got {action:?}"
        );
    }

    /// Non-interruptible movement. While we are in transit (current !=
    /// lock), the router must keep moving along the shortest path
    /// regardless of state changes.
    #[test]
    fn no_replanning_during_transit() {
        let sw = vec![
            vec![0.0, 1.0, -1.0],
            vec![1.0, 0.0, 1.0],
            vec![-1.0, 1.0, 0.0],
        ];
        let topo = Topology::from_matrix_with(&sw, SwitchoverDistribution::Deterministic);
        let r = router_for_test(
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![0.1, 0.1, 0.1],
            topo,
        );
        let q = [0usize, 1000, 5];
        let ctx = DecisionContext {
            locked_target: Some(2),
            ..Default::default()
        };
        let action = r.decide(1, &q, ctx);
        match action {
            Action::MoveTo { target, lock, .. } => {
                assert_eq!(target, 2);
                assert_eq!(lock, Some(2));
            }
            other => panic!("expected MoveTo to lock target Q2; got {other:?}"),
        }
    }

    /// Determinism — same state, same action.
    #[test]
    fn deterministic_decisions() {
        let topo = unit_full_topology(3, 1.0);
        let r1 = router_for_test(
            vec![1.0, 2.0, 3.0],
            vec![1.0, 1.0, 1.0],
            vec![0.1, 0.2, 0.3],
            topo.clone(),
        );
        let r2 = router_for_test(
            vec![1.0, 2.0, 3.0],
            vec![1.0, 1.0, 1.0],
            vec![0.1, 0.2, 0.3],
            topo,
        );
        let q = [4usize, 4, 4];
        let a1 = r1.decide(0, &q, DecisionContext::default());
        let a2 = r2.decide(0, &q, DecisionContext::default());
        assert_eq!(a1, a2, "same state should produce identical actions");
    }

    /// Numerical safety. λ very close to μ should not crash (just makes
    /// that target's β small / borderline).
    #[test]
    fn lambda_close_to_mu_does_not_crash() {
        let topo = unit_full_topology(2, 1.0);
        let r = router_for_test(vec![1.0, 1.0], vec![1.0, 1.0001], vec![0.1, 0.9999], topo);
        let q = [0usize, 5];
        let _ = r.decide(0, &q, DecisionContext::default());
    }
}
