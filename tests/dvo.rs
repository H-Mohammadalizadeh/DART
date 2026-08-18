//! End-to-end integration tests for the Duenyas-Van Oyen baseline.
//!
//! Unit tests in `src/dvo.rs` cover the formulas (β, η, thresholds,
//! tie-breaking, must-serve-on-arrival). These tests exercise the
//! router *through the full simulator* and verify the system-level
//! invariants we promised:
//!
//!   1. Movement is non-interruptible: once committed to target `j`,
//!      the server only emits a single switchover ending at `j`. The
//!      server cannot backtrack mid-trip and we should never observe
//!      "intermediate arrivals" away from the lock target.
//!   2. No transit service: the server never serves at a queue other
//!      than its commitment target.
//!   3. Throughput keeps up with arrivals over a long horizon
//!      (sanity: the policy is stable on a feasible workload).
//!   4. Variants dispatch correctly through the TOML registry.
//!
//! We rebuild the routers directly from `Config` via
//! `build_router_for_test` for the configuration tests, and run
//! `run_replication_traced` for the simulator-level tests so we can
//! inspect the full event stream.

#![allow(clippy::needless_range_loop)]

use dartsim::config::{Config, PolicyKind};
use dartsim::policy::{Action, DecisionContext};
use dartsim::simulation::{run_replication, run_replication_traced, TraceEvent};

fn three_queue_dvo_config(kind: &str) -> Config {
    // n=3, fully-connected, asymmetric travel times so the
    // graph-adapted variant differs from a uniform-distance baseline.
    // High-priority queue (Q0) has small λ and large w·μ; the others
    // are higher-load with lower priority.
    let toml = format!(
        r#"
[system]
n_queues = 3

[arrivals]
rates = [0.10, 0.30, 0.30]

[service]
rates = [1.0, 1.0, 1.0]

[priorities]
weights = [5.0, 1.0, 1.0]

[topology]
switchover_times = [
    [0.0, 1.0, 2.0],
    [2.0, 0.0, 1.0],
    [1.0, 2.0, 0.0],
]

[simulation]
horizon = 5000.0
warmup = 500.0
n_replications = 1
seed = 7

[policy]
kind = "{kind}"
"#
    );
    let cfg: Config = toml::from_str(&toml).expect("config parses");
    cfg.validate().expect("config validates");
    cfg
}

/// 1. The DVO variant parses, builds a router, and produces a decision
///    in a non-trivial state.
#[test]
fn dvo_variants_parse_and_decide() {
    let cfg = three_queue_dvo_config("dvo");
    assert_eq!(cfg.policy.kind, PolicyKind::Dvo);
    let topo = dartsim::topology::Topology::from_matrix(&cfg.topology.switchover_times);
    let router = dartsim::simulation::build_router_for_test(&cfg, &topo);
    let action = router.decide(0, &[0usize, 4, 0], DecisionContext::default());
    assert!(
        !matches!(action, Action::Serve),
        "DVO should not Serve at empty Q0; got {action:?}"
    );
}

/// 2. Per-queue stability is enforced: μ_i > λ_i must hold for every
///    queue, just like Tian's index requires.
#[test]
fn dvo_rejects_unstable_queue() {
    let toml = r#"
[system]
n_queues = 2

[arrivals]
rates = [1.5, 0.1]

[service]
rates = [1.0, 1.0]

[priorities]
weights = [1.0, 1.0]

[topology]
switchover_times = [[0.0, 1.0],[1.0, 0.0]]

[simulation]
horizon = 100.0
warmup = 0.0
n_replications = 1
seed = 1

[policy]
kind = "dvo"
"#;
    let cfg: Config = toml::from_str(toml).expect("toml parses");
    let err = cfg.validate().expect_err("should fail validation");
    assert!(
        err.contains("μ > λ") || err.contains("Duenyas-Van Oyen"),
        "expected stability-violation error, got: {err}"
    );
}

/// 3. Non-interruptible commitment: in a traced run, every committed
///    move ends with the server at exactly its lock target. We assert
///    the simpler-but-equivalent property: every `StartSwitchover`
///    event is paired with an `Arrived` event at the recorded `to`,
///    and no `Arrived` event ever lands at any node *other* than the
///    starting and ending queues — i.e., the simulator never wakes up
///    the router mid-path.
#[test]
fn dvo_does_not_replan_during_setup() {
    let cfg = three_queue_dvo_config("dvo");
    let (_stats, trace) = run_replication_traced(&cfg, 7);

    let mut last_switch_to: Option<usize> = None;
    for ev in &trace {
        match *ev {
            TraceEvent::StartSwitchover { from: _, to, .. } => {
                last_switch_to = Some(to);
            }
            TraceEvent::Arrived { queue, .. } => {
                if let Some(expected) = last_switch_to.take() {
                    assert_eq!(
                        queue, expected,
                        "arrived at {queue} but we last started a switchover toward {expected}; \
                         DVO must commit the entire path",
                    );
                }
            }
            _ => {}
        }
    }
}

/// 4. No transit service. With DVO, the server should never start
///    serving at a queue that is not its current commitment target.
///    Concretely: every `StartService { queue }` event must have a
///    matching most-recent `Arrived { queue }` (the server is now at
///    that queue, not in transit). The simulator already rejects
///    serving at the wrong queue, but we add a stronger property
///    here: the server should never serve while in a "Switching"
///    state — which is implicit because StartService follows arrival.
#[test]
fn dvo_only_serves_at_arrival_target() {
    let cfg = three_queue_dvo_config("dvo");
    let (_stats, trace) = run_replication_traced(&cfg, 11);

    // Track current location across the trace. The server starts at
    // queue 0 (default).
    let mut current: usize = 0;
    for ev in &trace {
        match *ev {
            TraceEvent::Started { queue } => current = queue,
            TraceEvent::Arrived { queue, .. } => current = queue,
            TraceEvent::StartService { queue, .. } => {
                assert_eq!(
                    queue, current,
                    "service started at queue {queue} but server is at {current}; \
                     DVO must not perform transit service",
                );
            }
            _ => {}
        }
    }
}

/// 5. Long-horizon throughput is feasible: with ρ ≪ 1, the policy
///    serves customers and does not stall. We don't assert tight
///    throughput-equals-rate (it depends on the warmup and horizon),
///    just that we observe substantial service activity at every
///    queue with a non-trivial λ.
#[test]
fn dvo_throughput_is_nontrivial() {
    let cfg = three_queue_dvo_config("dvo");
    let stats = run_replication(&cfg, 13);
    for i in 0..stats.served.len() {
        assert!(
            stats.served[i] > 100,
            "queue {i} served only {} customers in 5000 time units; expected >100",
            stats.served[i]
        );
    }
}

/// 6. Determinism: the same seed twice produces the same served counts.
#[test]
fn dvo_runs_are_seed_reproducible() {
    let cfg = three_queue_dvo_config("dvo");
    let s1 = run_replication(&cfg, 42);
    let s2 = run_replication(&cfg, 42);
    for i in 0..s1.served.len() {
        assert_eq!(
            s1.served[i], s2.served[i],
            "queue {i}: not reproducible ({} vs {})",
            s1.served[i], s2.served[i]
        );
    }
}
