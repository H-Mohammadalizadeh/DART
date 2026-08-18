//! Decision-level tests for DART.
//!
//! DART ranks every nonempty configuration by the urgency
//! `U_ij = w_j (â_j + d) / (1 + d + α r)` of Eq. (8), where
//! `â_j = a_j + (Q_j − 1)^+ / (μ_j − λ_j)` is the anticipated delay of
//! Eq. (7). It commits to the maximizer, serves the configuration it is
//! standing on while either guard of Eq. (11) holds, and otherwise takes
//! one hop along the shortest path. The properties checked here:
//!
//!  1. Parses from `kind = "dart"` and returns Serve, MoveTo or Idle.
//!  2. Every nonempty configuration competes; there is no priority filter.
//!  3. A high-weight configuration attracts the server when ages are equal.
//!  4. A low-weight configuration with a very old head-of-line job can
//!     still win, so nothing starves.
//!  5. Commitment: a still-valid lock is kept across decisions.
//!  6. Arriving at the committed target with backlog serves before
//!     re-deciding.
//!  7. A lock whose target has drained is released and the target rechosen.
//!  8. The delay guard drains a high-weight configuration rather than
//!     abandoning it for a comparable target.
//!  9. Everything empty gives Idle; a stale lock at the current location
//!     does not trap the policy in a loop.
//! 10. The anticipated delay lets a near-saturated configuration win on
//!     backlog growth before its age has caught up.

#![allow(clippy::needless_range_loop)]

use dartsim::config::{Config, PolicyKind};
use dartsim::policy::{Action, DecisionContext};

fn fmt_vec(xs: &[f64]) -> String {
    format!(
        "[{}]",
        xs.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}
fn fmt_mat(rows: &[Vec<f64>]) -> String {
    let lines: Vec<String> = rows
        .iter()
        .map(|r| format!("    {},", fmt_vec(r)))
        .collect();
    format!("[\n{}\n]", lines.join("\n"))
}

fn cfg_with(
    policy_block: &str,
    n: usize,
    sw: &[Vec<f64>],
    rates: &[f64],
    service: &[f64],
    weights: &[f64],
) -> Config {
    let toml_text = format!(
        r#"
[system]
n_queues = {n}
[arrivals]
rates = {rates}
[service]
rates = {service}
[priorities]
weights = {weights}
[topology]
switchover_times = {sw}
[simulation]
horizon = 100.0
warmup = 0.0
n_replications = 1
seed = 1
[policy]
kind = "dart"
start = 0
{policy_block}
"#,
        n = n,
        rates = fmt_vec(rates),
        service = fmt_vec(service),
        weights = fmt_vec(weights),
        sw = fmt_mat(sw),
        policy_block = policy_block,
    );
    let cfg: Config = toml::from_str(&toml_text).expect("toml should parse");
    cfg.validate().expect("config should validate");
    cfg
}

/// One decision, with explicit head-of-line ages and arrival flag.
fn decide(
    cfg: &Config,
    current: usize,
    q: &[usize],
    ages: &[f64],
    locked: Option<usize>,
    just_arrived: bool,
) -> Action {
    let topo = dartsim::topology::Topology::from_matrix(&cfg.topology.switchover_times);
    let router = dartsim::simulation::build_router_for_test(cfg, &topo);
    let ctx = DecisionContext {
        locked_target: locked,
        ages: Some(ages),
        server_just_arrived: just_arrived,
    };
    router.decide(current, q, ctx)
}

fn line_sw(n: usize, edge: f64) -> Vec<Vec<f64>> {
    let mut sw = vec![vec![-1.0_f64; n]; n];
    for i in 0..n {
        sw[i][i] = 0.0;
    }
    for i in 0..n - 1 {
        sw[i][i + 1] = edge;
        sw[i + 1][i] = edge;
    }
    sw
}

/// Return discount off, delay guard at the evaluation setting.
const KNOBS: &str = "alpha = 0.0\nbeta = 6.0";

#[test]
fn parses_and_decides() {
    let sw = line_sw(4, 1.0);
    let cfg = cfg_with(
        KNOBS,
        4,
        &sw,
        &[0.2, 0.2, 0.2, 0.2],
        &[1.0; 4],
        &[10.0, 1.0, 1.0, 10.0],
    );
    assert_eq!(cfg.policy.kind, PolicyKind::Dart);
    let action = decide(&cfg, 0, &[0, 0, 0, 3], &[0.0, 0.0, 0.0, 5.0], None, false);
    assert!(
        matches!(action, Action::MoveTo { .. } | Action::Serve),
        "got {action:?}"
    );
}

#[test]
fn all_queues_compete_no_anchor_filter() {
    // q0 high-weight but empty; only low-weight q2 has backlog -> move to it.
    let sw = line_sw(3, 1.0);
    let cfg = cfg_with(
        KNOBS,
        3,
        &sw,
        &[0.1, 0.1, 0.1],
        &[1.0; 3],
        &[10.0, 1.0, 1.0],
    );
    let action = decide(&cfg, 0, &[0, 0, 5], &[0.0, 0.0, 30.0], None, false);
    match action {
        Action::MoveTo { target, lock, .. } => {
            assert_eq!(target, 1, "next hop toward q2 on a 0-1-2 line is q1");
            assert_eq!(lock, Some(2), "should lock onto q2");
        }
        other => panic!("expected MoveTo toward q2, got {other:?}"),
    }
}

#[test]
fn high_weight_attracts_when_ages_equal() {
    // From the middle, q0 (w=10) and q3 (w=1) both have an aged HoL; the
    // weighted-delay urgency should pick the high-weight q0.
    let sw = line_sw(4, 1.0);
    let cfg = cfg_with(KNOBS, 4, &sw, &[0.05; 4], &[1.0; 4], &[10.0, 1.0, 1.0, 1.0]);
    let action = decide(&cfg, 2, &[3, 0, 0, 3], &[20.0, 0.0, 0.0, 20.0], None, false);
    match action {
        Action::MoveTo { target, lock, .. } => {
            assert_eq!(target, 1, "next hop toward q0 from q2 is q1");
            assert_eq!(lock, Some(0), "high-weight q0 should win the destination");
        }
        other => panic!("expected MoveTo toward q0, got {other:?}"),
    }
}

#[test]
fn very_old_low_weight_hol_can_win() {
    // q0 (w=10) freshly served (age ~0); q3 (w=1) HoL starving for a very
    // long time. Weighted delay lets the low-weight queue win.
    let sw = line_sw(4, 1.0);
    let cfg = cfg_with(KNOBS, 4, &sw, &[0.05; 4], &[1.0; 4], &[10.0, 1.0, 1.0, 1.0]);
    let action = decide(
        &cfg,
        2,
        &[1, 0, 0, 4],
        &[0.1, 0.0, 0.0, 4000.0],
        None,
        false,
    );
    match action {
        Action::MoveTo { target, lock, .. } => {
            assert_eq!(target, 3, "next hop toward q3 from q2 is q3");
            assert_eq!(lock, Some(3), "a long-starved low-weight queue must win");
        }
        other => panic!("expected MoveTo toward q3, got {other:?}"),
    }
}

#[test]
fn keeps_locked_destination() {
    // Lock = q3 (non-empty). A small fresh backlog at q1 must not steal
    // the commitment.
    let sw = line_sw(4, 1.0);
    let cfg = cfg_with(KNOBS, 4, &sw, &[0.05; 4], &[1.0; 4], &[5.0, 1.0, 1.0, 5.0]);
    let action = decide(
        &cfg,
        0,
        &[0, 1, 0, 4],
        &[0.0, 1.0, 0.0, 50.0],
        Some(3),
        false,
    );
    match action {
        Action::MoveTo { lock, .. } => assert_eq!(lock, Some(3), "must keep the q3 commitment"),
        Action::Serve => {} // serving q0 is impossible (empty); only acceptable if it routed
        other => panic!("expected MoveTo keeping lock 3, got {other:?}"),
    }
}

#[test]
fn serve_on_arrival_at_locked_target() {
    // Just arrived at the locked target q3 with backlog -> Serve, even if
    // another queue looks momentarily urgent (commitment is honoured).
    let sw = line_sw(4, 1.0);
    let cfg = cfg_with(KNOBS, 4, &sw, &[0.05; 4], &[1.0; 4], &[10.0, 1.0, 1.0, 1.0]);
    let action = decide(
        &cfg,
        3,
        &[5, 0, 0, 2],
        &[100.0, 0.0, 0.0, 5.0],
        Some(3),
        true,
    );
    assert!(
        matches!(action, Action::Serve),
        "arrival at locked target must Serve, got {action:?}"
    );
}

#[test]
fn dropped_lock_when_target_emptied() {
    // Lock = q3 but q3 is now empty; the server should re-route to the
    // only nonempty queue (q1), not keep walking to an empty q3.
    let sw = line_sw(4, 1.0);
    let cfg = cfg_with(KNOBS, 4, &sw, &[0.05; 4], &[1.0; 4], &[5.0, 1.0, 1.0, 5.0]);
    let action = decide(
        &cfg,
        0,
        &[0, 3, 0, 0],
        &[0.0, 40.0, 0.0, 0.0],
        Some(3),
        false,
    );
    match action {
        Action::MoveTo { lock, .. } => assert_eq!(lock, Some(1), "should re-route to nonempty q1"),
        other => panic!("expected re-route MoveTo toward q1, got {other:?}"),
    }
}

#[test]
fn delay_guard_keeps_draining_high_weight() {
    // At a high-weight configuration with an aged head-of-line job, facing
    // a comparable distant target, the delay guard keeps the server
    // draining where it stands rather than chasing.
    let sw = line_sw(4, 1.0);
    let cfg = cfg_with(KNOBS, 4, &sw, &[0.05; 4], &[1.0; 4], &[10.0, 1.0, 1.0, 1.0]);
    let action = decide(&cfg, 0, &[5, 0, 0, 3], &[15.0, 0.0, 0.0, 15.0], None, false);
    assert!(
        matches!(action, Action::Serve),
        "should keep draining here, got {action:?}"
    );
}

#[test]
fn all_empty_idles() {
    let sw = line_sw(4, 1.0);
    let cfg = cfg_with(KNOBS, 4, &sw, &[0.1; 4], &[1.0; 4], &[10.0, 1.0, 1.0, 10.0]);
    let action = decide(&cfg, 1, &[0, 0, 0, 0], &[0.0; 4], None, false);
    assert!(
        matches!(action, Action::Idle),
        "all-empty must Idle, got {action:?}"
    );
}

#[test]
fn lock_at_current_only_candidate_serves() {
    // Lock points at current q0 (stale), q0 is the only nonempty queue,
    // not just-arrived: should Serve (drain it), not loop or move.
    let sw = line_sw(4, 1.0);
    let cfg = cfg_with(KNOBS, 4, &sw, &[0.05; 4], &[1.0; 4], &[10.0, 1.0, 1.0, 1.0]);
    let action = decide(
        &cfg,
        0,
        &[4, 0, 0, 0],
        &[12.0, 0.0, 0.0, 0.0],
        Some(0),
        false,
    );
    assert!(
        matches!(action, Action::Serve),
        "only-candidate current must Serve, got {action:?}"
    );
}

#[test]
fn anticipated_delay_favours_the_saturated_configuration() {
    // From q2, both q0 and q3 have the same head-of-line age and weight,
    // but q0 is near saturation (λ = 0.8 against 0.1) with a large
    // backlog. Dividing that backlog by the net rate μ − λ makes q0's
    // anticipated delay far larger, so it wins despite the tied ages.
    let sw = line_sw(4, 1.0);
    let cfg = cfg_with(
        KNOBS,
        4,
        &sw,
        &[0.8, 0.05, 0.05, 0.1],
        &[1.0; 4],
        &[1.0, 1.0, 1.0, 1.0],
    );
    // q0: big backlog (near saturation), q3: tiny backlog, equal HoL ages.
    let action = decide(&cfg, 2, &[40, 0, 0, 2], &[5.0, 0.0, 0.0, 5.0], None, false);
    match action {
        Action::MoveTo { target, lock, .. } => {
            assert_eq!(target, 1, "next hop toward q0 from q2 is q1");
            assert_eq!(
                lock,
                Some(0),
                "the anticipated delay should favour the saturated q0"
            );
        }
        other => panic!("expected MoveTo toward the saturated q0, got {other:?}"),
    }
}
