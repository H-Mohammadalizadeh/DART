//! Every policy parses from its TOML key, builds a router, and makes a
//! sensible decision.
//!
//! Test state: three configurations on a fully connected unit graph, the
//! server at 0, queue lengths `[0, 5, 0]`. With 0 empty, the only correct
//! action is to move toward 1; `Serve` at an empty configuration and
//! `Idle` while work waits are both bugs.

#![allow(clippy::needless_range_loop)]

use dartsim::config::{Config, PolicyKind};
use dartsim::policy::{Action, DecisionContext};

fn make_cfg(policy_block: &str) -> Config {
    let toml_text = format!(
        r#"
[system]
n_queues = 3

[arrivals]
rates = [0.2, 0.2, 0.2]

[service]
rates = [1.0, 1.0, 1.0]

[priorities]
weights = [1.0, 1.0, 1.0]

[topology]
switchover_times = [
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
]

[simulation]
horizon = 10.0
warmup = 0.0
n_replications = 1
seed = 1

[policy]
{policy_block}
"#
    );
    let cfg: Config = toml::from_str(&toml_text).expect("toml should parse");
    cfg.validate().expect("config should validate");
    cfg
}

fn assert_targets_queue_1(cfg: &Config) {
    let topo = dartsim::topology::Topology::from_matrix(&cfg.topology.switchover_times);
    let router = dartsim::simulation::build_router_for_test(cfg, &topo);
    let name = cfg.policy.kind.name();
    match router.decide(0, &[0usize, 5, 0], DecisionContext::default()) {
        Action::MoveTo { target, .. } => assert_eq!(
            target, 1,
            "policy {name} should target queue 1 from current = 0 with Q = [0, 5, 0]"
        ),
        Action::Serve => panic!("policy {name} should not Serve at the empty current queue 0"),
        Action::Idle => panic!("policy {name} returned Idle while queue 1 has work"),
    }
}

#[test]
fn dart_parses_and_decides() {
    let cfg = make_cfg("kind = \"dart\"\nalpha = 0.5\nbeta = 6.0\n");
    assert_eq!(cfg.policy.kind, PolicyKind::Dart);
    assert_targets_queue_1(&cfg);
}

#[test]
fn tian_parses_and_decides() {
    let cfg = make_cfg("kind = \"tian\"\nk = 2\n");
    assert_eq!(cfg.policy.kind, PolicyKind::Tian);
    assert_targets_queue_1(&cfg);
}

#[test]
fn tian_transit_parses_and_decides() {
    let cfg = make_cfg("kind = \"tian_transit\"\nk = 2\neta = 1.0\n");
    assert_eq!(cfg.policy.kind, PolicyKind::TianTransit);
    assert_targets_queue_1(&cfg);
}

#[test]
fn dvo_parses_and_decides() {
    let cfg = make_cfg("kind = \"dvo\"\n");
    assert_eq!(cfg.policy.kind, PolicyKind::Dvo);
    assert_targets_queue_1(&cfg);
}

#[test]
fn every_policy_defaults_its_parameters() {
    // Only `kind` is mandatory: each policy must fill in its own defaults.
    for kind in ["dart", "tian", "tian_transit", "dvo"] {
        let cfg = make_cfg(&format!("kind = \"{kind}\"\n"));
        assert_targets_queue_1(&cfg);
    }
}

#[test]
fn unknown_policy_kind_is_rejected() {
    let toml_text = r#"
[system]
n_queues = 2

[arrivals]
rates = [0.1, 0.1]

[service]
rates = [1.0, 1.0]

[priorities]
weights = [1.0, 1.0]

[topology]
switchover_times = [[0.0, 1.0], [1.0, 0.0]]

[simulation]
horizon = 1.0
warmup = 0.0
n_replications = 1
seed = 1

[policy]
kind = "this_is_not_a_real_policy"
"#;
    let parsed: Result<Config, _> = toml::from_str(toml_text);
    assert!(
        parsed.is_err(),
        "an unknown policy kind must be rejected at parse time"
    );
}
