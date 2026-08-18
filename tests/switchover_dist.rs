//! Config-layer tests for the reconfiguration-time families.
//!
//! These are integration tests, so they do not link the library's RNG
//! dependencies; the statistical sampling tests (mean, CV, quantiles and
//! reproducibility of each family) live in a unit-test module inside
//! `src/topology.rs`. Covered here:
//!
//!  1. parsing the TOML shapes — a global family with its parameter, and
//!     per-edge `edge_overrides`;
//!  2. per-edge family resolution;
//!  3. rejection of missing, out-of-range and misplaced parameters;
//!  4. the mean-preservation property the evaluation rests on: swapping
//!     families changes the realized durations but not one routing
//!     decision;
//!  5. reproducibility under a heavy-tailed family.

use dartsim::config::Config;
use dartsim::dist::all_pairs_shortest_paths;
use dartsim::simulation::run_replication;
use dartsim::topology::{SwitchoverDistribution as Family, Topology};

/// A return-trap-shaped four-configuration config: the edge out of 0 is
/// short and the edge back into it is long. The caller supplies the rest
/// of the `[topology]` body and the `[policy]` body.
fn cfg_with(topology_body: &str, policy_body: &str) -> Result<Config, String> {
    let toml_text = format!(
        r#"
[system]
n_queues = 4
[arrivals]
rates = [0.75, 0.07, 0.07, 0.07]
[service]
rates = [1.0, 1.0, 1.0, 1.0]
[priorities]
weights = [10, 1, 1, 1]
[topology]
switchover_times = [
    [0.0, 0.2, -1.0, -1.0],
    [8.0, 0.0, 1.0, -1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [-1.0, -1.0, 1.0, 0.0],
]
{topology_body}
[simulation]
horizon = 2000.0
warmup = 200.0
n_replications = 2
seed = 7
[policy]
{policy_body}
"#
    );
    let cfg: Config = toml::from_str(&toml_text).map_err(|e| e.to_string())?;
    cfg.validate()?;
    Ok(cfg)
}

const DART: &str = "kind = \"dart\"\nstart = 0";

#[test]
fn parses_global_lognormal_and_pareto2() {
    let lognormal = cfg_with("distribution = \"lognormal\"\ndistribution_cv = 3.0", DART)
        .expect("lognormal config should validate");
    assert_eq!(
        lognormal.topology.resolve_global_dist().unwrap(),
        Family::Lognormal { cv: 3.0 }
    );
    let lomax = cfg_with("distribution = \"pareto2\"\ndistribution_alpha = 1.5", DART)
        .expect("pareto2 config should validate");
    assert_eq!(
        lomax.topology.resolve_global_dist().unwrap(),
        Family::ParetoII { alpha: 1.5 }
    );
}

#[test]
fn edge_overrides_resolve_per_edge() {
    // Global exponential, with the slow return edge (1,0) heavy-tailed.
    let body = "distribution = \"exponential\"\n\
                [[topology.edge_overrides]]\n\
                from = 1\nto = 0\nkind = \"pareto2\"\nalpha = 1.5";
    let cfg = cfg_with(body, DART).expect("override config should validate");
    let families = cfg.topology.resolved_edge_families(4).unwrap();
    assert_eq!(families[4], Family::ParetoII { alpha: 1.5 });
    assert_eq!(families[1], Family::Exponential);
    assert_eq!(families[2 * 4 + 3], Family::Exponential);
}

#[test]
fn rejects_bad_parameters_and_overrides() {
    // Lomax needs alpha > 1 for a finite mean.
    assert!(cfg_with("distribution = \"pareto2\"\ndistribution_alpha = 1.0", DART).is_err());
    // A coefficient of variation cannot be negative.
    assert!(cfg_with("distribution = \"lognormal\"\ndistribution_cv = -0.5", DART).is_err());
    // Neither parameterized family may be left without its parameter.
    assert!(cfg_with("distribution = \"lognormal\"", DART).is_err());
    assert!(cfg_with("distribution = \"pareto2\"", DART).is_err());
    // An override must name a real edge...
    let non_edge = "distribution = \"exponential\"\n\
                    [[topology.edge_overrides]]\n\
                    from = 0\nto = 2\nkind = \"exponential\"";
    assert!(cfg_with(non_edge, DART).is_err());
    // ...whose endpoints are in range.
    let out_of_range = "distribution = \"exponential\"\n\
                        [[topology.edge_overrides]]\n\
                        from = 0\nto = 9\nkind = \"exponential\"";
    assert!(cfg_with(out_of_range, DART).is_err());
    // DART's parameters must be finite and non-negative.
    assert!(cfg_with(
        "distribution = \"exponential\"",
        "kind = \"dart\"\nalpha = -1.0"
    )
    .is_err());
    assert!(cfg_with(
        "distribution = \"exponential\"",
        "kind = \"dart\"\nbeta = -1.0"
    )
    .is_err());
}

#[test]
fn deterministic_is_the_default() {
    // With no `distribution` key every edge takes exactly its matrix
    // value, and a zero-mean edge is then legal — it is only rejected
    // under a stochastic family.
    let cfg = cfg_with("", DART).expect("default (deterministic) config validates");
    assert_eq!(
        cfg.topology.resolve_global_dist().unwrap(),
        Family::Deterministic
    );
}

#[test]
fn family_swap_is_mean_preserving_and_leaves_routing_alone() {
    // The evaluation swaps families per edge at a matched mean. That is
    // only a fair manipulation if it moves no routing decision, which
    // holds because the shortest paths are computed on the means.
    let matrix = vec![
        vec![0.0, 6.0, 10.0],
        vec![6.0, 0.0, 5.0],
        vec![10.0, 5.0, 0.0],
    ];
    let uniform = Topology::from_matrix_with(&matrix, Family::Exponential);
    let mixed = Topology::from_matrix_with_overrides(
        &matrix,
        Family::Exponential,
        &[
            (0, 2, Family::ParetoII { alpha: 1.5 }),
            (1, 2, Family::Deterministic),
            (2, 0, Family::Lognormal { cv: 2.0 }),
        ],
    );
    for from in 0..3 {
        for to in 0..3 {
            assert_eq!(
                uniform.time(from, to),
                mixed.time(from, to),
                "edge ({from},{to}) must keep its mean under a family swap"
            );
        }
    }
    let before = all_pairs_shortest_paths(&uniform);
    let after = all_pairs_shortest_paths(&mixed);
    assert_eq!(before.dist, after.dist);
    assert_eq!(before.next_hop, after.next_hop);
}

#[test]
fn run_is_seed_reproducible_under_heavy_tails() {
    let cfg = cfg_with("distribution = \"pareto2\"\ndistribution_alpha = 1.5", DART)
        .expect("config validates");
    let first = run_replication(&cfg, 12345);
    let second = run_replication(&cfg, 12345);
    assert_eq!(
        first.weighted_sojourn_pooled, second.weighted_sojourn_pooled,
        "a heavy-tailed run must be reproducible under a fixed seed"
    );
    assert!(
        first.weighted_sojourn_pooled.iter().all(|x| x.is_finite()),
        "every weighted sojourn must be finite"
    );
}
