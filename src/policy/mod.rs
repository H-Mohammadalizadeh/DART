//! Scheduling policies.
//!
//! A [`Router`] answers one question: given where the server is, how much
//! work waits at each configuration, and a little context, what should the
//! server do next? The answer is an [`Action`]:
//!
//!   * [`Action::Serve`] — serve one job at the current configuration,
//!   * [`Action::MoveTo`] — start a reconfiguration along one edge,
//!   * [`Action::Idle`] — nothing to do anywhere; wait for an arrival.
//!
//! A router that commits to a multi-hop target emits `MoveTo` with
//! `lock = Some(target)`. The simulator stores that target and feeds it
//! back on the next decision through [`DecisionContext::locked_target`],
//! which is how commitment survives across hops without the router holding
//! mutable state. Routers are therefore pure functions of
//! `(location, queue lengths, context)` and are shared across replications.
//!
//! Four policies live here: [`dart::DartRouter`] (the proposed policy) and
//! the three baselines it is measured against — [`tian::TianRouter`] in its
//! plain and transit-serving forms, and [`dvo::DvoRouter`].

pub mod dart;
pub mod dvo;
pub mod tian;

pub use dart::DartRouter;
pub use dvo::DvoRouter;
pub use tian::TianRouter;

use crate::dist::PathInfo;
use crate::topology::Topology;

/// Everything a policy reads about the system it controls.
///
/// Built once per run and handed to whichever router the configuration
/// selects. Bundling it keeps the three constructors honest about the fact
/// that all of them see exactly the same system — in particular the same
/// `paths`, which is what makes the comparison in the paper fair.
pub struct SystemView {
    pub n: usize,
    /// Holding-cost weight `w_i`.
    pub weights: Vec<f64>,
    /// Service rate `μ_i`.
    pub service_rates: Vec<f64>,
    /// Arrival rate `λ_i`.
    pub arrival_rates: Vec<f64>,
    /// Shortest-path distances `d(i,j)` and their first hops.
    pub paths: PathInfo,
    /// The graph itself, for the mean of a single edge.
    pub topology: Topology,
}

impl SystemView {
    pub fn new(
        weights: Vec<f64>,
        service_rates: Vec<f64>,
        arrival_rates: Vec<f64>,
        paths: PathInfo,
        topology: Topology,
    ) -> Self {
        let n = topology.n;
        assert_eq!(weights.len(), n, "one weight per configuration");
        assert_eq!(service_rates.len(), n, "one service rate per configuration");
        assert_eq!(arrival_rates.len(), n, "one arrival rate per configuration");
        Self {
            n,
            weights,
            service_rates,
            arrival_rates,
            paths,
            topology,
        }
    }
}

/// What the server does next.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    /// Serve one job at the current configuration.
    Serve,
    /// Reconfigure along one edge toward `target`.
    ///
    /// `target` is the *next hop*, and `duration` is that edge's mean; the
    /// simulator draws the realized duration itself, so `duration` is
    /// advisory. `lock` carries the final destination for a committing
    /// policy, and is `None` for a policy that re-decides every hop.
    MoveTo {
        target: usize,
        duration: f64,
        lock: Option<usize>,
    },
    /// Every configuration is empty; wait in place.
    Idle,
}

/// State the simulator passes to a router that the queue lengths alone do
/// not carry.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecisionContext<'a> {
    /// Destination this policy committed to on an earlier decision, if it
    /// has not been reached yet.
    pub locked_target: Option<usize>,
    /// Per-configuration head-of-line age `a_i`: `now − arrival` of the
    /// oldest waiting job, `0` for an empty configuration. Supplied for
    /// every simulator run; `None` only in unit tests of backlog-based
    /// policies that ignore it.
    pub ages: Option<&'a [f64]>,
    /// `true` only on the decision taken immediately after the server
    /// physically arrives somewhere via a completed reconfiguration.
    ///
    /// A delay-based policy uses this to honour its commitment by serving
    /// at least once on arrival before re-deciding. Without it, an age-
    /// ranked policy can arrive at its target, observe that the age it just
    /// travelled to relieve is now the *second* largest, and leave again
    /// without serving — ping-ponging between two configurations while both
    /// tails grow.
    pub server_just_arrived: bool,
}

pub trait Router: Send + Sync {
    fn decide(&self, current: usize, queue_lens: &[usize], ctx: DecisionContext<'_>) -> Action;
}
