//! Discrete-event simulator.
//!
//! Entry points:
//!
//!   * [`run_replication`] — one replication, one seed.
//!   * [`run_replication_traced`] — the same, plus a full event trace.
//!   * [`run_parallel`] — every replication of a config, in parallel.
//!
//! The main loop pops events in time order, accumulates statistics over the
//! interval just elapsed, applies the event, and then asks the policy what
//! the server should do next. Three event kinds drive it: a job arrival, a
//! service completion, and the completion of a reconfiguration.
//!
//! # Reproducibility
//!
//! Each replication owns one `Xoshiro256++` stream seeded deterministically
//! from the config seed and the replication index, and every random draw in
//! the run comes from it in a fixed order. Two policies run under the same
//! seed therefore see the same arrival and service draws, which is what
//! makes the paired per-replication comparison in the evaluation valid.
//!
//! # Reconfiguration semantics
//!
//! A policy that commits to a multi-hop target emits one-edge moves and the
//! simulator walks the shortest path hop by hop, drawing each edge's
//! duration independently. A policy that returns a collapsed multi-hop move
//! is walked the same way, so its trip time is a sum of independent edge
//! draws rather than a single draw at the path mean — the two differ in
//! variance, and the physical model is the sum.

use std::collections::{BinaryHeap, VecDeque};

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::config::{Config, PolicyKind};
use crate::dist::{all_pairs_shortest_paths, NextHopMatrix};
use crate::event::{Event, EventKind};
use crate::policy::{
    dart, tian, Action, DartRouter, DecisionContext, DvoRouter, Router, SystemView, TianRouter,
};
use crate::stats::{Aggregate, ReplicationStats};
use crate::topology::Topology;

#[derive(Clone, Copy, Debug)]
enum ServerState {
    Idle,
    Serving,
    Switching,
}

/// Event trace for tests and external observers. Only populated by
/// [`run_replication_traced`]; the production path allocates nothing extra.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraceEvent {
    /// Initial server location, before any event has fired.
    Started { queue: usize },
    /// The server arrived at `queue` at the end of a reconfiguration.
    Arrived { time: f64, queue: usize },
    /// A service started at `queue`.
    StartService { time: f64, queue: usize },
    /// A service completed at `queue`.
    EndService { time: f64, queue: usize },
    /// A reconfiguration began.
    StartSwitchover {
        time: f64,
        from: usize,
        to: usize,
        duration: f64,
    },
}

pub fn run_replication(cfg: &Config, seed: u64) -> ReplicationStats {
    run_replication_inner(cfg, seed, None)
}

pub fn run_replication_traced(cfg: &Config, seed: u64) -> (ReplicationStats, Vec<TraceEvent>) {
    let mut trace = Vec::new();
    let stats = run_replication_inner(cfg, seed, Some(&mut trace));
    (stats, trace)
}

/// Build the topology with the resolved global and per-edge families.
/// Validation has already checked the parameters, so resolution cannot fail.
fn build_topology(cfg: &Config) -> Topology {
    let global = cfg
        .topology
        .resolve_global_dist()
        .expect("config validated: global distribution resolves");
    let overrides = cfg
        .topology
        .resolved_edge_overrides()
        .expect("config validated: edge overrides resolve");
    Topology::from_matrix_with_overrides(&cfg.topology.switchover_times, global, &overrides)
}

/// Construct the router for this config.
///
/// Every policy is handed the same [`SystemView`], including the same mean
/// shortest-path distances. That is what makes the comparison fair: no
/// baseline is penalized for the graph being multi-hop, and none of them
/// gets a routing metric the others do not.
fn build_router(cfg: &Config, topo: &Topology) -> Box<dyn Router> {
    let system = SystemView::new(
        cfg.priorities.weights.clone(),
        cfg.service.rates.clone(),
        cfg.arrivals.rates.clone(),
        all_pairs_shortest_paths(topo),
        topo.clone(),
    );
    let policy = &cfg.policy;
    match policy.kind {
        PolicyKind::Dart => Box::new(DartRouter::new(
            system,
            policy.alpha.unwrap_or(dart::DEFAULT_ALPHA),
            policy.beta.unwrap_or(dart::DEFAULT_BETA),
        )),
        PolicyKind::Tian | PolicyKind::TianTransit => {
            let router = TianRouter::new(
                system,
                policy.k.unwrap_or(tian::DEFAULT_K),
                policy.epsilon.unwrap_or(tian::DEFAULT_EPSILON),
                policy.max_sequences.unwrap_or(tian::DEFAULT_MAX_SEQUENCES),
            );
            if policy.kind == PolicyKind::TianTransit {
                Box::new(router.with_transit_service(policy.eta.unwrap_or(tian::DEFAULT_ETA)))
            } else {
                Box::new(router)
            }
        }
        PolicyKind::Dvo => Box::new(DvoRouter::new(system)),
    }
}

/// Public entry point for tests that build a router from a `Config` without
/// running the simulator.
pub fn build_router_for_test(cfg: &Config, topo: &Topology) -> Box<dyn Router> {
    build_router(cfg, topo)
}

#[inline]
fn rec(trace: &mut Option<&mut Vec<TraceEvent>>, ev: TraceEvent) {
    if let Some(v) = trace.as_deref_mut() {
        v.push(ev);
    }
}

/// State threaded through the per-event handlers.
struct Sim<'a> {
    rng: Xoshiro256PlusPlus,
    /// Per-configuration inter-arrival and service distributions.
    arrivals: Vec<Exp<f64>>,
    services: Vec<Exp<f64>>,
    topo: Topology,
    next_hop: NextHopMatrix,
    router: Box<dyn Router>,
    /// Arrival timestamps of the jobs waiting at each configuration, oldest
    /// first, so the head-of-line age is a front lookup.
    queues: Vec<VecDeque<f64>>,
    queue_lens: Vec<usize>,
    state: ServerState,
    server_loc: usize,
    locked_target: Option<usize>,
    /// Scratch buffer for the head-of-line ages handed to the router,
    /// refilled each decision to avoid a per-decision allocation.
    ages_buf: Vec<f64>,
    /// Start of the most recent observed visit to each configuration; NaN
    /// until the first one. Differences give the inter-visit gaps.
    last_visit_open_time: Vec<f64>,
    /// Did the current visit begin after warm-up? Only such visits are
    /// counted, so visit statistics all come from the same window.
    visit_observed: bool,
    heap: BinaryHeap<Event>,
    seq: u64,
    stats: ReplicationStats,
    trace: Option<&'a mut Vec<TraceEvent>>,
    warmup: f64,
}

impl Sim<'_> {
    /// Ask the policy what to do next and act on it. Called after every
    /// event that frees the server. `just_arrived` is true only when this
    /// decision follows a completed reconfiguration.
    fn dispatch(&mut self, t: f64, just_arrived: bool) {
        for i in 0..self.queue_lens.len() {
            self.ages_buf[i] = match self.queues[i].front() {
                Some(&arrived_at) => (t - arrived_at).max(0.0),
                None => 0.0,
            };
        }
        let action = self.router.decide(
            self.server_loc,
            &self.queue_lens,
            DecisionContext {
                locked_target: self.locked_target,
                ages: Some(&self.ages_buf),
                server_just_arrived: just_arrived,
            },
        );
        match action {
            Action::Serve => self.start_service(t),
            Action::MoveTo { target, lock, .. } => {
                self.close_visit();
                self.locked_target = lock;
                // The router's `duration` is the edge mean; the realized
                // duration is drawn here, edge by edge along the path.
                let duration = self.sample_path_duration(self.server_loc, target);
                self.start_switchover(t, target, duration);
            }
            Action::Idle => {
                // Wait in place. The visit is still open, and the lock is
                // kept: if the visit later ends in a move, the policy still
                // needs to know which target it committed to.
                self.state = ServerState::Idle;
            }
        }
    }

    fn close_visit(&mut self) {
        if self.visit_observed {
            self.stats.visits[self.server_loc] += 1;
        }
    }

    fn open_visit(&mut self, t: f64) {
        self.visit_observed = t >= self.warmup;
        if self.visit_observed {
            let i = self.server_loc;
            let previous = self.last_visit_open_time[i];
            if previous.is_finite() {
                let gap = (t - previous).max(0.0);
                let r: u64 = self.rng.gen();
                self.stats.record_intervisit(i, gap, r);
            }
            self.last_visit_open_time[i] = t;
        }
    }

    /// Walk from `current` to `target` along shortest-path next hops,
    /// drawing each edge independently and summing. One iteration for a
    /// one-edge move; for a collapsed multi-hop move this gives the correct
    /// sum-of-edge-draws variance.
    fn sample_path_duration(&mut self, current: usize, target: usize) -> f64 {
        let mut total = 0.0;
        let mut at = current;
        while at != target {
            let hop = self.next_hop[at][target].expect("strong connectivity guarantees a next hop");
            total += self.topo.sample_time(at, hop, &mut self.rng);
            at = hop;
        }
        total
    }

    fn start_service(&mut self, t: f64) {
        let i = self.server_loc;
        let arrived_at = *self.queues[i]
            .front()
            .expect("Action::Serve at empty queue");
        if t >= self.warmup {
            let wait = t - arrived_at;
            self.stats.wait_sum[i] += wait;
            self.stats.wait_count[i] += 1;
            let r: u64 = self.rng.gen();
            self.stats.record_wait(i, wait, r);
            // En-route service: serving here while a *different*
            // configuration is the committed target. Always zero for a
            // policy that never locks a distant target.
            if let Some(target) = self.locked_target {
                if target != i {
                    self.stats.transit_services += 1;
                }
            }
        }
        rec(
            &mut self.trace,
            TraceEvent::StartService { time: t, queue: i },
        );
        let service_time = self.services[i].sample(&mut self.rng);
        self.push_event(t + service_time, EventKind::ServiceCompletion);
        self.state = ServerState::Serving;
    }

    fn start_switchover(&mut self, t: f64, target: usize, duration: f64) {
        rec(
            &mut self.trace,
            TraceEvent::StartSwitchover {
                time: t,
                from: self.server_loc,
                to: target,
                duration,
            },
        );
        self.push_event(t + duration, EventKind::SwitchoverCompletion(target));
        self.state = ServerState::Switching;
    }

    #[inline]
    fn push_event(&mut self, time: f64, kind: EventKind) {
        self.heap.push(Event {
            time,
            seq: self.seq,
            kind,
        });
        self.seq += 1;
    }

    /// Accumulate the time-weighted statistics over `[a, b]`, both already
    /// clipped to the post-warm-up window.
    fn accumulate(&mut self, a: f64, b: f64) {
        if b <= a {
            return;
        }
        let dt = b - a;
        for i in 0..self.stats.n {
            self.stats.queue_area[i] += self.queue_lens[i] as f64 * dt;
            self.stats.record_qlen(i, self.queue_lens[i], dt);
        }
        match self.state {
            ServerState::Serving => self.stats.busy_time += dt,
            ServerState::Switching => self.stats.switch_time += dt,
            ServerState::Idle => self.stats.idle_time += dt,
        }
    }
}

fn run_replication_inner(
    cfg: &Config,
    seed: u64,
    mut trace: Option<&mut Vec<TraceEvent>>,
) -> ReplicationStats {
    let n = cfg.system.n_queues;
    let horizon = cfg.simulation.horizon;
    let warmup = cfg.simulation.warmup;
    let weights = &cfg.priorities.weights;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let arrivals: Vec<Exp<f64>> = cfg
        .arrivals
        .rates
        .iter()
        .map(|&r| Exp::new(r).expect("arrival rate must be > 0"))
        .collect();
    let services: Vec<Exp<f64>> = cfg
        .service
        .rates
        .iter()
        .map(|&r| Exp::new(r).expect("service rate must be > 0"))
        .collect();

    let topo = build_topology(cfg);
    let next_hop = all_pairs_shortest_paths(&topo).next_hop;
    let router = build_router(cfg, &topo);
    let server_loc = cfg.policy.start.unwrap_or(0);

    rec(&mut trace, TraceEvent::Started { queue: server_loc });

    let mut heap: BinaryHeap<Event> = BinaryHeap::with_capacity(4 * n + 16);
    let mut seq: u64 = 0;
    for (i, arrival) in arrivals.iter().enumerate() {
        heap.push(Event {
            time: arrival.sample(&mut rng),
            seq,
            kind: EventKind::Arrival(i),
        });
        seq += 1;
    }

    let mut sim = Sim {
        rng,
        arrivals,
        services,
        topo,
        next_hop,
        router,
        queues: (0..n).map(|_| VecDeque::new()).collect(),
        queue_lens: vec![0; n],
        state: ServerState::Idle,
        server_loc,
        // Treat the starting location as an already-committed target so the
        // policy serves there before moving, exactly as it would after any
        // other arrival at a committed target.
        locked_target: Some(server_loc),
        ages_buf: vec![0.0; n],
        last_visit_open_time: vec![f64::NAN; n],
        // The opening visit begins at t = 0, so it counts only when there
        // is no warm-up to discard.
        visit_observed: warmup == 0.0,
        heap,
        seq,
        stats: ReplicationStats::new(n),
        trace,
        warmup,
    };

    let mut last_time = 0.0_f64;
    while let Some(ev) = sim.heap.pop() {
        if ev.time > horizon {
            break;
        }
        sim.accumulate(last_time.max(warmup), ev.time.max(warmup));
        last_time = ev.time;
        let t = ev.time;

        match ev.kind {
            EventKind::Arrival(i) => {
                sim.queues[i].push_back(t);
                sim.queue_lens[i] += 1;
                let next = t + sim.arrivals[i].sample(&mut sim.rng);
                sim.push_event(next, EventKind::Arrival(i));
                // An arrival changes what the server does only if it was
                // idle; otherwise the in-flight service or reconfiguration
                // finishes first and the policy reconsiders then.
                if matches!(sim.state, ServerState::Idle) {
                    sim.dispatch(t, false);
                }
            }
            EventKind::ServiceCompletion => {
                let i = sim.server_loc;
                let arrived_at = sim.queues[i].pop_front().expect("served from empty queue");
                sim.queue_lens[i] -= 1;
                if t >= warmup {
                    let sojourn = t - arrived_at;
                    sim.stats.sojourn_sum[i] += sojourn;
                    sim.stats.sojourn_count[i] += 1;
                    sim.stats.served[i] += 1;
                    let r: u64 = sim.rng.gen();
                    sim.stats.record_sojourn(i, sojourn, r);
                    // The objective sample: `w_i·S`, recorded once per
                    // completion so the pooled 0.99 quantile is unbiased.
                    sim.stats.weighted_sojourn_pooled.push(weights[i] * sojourn);
                }
                rec(&mut sim.trace, TraceEvent::EndService { time: t, queue: i });
                sim.dispatch(t, false);
            }
            EventKind::SwitchoverCompletion(target) => {
                sim.server_loc = target;
                if t >= warmup {
                    sim.stats.switchovers += 1;
                }
                rec(
                    &mut sim.trace,
                    TraceEvent::Arrived {
                        time: t,
                        queue: target,
                    },
                );
                sim.open_visit(t);
                sim.dispatch(t, true);
            }
        }
    }

    sim.accumulate(last_time.max(warmup), horizon);
    sim.stats.observation_time = horizon - warmup;
    sim.stats
}

/// Derivation of each replication's seed from the config's base seed.
///
/// Both derivations below are fixed parts of the published protocol, so a
/// rerun reproduces earlier results exactly. Each is a pure function of
/// `(base, index)`, independent of thread scheduling, so the same figure
/// comes out of a 1-thread and a 64-thread machine.
///
/// The two binaries use different streams: `dartsim` mixes the index
/// through a large odd constant, which decorrelates neighbouring base
/// seeds, while `dartsim-samples` offsets it directly. Both give
/// independent replications of the same system and are therefore both
/// unbiased; they simply do not draw the *same* replications, so a summary
/// from `dartsim` and a distribution from `dartsim-samples` are two
/// independent estimates rather than two views of one run.
pub mod seeding {
    /// Aggregate runs: the `dartsim` binary and [`super::run_parallel`].
    #[inline]
    pub fn aggregate(base: u64, index: usize) -> u64 {
        base.wrapping_add(index as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// Per-job sample dumps: the `dartsim-samples` binary.
    #[inline]
    pub fn samples(base: u64, index: usize) -> u64 {
        base.wrapping_add(index as u64)
    }
}

/// Run every replication of `cfg` in parallel and aggregate the results.
///
/// Replications go to rayon's global pool, which sizes itself from
/// `RAYON_NUM_THREADS` and otherwise from the available parallelism. That
/// matters when many configurations are driven at once, as the reproduction
/// pipeline does: it takes its parallelism across runs and pins each run to
/// a single thread. A simulator that ignored the variable would instead
/// oversubscribe the machine by the width of its own pool.
///
/// The result does not depend on how many threads are used: each
/// replication's seed is a pure function of its index, and `collect`
/// restores order, so the same aggregate comes out on one thread and on
/// sixty-four.
pub fn run_parallel(cfg: &Config) -> Aggregate {
    let base_seed = cfg.simulation.seed;
    let reps: Vec<ReplicationStats> = (0..cfg.simulation.n_replications)
        .into_par_iter()
        .map(|i| run_replication(cfg, seeding::aggregate(base_seed, i)))
        .collect();
    Aggregate::from_replications(reps, &cfg.priorities.weights)
}
