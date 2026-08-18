//! DARTsim — a discrete-event simulator for tail-delay control in
//! reconfigurable systems.
//!
//! A single server moves over a directed graph of *configurations*. Each
//! configuration holds its own queue of jobs, and moving between two of
//! them costs a random reconfiguration time whose distribution depends on
//! the direction. Reaching a target may require crossing intermediate
//! configurations, which the server may also serve on the way. The
//! objective is the high-percentile *weighted* sojourn time.
//!
//! # Layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`config`] | TOML schema and validation |
//! | [`topology`] | the reconfiguration graph and its mean-preserving time families |
//! | [`dist`] | all-pairs shortest paths, the distances `d(i,j)` every policy routes on |
//! | [`policy`] | DART and the three baselines |
//! | [`event`] | the event type and its time ordering |
//! | [`simulation`] | the event loop |
//! | [`stats`] | per-replication accumulators and cross-replication aggregation |
//!
//! # Example
//!
//! ```no_run
//! use dartsim::{config::Config, simulation::run_parallel, stats::Aggregate};
//! use std::path::Path;
//!
//! let cfg = Config::load(Path::new("run.toml"))?;
//! let agg = run_parallel(&cfg);
//! println!("weighted P99 = {:.1}", agg.weighted_sojourn_pct_mean[Aggregate::pct_idx(0.99)]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod config;
pub mod dist;
pub mod event;
pub mod policy;
pub mod simulation;
pub mod stats;
pub mod topology;
