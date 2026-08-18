//! `dartsim-samples` — dump individual sojourn observations as CSV.
//!
//! The `dartsim` binary reports summary statistics. This one emits the raw
//! per-job observations behind them, for figures that need a distribution
//! rather than a number: the weighted-sojourn CCDF and the per-class
//! percentiles.
//!
//! ```text
//! dartsim-samples --config run.toml [--mode weighted|per-queue] [--cap N]
//! ```
//!
//! * `weighted` (the default) emits the pooled weighted sojourn `w_i·S` of
//!   every post-warm-up completion. Columns: `policy,replication,weighted_sojourn`.
//! * `per-queue` emits the per-configuration sojourn reservoirs. Columns:
//!   `policy,replication,seed,queue,sojourn`.
//!
//! `--cap N` bounds the rows emitted per replication (per configuration in
//! `per-queue` mode) by taking a fixed stride through the observations.
//! Because they are in completion order, a fixed stride is an unbiased
//! sample of the stationary post-warm-up distribution. `--cap 0` disables
//! the bound.

use std::env;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use rayon::prelude::*;

use dartsim::config::Config;
use dartsim::simulation::{run_replication, seeding};

const USAGE: &str =
    "usage: dartsim-samples --config <path-to-toml> [--mode weighted|per-queue] [--cap N]";

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Pooled weighted sojourn `w_i·S`, the objective's own population.
    Weighted,
    /// Per-configuration sojourn reservoirs.
    PerQueue,
}

struct Args {
    config: PathBuf,
    mode: Mode,
    /// Rows per replication (per configuration in `per-queue` mode);
    /// `0` means no bound.
    cap: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let mut config: Option<PathBuf> = None;
    let mut mode = Mode::Weighted;
    let mut cap = 40_000usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-c" | "--config" => config = args.next().map(PathBuf::from),
            "-m" | "--mode" => {
                let v = args.next().ok_or("missing value after --mode")?;
                mode = match v.as_str() {
                    "weighted" => Mode::Weighted,
                    "per-queue" => Mode::PerQueue,
                    other => {
                        return Err(format!(
                            "unknown --mode {other:?}; expected weighted|per-queue"
                        ))
                    }
                };
            }
            "--cap" => {
                cap = args
                    .next()
                    .and_then(|s| s.parse::<usize>().ok())
                    .ok_or("--cap needs a non-negative integer")?;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    let config = config.ok_or_else(|| format!("missing --config\n{USAGE}"))?;
    Ok(Args { config, mode, cap })
}

/// Stride that keeps at most `cap` of `len` observations.
#[inline]
fn stride(len: usize, cap: usize) -> usize {
    if cap == 0 || len <= cap {
        1
    } else {
        (len / cap).max(1)
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let cfg = match Config::load(&args.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config error: {e}");
            return ExitCode::from(2);
        }
    };

    match emit(&cfg, args.mode, args.cap) {
        Ok(()) => ExitCode::SUCCESS,
        // A reader that stops early closes the pipe; that is a normal way
        // to end a dump of several hundred thousand rows.
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("write error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn emit(cfg: &Config, mode: Mode, cap: usize) -> io::Result<()> {
    let policy = cfg.policy.kind.name();
    let n_reps = cfg.simulation.n_replications;
    let base_seed = cfg.simulation.seed;
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    match mode {
        Mode::Weighted => {
            let rows: Vec<(usize, f64)> = (0..n_reps)
                .into_par_iter()
                .flat_map_iter(|r| {
                    let pooled = run_replication(cfg, seeding::samples(base_seed, r))
                        .weighted_sojourn_pooled;
                    let step = stride(pooled.len(), cap);
                    pooled.into_iter().step_by(step).map(move |w| (r, w))
                })
                .collect();
            writeln!(out, "policy,replication,weighted_sojourn")?;
            for (r, w) in rows {
                writeln!(out, "{policy},{r},{w}")?;
            }
        }
        Mode::PerQueue => {
            let rows: Vec<(usize, u64, usize, f64)> = (0..n_reps)
                .into_par_iter()
                .flat_map_iter(|r| {
                    let seed = seeding::samples(base_seed, r);
                    run_replication(cfg, seed)
                        .sojourn_samples
                        .into_iter()
                        .enumerate()
                        .flat_map(move |(queue, xs)| {
                            let step = stride(xs.len(), cap);
                            xs.into_iter()
                                .step_by(step)
                                .map(move |sojourn| (r, seed, queue, sojourn))
                        })
                })
                .collect();
            writeln!(out, "policy,replication,seed,queue,sojourn")?;
            for (r, seed, queue, sojourn) in rows {
                writeln!(out, "{policy},{r},{seed},{queue},{sojourn}")?;
            }
        }
    }
    out.flush()
}
