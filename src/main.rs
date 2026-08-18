//! `dartsim` — run one configuration and report the result.
//!
//! ```text
//! dartsim --config run.toml [--output human|json|csv]
//! ```
//!
//! `human` prints a readable summary, `json` the full aggregate for
//! programmatic use, and `csv` a single header-plus-row record convenient
//! for appending many runs into one table.

use std::env;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use serde::Serialize;

use dartsim::config::Config;
use dartsim::simulation::run_parallel;
use dartsim::stats::{Aggregate, PERCENTILES};

#[derive(Clone, Copy, Debug)]
enum OutputMode {
    Human,
    Json,
    Csv,
}

const USAGE: &str = "usage: dartsim --config <path-to-toml> [--output human|json|csv]";

fn parse_args() -> Result<(PathBuf, OutputMode), String> {
    let mut args = env::args().skip(1);
    let mut config: Option<PathBuf> = None;
    let mut mode = OutputMode::Human;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-c" | "--config" => config = args.next().map(PathBuf::from),
            "-o" | "--output" => {
                let v = args.next().ok_or("missing value after --output")?;
                mode = match v.as_str() {
                    "human" => OutputMode::Human,
                    "json" => OutputMode::Json,
                    "csv" => OutputMode::Csv,
                    other => {
                        return Err(format!(
                            "unknown --output {other:?}; expected human|json|csv"
                        ))
                    }
                };
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    let config = config.ok_or_else(|| format!("missing --config\n{USAGE}"))?;
    Ok((config, mode))
}

/// The JSON document: the aggregate, plus the percentile grid its arrays
/// are indexed by and the wall-clock cost of producing it.
#[derive(Serialize)]
struct JsonReport<'a> {
    #[serde(flatten)]
    aggregate: &'a Aggregate,
    percentile_pts: &'a [f64],
    wall_time_s: f64,
}

/// One CSV record summarizing the run: identification, the objective, the
/// server's time budget, and per-configuration delay and visit structure.
fn write_csv(cfg: &Config, agg: &Aggregate, out: &mut impl Write) -> io::Result<()> {
    use std::fmt::Write as _;
    let n = agg.n;
    let p50 = Aggregate::pct_idx(0.50);
    let p95 = Aggregate::pct_idx(0.95);
    let p99 = Aggregate::pct_idx(0.99);
    let p999 = Aggregate::pct_idx(0.999);
    let load: f64 = cfg
        .arrivals
        .rates
        .iter()
        .zip(cfg.service.rates.iter())
        .map(|(&lam, &mu)| lam / mu)
        .sum();

    let mut header = String::from(
        "policy,n,load,seed,horizon,warmup,n_replications,\
         weighted_sojourn_p50,weighted_sojourn_p95,weighted_sojourn_p99,\
         weighted_sojourn_p99_ci95,weighted_sojourn_p999,\
         sojourn_p50,sojourn_p95,sojourn_p99,\
         mean_weighted_queue_length,\
         serve_fraction,switch_fraction,idle_fraction,\
         switches_per_time,transit_services_per_time",
    );
    for prefix in [
        "mean_queue",
        "sojourn_p99_queue",
        "intervisit_mean_queue",
        "intervisit_p99_queue",
    ] {
        for i in 0..n {
            let _ = write!(header, ",{prefix}_{i}");
        }
    }
    writeln!(out, "{header}")?;

    let mut row = String::new();
    let _ = write!(
        row,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        cfg.policy.kind.name(),
        n,
        load,
        cfg.simulation.seed,
        cfg.simulation.horizon,
        cfg.simulation.warmup,
        agg.n_reps,
        agg.weighted_sojourn_pct_mean[p50],
        agg.weighted_sojourn_pct_mean[p95],
        agg.weighted_sojourn_pct_mean[p99],
        agg.weighted_sojourn_pct_ci95[p99],
        agg.weighted_sojourn_pct_mean[p999],
        agg.sojourn_percentiles_overall[p50],
        agg.sojourn_percentiles_overall[p95],
        agg.sojourn_percentiles_overall[p99],
        agg.mean_weighted_qlen,
        agg.utilization,
        agg.switch_frac,
        agg.idle_frac,
        agg.switches_per_time,
        agg.transit_services_per_time,
    );
    for i in 0..n {
        let _ = write!(row, ",{}", agg.mean_qlen[i]);
    }
    for i in 0..n {
        let _ = write!(row, ",{}", agg.sojourn_percentiles[i][p99]);
    }
    for i in 0..n {
        let _ = write!(row, ",{}", agg.mean_intervisit[i]);
    }
    for i in 0..n {
        let _ = write!(row, ",{}", agg.intervisit_percentiles[i][p99]);
    }
    writeln!(out, "{row}")
}

fn main() -> ExitCode {
    let (path, mode) = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let cfg = match Config::load(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config error: {e}");
            return ExitCode::from(2);
        }
    };

    let start = Instant::now();
    let agg = run_parallel(&cfg);
    let elapsed = start.elapsed().as_secs_f64();

    match emit(&cfg, &agg, elapsed, mode) {
        Ok(()) => ExitCode::SUCCESS,
        // A reader that stops early -- `dartsim ... | head` -- closes the
        // pipe. That is a normal way to end, not a failure to report.
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("write error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn emit(cfg: &Config, agg: &Aggregate, elapsed: f64, mode: OutputMode) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    match mode {
        OutputMode::Human => {
            agg.write_summary(&mut out)?;
            writeln!(out)?;
            writeln!(out, "wall time: {elapsed:.3}s")?;
        }
        OutputMode::Json => {
            let report = JsonReport {
                aggregate: agg,
                percentile_pts: &PERCENTILES,
                wall_time_s: elapsed,
            };
            serde_json::to_writer(&mut out, &report)?;
            writeln!(out)?;
        }
        OutputMode::Csv => write_csv(cfg, agg, &mut out)?,
    }
    out.flush()
}
