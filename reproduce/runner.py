"""Driving the simulator from Python: compose, run, cache, parallelise.

The simulator is a command-line program, so a "run" here is a subprocess
call on a temporary configuration file. Two properties make that cheap
enough to drive a whole figure set:

* **Determinism.** A run is a pure function of its configuration text, so
  results are cached under a hash of that text. Re-rendering a figure after
  a styling change costs nothing.
* **Independence.** Runs do not interact, so they are spread across a
  process pool.

Parallelism is taken at whichever level has work. Cached runs are resolved
in this process, without spawning anything; the runs that are left divide
the machine between them, so a batch of forty gives each simulator one
thread while a batch of one gives it every core. Fixing the split in
advance would either oversubscribe the machine or leave it idle at the
tail of a batch, and the tail is where the expensive runs end up.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .protocol import config_text
from .scenarios import Scenario

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "target" / "release"
SIMULATOR = BIN_DIR / "dartsim"
SAMPLER = BIN_DIR / "dartsim-samples"
CACHE_DIR = ROOT / ".cache"

#: Cores available to the whole pipeline.
CORES = os.cpu_count() or 1

#: Cap on concurrent simulator processes.
MAX_WORKERS = min(16, CORES)

#: Threads the current process lets one simulator run use. Set per batch by
#: :func:`run_all` and installed into each pool worker.
_THREADS_PER_RUN = 1


def _set_threads_per_run(threads: int) -> None:
    global _THREADS_PER_RUN
    _THREADS_PER_RUN = threads


class _NotCached(Exception):
    """Raised instead of running, while :func:`run_all` probes the cache."""


#: True while probing: `_cached` then reports a miss rather than computing.
_PROBING = False


class SimulatorMissing(RuntimeError):
    pass


def require_binaries() -> None:
    """Fail early, with the command to fix it, if the binaries are absent."""
    missing = [p.name for p in (SIMULATOR, SAMPLER) if not p.exists()]
    if missing:
        raise SimulatorMissing(
            f"{', '.join(missing)} not built. Run `cargo build --release` in {ROOT}."
        )


def _cache_path(text: str, *parts: object) -> Path:
    key = hashlib.sha256("\0".join([text, *map(str, parts)]).encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _invoke(binary: Path, config: str, extra_args: Sequence[str]) -> str:
    """Run `binary` on a temporary file holding `config` and return stdout."""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as handle:
        handle.write(config)
        path = handle.name
    try:
        # The simulator parallelises over its own replications, which would
        # multiply with the process pool driving it. `run_all` decides how
        # to divide the machine between the two and sets the share here.
        env = dict(os.environ, RAYON_NUM_THREADS=str(_THREADS_PER_RUN))
        result = subprocess.run(
            [str(binary), "--config", path, *extra_args],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{binary.name} failed:\n{exc.stderr}") from exc
    finally:
        os.unlink(path)
    return result.stdout


def _cached(text: str, tag: object, produce: Callable[[], object]) -> object:
    path = _cache_path(text, tag)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass  # truncated by an interrupted run; recompute
    if _PROBING:
        raise _NotCached
    value = produce()
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(value))
    return value


# ---------------------------------------------------------------------------
# The three things we ask of the simulator
# ---------------------------------------------------------------------------


def summary(scenario: Scenario, policy_block: str, **protocol) -> dict:
    """Full aggregate statistics for one (scenario, policy) pair."""
    text = config_text(scenario, policy_block, **protocol)
    return _cached(text, "summary", lambda: json.loads(_invoke(SIMULATOR, text, ["--output", "json"])))


def weighted_sojourn_samples(
    scenario: Scenario, policy_block: str, cap: int = 1500, **protocol
) -> list[float]:
    """Sorted pooled weighted sojourn `w_i·S`, for the CCDF."""
    text = config_text(scenario, policy_block, **protocol)

    def produce() -> list[float]:
        out = _invoke(SAMPLER, text, ["--mode", "weighted", "--cap", str(cap)])
        values = [float(line.rsplit(",", 1)[1]) for line in out.splitlines()[1:] if line]
        values.sort()
        return values

    return _cached(text, ("weighted-ccdf", cap), produce)


def per_queue_samples(
    scenario: Scenario, policy_block: str, cap: int = 2000, **protocol
) -> dict[tuple[int, int], list[float]]:
    """Sojourn samples keyed by ``(replication, configuration)``.

    Keeping the replication index lets per-class percentiles be taken one
    replication at a time, so their confidence intervals use the same
    Student-t method as every other figure.
    """
    text = config_text(scenario, policy_block, **protocol)

    def produce() -> dict[str, list[float]]:
        out = _invoke(SAMPLER, text, ["--mode", "per-queue", "--cap", str(cap)])
        grouped: dict[str, list[float]] = {}
        for line in out.splitlines()[1:]:
            if not line:
                continue
            _, replication, _, queue, sojourn = line.split(",")
            grouped.setdefault(f"{replication},{queue}", []).append(float(sojourn))
        return grouped

    raw = _cached(text, ("per-queue", cap), produce)
    return {tuple(int(part) for part in key.split(",")): values for key, values in raw.items()}


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


def _call(job):
    key, function, args, kwargs = job
    return key, function(*args, **kwargs)


def _probe(job):
    """The job's cached result, or `None` if it has not been run yet."""
    global _PROBING
    _PROBING = True
    try:
        return _call(job)
    except _NotCached:
        return None
    finally:
        _PROBING = False


def run_all(jobs: Iterable[tuple]) -> dict:
    """Execute ``(key, function, args, kwargs)`` jobs and collect by key.

    Cached jobs are resolved here, so a re-render of a figure set costs
    milliseconds and never starts a process. Whatever is left divides the
    machine between however many runs there are.
    """
    results = {}
    pending = []
    for job in jobs:
        hit = _probe(job)
        if hit is None:
            pending.append(job)
        else:
            results[hit[0]] = hit[1]
    if not pending:
        return results

    workers = min(MAX_WORKERS, len(pending))
    threads = max(1, CORES // workers)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_set_threads_per_run,
        initargs=(threads,),
    ) as pool:
        futures = [pool.submit(_call, job) for job in pending]
        for future in as_completed(futures):
            key, value = future.result()
            results[key] = value
    return results


def weighted_p99(report: dict) -> tuple[float, float]:
    """The objective and its 95% interval half-width, from a summary."""
    from .protocol import P99

    return (
        report["weighted_sojourn_pct_mean"][P99],
        report["weighted_sojourn_pct_ci95"][P99],
    )


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile, matching the simulator's own method."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    frac = index - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac
