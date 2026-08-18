"""Loading of the scenario files in ``scenarios/``.

A scenario file describes a system — configurations, weights, arrival and
service rates, and the reconfiguration graph — but not how it is measured
or by which policy. Those two sections are appended at run time by
:mod:`reproduce.protocol`, so every policy provably sees the same system:
the scenario text is copied verbatim, never regenerated per policy.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "scenarios"

#: The six stress topologies, in the order they appear in the paper. Each
#: isolates one difficulty in tail control.
STRESS_LABELS = ["A", "B", "C", "D", "E", "F"]

#: Generalization batteries: directory name and the human-readable name
#: used in reports and figure legends.
BATTERIES = {
    "random": "random topology",
    "regime": "regime boundary",
    "scaling": "scalability",
}


@dataclass(frozen=True)
class Scenario:
    """One scenario file, parsed."""

    path: Path
    label: str
    name: str
    text: str
    n: int
    arrival_rates: list[float]
    service_rates: list[float]
    weights: list[float]
    switchover_times: list[list[float]]

    @property
    def offered_load(self) -> float:
        """Total offered load ``Σ_i λ_i / μ_i``."""
        return sum(lam / mu for lam, mu in zip(self.arrival_rates, self.service_rates))

    def high_priority(self) -> list[int]:
        """Configurations in the high-priority class.

        The split is at half the largest weight, which separates the two
        classes cleanly in every scenario used here.
        """
        cut = 0.5 * max(self.weights)
        return [i for i, w in enumerate(self.weights) if w >= cut]

    def low_priority(self) -> list[int]:
        cut = 0.5 * max(self.weights)
        return [i for i, w in enumerate(self.weights) if w < cut]

    def with_arrival_rates(self, rates: list[float]) -> "Scenario":
        """A copy whose arrival rates are replaced, for the load sweep.

        The rates are substituted in the scenario *text* as well, so the
        composed configuration stays a faithful rendering of this object.
        """
        old = _format_rates(self.arrival_rates)
        new = _format_rates(rates)
        marker = f"[arrivals]\nrates = {old}"
        if marker not in self.text:
            raise ValueError(f"{self.path.name}: cannot locate the arrival rates to substitute")
        return Scenario(
            path=self.path,
            label=self.label,
            name=self.name,
            text=self.text.replace(marker, f"[arrivals]\nrates = {new}", 1),
            n=self.n,
            arrival_rates=list(rates),
            service_rates=self.service_rates,
            weights=self.weights,
            switchover_times=self.switchover_times,
        )


def _format_rates(rates: list[float]) -> str:
    return "[" + ", ".join(_format_number(r) for r in rates) + "]"


def _format_number(x: float) -> str:
    """Shortest text that round-trips to the same float."""
    x = float(x)
    return f"{int(x)}.0" if x == int(x) and abs(x) < 1e15 else repr(x)


def load(path: Path) -> Scenario:
    """Parse one scenario file."""
    text = path.read_text()
    doc = tomllib.loads(text)
    meta = doc.get("meta", {})
    return Scenario(
        path=path,
        label=meta.get("label", path.stem),
        name=meta.get("name", path.stem),
        text=text,
        n=doc["system"]["n_queues"],
        arrival_rates=[float(x) for x in doc["arrivals"]["rates"]],
        service_rates=[float(x) for x in doc["service"]["rates"]],
        weights=[float(x) for x in doc["priorities"]["weights"]],
        switchover_times=[[float(v) for v in row] for row in doc["topology"]["switchover_times"]],
    )


@lru_cache(maxsize=None)
def stress() -> dict[str, Scenario]:
    """The six stress topologies, keyed by label ``A``…``F``."""
    found = {}
    for path in sorted((SCENARIO_DIR / "stress").glob("*.toml")):
        scenario = load(path)
        found[scenario.label] = scenario
    missing = [label for label in STRESS_LABELS if label not in found]
    if missing:
        raise FileNotFoundError(f"missing stress scenarios: {', '.join(missing)}")
    return {label: found[label] for label in STRESS_LABELS}


@lru_cache(maxsize=None)
def generalization(battery: str) -> list[Scenario]:
    """One generalization battery, in sorted case order."""
    if battery not in BATTERIES:
        raise KeyError(f"unknown battery {battery!r}; expected one of {sorted(BATTERIES)}")
    directory = SCENARIO_DIR / "generalization" / battery
    return [load(path) for path in sorted(directory.glob("*.toml"))]
