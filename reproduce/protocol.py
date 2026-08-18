"""The measurement protocol and the policies under comparison.

Every number in the paper comes out of the settings in this module. A run
is a scenario file (see :mod:`reproduce.scenarios`) with a ``[simulation]``
and a ``[policy]`` section appended, so changing a protocol constant here
changes it identically for all four policies.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scenarios import Scenario

# ---------------------------------------------------------------------------
# Simulation protocol
# ---------------------------------------------------------------------------

#: Length of one replication, in simulated time units. Long enough that the
#: P99 of the weighted sojourn is stable.
HORIZON = 200_000.0

#: Discarded leading interval, so statistics come from steady state.
WARMUP = 30_000.0

#: Independent replications per point. Error bars are 95% Student-t
#: intervals over these.
REPLICATIONS = 16

#: Base seed. All policies share the derived per-replication seed stream,
#: so each configuration sees the same arrivals and service draws under
#: every policy and the comparison is paired.
SEED = 2027

#: Shorter protocol for the generalization batteries, which cover 41 cases
#: rather than 6 and do not feed the headline numbers.
GENERALIZATION_HORIZON = 150_000.0
GENERALIZATION_WARMUP = 20_000.0

#: Index of the 0.99 quantile in the simulator's percentile grid
#: ``[0.5, 0.9, 0.95, 0.99, 0.999]``.
P99 = 3


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    """One policy as it is configured for the evaluation."""

    key: str
    #: Name used in figure legends.
    label: str
    #: ``[policy]`` body.
    block: str
    #: Figure colour.
    color: str


#: DART's two parameters, fixed across every topology: ``α`` discounts the
#: return time in the urgency, ``β`` sets the strength of the delay guard.
DART_ALPHA = 0.5
DART_BETA = 6.0

#: Tian's lookahead. A larger value rarely changes the selected path on
#: these graphs.
TIAN_K = 4

#: Tian-T's transit reluctance, at its neutral setting.
TIAN_T_ETA = 1.0


def dart_block(alpha: float = DART_ALPHA, beta: float = DART_BETA) -> str:
    return f'kind = "dart"\nstart = 0\nalpha = {alpha}\nbeta = {beta}'


def tian_transit_block(eta: float = TIAN_T_ETA) -> str:
    return f'kind = "tian_transit"\nstart = 0\nk = {TIAN_K}\neta = {eta}'


DART = Policy("dart", "DART", dart_block(), "#2E5C8A")
TIAN = Policy("tian", "Tian K=4", f'kind = "tian"\nstart = 0\nk = {TIAN_K}', "#9E2A2B")
TIAN_T = Policy("tian_transit", "Tian-T", tian_transit_block(), "#C9A227")
DVO = Policy("dvo", "DVO", 'kind = "dvo"\nstart = 0', "#7E8587")

#: Order used consistently in every figure: the proposed policy first, then
#: the three baselines.
POLICIES = [DART, TIAN, TIAN_T, DVO]
BASELINES = [TIAN, TIAN_T, DVO]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def config_text(
    scenario: Scenario,
    policy_block: str,
    *,
    horizon: float = HORIZON,
    warmup: float = WARMUP,
    replications: int = REPLICATIONS,
    seed: int = SEED,
) -> str:
    """A complete simulator configuration for one (scenario, policy) pair.

    The scenario text is copied verbatim; only the protocol and the policy
    are appended.
    """
    return (
        f"{scenario.text.rstrip()}\n\n"
        "[simulation]\n"
        f"horizon = {horizon}\n"
        f"warmup = {warmup}\n"
        f"n_replications = {replications}\n"
        f"seed = {seed}\n\n"
        "[policy]\n"
        f"{policy_block.strip()}\n"
    )
