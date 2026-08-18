# DARTsim

A discrete-event simulator for **tail-delay control in reconfigurable
systems**, and the code that reproduces every figure in

> H. Mohammadalizadeh and H. Karl,
> *DART: Aiming for Tail-Delay Control in Reconfigurable Networks.*

A single server moves over a directed graph of **configurations**. Each
configuration holds its own queue of jobs; moving between two of them costs
a random **reconfiguration time** whose distribution depends on the
direction; and reaching a target may require crossing intermediate
configurations, which the server may also serve on the way. The objective
is the high-percentile **weighted** sojourn time — a target that couples
three decisions: which configuration to aim for, which path to take, and
whether to pause en route.

The simulator implements **DART** and the three baselines it is measured
against, and nothing else.

---

## Quick start

```sh
cargo build --release            # build the simulator (~15 s)
cargo test  --release            # 66 tests

# One run, printed as a summary
./target/release/dartsim --config examples/return_trap.toml

# The paper's evaluation: 8 figures into figures/
python3 -m reproduce
```

`python3 -m reproduce` needs Python 3.11 or newer (for `tomllib`) and
`matplotlib` and `numpy`. `make figures` does the build and the
reproduction in one step; `make help` lists the other targets.

---

## What is here

```
src/                the simulator
  config.rs           TOML schema and validation
  topology.rs         the reconfiguration graph and its time families
  dist.rs             all-pairs shortest paths — the distances d(i,j)
  policy/             DART and the three baselines
  event.rs            the event type and its time ordering
  simulation.rs       the event loop
  stats.rs            per-replication accumulators, cross-replication aggregation
  main.rs             the `dartsim` binary
  bin/samples.rs      the `dartsim-samples` binary

scenarios/          the systems under study, as data
  stress/             the six stress topologies A-F
  generalization/     41 further cases: random, regime, scaling

reproduce/          the pipeline that turns scenarios into the paper's figures
tests/              integration tests, one file per policy
figures/            the output, and the committed copies used in the paper
examples/           small configurations to run by hand
```

A **scenario file** describes a system but not how it is measured or by
which policy. `reproduce/` appends a `[simulation]` and a `[policy]`
section to the scenario text verbatim, so every policy provably sees the
same system, the same graph, and the same seeds.

---

## Terminology

The paper is written for reconfigurable networks; the simulator uses the
queueing-theory vocabulary its baselines come from. They line up as:

| Paper | Code |
|---|---|
| configuration | queue (`n_queues`, `queue_lens`) |
| reconfiguration time `τ_ij` | switchover time (`switchover_times`) |
| mean reconfiguration matrix `T` | `topology.switchover_times` |
| holding-cost weight `w_i` | `priorities.weights` |
| high-priority / low-priority class | weight at least / below half the largest |

---

## Configuration format

A complete configuration is a scenario plus a protocol plus a policy:

```toml
[meta]                          # optional; the simulator ignores it
label = "C"
name = "return-trap"

[system]
n_queues = 4

[arrivals]
rates = [0.75, 0.07, 0.07, 0.07]     # Poisson rate λ_i

[service]
rates = [1.0, 1.0, 1.0, 1.0]         # exponential rate μ_i, must exceed λ_i

[priorities]
weights = [10.0, 1.0, 1.0, 1.0]      # holding cost w_i

[topology]
# Mean reconfiguration times T_ij; a negative entry marks an absent edge.
switchover_times = [
    [0.0, 0.2, -1.0, -1.0],
    [8.0, 0.0,  1.0, -1.0],
    [-1.0, 1.0, 0.0,  1.0],
    [-1.0, -1.0, 1.0, 0.0],
]
distribution = "exponential"         # default family for every edge

[[topology.edge_overrides]]          # ...and per-edge exceptions
from = 1
to = 0
kind = "pareto2"
alpha = 3.0

[simulation]
horizon = 200000.0
warmup = 30000.0
n_replications = 16
seed = 2027

[policy]
kind = "dart"
alpha = 0.5
beta = 6.0
```

Every section rejects unknown keys, so a misspelled parameter is an error
rather than a silent fallback to the default:

```
config error: unknown field `dart_w_gamma`, expected one of
`kind`, `start`, `alpha`, `beta`, `k`, `eta`, `epsilon`, `max_sequences`
```

### Reconfiguration-time families

Every family is **mean-preserving**: its parameters are derived from the
matrix entry so that `E[τ_ij]` equals it. Switching family therefore
changes the spread of a reconfiguration time and nothing else — in
particular it moves no routing decision, because the shortest paths are
computed on the means. A test asserts this.

| `kind` | Parameter | Shape |
|---|---|---|
| `deterministic` | — | exactly the mean; a quick, well-characterised reconfiguration |
| `exponential` | — | CV = 1; occasional long reconfigurations |
| `lognormal` | `cv` | all moments finite, subexponential; slow but moderately variable |
| `pareto2` | `alpha` | Lomax, regularly varying; finite variance only for `alpha > 2` |

### Policies

| `kind` | Policy | Parameters |
|---|---|---|
| `dart` | **DART** — opportunistic commitment | `alpha` (return discount, default `0.5`), `beta` (delay guard, default `6.0`) |
| `tian` | Tian & Shone's `K`-stop network index | `k` (default `1`) |
| `tian_transit` | Tian `K`-stop plus DART's en-route service | `k`, `eta` (transit reluctance, default `1.0`) |
| `dvo` | Duenyas & Van Oyen's reconfiguration-aware index | — |

All four route on the same shortest-path distances `d(i,j)`, and any
reconfiguration time in a baseline's formula is replaced by the matching
distance. No baseline is penalized for the graph being multi-hop.

DART's rule, with `d = d(i,j)` the time to reach `j` and `r = d(j,i)` the
time to leave it again:

```
anticipated delay   â_j  = a_j + (Q_j − 1)⁺ / (μ_j − λ_j)
urgency             U_ij = w_j (â_j + d) / (1 + d + α r)
target              j*   = argmax_j U_ij   over nonempty j ≠ i

serve a traversed k while either guard holds:
  delay guard    w_k (a_k + β(d + r))  ≥  U_kj*
  backlog guard  w_k Q_k μ_k           ≥  w_j* (Q_j* + λ_j* d) μ_j* / (1 + μ_j* d + α μ_j* r)
```

`α` and `β` are fixed across every topology in the evaluation; nothing is
tuned per graph.

---

## The binaries

**`dartsim`** runs one configuration and reports the result.

```sh
dartsim --config run.toml [--output human|json|csv]
```

`human` prints a readable summary, `json` the full aggregate, and `csv` one
header-plus-row record convenient for appending many runs into a table.

**`dartsim-samples`** dumps the individual observations behind those
summaries, for figures that need a distribution rather than a number.

```sh
dartsim-samples --config run.toml [--mode weighted|per-queue] [--cap N]
```

---

## Reproducing the paper

```sh
python3 -m reproduce              # everything
python3 -m reproduce --list       # what can be built
python3 -m reproduce --figures E2,E3
```

| Figure | Paper | Shows |
|---|---|---|
| `E1_topologies` | Fig. 3, sec. V-B | the six stress topologies and their reconfiguration matrices |
| `E2_wp99_bars` | Fig. 4, sec. V-D | system-wide weighted P99 on the six topologies |
| `E3_weighted_ccdf` | Fig. 5, sec. V-D | complementary CDF of the weighted sojourn |
| `E6_class_balance` | Fig. 6, sec. V-E | low- against high-priority tail on the two hardest graphs |
| `E4_per_class` | Fig. 7, sec. V-E | per-class weighted P99 across the six topologies |
| `E5_load_sweep` | Fig. 8, sec. V-F | weighted P99 against offered load on topology F |
| `E10_knob_single` | Fig. 9, sec. V-F | each method swept over its own knob |
| `E8_regime_heatmap` | Fig. 10, sec. V-F | margin over the best baseline across the regime grid |

`E8` additionally writes `figures/generalization.json`, the full
case-by-case result of the three generalization batteries.

The `figures/` directory holds the output, and what is committed there is
what the paper prints. Re-running reproduces it.

Results are cached under `.cache/`, keyed by the exact configuration text
of each run, so re-rendering after a styling change is instant. Delete that
directory to force a full recomputation, which takes roughly an hour on 16
cores. Most of that is the generalization batteries, and most of *that* is
Tian's `K`-stop lookahead on the largest graph: it enumerates ordered
sequences of `K` distinct stops out of the other `N−1` configurations,
which at `N = 10, K = 4` is 9·8·7·6 = 3024 candidates per decision.

### Why the numbers come out the same every time

* A run is a pure function of its configuration text. Every random draw
  comes from one `Xoshiro256++` stream per replication, seeded
  deterministically from the base seed and the replication index.
* Replications are distributed across threads, but their seeds are not, so
  a 1-thread and a 64-thread machine produce identical output.
* All four policies share the derived seed stream, so each configuration
  sees the same arrivals and service draws under every policy. The
  comparison is paired, and the reported significance test is the paired
  per-replication difference.

The two binaries use different seed streams from one another — see
`simulation::seeding`, which documents both. Each gives independent
replications of the same system; they simply do not draw the *same*
replications, so a summary from `dartsim` and a distribution from
`dartsim-samples` are two independent estimates rather than two views of a
single run.

---

## Statistics

The objective is the system-wide weighted P99: all weighted sojourns
`w_i·S_i` are pooled into one distribution and its 0.99 quantile taken.
Every replication keeps that pooled vector in full — not a reservoir — so
its quantile is exact; the reported figure is the mean over replications
with a 95% Student-t interval on `n_reps − 1` degrees of freedom. Per-class
percentiles are computed the same way, one replication at a time, so every
error bar in the paper comes from one method.

---

## License

MIT. See [LICENSE](LICENSE).
