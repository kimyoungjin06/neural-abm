# N15 parity: recorded Pivot runs through the package's policy-ladder semantics

Date: 2026-10-20

Status: fixed before the parity runs.

## Why

On 2026-10-20 the user approved implementing
[Decision 0017](../decisions/0017-policy-ladder-semantics-extraction.md) (N15).
The package now has two new modules:

- `neural_abm.policy_ladder` (torch-free):
  - the prior rule, keyed randomness, and the exploration floor;
  - the outcome record;
  - prior-boosted trees, which import scikit-learn on first use.
- `neural_abm.policy_learners` (the `torch` extra): the residual linear and
  MLP learners.

The decision asks for two things before it is accepted:

- the package must reproduce the recorded Pivot run that N14 reproduced
  through the frozen core;
- the outcome record's ordering rule must be fixed against a recorded Pivot
  run with delays (open item O27).

## What the package changes from the frozen core

**No global state.**
- The core's MLP set torch's global seed; the package's MLP draws its
  initial weights from a private generator, with the same draws.
- The core's training set the thread count and deterministic algorithms; the
  package leaves both to the caller.

**The ordering rule.** The outcome record returns rows ordered by the step
from which they are visible, then by insertion. For a caller that adds
outcomes as they are released, this is the core's insertion order.

## Checks before the freeze

**Package tests** (`tests/test_policy_ladder_package.py`):
- the package reproduces the core byte for byte:
  - keyed randomness, the floor, and the prior;
  - the linear learner, the MLP under two settings, and the trees;
  - the record;
- fitting leaves torch's random state, thread count, and determinism flag
  unchanged;
- the information-spread simulator gives the same decision digests on the
  package's record.

**Smoke checks of the runner**
(`scripts/reexpress_pivot_through_policy_ladder_package_v1.py`):

- v3 smoke tasks match Pivot's rows.
- The delayed field arms match Pivot's rows with either insertion mode.
- Event insertion on the core's insertion-order record differs from Pivot's
  rows under both failure delays, and matches under immediate visibility.
  So the event mode tests the ordering rule.

## Runs and criteria

All runs are on the workstation where the targets were recorded. They use
16 spawned workers, each with one thread, as recorded.

| Run | Target | Criterion |
|---|---|---|
| `v3` | The Pivot LLM-prior pilot v3 (`87c90eb`): 288 tasks, its three field-public learner arms through the package | The whole `runs.jsonl` equals the record (sha256 `1f532d98…`) |
| `abm2 --insertion release` | The nonlinear policy ABM v2 (`c56d741`): its linear, tree, and MLP field-public arms, 16 replicates × 3 visibility regimes (immediate, failure lag 2, failure lag 4) × 3 worlds | All 432 rows equal the recorded rows |
| `abm2 --insertion event` | The same, with outcomes added when realized and tagged with their release step | All 432 rows equal the recorded rows |

**What follows.**
- **If all three hold:**
  - Decision 0017 is accepted for the `main` / next-alpha candidate;
  - O27 is answered by the ordering rule;
  - the outcome record is extracted with the other primitives.
- **If any fails:** the mismatched rows are reported, and the decision stays
  proposed.

## Boundary

The runs call no LLM and read no empirical data. They reinterpret no Pivot
result, and they do not change a frozen script or a released tag.
