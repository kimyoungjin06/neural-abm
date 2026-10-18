# Information-spread policy ladder: D2d specification

Date: 2026-10-18

Status: fixed before the D2d run. Controls only; no study cell is scored.

## Why D2d

[D4](../../experiments/evidence/results/info_spread_policy_ladder_study_v1_d4_20261016.summary.md)
failed its null-world check. The cause was found by an
[exploratory diagnosis](../../experiments/evidence/results/info_spread_policy_ladder_null_leak_diagnosis_20261018.summary.md):

- an MLP learner recognizes items it has seen fact-checked;
- so share precision over all items rewards that recognition, even where
  the features carry no information.

On 2026-10-18 the user chose a new primary measure: **share precision on
unverified items**. These are items whose fact-check is not public at the
decision, because the item is never checked or its check is still pending.
The world and the learners stay as they are.

D2d checks the group rule on this measure before a new freeze, as D2c did
for D3a.

## Measure

`scripts/info_spread_unverified_measure_v1.py` measures without changing the
frozen simulator.

- A wrapper logs every decision of a probability policy.
- It reapplies the exploration floor's keyed draws, as the simulator does.
- It classifies each decision by the item's fact-check state at that step.
- Every run checks that the reconstructed shares per item, and the decision
  count, equal the simulator's own.

F0 runs in probability form (1 at or above the threshold, else 0), which
makes the threshold rule's decisions exactly. The old measure, share
precision over all items, is recorded beside the new one.

## Controls

| Control | Pairs | World | Known answer |
|---|---|---|---|
| NEG | T0-1a − T0-1b, T0-4a − T0-4b (donor seeds 1000–1003 against 2000–2003) | Linear | No difference: both arms are tables written before the run |
| POS-sign | L0 − T0-1, L0 − T0-4 | Linear, with the credibility weight +3 or −3 drawn per replicate | L0 learns its own run's sign; tables cannot |
| NULL | L0 − F1, L0 − T0-1, L0 − T0-4 | Null: features carry no information | No learner gain on unverified items |

**Groups.** Each control runs in four groups:
- Regime A: `linear_exact` and `mlp`;
- Regime B: `partial_linear_exact` and `partial_mlp`.

**Size.** 20 cells per group, each with its own world seed (5000 + cell
index), and 16 replicates per cell.

**Fixed from D3a.** These are D3a's:
- the world sizes (1,000 agents, 60 steps, 20 items per step, ε = 0.02);
- observability (public fact-checks, one row per agent and item);
- the learners (exact linear fits; MLPs at the Pivot settings on GPU 2;
  partial pooling in Regime B).

## Rule

The group rule of D2b–D4 applies on the new measure:

- the cell-resampled 95% interval of the mean cell Δ excludes zero;
- and that mean is at least 0.01 in magnitude.

**Adequate** if all three hold on the unverified measure:

1. no NEG pair is material (8 pairs);
2. L0 − T0-1 in POS-sign is material and positive in all 4 groups;
3. L0 − F1 in NULL is not material and positive in any of the 4 groups.

**Reported, not judged.** NULL on the old measure. D4's defect should
reappear there in at least the B/partial_mlp group. That would show the
control can detect it.

## Run

- **Host.** GPU-Server, GPU 2 for the MLPs, 24 workers, with per-task
  progress.
- **Pilot.** A full-load pilot (2 cells per group) sets the time estimate,
  quoted as a range.
- **Record.** A summary and a result test.

**If D2d is adequate.** The next steps, as the user's choice entails:

- freeze study v2 (D3a v2), with the unverified measure as the primary
  measure, the null check on it, and fresh world seeds;
- freeze the LLM arms (D3b) on the same measure;
- run D4 v2.

**If it is not adequate.** No freeze follows, and the finding goes to the
user.
