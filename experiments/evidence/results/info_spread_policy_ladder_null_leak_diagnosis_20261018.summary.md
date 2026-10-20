# Information-spread policy ladder: diagnosis of D4's null-world defect

Date: 2026-10-18

Status: `EXPLORATORY_DIAGNOSIS_NOT_REGISTERED_NULL_CELLS_ONLY`

Code: `scripts/diagnose_info_spread_null_leak_v1.py`, run against D4's frozen
contract and runner (`daf5d80`). No study cell was read.

## Question

[D4](info_spread_policy_ladder_study_v1_d4_20261016.summary.md) found
L0 − F1 material and positive in B/partial_mlp/null. In that world the
features carry no information about veracity. Where does the learner's gain
come from?

## Method

- **Cells.** Null cells only: the first 6 cells × 4 replicates of three
  groups.
  - A/mlp and B/partial_mlp, the groups with a positive interval in D4;
  - B/partial_linear_exact, as a contrast.
- **Measure.** Each policy's own share decisions (probability at least the
  cutoff, before the exploration floor). They are split by the state of the
  item's fact-check at the decision:
  - unrevealed: the item is never fact-checked;
  - pending: it will be, but the check is not yet public;
  - visible: its check is already public.
- **Two routes to separate.**
  - (a) An item's four features identify it, and an MLP can memorize them.
  - (b) In Regime B, the trust feature of an item's sharers moves with that
    item's own fact-check.
- **The trust variant.** `L0_no_trust` holds the trust feature at a constant,
  in both its decisions and its training rows.

## Result

The table shows the precision of L0's own shares minus F1's, as a mean over
6 cells, with the number of cells where it is positive.

| Group / arm | Unrevealed | Pending | Visible |
|---|---:|---:|---:|
| A/mlp: L0 | −0.003 | −0.008 | +0.030 (5 of 6) |
| B/partial_mlp: L0 | +0.000 | −0.011 | **+0.068 (6 of 6)** |
| B/partial_mlp: L0_no_trust | +0.005 | −0.009 | **+0.064 (6 of 6)** |
| B/partial_linear_exact: L0 | +0.005 | −0.006 | +0.003 |

Pooled over runs, L0's share precision on visible items is:

- 0.470 in B/partial_mlp and 0.433 in A/mlp;
- about 0.40 for F1 in every state.

## Reading

- **The gain arises only on items whose fact-check is already public.** On
  the other items, L0 is at the base rate.
- **Route (b) does not explain it.** Holding the trust feature constant
  leaves the gain almost unchanged (+0.064 against +0.068).
- **What remains is route (a).** The MLP recognizes an item whose
  fact-check it has trained on, from the item's features. The exact linear
  learner shows no comparable gain: a linear rule cannot single out one item.
- **The same mechanism works in the study worlds, for L0 only.** T0's
  tables come from other streams. So D4's comparison of L0 with T0 mixes two
  things:
  - learning the relation within the run;
  - recognizing items already fact-checked.
  The null check caught this as designed.

**Limits.**
- Exploratory: 6 cells × 4 replicates, with no registered rule.
- The split counts policy decisions before the exploration floor, not the
  registered share precision.

## Next

The user chose on 2026-10-18 to make the primary measure share precision on
items whose fact-check is not public at the decision. The world and the
learners stay as they are.
