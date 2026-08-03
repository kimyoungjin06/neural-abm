# OpenAlex G-1 Work-Topic Primary-Score Successor

- Status: `PASS_CURRENT_SNAPSHOT_WORK_TOPIC_RELATION_COMPONENT_WITH_PRIMARY_EXACT_SCORE_ADJUDICATION`
- Full current-snapshot scope: `true`
- GPU used: `false`
- Topic classification rerun: `false`
- Raw rows mutated: `false`
- Parent raw topic rows: `928,562,546`
- Canonical relation rows: `928,561,783`
- Same-key duplicate groups: `714`
- Duplicate row excess: `763`
- Different-score conflict groups: `2`
- Primary-exact adjudicated groups: `2`
- Unexpected or unresolved groups: `0`
- Canonical relation SHA-256: `617a508c458ae664689745e8bcf9de0b9b416f1a2bb961a406b96cf74f43873d`
- Output receipt SHA-256: `124ee59348a4f3c73cac8e0d567f764a57cf8ba6700c02ffa05a276e05562b5d`

## What changed

The blocked parent audit remains immutable evidence. This successor rehashed all
1,430 parent receipt artifacts and reused its exact sorted event surface. It
collapsed same-key, same-score duplicates. For the two different-score groups,
it selected a score only because the frozen main row named the same primary
topic and its exact float64 score appeared among that edge's child rows:

| Work | Topic | Raw scores | Selected |
| ---: | ---: | --- | ---: |
| `W5401631` | `T13112` | `0.0805`, `0.0806` | `0.0806` |
| `W26888826` | `T10455` | `0.7346`, `0.7347` | `0.7347` |

Every conflicting raw row and locator remains in the receipt-bound sidecar.
There is no unconditional first, maximum, minimum, or mean score rule.

## Independent verification

The separate read-only verifier rehashed the successor output and frozen parent
controls, replayed both frozen main rows, and compared all 1,420,855,752 parent
events with all 928,561,783 canonical relation records. Counts, keys, and exact
score bits matched. Its status was
`PASS_FULL_READ_ONLY_INDEPENDENT_PRIMARY_ADJUDICATION_VERIFICATION`.

## Claim boundary

This passes only the current-snapshot work-topic membership and narrow canonical
score-selection component. Raw source-ID validity remains `BLOCKED`, `W0`
entity validity remains `UNRESOLVED`, parent historical content identity remains
`UNVERIFIED`, and G-1/G0 remain `PENDING`. It does not establish temporal,
author, candidate, empirical G1, learned-NABM, or scientific-effect claims.
