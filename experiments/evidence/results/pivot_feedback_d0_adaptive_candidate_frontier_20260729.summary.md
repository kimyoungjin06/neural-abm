# Pivot-feedback D0 adaptive candidate frontier

Status:
`PASS_INDEPENDENT_ADAPTIVE_FRONTIER_STATE_MACHINE_EXHAUSTIVE_SMALL_FIXTURES_AND_INITIAL_STRUCTURE`

The adaptive state machine was independently replayed over 299,584 exhaustive
small fixtures covering candidate lengths two through six, two subfields, two
entry-year positions, and every binary content assignment.

- All 118,404 fixtures with an eager compatible pair produced the same lazy
  certificate pair.
- All 331,424 mutations that changed a required state to
  `CONTENT_UNRESOLVED` blocked certification.
- All 146,832 checks that left a structurally incompatible earlier row
  unevaluated preserved the pair.

This is an exhaustive bounded fixture proof of the implemented state machine,
not a claim of a general formal mathematical proof.

The initial rank-1-pass structural branches contain 51 rows:

| Field | Frontier | Structural skip | Content required |
|---|---:|---:|---:|
| Computer Science | 45 | 43 | 2 |
| Medicine | 6 | 4 | 2 |
| Total | 51 | 47 | 4 |

The four content-required rows are the existing topics `T14430`, `T10579`,
`T12505`, and `T12421`, all of which have frozen cards. The rank-1-pass branch
is only a structural assumption. It is not a content disposition.

All four required rows remain unresolved. No selection certificate,
reselection, raw title scan, new API request, outcome read, or model run was
authorized or executed.
