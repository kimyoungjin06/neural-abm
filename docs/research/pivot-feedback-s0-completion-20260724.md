# Pivot-feedback S0 completion

Date: 2026-07-24

Status:
`PASS_S0_THRESHOLD_FREE_FIELD_AUTHOR_GRAPH_AND_AFFILIATION_INVENTORY_D0_DISCOVERY_ONLY_AUTHORIZED`

## What passed

S0 now has one independently verified completion receipt over all three
predeclared threshold-free components:

| Component | Full scope | Compact rows | Independent status |
|---|---:|---:|---|
| core field-year and attention inventory | 492,293,206 works; 779,630,258 author events | 18,544 core; 18,518 attention | pass |
| ORCID and affiliation projection | 1,361,551,038 authorship rows | 16,758 field-year | pass |
| non-ranked coauthor graph profiles | 4,096 author-event artifacts | 55,632 cell-cap | pass |

The completion verifier rehashed 13 current artifacts and receipts plus three
external verification reports. It also required the three component verifier
hashes and its own hash to match the current repository. The compact receipt
is
`experiments/evidence/results/pivot_feedback_s0_completion_full_20260724.summary.json`
with SHA-256
`3dafee229190047ae14dfeb19f7d285a803138f4a446975993261956d5c00d93`.

## ORCID and affiliation structure

The full raw source contained 458,308,219 non-null ORCID rows. Of these,
458,276,860 passed the frozen syntax rule and 458,275,464 also passed the
checksum rule. Exactly 8,324,011 current OpenAlex author tokens had one
distinct checksum-valid ORCID; no token had multiple valid ORCIDs.

That apparent within-token stability is not person-identity proof. The audit
also found 35,765 valid ORCIDs shared across 78,963 stable author tokens, with
as many as 37 tokens on one ORCID. The primary scientific unit therefore
remains a current-snapshot OpenAlex author token, with the ORCID-stable subset
reserved for sensitivity analysis.

The projection counted 487,064,370 grammar-valid author-work rows with a
nonempty raw affiliation and 442,695,661 with at least one grammar-valid
institution token. It observed 103,574 distinct current-snapshot institution
tokens. These are coverage and token diagnostics, not historically observed
affiliations.

## Exact graph structure

All three finite declared-team caps were frozen before the full output was
inspected and remain non-ranked:

| Cap | Field-year-summed nodes | Distinct undirected edges | Connected components |
|---:|---:|---:|---:|
| 10 | 352,175,135 | 776,875,376 | 97,445,391 |
| 50 | 378,768,092 | 1,207,070,104 | 96,253,316 |
| 100 | 380,010,841 | 1,289,015,377 | 96,201,999 |

The graph product restored the author-event year-zero sentinel to 78 null-year
profile rows, all with nonzero node support. It required materialized node and
edge counts to equal the compact sums at every cap, then deleted all raw
author and edge intermediates before writing the receipt.

These counts are summed within field-year cells. They are not global unique
people, one global collaboration graph, or evidence that one cap is best.
The unrestricted 47,627,187,584 declared pair-incidence profile remains
diagnostic only and was not edge-materialized.

## Gate transition

The immutable v2 ledger remains the historical partial-S0 record. Its additive
successor is
`experiments/configs/pivot_feedback_gate_state_v3.yaml`, whose validator
returns:

```text
PASS_G_MINUS_1A_AND_S0_COMPLETE_D0_DISCOVERY_ONLY_AUTHORIZED
```

The live state is:

```text
G-1A  PASS_BOUNDED
S0    PASS_THRESHOLD_FREE_INVENTORY
D0    AUTHORIZED_NOT_RUN, entry year <= 2012
G0    PENDING / NOT_AUTHORIZED
G0.5  NOT_RUN / NOT_AUTHORIZED
G1-G3 NOT_RUN
G4    SEALED_NOT_OPENED
G5    NOT_RUN_OPTIONAL
```

D0 may calibrate measurements on discovery outcomes through 2012. It may not
select final fields by effect size, change confirmatory periods, materialize a
final panel, fit the confirmatory model, or open G4.

## Claim boundary

S0 is a structural data-readiness pass, not an empirical finding. It does not
establish natural-person identity, ORCID ownership, historical affiliation or
collaboration observability, candidate eligibility, information receipt,
belief updating, a preferred graph cap, final fields or thresholds, an entry
or persistence association, a causal effect, or a scientific effect.

Successor update, 2026-07-24: D0-A has since frozen two outcome-blind pilot
fields and four provisional attention episodes and passed a second full
source replay. The additive v4 ledger now authorizes only D0-B
outcome-blind author-opportunity and leave-out measurement construction.
Entry and persistence outcomes remain unopened, and D0 remains incomplete.
See
`docs/research/pivot-feedback-d0a-structural-pilot-freeze-20260724.md`.

The next executable work is therefore D0-B on the four frozen episodes. It is
primarily CPU/I/O work; GPU training is not yet authorized.
