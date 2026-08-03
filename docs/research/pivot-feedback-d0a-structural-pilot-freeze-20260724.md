# Pivot-feedback D0-A structural pilot freeze

Date: 2026-07-24

Status:
`PASS_INDEPENDENT_D0A_STRUCTURAL_FREEZE_D0_PARTIAL_D0B_ONLY_AUTHORIZED`

## What question this stage answers

D0-A asks a deliberately narrower question than the empirical study:

> Can we choose two structurally well-supported fields and two separable
> attention-acceleration episodes in each field without looking at whether
> researchers entered, persisted, succeeded, or improved a model score?

The answer is yes for this current-snapshot OpenAlex projection. That result
is a measurement-feasibility pass, not evidence that collaborators influence
entry or that an attention episode caused a pivot.

## Outcome-blind field freeze

The field selector aggregated only S0 rows with non-null publication years
through 2012. It ranked all 26 fields on eight structural quantities:

- work volume and usable-date coverage;
- usable author-history coverage;
- ORCID-stable author-work density;
- affiliation and institution row density; and
- cap-100 coauthor graph node and edge support.

Each quantity was converted to an exact empirical percentile. The primary
rank was the minimum percentile across the eight quantities, followed by the
mean percentile and numeric field ID. The first pilot field was the global
leader. The second was the highest-ranked field in a different domain. No
entry, persistence, citation-success, overshoot, predictive, model-selection,
or G4 value entered the rule.

The frozen pilot fields are:

| Order | Field | Domain | Global rank | Minimum percentile | Mean percentile |
|---:|---|---|---:|---:|---:|
| 1 | Computer Science (`17`) | Physical Sciences | 1 | 0.64 | 0.77 |
| 2 | Medicine (`27`) | Health Sciences | 3 | 0.52 | 0.75 |

Materials Science (`25`) was global rank 2 but shared the first field's
domain, so the predeclared cross-domain rule skipped it for this two-field
pilot. The full 26-field reserve order remains in the output receipt.

## Provisional episode freeze

For the two frozen fields, the producer rescanned all 4,096 normalized
work-metadata partitions and counted usable primary-topic works through 2011.
For candidate entry year \(t \le 2012\), the structural attention comparison
was:

```text
baseline = works in t-6 through t-4
recent   = works in t-3 through t-1
gain     = recent - baseline
```

Only positive-baseline, positive-gain candidates were retained. Ranking used
the exact rational quantity

```text
gain^2 / (baseline + 1)
```

by integer cross multiplication. The displayed square-root score is
descriptive only and cannot change the order through floating-point
rounding. Within each field, the second episode had to belong to a different
current-snapshot subfield and be at least nine entry years away, making the
full `t-6` through `t+2` windows calendar-disjoint.

| Field | Episode | Topic | Entry year | Current subfield | Baseline | Recent |
|---|---:|---|---:|---|---:|---:|
| Computer Science | 1 | T14430, Higher Education and Teaching Methods | 2003 | Information Systems | 689 | 12,575 |
| Computer Science | 2 | T10579, Cognitive Radio Networks and Spectrum Sensing | 2012 | Computer Networks and Communications | 1,998 | 6,664 |
| Medicine | 1 | T11361, Digital Radiography and Breast Imaging | 2007 | Pulmonary and Respiratory Medicine | 1,557 | 23,073 |
| Medicine | 2 | T12505, Peptidase Inhibition and Analysis | 1991 | Oncology | 2,137 | 13,023 |

These are provisional discovery episodes, not final G0 shocks.

## Independent replay

The producer output is
`/ssd/pivot-feedback-d0-structural-pilot-freeze-v1-full-20260724`. Its
receipt SHA-256 is
`a1d13a21304742437b44400e16544e40e7da4a7f4084f2fc45316aa1ce014a94`.

The independent verifier imported no producer module. It rehashed all 4,096
source artifacts, replayed all 492,293,206 work rows, independently rebuilt
the field ranking, 109,659 selected-field topic-year cells, and 76,062
episode candidates, and recovered the same two fields and four episodes. The
current external verification report is
`/ssd/pivot-feedback-d0-structural-pilot-freeze-v1-full-20260724.verification-v2.json`
with SHA-256
`d46fd953207a483f0f8e0658f2fb2689fa5d358531cb8124f093ec7cdaa4046a`.

The compact tracked summary is
`experiments/evidence/results/pivot_feedback_d0a_structural_pilot_freeze_full_20260724.summary.json`.

## New risks exposed by the freeze

D0-A makes several remaining weaknesses more concrete.

First, primary topics and subfields are current-snapshot labels applied
retrospectively. Their names and parent assignments can be broad or
semantically surprising. A sharp count increase can reflect indexing,
coverage, or classification change rather than real scientific attention.
The selected labels therefore need an outcome-blind content-coherence and
indexing-discontinuity audit before any outcome opens.

Second, these counts are not yet incumbent-only. They do not exclude the
focal candidate, the candidate's local network, or researchers whose entry is
part of the response. D0-B must reconstruct the signal from authors active
before the signal window and implement the registered focal and network
leave-outs.

Third, different subfields and disjoint calendar windows reduce obvious
dependence but do not establish independent author populations or shock
families. Author overlap and common field-wide events remain unmeasured.

Finally, a high structural rank favors fields with strong observable
bibliographic and network surfaces. It improves pilot feasibility but does
not make the fields representative of science as a whole.

## Gate transition

The additive v4 ledger,
`experiments/configs/pivot_feedback_gate_state_v4.yaml`, validates as:

```text
PASS_G_MINUS_1A_S0_D0A_STRUCTURAL_FREEZE_D0B_ONLY_AUTHORIZED
```

The live boundary is:

```text
G-1A  PASS_BOUNDED
S0    PASS_THRESHOLD_FREE_INVENTORY
D0-A  PASS_OUTCOME_BLIND_STRUCTURAL_FREEZE
D0    PARTIAL / NOT COMPLETE
D0-B  AUTHORIZED_OUTCOME_BLIND_AUTHOR_PANEL_ONLY
G0    PENDING / NOT AUTHORIZED
G0.5  NOT_RUN / NOT AUTHORIZED
G1-G3 NOT_RUN
G4    SEALED_NOT_OPENED
```

D0-B may build author opportunities, incumbent-only attention, focal and
network leave-outs, prior-peer activity, recent-collaboration, and
same-affiliation measurements for only the four frozen episodes. It may not
open entry or persistence outcomes. Those outcomes require a later
receipt-bound transition after the measurement panel and temporal leakage
audit pass.

This stage establishes no empirical association, causal effect, information
receipt, belief update, NABM necessity, or scientific effect.

Successor update, 2026-07-24: the first D0-B component has since materialized
83,565 exact pre-decision target-topic works and 5,849,627 focal-field-active
author-episode rows. An independent Arrow `group_by` replay matched all
16,384 output row groups after replaying 1,271,923,464 input rows. The additive
v5 ledger marks only the author-opportunity foundation complete; incumbent
attention, leave-outs, network and affiliation bridges, content/indexing
audits, D0-B, and D0 remain incomplete. See
`docs/research/pivot-feedback-d0b-author-opportunity-foundation-20260724.md`.

Second successor update, 2026-07-24: the work-to-author linkage audit found a
severe heterogeneous observation-universe gap, dominated by author-unlinked
type-`other` records in F27 T11361. The additive v6 ledger therefore retains
the D0-A result as a historical provisional freeze but does not authorize its
four episodes for further leave-out or peer measurement. Only outcome-blind
work-type/linkage ABI redesign and deterministic episode reselection are
authorized. See
`docs/research/pivot-feedback-d0b-selected-work-author-linkage-audit-20260724.md`.
