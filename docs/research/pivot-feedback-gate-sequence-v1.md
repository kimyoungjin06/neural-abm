# Pivot-feedback acyclic gate sequence v1

Status: `PASS_ACYCLIC_GATE_SEQUENCE_CONTRACT_ONLY`

The machine-readable amendment is
`experiments/configs/pivot_feedback_gate_sequence_v1.yaml`. The read-only
validator is `scripts/verify_pivot_feedback_gate_sequence_v1.py`.

## Why the amendment is necessary

The legacy pending contracts contain a logical cycle when read as one
execution order:

- G0 requires a passed G-1 readiness receipt; while
- the legacy G-1 registry requires a final panel whose eligibility,
  maturation, sampling, and weighting rules must be resolved by G0.

The amendment does not rewrite those evidence-bearing v0 files. It binds their
exact SHA-256 values and separates their mixed responsibilities into ordered
successor gates.

## Validated order

```text
G-1A  primitive and source-fidelity readiness
  |
  v
S0    threshold-free structural inventory
  |
  v
D0    discovery-only calibration through 2012
  |
  v
G0    quantitative measurement freeze and holdout seal
  |
  v
G0.5  final-panel conformance, leakage, and support audit
  |
  v
G1    empirical replication, 2013-2016
  |
  v
G2    explicit baselines and model selection, 2017-2019
  |
  v
G3    conditional NABM or simpler-model resolution
  |
  v
G4    sealed temporal, domain, and author-cold holdout, 2020-2022
  |
  v
G5    optional external causal axis
```

The validator found 10 nodes, 9 forward edges, and no cycle. The final cutoff
panel has one owner, G0.5. It is explicitly not a prerequisite for G-1A or G0.

## Field-structure split

S0 can measure only threshold-free structure. It cannot use eligible-author
count, matured opportunity count, fixed-horizon outcome coverage, independent
shock count, effect sizes, overshoot timing, or predictive performance.

After G0 freezes the missing definitions, G0.5 may measure the corresponding
threshold-dependent eligibility, opportunity, horizon, episode, and
signal-by-bridge support. This prevents a nominally outcome-blind pre-G0 audit
from silently selecting unresolved measurement thresholds.

D0 may inspect outcomes only through 2012 to calibrate measurements. It cannot
select final fields by effect size, move the confirmatory periods, pass G1, or
open G4.

## Bound evidence and immutable parents

The amendment binds:

- the legacy fail-closed G-1 registry;
- the pending G0 overlay;
- the cross-field matrix;
- the peer-mechanism registry;
- the full normalized cutoff-primitives result; and
- the completed full source-replay result.

The four legacy configs remain byte-for-byte unchanged. The amendment contract
SHA-256 is
`3a3a655234c4efb74a4900e65624c903176615cee19da760cd06b1416e3fd365`.

## Current boundary

The v1 amendment pass itself validates an acyclic prospective contract only.
It does not by itself issue receipts or establish an empirical effect.

Subsequent execution on 2026-07-23 issued the bounded G-1A receipt and completed
the normalized-surface S0 core inventory. The latter does not pass S0:
ORCID/token stability, author-work affiliation, and distinct coauthor graph
topology remain `NOT_RUN`. D0, G0, G0.5, model fitting, G4, empirical
signatures, and scientific effects therefore remain unauthorized, pending, or
not established. Current evidence is summarized in
`docs/research/pivot-feedback-g-minus-1a-and-s0-core-20260723.md`.

Successor update, 2026-07-24: the historical statement above remains the
correct boundary of the v2 partial ledger, but it has now been superseded by
an additive v3 state. The full ORCID/affiliation and three non-ranked graph
profiles each passed independent verification; their three-component
completion receipt passes S0 and authorizes only pre-2013 D0 discovery. G0,
G0.5, model fitting, G4, empirical effects, and scientific effects remain
closed. See
`docs/research/pivot-feedback-s0-completion-20260724.md`.

Second successor update, 2026-07-24: D0-A then selected Computer Science and
Medicine by pre-2013 threshold-free structural coverage and froze two
calendar-disjoint current-snapshot primary-topic episodes per field. An
independent verifier replayed all 4,096 cutoff partitions and recovered the
same selection. The additive v4 ledger records D0 as partial and authorizes
only D0-B outcome-blind author-opportunity and leave-out construction. Entry
and persistence outcomes, G0, model fitting, and G4 remain closed. See
`docs/research/pivot-feedback-d0a-structural-pilot-freeze-20260724.md`.

Third successor update, 2026-07-24: the outcome-blind D0-B
author-opportunity foundation then selected 83,565 exact target-topic works
through `t-1` and materialized 5,849,627 author-episode rows. A replay
independent of the producer used Arrow `group_by` and matched all 16,384
output row groups after replaying 1,271,923,464 source rows. The additive v5
ledger records this one
component as complete but keeps D0-B and D0 partial. It authorizes only the
remaining outcome-blind content, incumbent, leave-out, peer, collaboration,
affiliation, ORCID-subset, and leakage measurements. Entry and persistence
outcomes remain closed. See
`docs/research/pivot-feedback-d0b-author-opportunity-foundation-20260724.md`.

Fourth successor update, 2026-07-24: joining the 83,565 selected works to
valid author-token events exposed a severe heterogeneous gap. F27 T11361 has
23,073 signal-window works but only 1,790 linked works; 21,044 of those signal
works are type `other`, with only 9 linked. A producer-independent replay
reconstructed all 106,148 author incidences and all 1,200 profile cells. The
additive v6 ledger supersedes v5, pauses leave-out/network/bridge construction,
and authorizes only outcome-blind work-type/linkage ABI redesign followed by a
deterministic episode-reselection successor. Entry and persistence remain
closed. See
`docs/research/pivot-feedback-d0b-selected-work-author-linkage-audit-20260724.md`.

Fifth successor update, 2026-07-30: the content-validity frontier closed after
an exact nine-row, deterministic accept-only external-title/PubMed replay.
The rule adds two HCV-specific accepts, moves Medicine rank 5 to
`CONTENT_PASS` at the frozen 32-row lower-bound threshold, and leaves seven
rows unresolved rather than converting them to rejects. The amendment is
explicitly retrospective and post-diagnostic, not out-of-sample validation.
Independent selection replay issued field certificates for Computer Science
ranks `[1, 45]` and Medicine ranks `[5, 6]`, plus the global selection
certificate. Additive gate v36 authorizes only deterministic materialization
of those exact four episodes. It has not executed that reselection or opened
author-panel rebuilding, pivot classification, outcomes, fitting, or G4. See
`docs/research/pivot-feedback-d0-rank5-post-diagnostic-certificate-closeout-v1.md`.

Sixth successor update, 2026-07-30: the certificate-bound deterministic
episode reselection materialized the exact four measurement-ABI candidate
rows and passed an independent replay over all 70,277 source candidate rows
and 51 frontier rows. Computer Science ranks `[1,45]` and Medicine ranks
`[5,6]` are now the active selection; three prior episodes are retained,
Medicine rank 5 is newly selected, and provisional Medicine rank 1 is
deselected. Gate v37 authorizes only the outcome-blind selected-work manifest
and author-opportunity panel rebuild for those four episodes. The old panel is
superseded provisional evidence. Leave-out/network/bridge construction,
pivot classification, outcomes, fitting, and G4 remain closed. See
`docs/research/pivot-feedback-d0-certified-episode-reselection-v1.md`.
