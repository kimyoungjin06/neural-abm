# Pivot-feedback D0-B selected-work author-linkage audit

Date: 2026-07-24

Status:
`PASS_INDEPENDENT_D0B_LINKAGE_AUDIT_MEASUREMENT_REVIEW_REQUIRED_LEAVEOUT_BUILD_PAUSED`

## Why this audit was necessary

The first D0-B component correctly reconstructed two separate surfaces:

- 83,565 current-snapshot primary-topic works available by `t-1`; and
- 5,849,627 focal-field-active author-episode opportunity rows.

Those surfaces were each independently reproducible, but their join had not
been audited. That join is essential. An all-work attention surge and an
author-linked peer or network measure can only be compared as parts of the same
mechanism if they describe a sufficiently coherent observation universe.

This audit therefore asks:

> How many frozen attention works link to at least one valid, distinct
> current-snapshot OpenAlex author token, and is that linkage stable across
> episodes, calendar windows, work types, and team caps?

No entry, persistence, realized-citation, predictive, model-selection, or G4
outcome was opened.

## Exact linkage result

The production root is
`/ssd/pivot-feedback-d0b-selected-work-author-linkage-audit-v1-full-20260724`.
Its receipt SHA-256 is
`fc9f18cfdaaa9a73e39592d5001e4c8abe88d90bc569d27f6bb772e9451e345b`.

Across the four episodes:

- selected pre-decision works: 83,565;
- works with at least one valid author token: 48,245;
- zero-valid-token works: 35,320;
- declared author slots: 145,682;
- distinct valid author-work incidences: 106,148; and
- declared slots minus distinct valid token incidences: 39,534.

The last difference is not labeled as exclusively null author IDs. It can also
include repeated token slots. The audit establishes the join deficit, not a
complete decomposition of its raw-source causes.

| Episode | Overall linked / selected | Reference linked / selected | Signal linked / selected |
| --- | ---: | ---: | ---: |
| F17 T14430, 2003 | 15,109 / 15,508 | 663 / 689 | 12,538 / 12,575 |
| F17 T10579, 2012 | 8,111 / 9,280 | 1,720 / 1,998 | 6,190 / 6,664 |
| F27 T11361, 2007 | 8,265 / 31,846 | 1,276 / 1,557 | 1,790 / 23,073 |
| F27 T12505, 1991 | 16,760 / 26,931 | 1,568 / 2,137 | 9,243 / 13,023 |

## Decisive failure mode

The F27 T11361 episode is not a modest missing-data case. Its signal window has
23,073 all-work records but only 1,790 author-linked records; 21,283 have no
valid author-token event.

Work-type decomposition explains most of the discontinuity:

- `other`: 22,451 pre-decision works, 14 author-linked;
- `article`: 8,332 pre-decision works, 7,347 author-linked; and
- in the signal window specifically, `other` contributes 21,044 works with
  only 9 author-linked, while `article` contributes 1,786 with 1,576 linked.

The original all-work surge of 23,073 versus 1,557 works is therefore largely
an `other`-type, author-unlinked influx. The article surface changes much less:
1,786 signal-window works versus 1,229 reference-window works. This does not
show that T11361 has no scientific activity. It shows that the all-work episode
cannot be used unqualified as the field-attention parent of author-level
incumbent or network measurements.

T12505 also has material, though less extreme, linkage loss. By contrast, the
two Computer Science episodes retain 97.4% and 87.4% of pre-decision works.
The gap is thus heterogeneous and cannot be repaired by multiplying every
episode by one global coverage factor.

## Independent verification

The external create-only report is
`/ssd/pivot-feedback-d0b-selected-work-author-linkage-audit-v1-full-20260724.verification-v1.json`.
Its SHA-256 is
`4151139725ba2306b5e57dd31528ac7d7c3160dc823f017cf4049cff26514a91`.

The verifier:

- imports no producer module;
- rehashes all 8,192 cutoff and author-event source partitions;
- replays 1,271,923,464 source rows;
- independently reconstructs all 106,148 author incidences with Python
  sets and work-level grouping;
- matches work metadata and linkage flags exactly; and
- independently reproduces all 1,200 episode-by-work-type-by-window-by-cap
  profile cells.

The verification status is:

```text
PASS_INDEPENDENT_D0B_SELECTED_WORK_AUTHOR_LINKAGE_AUDIT_SEVERE_HETEROGENEOUS_LINKAGE_GAP_REQUIRES_MEASUREMENT_REVIEW
```

## Decision

No episode has been excluded and no eligible work-type or linkage threshold has
been selected. Instead, the current leave-out/network branch is paused.

The reason is conceptual, not merely statistical:

```text
current all-work attention
        !=
stable author-linked observation universe
        -> incumbent/network leave-out is not yet well-defined
```

The next authorized step is a versioned, outcome-blind measurement-ABI
successor that:

1. freezes eligible scholarly work types;
2. freezes paratext and retraction treatment;
3. freezes the valid-author-linkage requirement and missingness reporting;
4. recomputes the complete field-topic-year candidate universe under that ABI;
5. reruns the deterministic episode selection or fails coverage;
6. performs content-coherence and indexing-discontinuity review; and
7. independently replays the successor freeze.

Only after that successor passes may incumbent attention, focal/network
leave-outs, prior-peer activity, or collaboration and affiliation bridges
resume.

## Gate state

The additive v6 ledger,
`experiments/configs/pivot_feedback_gate_state_v6.yaml`, validates as:

```text
PASS_D0_LINKAGE_GAP_REVIEW_ABI_REDESIGN_ONLY_AUTHORIZED_LEAVEOUT_AND_OUTCOMES_FAIL_CLOSED
```

The live boundary is:

```text
G-1A  PASS_BOUNDED
S0    PASS_THRESHOLD_FREE_INVENTORY
D0-A  HISTORICAL PROVISIONAL FREEZE
D0-B  PARTIAL / LINKAGE AUDIT PASS / MEASUREMENT REVIEW REQUIRED
D0    PARTIAL / NOT COMPLETE
G0    PENDING / NOT AUTHORIZED
G0.5  NOT_RUN / NOT AUTHORIZED
G1-G3 NOT_RUN
G4    SEALED_NOT_OPENED
```

The prior v5 authorization for remaining D0-B measurements is superseded by
this narrower v6 state. Only outcome-blind ABI redesign and a deterministic
episode-reselection successor are authorized. Entry and persistence outcomes,
leave-out/network/bridge construction, G0, model fitting, G4, empirical
associations, causal claims, and scientific effects remain paused,
unauthorized, unopened, or not established.
