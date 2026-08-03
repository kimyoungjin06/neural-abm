# Pivot-feedback D0-B author-opportunity foundation

Date: 2026-07-24

Status:
`PASS_INDEPENDENT_D0B_AUTHOR_OPPORTUNITY_FOUNDATION_D0_AND_D0B_PARTIAL_REMAINING_MEASUREMENTS_ONLY_AUTHORIZED`

## What this component asks

D0-A froze four topic-year episodes without reading entry or persistence
outcomes. This first D0-B component asks the next narrower question:

> For each frozen episode, can we reconstruct exactly which target-topic works
> and which focal-field-active author tokens were observable before the entry
> decision year?

The answer is yes for the current-snapshot OpenAlex projections. This result is
a measurement foundation, not a researcher sample or an empirical effect.

## Frozen input and timing rules

The component is bound to the additive v4 ledger and the four D0-A episodes:

| Field | Topic | Decision year | Portfolio window | Pre-signal incumbent cutoff |
| --- | ---: | ---: | --- | --- |
| Computer Science | T14430 | 2003 | 2000–2002 | first target work before 1997 |
| Computer Science | T10579 | 2012 | 2009–2011 | first target work before 2006 |
| Medicine | T11361 | 2007 | 2004–2006 | first target work before 2001 |
| Medicine | T12505 | 1991 | 1988–1990 | first target work before 1985 |

Only temporal-disposition codes 2 and 3 are used. Selected target-topic works
must be published no later than `t-1`. An author token enters the opportunity
base after at least one focal-field work in `t-3` through `t-1`. The component
reports cap-10, cap-50, cap-100, and all-team profiles without ranking or
selecting one.

## Production result

The production root is
`/ssd/pivot-feedback-d0b-author-opportunity-foundation-v1-full-20260724`.
Its receipt SHA-256 is
`2261e1aba8b3f4f8edea27301603a1ebb6d0931b424cbc5c0ed0a5d2cefef92c`.

The producer scanned 492,293,206 cutoff work rows and 779,630,258 author-event
rows. It materialized:

- 83,565 exact selected target-topic works through `t-1`;
- 5,849,627 author-episode rows;
- four non-ranked team-cap profiles for pre-entry and portfolio-window work
  histories; and
- prior target-topic count, first target-topic year, and pre-signal incumbent
  flags.

| Episode | Selected works | Field-active author rows | No prior target | Prior target | Pre-signal incumbent |
| --- | ---: | ---: | ---: | ---: | ---: |
| F17 T14430, 2003 | 15,508 | 553,419 | 541,917 | 11,502 | 182 |
| F17 T10579, 2012 | 9,280 | 1,231,799 | 1,222,909 | 8,890 | 130 |
| F27 T11361, 2007 | 31,846 | 2,805,352 | 2,797,372 | 7,980 | 3,368 |
| F27 T12505, 1991 | 26,931 | 1,259,057 | 1,246,641 | 12,416 | 5,449 |

The last column is deliberately stricter than the number of all authors ever
seen on pre-signal target-topic works. It requires both pre-signal target-topic
incumbency and focal-field activity in the decision portfolio window. The very
small Computer Science intersections, 182 and 130, are therefore a real
measurement warning. They are not yet a reason to choose a threshold, discard
an episode, or open outcomes.

## Independent replay

The external create-only verification report is
`/ssd/pivot-feedback-d0b-author-opportunity-foundation-v1-full-20260724.verification-v1.json`.
Its SHA-256 is
`6da5e48ffe84ab9d96fc4ba3268751598250c2029a39a3639cb575912f062a99`.

The verifier:

- imports no producer module;
- rehashes all 4,096 cutoff and 4,096 author-event partitions;
- replays 1,271,923,464 source rows in total;
- reconstructs author metrics with Arrow `group_by`, separately from the
  producer's NumPy cumulative-count implementation;
- checks every one of the 16,384 author-output row groups; and
- obtains exact work and author output matches with all forbidden outcome
  columns absent.

## What remains unresolved

This component does not yet produce the registered attention or peer measures.
The remaining D0-B work is:

1. audit current-snapshot topic content coherence and possible indexing
   discontinuities;
2. freeze incumbent membership and construct incumbent-only attention;
3. calculate focal-author and prior-network leave-outs;
4. construct prior-peer target activity and a stricter recent-collaboration
   bridge;
5. add the ORCID-stable subset flag;
6. rescan the raw authorship surface into a receipt-bound
   author-work-institution relation before claiming a same-affiliation bridge;
   and
7. independently audit every `available_at <= t-1` boundary.

The author-history output does not preserve normalized author-work-institution
rows. Same-affiliation therefore cannot be recovered from the current D0-B
foundation alone. It requires a new bounded raw-authorship scan rather than an
inference from field-year affiliation coverage.

## Gate state

The additive v5 ledger,
`experiments/configs/pivot_feedback_gate_state_v5.yaml`, validates as:

```text
PASS_G_MINUS_1A_S0_D0A_D0B_AUTHOR_FOUNDATION_REMAINING_D0B_ONLY_AUTHORIZED
```

The live boundary is:

```text
G-1A  PASS_BOUNDED
S0    PASS_THRESHOLD_FREE_INVENTORY
D0-A  PASS_OUTCOME_BLIND_STRUCTURAL_FREEZE
D0-B  PARTIAL / AUTHOR-OPPORTUNITY FOUNDATION PASS
D0    PARTIAL / NOT COMPLETE
G0    PENDING / NOT AUTHORIZED
G0.5  NOT_RUN / NOT AUTHORIZED
G1-G3 NOT_RUN
G4    SEALED_NOT_OPENED
```

Only the remaining outcome-blind D0-B measurements on the four frozen episodes
are authorized. Entry and persistence outcomes, threshold or team-cap
selection, final candidate eligibility, G0, model fitting, G4, empirical
associations, causal claims, and scientific effects remain unopened,
unauthorized, or not established.

Successor update, 2026-07-24: a selected-work-to-author-token linkage audit
found that only 48,245 of 83,565 selected works have a valid author-token event.
The gap is severely heterogeneous: F27 T11361 has 23,073 signal-window works
but only 1,790 linked works, and its `other` work type has 22,451 pre-decision
works with only 14 linked. The independently verified additive v6 ledger
therefore supersedes the v5 authorization and pauses leave-out/network/bridge
construction. Only outcome-blind measurement-ABI redesign and deterministic
episode reselection are now authorized. See
`docs/research/pivot-feedback-d0b-selected-work-author-linkage-audit-20260724.md`.
