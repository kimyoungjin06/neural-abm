# Lean H3 set-identified prospectus v2 analytic result

- Completed: `2026-08-07T02:03:08Z`
- Definition precommit: `9747e5b77157af24838f4f79596832050c5d689d`
- Status: `COMPLETE_SET_IDENTIFIED_ANALYTIC_CONFORMANCE_NO_STOCHASTIC_CANDIDATE_NO_SELECTION`
- Fixtures: 7/7 definition-conformant
- Opened challenge and empirical rows read: `0`
- Candidate, estimator, and support selection: `NO`
- Fresh blind authorized: `false`
- Scientific effect: `NOT_ESTABLISHED`

## What changed

The v1 peak-set candidate treated diffuse or tied peaks mainly as reasons to
drop an episode. The v2 prospectus asks a different question: can ambiguity be
carried forward as a set instead of being forced into an earliest argmax or
discarded?

On closed analytic curves, every event time within an inclusive probability
margin of `0.01` from the maximum belongs to the practical peak set. Entry and
persistence peak sets induce an episode gap set. Those lower and upper bounds
are averaged within each field, producing a four-field interval vector. A
field-balanced interval is retained only as a secondary summary, never as a
replacement for the vector.

The margin was not chosen from opened peak-set metrics. It reuses the
outcome-blind G0 v4 one-percentage-point practical-null as a provisional,
dimensionally compatible anchor. The fixtures do not validate that transfer;
entry and conditional-persistence scales still require a separate audit.

## Analytic findings

| Fixture | Primary result | Meaning |
|---|---|---|
| Clear consistent peak | `[[1,1],[1,1],[1,1],[1,1]]` | Point-identified and directionally consistent. |
| Adjacent near tie | `[[1,2],[1,2],[1,2],[1,2]]` | Ambiguity survives as an interval; no forced tie-break. |
| Staggered fields | `[[-1,-1],[0,0],[1,1],[2,2]]` | The scalar average `0.5` hides mixed field directions. |
| Interior multimodal | `[[0,2],[0,2],[0,2],[0,2]]` | Multimodality survives without a scalar peak. |
| Boundary | `INVALID` | The peak is censored by the event window. |
| Missing registered time | `INVALID` | Informative missingness remains unidentified. |
| Practically flat | `INVALID` | A full flat curve is not rescued by set expansion. |

All seven expected states were reproduced and independently replayed. Four
fixtures were valid, including two set-identified cases; three structural cases
remained invalid.

## Why this is not another gate

`PASS_ANALYTIC_DEFINITION_CONFORMANCE` means the implementation matches its
closed fixture oracle. It is not an estimator performance pass. There is no
coverage floor, power target, validity minimum, or admissibility decision in
this result.

The change actually removes an unnecessary binary: near ties and field
disagreement no longer have to become either a scalar estimate or a terminal
failure. The result can truthfully be an interval or a mixed field vector.
Missingness, flatness, and boundary censoring remain invalid because the
registered window contains insufficient information, not because a governance
threshold was missed.

## What remains unresolved

This prototype uses exact analytic probabilities. It does not say how finite
counts should enter a posterior practical peak set, how simultaneous
uncertainty across five event times should be controlled, or how repeated
authors, episode dependence, and overdispersion should be represented. It also
does not validate `0.01` as a scientifically meaningful peak contrast on both
entry and persistence scales.

The next gate therefore has to be a separate stochastic precommit, not an
opened-panel replay. It must contain:

1. a scale audit for entry and conditional persistence margins;
2. a posterior practical peak-set rule with simultaneous uncertainty;
3. an explicit dependence/overdispersion model or an unresolved boundary;
4. a disjoint nonblind development generator not calibrated to opened metrics;
5. a fresh held-out blind generator only after the stochastic candidate is
   frozen.

The development generator should vary probability scale and denominator
support, contrast relative to the provisional margin, peak-set geometry, and
concordant versus mixed field vectors. No performance threshold or candidate
selection is authorized yet.

The canonical result SHA-256 is
`4e5653e1aa6ce838fc822d9f53c7277daac8c16018db3a4e3bcc5ab5038ba0ed`;
the independent verification SHA-256 is
`62942dcf434264bc4286aea3855dfd08b534ed15359d220129a66c15870d4a8a`.

This is analytic definition conformance only. It is not stochastic estimator
validation, blind confirmation, empirical H3, a mechanism or causal result,
predictive validation, NABM necessity, or a scientific effect.
