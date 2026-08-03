# Pivot-feedback current-status and evidence ledger v1

Date: 2026-08-03

Status: `CURRENT_HUMAN_FACING_INDEX_V61_NO_EXECUTION_AUTHORITY`

## Authority rule

This is the canonical human-facing index for the current pivot-feedback work.
It consolidates the latest evidence and claim boundaries, but it is not a gate,
receipt, preregistration, or execution authorization.

The machine-readable authority is
[`pivot_feedback_gate_state_v61.yaml`](../../experiments/configs/pivot_feedback_gate_state_v61.yaml),
SHA-256
`253c661fd529feb0d70733174197be449dfd27403b5881c6510bd7c8257c2715`.
If this document conflicts with that immutable gate, the gate controls and this
ledger must be updated. Earlier gate states, readiness inventories, protocol
status paragraphs, and planning notes remain provenance or narrative context;
they are not the current authorization source.

## Two evidence tracks

The repository contains two separate tracks that must not be promoted into one
another.

| Track | Current supported conclusion | Prohibited promotion |
| --- | --- | --- |
| Method-first NABM | The unit lifecycle can carry bounded classical instances, neural or fixed policies, domain-owned transitions, replicated scenarios, and audit traces. | General superiority over classical ABMs, external validity, or a real-world causal mechanism. |
| Empirical pivot-feedback | Current-snapshot OpenAlex primitives support a reproducible discovery-only entry and three-year persistence measurement chain, and twelve outcome-blind G0 policy choices are serialized. | A passed empirical signature, peer learning, causal influence, NABM necessity, holdout transport, or a scientific effect. |

The current method-first manuscript boundary remains in
[`paper/claim-matrix.md`](../../paper/claim-matrix.md). The prospective
scientific hypotheses and stop rules remain in the
[`pivot-feedback empirical protocol`](../case-studies/researcher-pivot/pivot-feedback-empirical-nabm-protocol.md).
Their dated operational status text predates gate v61 and must not override this
ledger or the current gate.

## Current evidence ladder

| Layer | Current state | What the evidence establishes | What it does not establish |
| --- | --- | --- | --- |
| G-1A bounded source prerequisite | Passed in its bounded role; legacy G-1 remains unpassed. | Exact binding of current-snapshot canonical work time, author-token history, work-topic relation, bibliographic citation chronology proxy, cutoff primitives, and independent source replay. | Natural-person identity, historical OpenAlex visibility, candidate eligibility, G0, or an effect. |
| D0 episode selection | Complete for discovery only. | Four certified episodes: `(17, 14430, 2003)`, `(17, 10579, 2012)`, `(27, 10151, 1993)`, and `(27, 12421, 1953)`. | The three-field G1 panel, the G4 field, or confirmatory episode selection. |
| D0 predecision measurement | Independently verified. | 3,256,271 registered author-episode rows, 74 columns, and explicit temporal-availability classes with zero replay mismatch. | A natural-person panel, final eligibility, a selected graph cap, or an outcome result. |
| D0 entry classification | Independently verified. | An exhaustive observable-history four-state classification over all 3,256,271 rows. | True lifetime-first entry, association, timing-gap evidence, or causality. |
| D0 persistence calibration | Independently verified and descriptive. | Three-year current-snapshot-annotated persistence for the exact 4,301 `NEW_TOPIC_ENTRY` rows. | Adjusted episode comparison, exposure association, researcher productivity or quality, or a scientific effect. |
| G0 v4 | Pending and unsealed. | Twelve outcome-blind policy choices are serialized and independently replayed. | A freeze-ready measurement system, permission to fit a model, or a G0 pass. |
| G1-G3 | Not run. | Prospective hypotheses, baseline ladder, stop rules, and future decision thresholds only. | Any empirical signature, predictive advantage, simulation adequacy, or NABM necessity. |
| G4 | Logically reserved, not byte-sealed, and unopened. | The intended 2020-2022 entry period, follow-up through 2025, and domain/author-cold rule are declared. | A selected G4 domain or field, materialized artifact set, production preflight, byte seal, receipt, or holdout result. |

## Exact D0 descriptive accounting

The independently replayed entry classification is:

| Observable-history state | Rows |
| --- | ---: |
| `NO_FOCAL_ENTRY` | 3,246,984 |
| `NEW_TOPIC_ENTRY` | 4,301 |
| `RETURN` | 125 |
| `CONTINUATION` | 4,861 |
| Total | 3,256,271 |

`NEW_TOPIC_ENTRY` is 0.132084% of the registered author-episode rows. It means
positive focal-year target-topic work and zero prior observed target-topic work
under the frozen current-snapshot ABI. It is not a true lifetime-first event.

Among the 4,301 entry rows, 1,487 have at least one eligible target-topic work
at `t+1..t+3` and 2,814 do not. The unadjusted descriptive persistence rate is
34.573355%. Episode rates are 30.42%, 39.69%, 36.69%, and 20.73% in certified
episode order. These episodes differ in field, topic, year, and opportunity
structure, so the rates cannot be used to rank or select episodes.

Primary source artifacts:

- [`G-1A receipt`](../../experiments/evidence/results/pivot_feedback_g_minus_1a_receipt_20260723.summary.json), SHA-256 `8383e1884c3679df225b743f775832f02e4d767ac5a6b047dbde8aacf9e12802`;
- [`D0 episode reselection`](../../experiments/evidence/results/pivot_feedback_d0_certified_episode_reselection_20260730.summary.json), SHA-256 `b128f12d2e91510a7da5a689c16667f7fd4ec9329f3ff68ceda32adf3533ec25`;
- [`D0 predecision measurement`](../../experiments/evidence/results/pivot_feedback_d0_certified_predecision_author_measurement_20260731.summary.json), SHA-256 `a9ecd79274139e5b9a375276eacdc51703fd561cabe38f8e8ce4356c3c7c311f`;
- [`D0 entry classification`](../../experiments/evidence/results/pivot_feedback_d0_production_pivot_classification_20260731.summary.json), SHA-256 `f98ebf278bc3b6bbdc2fef59bf2bd5a8dd4e1c7da3d14f9fccde454ffc94c4c7`;
- [`D0 persistence source`](../../experiments/evidence/results/pivot_feedback_d0_persistence_source_production_20260731.summary.json), SHA-256 `bdb3356bfbde854305972448f0e5a45f889437f2e2f8f9702483a8cb0f357442`;
- [`D0 persistence readout`](../../experiments/evidence/results/pivot_feedback_d0_persistence_calibration_readout_20260731.summary.json), SHA-256 `8cc7aa7190fd96e79d338303083da1cb65baab1d170df2fb3d5a94d3ce65a620`.

## Exact G0 v4 state

The current contract is
[`pivot_feedback_g0_measurement_freeze_v4.yaml`](../../experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml),
SHA-256
`cd49a31ab67e7a66d038b3198493df8b3a309aa14ee9de80410557a194581ce4`.
Its independent binding summary is
[`pivot_feedback_g0_measurement_freeze_v4_binding_20260731.summary.json`](../../experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v4_binding_20260731.summary.json),
SHA-256
`f50d668aca02e284943d71108d1e7cbf71ff84af5947a179edb29c02f145ba05`.

Twelve choices are resolved:

1. no minimum-history threshold for the D0 observable-history classifier;
2. the exact four-state pivot definition;
3. incumbent-only robust attention acceleration with modified `z >= 3.0`;
4. an age-three field-by-publication-year top-decile-anchored citation proxy;
5. three complete calendar years for citation maturation;
6. a `[-0.01, 0.01]` marginal-probability practical-null interval with TOST;
7. at most ten controls per case within the exact risk set;
8. nearest-rank P01/P99 stabilized-weight winsorization and mean-one normalization;
9. a 500/100/100 cohort floor;
10. suppression of counts 1 through 10 and derivable complements;
11. a 50/11/11 floor in every signal-by-capacity cell; and
12. centered weighted VIF strictly below 5 for signal, capacity, and interaction.

Items 3-12 are conservative outcome-blind policies. They are not
literature-mandated truths, efficiency or power guarantees, production results,
or empirical effects. Support or VIF failure collapses the mechanism block to
`ANY_PEER_SIGNAL_OR_CAPACITY` and forfeits the separate information-versus-
capacity claim.

Seven quantitative choices remain unresolved:

- `g1_replication_tolerance`;
- `g2_score_margin`;
- `g2_maximum_calibration_degradation`;
- `g2_macro_trajectory_tolerance`;
- `simulation_draws_or_mcse_stopping_rule`;
- `parameter_budget`; and
- `tuning_budget`.

All fifteen required sections remain not freeze-ready. The holdout receipt is
absent, `pass_g0` is false, confirmatory and holdout access are unauthorized,
no model has been fitted, and G4 is unopened.

## Scientific claim disposition

### Supported now

- The method-first unit is reusable and auditable within its tested surfaces.
- Bounded deterministic classical instances and explicitly labeled near
  variants run through the shared lifecycle.
- Targeted toy and adapter studies demonstrate bounded robustness and
  extensibility, not general superiority.
- The hand-set researcher-pivot studies demonstrate replicated scenario and
  adaptive-lifecycle machinery, not empirical estimates.
- The OpenAlex pipeline reproducibly constructs the bounded D0 discovery
  measurement and descriptive persistence accounting above.
- Twelve prospective policy choices have been declared outcome-blind and
  fixture-tested.

### Not established

- that an attention surge predicts or causes later entry;
- that prior-collaborator public success adds prediction beyond peer activity;
- that entry peaks after later-cohort persistence;
- that a capacity bridge predicts persistence conditional on the signal proxy;
- that any observed association is social learning or causal influence;
- that NABM improves on the registered simpler baseline ladder;
- that a simulation phase boundary is robust;
- that any result transports to a domain-and-author-cold G4 holdout; or
- that the empirical study has established a scientific or policy effect.

## Holdout boundary

The current holdout-input resolution is
[`pivot_feedback_holdout_production_input_resolution_20260731.summary.json`](../../experiments/evidence/results/pivot_feedback_holdout_production_input_resolution_20260731.summary.json),
SHA-256
`ce2f018f35d9b383fe2f067f9c4c974f21fc754af80f07c04320edfeac29e267`.
It reports no selected G4 domain, no selected G4 field, no materialized artifact
set, no source-root binding, no complete artifact list, no production
preflight, no byte seal, and no receipt. A logical temporal reservation is not
a byte seal.

## Next authorized work

Gate v61 authorizes exactly one next package:

`G0_G1_REPLICATION_TOLERANCE_PRECOMMITMENT_CLOSED_FIXTURE`

That package may define only `g1_replication_tolerance`, outcome-blind, with its
estimand scale, direction, interval, failure rule, policy-value label, and an
independent closed synthetic fixture. It may not modify G0 v4, bind the value
into a G0 successor, resolve G2 or G3 choices, complete required sections, read
G1 or holdout rows, choose or materialize G4, create a production seal, pass
G0, fit a model, or establish an effect.

## Update contract

Update this ledger only when a new independently verified gate supersedes v61
or when a bound evidence artifact is formally replaced. An update must:

1. name the new machine-readable authority and its exact hash;
2. preserve the two-track boundary;
3. distinguish policy choices from empirical results;
4. retain `PENDING`, `NOT_RUN`, `UNVERIFIED`, and `BLOCKED` labels where true;
5. reconcile every D0, G0, G1-G4, holdout, and scientific-claim row; and
6. pass the ledger drift tests before it is treated as current.
