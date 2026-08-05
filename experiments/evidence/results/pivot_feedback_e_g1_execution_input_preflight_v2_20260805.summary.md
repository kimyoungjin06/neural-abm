# E/G1 execution-input preflight v2 independent verification

Date: 2026-08-05

Status:
`PASS_INDEPENDENT_E_G1_EXECUTION_INPUT_PREFLIGHT_V2_COMPLETE_NO_G1_RUN`

## Outcome

The additive pre-read completeness correction passed independent replay. It
binds the three input roles that the v1 preflight omitted: citation chronology
for age-3 availability and the frozen future-citation control, the complete
topic-to-subfield taxonomy for the alternative-pivot and unrelated-topic
control, and raw authorship data for the required ORCID-linked and affiliation
sensitivities.

This correction was detected and completed before the first confirmatory read.
It changes no scientific rule, field, episode, year, threshold, control,
estimand, or decision rule.

## Exact added inputs

| Input | Independently replayed identity |
| --- | --- |
| Citation chronology | 4,096 artifacts; 22,721,388,007 bytes; canonical inventory `afac933e0fb067d4b69486b8f038cf1cd5fc3d3d1de04fa13f33b165d8dfc98b` |
| Topic taxonomy | 4,516 rows; SHA-256 `566facdbc02bf4a37783fe64db4c1d104d00b73e250999bd824d560083827ba5` |
| Raw authorship | 1,363 files; 103,412,497,044 bytes; path-size inventory `5ca11e55066c14751a0fd2dc3d8631ef75621109026e8b5fb7250f6ebdaf90a9` |

The raw authorship source also replays the pre-existing content-freeze and
checkpoint-tree hashes. Only source-file stats and one taxonomy Parquet footer
were inspected; no source row, analytic row, confirmatory row, or holdout row
was read.

## Effective execution boundary

The execution authority is now the conjunction of immutable preflight v1 and
independently verified v2. It contains seven effective input groups and eight
output schemas, adding only the aggregate-suppressed ORCID/affiliation
sensitivity artifact. ORCID-linked analysis remains a required sensitivity,
not a primary replacement, and affiliation remains a current-snapshot proxy,
not historical truth.

The registered G1 scope remains 2013-2016, fields 16, 32, and 11, 28 episodes,
and 13 families. G4 field/year data and private G4 artifacts remain excluded;
the seal remains unopened.

## Fail-closed state and next action

No G1 result exists. `empirical_g1_status=NOT_RUN`, `pass_e=false`,
`pass_g1=false`, confirmatory and holdout rows read are zero, no label was
materialized, no estimator ran, `g4_open=false`, and
`scientific_effect=NOT_ESTABLISHED`.

The exact G1 execution successor is authorized only from parent v1 plus this
independent v2 PASS. It must accept PASS, registered-decision FAIL, or BLOCKED
without redesign after confirmatory access.
