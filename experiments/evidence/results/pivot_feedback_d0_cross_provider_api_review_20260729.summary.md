# D0 cross-provider API review — 2026-07-29

Status: `PASS_INDEPENDENT_D0_CROSS_PROVIDER_RESULT_VERIFICATION_HUMAN_REVIEW_PENDING`

OpenAI and Claude each produced 384 complete, structured title-only decisions
for the same frozen items. The independent verifier found no missing IDs,
provider errors, incomplete responses, or invalid structured results.

The provider labels agree on 306 of 384 rows (79.7%). Both providers assign
`ON_TOPIC` with `HIGH` confidence to 197 rows; these are API-screen
auto-accept candidates. The frozen fail-closed policy escalates 187 rows. When
those rows are unioned with the fixed 96-row calibration sample, with 50 rows
overlapping, each blinded human packet contains 233 rows.

The first v3 submission attempt failed before Claude batch creation because
the provider-prefixed custom IDs exceeded Claude's 64-character limit. Its
OpenAI batch is now confirmed `cancelled` with zero requests. In v4, Claude
returned 384 valid rows, while the first OpenAI run was invalid because 241
responses hit the output-token ceiling. The v5 OpenAI-only retry preserved the
model, input, prompt, schema, and IDs; it changed only reasoning effort and
the output cap, and returned 384 valid rows. Claude was not resubmitted.

The cumulative API cost estimate, including the invalid v4 OpenAI run, is
`$0.352772`, below the frozen `$1.00` ceiling.

Human calibration and adjudication have not been run. D0,
complete-candidate-universe screening, outcomes, G0, model fitting, G4, and
any scientific effect remain closed or not established.
