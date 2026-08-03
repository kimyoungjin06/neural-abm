# OpenAlex W0 Positive-ID Sensitivity

- Status: `PASS_W0_POSITIVE_ID_EXCLUSION_SENSITIVITY_COMPONENT`
- Work files read: `2`
- Rows projected: `2,000,000`
- GPU used: `false`
- Scientific effect estimated: `false`

## Profile Delta

| Quantity | Grammar-valid view | Positive-ID view | Delta |
| --- | ---: | ---: | ---: |
| Master works | 492,293,206 | 492,293,205 | -1 |
| Classified primary field | 354,145,248 | 354,145,248 | 0 |
| Null primary field | 138,147,958 | 138,147,957 | -1 |

All 26 non-null field counts are unchanged. The excluded W0 row is in
the null-primary-field, 2024, one-declared-author, no-usable-author-ID stratum.

## Gate Status

| Gate | Status |
| --- | --- |
| `author_entity_validity` | `PENDING` |
| `candidate_eligibility` | `PENDING` |
| `current_main_w0_projection` | `PASS` |
| `current_topic_w0_incidence` | `PASS_REVALIDATED` |
| `empirical_g1_signature` | `NOT_RUN` |
| `frozen_source_propagation_ledger` | `PASS_REVALIDATED` |
| `g0_measurement_freeze` | `PENDING` |
| `g_minus_1_data_readiness` | `PENDING` |
| `nonnull_field_margins` | `UNCHANGED` |
| `parent_content_identity` | `UNVERIFIED` |
| `pass_g1` | `false` |
| `positive_id_exclusion_sensitivity` | `PASS` |
| `publication_date_validity` | `PENDING` |
| `w_zero_entity_validity` | `UNRESOLVED` |

## Claim Boundary

A pass quantifies the predeclared positive-work-ID sensitivity for the single retained W0 record and propagates its one-row removal through the already frozen cross-field profile margins. It does not decide whether W0 denotes a real or invalid entity, establish historical or current full-root byte identity, validate publication dates or authors, select fields or candidate researchers, pass G-1 or G0, run empirical G1, or establish an effect.

- Result fingerprint: `3056af90867d813ef70b9bcf11c67385c3b288d2f21d89de150c4f2ddbc30190`
