# OpenAlex Current-Snapshot Taxonomy Reference Audit

## Status and boundary

- Status: `PASS_CURRENT_SNAPSHOT_TAXONOMY_REFERENCE_COMPONENT`
- Full scope: `true`
- Files: `1,423` / `1,423`
- Projected rows: `1,420,855,753` / `1,420,855,753`
- Workload: `local_cpu_io`; GPU used: `false`
- Scientific effect estimation: `false`
- Resume verification: `REHASH_CURRENT_PROJECTED_PAYLOAD`; metrics reused only after projected-payload rehash: `true`

A bounded run is pilot evidence only. Even a full component pass is not
G-1, G0, empirical G1, or evidence of a scientific effect.

## Reference accounting

| Category | Exact rows |
| --- | ---: |
| All-null hierarchy | 138,147,958 |
| Partial hierarchy | 0 |
| Complete hierarchy | 1,282,707,794 |
| Valid complete reference | 1,282,707,794 |
| Invalid complete reference | 0 |

## Per-table accounting

| Table | Files | Rows | W-1 | W0 | All null | Partial | Complete | Invalid complete | Conserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| openalex_works_20260330 | 493 | 492,293,207 | 1 | 1 | 138,147,958 | 0 | 354,145,248 | 0 | true |
| openalex_works_20260330__topics | 930 | 928,562,546 | 0 | 0 | 0 | 0 | 928,562,546 | 0 | true |

## Cross-field margin check

- Status: `PASS`
- Parent work-source field/status margins equal the compact cross-field profile: `true`
- Observed main field works equal the compact profile: `true`
- Observed main null-field works equal the compact profile: `true`

## Gate status

| Gate | State |
| --- | --- |
| author_entity_validity | PENDING |
| candidate_eligibility | PENDING |
| current_snapshot_taxonomy_reference_component | PASS |
| duplicate_work_topic_edges | PENDING |
| empirical_g1_signature | NOT_RUN |
| g0_measurement_freeze | PENDING |
| g_minus_1_data_readiness | PENDING |
| parent_content_identity | UNVERIFIED |
| pass_g1 | false |
| primary_topic_child_inclusion | PENDING |
| publication_date_validity | PENDING |
| raw_source_id_validity | BLOCKED |
| temporal_available_at | PENDING |
| w_zero_entity_validity | UNRESOLVED |

## Remaining boundaries

| Pending item | Gate | Current state |
| --- | --- | --- |
| duplicate_work_topic_edge_audit | duplicate_work_topic_edges | PENDING |
| primary_topic_present_in_work_topic_rows | primary_topic_child_inclusion | PENDING |
| publication_date_and_year_validity | publication_date_validity | PENDING |
| cutoff_aware_available_at_validation | temporal_available_at | PENDING |
| author_entity_and_split_merge_validity | author_entity_validity | PENDING |
| candidate_researcher_eligibility | candidate_eligibility | PENDING |
| deterministic_representative_pilot | none | PENDING |
| g_minus_1_data_readiness | g_minus_1_data_readiness | PENDING |
| g0_measurement_freeze | g0_measurement_freeze | PENDING |
| empirical_g1_signature | empirical_g1_signature | NOT_RUN |

## Provenance

- Task fingerprint: `bbb7a2eaa80c7a5f7cfa791fa211543b11a00a4137bba62fd3204255c96a316d`
- Result fingerprint: `36209d119cc32802d0e7391b92425dcef931e9513841afa19e8be7b0bc06b30b`
- Checkpoint tree: `bb65e3e0268d251a06af74542aacddc56331fcc954096aca796c2016e41c4e4d`
- Taxonomy canonical SHA-256: `8b9f4e2f492eb6939e1dbb6a8187f06d4ea8fdebb9c1bafce017cd39e54401b4`

## Claim boundary

A full component pass establishes current-snapshot, projected-column row conservation and validates non-null primary-topic and work-topic hierarchy references against the seven-file canonical taxonomy graph, with the frozen W-1 main-row quarantine and retained W0 warning unchanged. It does not prove historical inventory-era byte identity, deduplicate work-topic edges, prove that a primary topic occurs among each work's child topic rows, validate publication dates or cutoff-aware available_at, validate author entities or candidate eligibility, establish G-1 or G0, run empirical G1, or establish a scientific effect. A max-files run is PILOT_REFERENCE_ONLY and cannot be promoted to the full component status.
