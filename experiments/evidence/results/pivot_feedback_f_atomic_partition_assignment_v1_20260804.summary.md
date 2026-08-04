# Pivot-feedback atomic partition assignment v1 summary

Status:
`PASS_OUTCOME_BLIND_ATOMIC_EPISODE_FAMILY_PARTITION_ASSIGNMENT_PRIVATE_HOLDOUT_MATERIALIZATION_AUTHORIZED_NO_F_PASS`

The verified final panel now has an atomic metadata assignment. All 65 shock
episodes and 29 field-subfield families remain whole: 28 episodes in 13
families are assigned to the 2013--2016 G1 replication partition, and 37
episodes in 16 families are assigned to the 2020--2022 operator-only G4
partition. No family crosses G1 and G4.

| Partition | Entry years | Episodes | Families | State |
| --- | --- | ---: | ---: | --- |
| Discovery | through 2012 | 0 | 0 | registered empty final-shock cell |
| G1 replication | 2013--2016 | 28 | 13 | atomically assigned |
| Model selection | 2017--2019 | 0 | 0 | registered empty final-shock cell; row folds deferred |
| G4 holdout | 2020--2022 | 37 | 16 | atomically assigned; row enforcement pending |

Thirty-nine episodes have at least one registered `k=0,1,2` response year
outside their event partition's entry-year interval. They remain wholly
assigned by event year, as frozen; none was split or moved to fill the
model-selection cell.

The zero model-selection episode count is not a conformance failure. The
frozen final shock panel contains G1 event years and G4 event years only. The
five-fold connected-component recipe for later model-selection rows remains
bound, but no author-token component or row-level fold was materialized and no
score was read.

Independent verification returned
`PASS_INDEPENDENT_OUTCOME_BLIND_ATOMIC_EPISODE_FAMILY_PARTITION_ASSIGNMENT_NO_F_PASS`.
It reconstructed all episode, family, and partition rows directly from the
verified final-panel parent and confirmed the absence of raw identifiers.

Operator-only private holdout materialization is now the only next authorized
artifact. That successor must enforce the G4 author-cold rule at row level.
Preflight, byte sealing, analytic reads, model fitting, and G4 opening remain
unauthorized. `pass_f=false`, `pass_g0=false`.
