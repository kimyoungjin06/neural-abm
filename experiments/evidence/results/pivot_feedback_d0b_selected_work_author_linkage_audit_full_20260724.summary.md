# D0-B selected-work author-linkage audit

Status:
`PASS_INDEPENDENT_D0B_LINKAGE_AUDIT_MEASUREMENT_REVIEW_REQUIRED_LEAVEOUT_BUILD_PAUSED`

The audit joined all 83,565 D0-A-selected pre-decision works to valid,
distinct current-snapshot OpenAlex author-token events. Only 48,245 works
(57.7%) have at least one such token. The gap is strongly episode- and
work-type-dependent rather than a uniform loss.

| Episode | Overall linked / selected | Reference linked / selected | Signal linked / selected |
| --- | ---: | ---: | ---: |
| F17 T14430, 2003 | 15,109 / 15,508 | 663 / 689 | 12,538 / 12,575 |
| F17 T10579, 2012 | 8,111 / 9,280 | 1,720 / 1,998 | 6,190 / 6,664 |
| F27 T11361, 2007 | 8,265 / 31,846 | 1,276 / 1,557 | 1,790 / 23,073 |
| F27 T12505, 1991 | 16,760 / 26,931 | 1,568 / 2,137 | 9,243 / 13,023 |

The decisive failure is F27 T11361. Its signal window contains 23,073 selected
works, but 21,283 have no valid author-token event. Across its full
pre-decision history, work type `other` contributes 22,451 works and only 14
link to a valid author token. The apparent all-work attention surge therefore
does not share a stable observation universe with an author-level incumbent or
network measure.

The independent verifier rehashed all 8,192 source partitions, replayed
1,271,923,464 source rows, reconstructed all 106,148 author incidences, and
matched work metadata, work linkage counts, and all 1,200 work-type/window/cap
profile cells exactly without importing the producer.

No episode has been excluded and no work-type or linkage threshold has been
selected. Further focal/network leave-out construction is paused. The next
authorized work is an outcome-blind measurement-ABI successor that freezes
eligible scholarly work types and author-linkage requirements, then recomputes
the episode candidate universe before any entry or persistence outcome is
opened.
