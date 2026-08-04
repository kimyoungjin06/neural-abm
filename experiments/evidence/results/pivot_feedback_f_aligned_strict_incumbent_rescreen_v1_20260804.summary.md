# Pivot-feedback F aligned strict-incumbent rescreen v1 summary

Status: `PASS_OUTCOME_BLIND_STRICT_INCUMBENT_ALIGNED_ALL_TOPIC_RESCREEN_FINAL_LEAVEOUT_REPLAY_AUTHORIZED_NO_F_PASS`

The production replay recomputed the closed pre-signal incumbent-team attention
measure from the normalized work and author-event inputs for every eligible
topic in the 12 frozen field attempts. It produced 7,099 positive-reference
topic-by-field-by-year rows across all 45 registered normalization groups. No
group had zero or nonfinite MAD. The unchanged positive-growth and inclusive
`z >= 3` rules retained 99 topic-year candidates representing 70 topics.

The raw implementation reproduced the strict reference and recent counts for
all 194 predecessor episodes exactly. It did not reuse aggregate counts or
aggregate field-year median/MAD. The independently recovered 2026-03-30 topic
taxonomy contains all 4,516 topics and agrees exactly with 476 distinct frozen
aggregate mappings and 994 distinct frozen D0 mappings; all 70 new candidate
topics have a subfield family.

| Slot | Frozen order | Aligned field | Episodes | Families |
| --- | --- | ---: | ---: | ---: |
| Physical G1 | 25, 17, 16 | 16 | 8 | 3 |
| Social G1 | 32, 18, 20 | 32 | 3 | 3 |
| Life G1 | 28, 11, 24 | 11 | 17 | 7 |
| Health G4 operator-only | 36, 29, 27 | 27 | 37 | 16 |

The first field meeting the unchanged three-family rule was selected in each
frozen order. The resulting aligned panel contains 65 topic-year episodes over
45 field-topic pairs. This is a structural screen result; it is not a
leaveout-conformed final panel and does not pass F.

Independent verification status:
`PASS_INDEPENDENT_STRICT_INCUMBENT_ALIGNED_ALL_TOPIC_RESCREEN_NO_F_PASS`.
The verifier independently recomputed every median, MAD, modified z-score,
candidate flag, taxonomy mapping, frozen-order selection, and predecessor
count closure.

Exactly one next artifact is authorized: the outcome-blind final-panel focal
and preexisting-peer leaveout replay on these 65 selected episodes. `pass_f=false`,
`pass_g0=false`; no confirmatory label, realized effect, holdout byte seal,
model fit, or G4 analytic outcome was opened.
