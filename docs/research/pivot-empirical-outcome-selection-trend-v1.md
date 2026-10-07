# Pivot empirical outcome-selection trend v1 (N10)

Date: 2026-10-07

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_empirical_outcome_selection_trend_v1.yaml`](../../experiments/configs/pivot_empirical_outcome_selection_trend_v1.yaml)

## Why this study exists

[Consolidation v4](pivot-policy-ladder-consolidation-v4.md) carried one
indicator to the empirical stage: the **outcome selection trend**. It is the
rank correlation, within a pivot episode, between when researchers entered and
whether they persisted. In the synthetic worlds a clearly positive value
appeared only when learned policy changed who entered. Without learning, later
cohorts persisted less. The reading is one-sided: a positive value is a sign,
and its absence is not evidence against.

The source is now settled. The exact 2026-06-26 OpenAlex snapshot is complete
on `/ssd` ([record](../../experiments/evidence/results/pivot_feedback_lean_h3_openalex_full_core_resume_v3_20261007.summary.md)).
This document fixes every claim-bearing choice before any empirical row is
read, following the requirements of the
[lean-H3 successor design](pivot-feedback-lean-h3-successor-design-v1.md).

## Episodes and years

| Choice | Value | Why |
|---|---|---|
| Episode | Field × primary topic × decision year | The lineage's episode key |
| Surge rule | Incumbent-only attention acceleration, modified z ≥ 3.0 | The frozen F-phase rule, unchanged |
| Fields | 16 Chemistry, 32 Psychology, 11 Agricultural and Biological Sciences | Selected earlier by an outcome-blind structural rule for the three non-G4 domains |
| Decision years | 2000–2012 | The only registered window open to a new study |
| Response cohorts | Entry in years t, t+1, t+2 | The lineage's response cohorts |
| Spacing | Two episodes of one topic at least 3 years apart | Their cohorts must not overlap |

The years are not a free choice:

- 2013–2016 is the opened G1 cohort, and may not be retried.
- 2017–2019 may not be reselected or repurposed.
- 2020–2022 is the sealed G4 holdout.

Field 27 (the G4 field) and field 17 (the D0 field, with opened episodes) are
excluded. The four D0 episodes and every episode key of the earlier partition
are excluded. A structural indexing screen drops a surge when the median
topic score falls by 0.15 and one source's share rises by 0.25 at the same
time. It needs two columns the footer check has not confirmed. If they are
absent, the screen is reported as unavailable and the claim says so.

## Partition and author overlap

Discovery and confirmatory episodes are separated by subfield family, so that
related topics never straddle the two. Within each field, families are ranked
by a SHA-256 of a registered seed. The first third, rounded up, go to
discovery.

- **G4 guard first.** The sealed G4 author list is anti-joined before any entry
  label. Only counts are reported.
- **Discovery authors.** A confirmatory entrant who also entered a discovery
  episode is excluded before any persistence label is read.
- **Repeated authors** across confirmatory episodes are kept and counted.
  Episodes are the resampling unit, and author-level dependence is a
  registered limitation.

## Entry, persistence, and the indicator

- **Entrant.** A researcher's first eligible work in the episode topic, in
  year t, t+1, or t+2. The researcher also needs an eligible work in the same
  field in the three preceding years, so that the entry is a pivot rather
  than a first paper.
- **Persistence.** At least one eligible work in the topic in the three years
  after entry. This is the D0 definition, unchanged. Every entry in the window
  matures by 2017, and the snapshot is complete through 2025.
- **Episode statistic.** Spearman's ρ between entry offset (0, 1, 2) and
  persistence, the synthetic statistic on the empirical clock. An episode
  needs at least 30 entrants in at least two cohorts.
- **Estimate.** The mean ρ of each field's valid episodes. The primary
  estimate is the mean of the three field values.

The synthetic persistence is "not exited four steps after entry". The primary
empirical persistence is "returned at least once in three years". These are
not the same object. A horizon-end version (a work in year +3) is the closer
analogue. It is reported as a sensitivity, not decided.

## Stage T: transferring the threshold

The synthetic threshold, 0.098417, was set on an 18-step entry clock. Data has
three entry years, and coarser time compresses a rank correlation. Stage T
therefore recomputes the indicator from the stored synthetic runs of N05, N08,
and N09, with each run's entry steps cut into three equal-width bins. It reads
no empirical row.

1. Every run's fine-clock value is rebuilt from its stored cohorts and must
   equal the stored value. Any mismatch stops the stage.
2. The coarse threshold is the largest coarsened value among inactive cells,
   across all 72 cells. This is the rule that gave no false actives.
3. The **data threshold** is the larger of the coarse threshold and 0.098417.
4. The fixed-rule (no-learning) control must stay at or below the data
   threshold in every cell. Otherwise the threshold does not transfer, and
   data may report the indicator only descriptively.

## Precision and decision

Discovery persistence fixes the between-episode spread of ρ in each field.
Together with the confirmatory entrant counts, it projects the confirmatory
one-sided 95% half-width. The projection must not exceed the data threshold:
then any estimate past the threshold is projected to have a positive lower
bound. If it does exceed it, the study stops before reading confirmatory
persistence.

| Disposition | Condition |
|---|---|
| `INDICATOR_POSITIVE` | Estimate above the data threshold and one-sided 95% lower bound above 0 |
| `NOT_POSITIVE_NO_INFERENCE` | Otherwise. A 95% upper bound below 0 is flagged as a later-cohort decline |
| `INVALID_FIELD_SUPPORT` | A field has fewer than 4 valid confirmatory episodes |

The bounds come from a field-stratified episode bootstrap: 9,999 resamples,
seed 20261007. The following are reported but never decided:

- two persistence sensitivities;
- one episode per author;
- surge thresholds of 2.5 and 3.5;
- a matched non-surge placebo.

Disagreement with the primary downgrades the claim. A positive placebo means
the trend is not specific to attention surges.

## Stages and authorization

| Stage | Reads | Authorization |
|---|---|---|
| T | Stored synthetic runs | This precommit |
| S1 | Column-limited Works rows: eligibility, attention, episodes, partition, entrant counts. No persistence label | Separate, with a frozen executable and a byte budget checked against free space plus 100 GiB |
| S2 | Discovery persistence only. Precision projection, go or stop | Separate |
| S3 | Confirmatory persistence. The decision | Separate |

A stop is never read as a null or an effect. A definition can change only by a
new versioned precommit before the affected labels are read.

## Claim ceiling

> In the preregistered OpenAlex episodes, later post-surge entrants were more
> likely to persist in the topic, past the threshold transferred from the
> synthetic worlds. In those worlds this sign appeared only when learned policy
> changed who entered.

The study does not establish learning in data, feedback, a mechanism, a causal
effect, the H3 timing gap, transport beyond these fields and years, or NABM
necessity.

## Boundary

The executable must be committed before stage T runs. A deterministic rerun
must reproduce every stage-T artifact byte for byte. This precommit reads no
empirical row. It does not authorize row decoding, materialization, G1,
model-selection, or G4 access, or model fitting. It leaves current state v4
and `g4_open = false` unchanged.
