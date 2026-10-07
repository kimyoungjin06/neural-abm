# Pivot empirical outcome-selection trend v2 (N10)

Date: 2026-10-07

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_empirical_outcome_selection_trend_v2.yaml`](../../experiments/configs/pivot_empirical_outcome_selection_trend_v2.yaml)

Amends: [v1](pivot-empirical-outcome-selection-trend-v1.md) (`49f5345`), whose
contract, script, and stage-T result are bound by hash.

## Why v2

v1's [stage T](../../experiments/evidence/results/pivot_empirical_outcome_selection_trend_v1_20261007.summary.md)
transferred the synthetic threshold to three yearly entry levels. On that
clock the indicator caught only 12 of 33 active synthetic cells, against 29 of
33 on the 18-step synthetic clock. Early-concentrated entry falls into the
first year, and the signal is lost. A result that is not positive would say
almost nothing.

No empirical row has been read, so the clock can still change. v2 is informed
by v1's synthetic stage T and is labelled a result-informed successor.

## What changes

| | v1 | v2 |
|---|---|---|
| Entry time | Response year, 3 levels | Publication-date quarter within the three response years, 12 levels |
| Entry date | — | Earliest eligible topic work in the entry year |
| Persistence | A work in calendar years e+1 to e+3 | A work in the 12 quarters after the entry quarter |
| Imprecise dates | — | Entrants dated January 1 are excluded from the primary statistic |
| Stage T | 3 bins | 12 bins, same rule |
| Episode floor | 30 entrants in 2 cohorts | 30 precise entrants in 2 response years |

Everything else is v1's and is checked at load: source, fields, years,
episodes, partition, author overlap, precision, uncertainty, decision rule,
stages, and claim ceiling.

**Why persistence moves with the clock.** With quarter entry and a
calendar-year window, an entrant in the fourth quarter would start follow-up
almost at once. An entrant in the first quarter would wait nine months.
Persistence falls with time since entry, so this alone would create a positive
within-year trend: a false sign. A window aligned to the entry quarter gives
every entrant the same 12 quarters. It is also closer to the synthetic
definition, which counts from entry.

**Why January 1 is excluded.** OpenAlex records a year-only publication date
as January 1, so the quarter of such an entry is unknown. True January 1
publications cannot be told apart and are excluded too. Each episode's
excluded share is reported. Follow-up works are counted at their recorded
date.

## Stage T on twelve levels

The rule is v1's, with twelve equal-width bins over the 18 full-horizon
synthetic entry steps:

1. Every run's fine-clock value must be rebuilt exactly from its cohorts.
2. The coarse threshold is the largest twelve-level value among inactive cells.
3. The data threshold is the larger of that and 0.098417.
4. The no-learning control must stay at or below it in every cell.

The share of active cells past the threshold is reported but not decided.

## Report-only additions

- **S5:** v1's three-level clock with D0 persistence, against v1's data
  threshold of 0.105635.
- **S6:** imprecise entrants included at a mid-year rank, with a calendar-year
  window.
- **S1:** the horizon-end sensitivity is now quarters 9 to 12 after entry.

## Boundary

The executable must be committed before stage T runs. A rerun must reproduce
its artifacts byte for byte. No empirical row is read. S1, S2, and S3 each
need separate authorization. Row decoding, materialization, and G1,
model-selection, or G4 access stay unauthorized.
