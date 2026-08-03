# Pivot-feedback G0 readiness inventory v1

Date: 2026-07-31

Status:
`PASS_METADATA_ONLY_G0_READINESS_INVENTORY_G0_NOT_READY`

## Decision

G0 is not ready for a measurement freeze.

The inventory preserves the pending G0 overlay byte-for-byte and checks it
against the completed D0 entry-and-persistence measurement chain. It validates
nine repository documents and eight operator receipt or independent-
verification JSON files. It opens no Parquet file and reads no production,
confirmatory, or holdout row.

Four hard blockers remain:

1. the G-1 readiness receipt binding is absent;
2. the holdout-access seal receipt binding is absent;
3. 17 of 19 quantitative choices remain unresolved; and
4. all 15 required G0 sections remain not freeze-ready.

## Quantitative choices

Two choices now have D0-resolved values that may be copied into a versioned G0
successor:

| Choice | D0-resolved successor value |
|---|---|
| `author_minimum_history_works` | `NONE` |
| `primary_pivot_thresholds` | threshold-free exact four-state rules; primary `NEW_TOPIC_ENTRY` is focal count >= 1 and pre-entry count = 0 |

These values were not selected from persistence rates. They preserve the
observable-history—not true lifetime-first—claim boundary.

The other 17 choices remain unresolved:

- attention-surge and optional public-success rules;
- citation maturation and negative-control equivalence margins;
- risk-set sampling, weighting, support, suppression, and collinearity rules;
- G1 replication tolerance;
- the three G2 predictive thresholds; and
- the G3 Monte Carlo stopping rule, parameter budget, and tuning budget.

An unresolved value stays `null`; the inventory validator fails if one is
silently promoted.

## Required sections

| Disposition | Sections | Count |
|---|---|---:|
| Partial, not freeze-ready | population/author eligibility; pivot/robustness; topic attention/surge; temporal availability/maturation; peer exposure/source assignment; capacity bridge/entry mode; outcome horizon/H3 timing; negative controls; partition/atomic folds; hierarchical multiplicity | 10 |
| Unresolved, not freeze-ready | risk-set sampling/weights; minimum support/suppression; estimators/uncertainty; G2 predictive gate; G3 simulation gate | 5 |
| Freeze-ready | none | 0 |

The prospective peer registry, temporal periods, H3-first multiplicity plan,
and seven negative controls are useful design inputs, but declaration alone is
not a frozen executable decision rule.

## Boundary and successor

The final cutoff panel is not a G0 prerequisite and remains owned by G0.5.
This avoids reintroducing the previously removed G-1/G0 dependency cycle.

The safe successor is a metadata-only versioned G0 skeleton. It may copy the
fixed prospective design and serialize the two D0-resolved choices, but it
must leave the other 17 choices pending and all 15 sections not freeze-ready.
It may not open operator Parquet, inspect confirmatory or holdout rows, choose
values from outcomes, pass G0, fit a model, or open G4.
