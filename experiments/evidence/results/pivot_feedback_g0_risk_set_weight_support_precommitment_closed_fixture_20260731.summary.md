# G0 risk-set, weight, support, and stability closed-fixture summary

- Status: `PASS_SIX_OUTCOME_BLIND_RISK_SET_WEIGHT_SUPPORT_CHOICES_AND_CLOSED_FIXTURE`
- Frozen choices: 6
- Negative sampling: 10 controls per case within exact risk sets
- Sampling: without replacement, deterministic SHA-256 priority
- Weight truncation: nearest-rank P01/P99 winsorization, then mean-one normalization
- Minimum cohort: 500 opportunities, 100 events, and 100 non-events
- Report suppression: counts 1 through 10, plus derivable complements
- Capacity-cell support: 50 total, 11 events, and 11 non-events in every cell
- Collinearity gate: every centered weighted exposure-block VIF strictly below 5
- Synthetic sampling strata: 2
- Synthetic selected controls: 25
- Synthetic upper-tail weights clipped: 1
- Synthetic cohort failures: 3
- Synthetic suppressed report cells: 2
- Synthetic capacity support failures: 1
- Synthetic VIF failures: 1
- Synthetic separate-channel passes: 1
- Discovery production application: `NOT_RUN`
- G0 v3 changed: no
- Remaining choices after a future binding successor: 7
- Confirmatory or holdout rows read: 0
- Model fitted: no
- G4 opened: no

The values are explicit outcome-blind design policies. They are not
literature-mandated truths, power guarantees, production effects, a G0 pass,
or authorization to open or score confirmatory data.
