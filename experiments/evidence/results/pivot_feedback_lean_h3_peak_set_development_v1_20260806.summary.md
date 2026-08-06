# Lean H3 peak-set development diagnostic v1

- Completed: `2026-08-06T10:19:53Z`
- Candidate precommit: `ea9bac8e3a94cba190e23c2a8055b5873105ada1`
- Status: `COMPLETE_OPENED_DISCOVERY_DIAGNOSTIC_CONDITIONAL_CALIBRATION_LOW_UNEVEN_VALIDITY_NO_SELECTION`
- Candidate admissibility: `NOT_EVALUATED`
- Estimator and support selection: `NO`
- Empirical and sealed/private research rows read: `0`
- Scientific effect: `NOT_ESTABLISHED`

## Result

The uncertainty-aware peak-set candidate did what the raw estimator could not:
it represented posterior uncertainty in each episode's peak location and could
return `INVALID` instead of forcing an argmax. On the already-opened synthetic
challenge, its 95% interval covered the truth in 1,054 of 1,084 valid,
identifiable panels (`0.9723`). It detected 571 of 609 meaningful positives
among valid panels (`0.9376`), produced no positive decision among 359 valid
nonpositive panels, and reversed sign in 8 of 1,031 valid signed panels
(`0.0078`).

That improvement is conditional on a much narrower analysis set. Only 1,084 of
2,560 identifiable panels were valid (`0.4234`). Overall, 1,156 of 4,096 panels
were valid once the 72 valid nonidentifiable plateau panels are included. The
candidate therefore replaces universal forced estimation with better-calibrated
estimation on a strongly selected subset. It has not demonstrated general
admissibility or superiority.

## Where it worked and where it did not

| Generator family | Validity | Conditional coverage | Reading |
|---|---:|---:|---|
| Asymmetric peak and tail | 255/512 (`0.4980`) | 234/255 (`0.9176`) | Peak uncertainty repairs conditional coverage, but half the panels are excluded. |
| Correlated episode shift | 144/512 (`0.2812`) | 137/144 (`0.9514`) | Calibration is conditional on narrow applicability. |
| No-feedback observational mimic | 427/512 (`0.8340`) | 427/427 (`1.0000`) | 375/427 valid panels still pass; timing cannot identify feedback. |
| Rounded/heaped measurement | 256/512 (`0.5000`) | 255/256 (`0.9961`) | Strong conditional coverage after half fail closed. |
| Staggered field mixture | 2/512 (`0.0039`) | 1/2 (`0.5000`) | Scalar field balancing does not resolve heterogeneous field timing. |

The structural nonidentifiability rules were partly successful. All 512
informative-missingness panels and all 512 monotone-boundary panels returned
`INVALID`. For plateau/near-tie panels, 440 of 512 (`0.8594`) returned
`INVALID`; the other 72 were treated as valid, although none produced a
positive decision. Thus the candidate prevents the earlier false-positive
behavior but does not yet fully encode structural near-tie nonidentifiability.

## Why this is not another excessive gate

No new admissibility threshold was applied. This run was deliberately
descriptive, with no pass/fail outcome. The `0.4234` identifiable validity is
an output of the estimator's precommitted fail-closed definition, not a
governance gate imposed after seeing results.

The raw v1 coverage (`0.8133` overall and `0.5352` for asymmetric curves) and
the candidate's conditional coverage (`0.9723` and `0.9176`) are useful context
but are not a direct contest. Raw estimated every panel; the new candidate
excluded 57.66% of identifiable panels. The scientific tradeoff is therefore
calibration versus applicability, not a simple win.

## What the surprising cases teach us

First, the near-tie problem is not fully solved by requiring a `0.60` modal
argmax probability and a narrow 90% peak set. Sampling variation can still make
a structurally near-tied curve look locally decisive. A future candidate needs
a prospectively justified practical peak-contrast or equivalence rule, not a
threshold tuned against these opened panels.

Second, the staggered-field result shows that a field-balanced scalar mean is
the wrong final object when fields peak at different times. A field-vector or
partial-identification result that can report heterogeneity without collapsing
it should be designed prospectively.

Third, the no-feedback mimic passed in 375 of 427 valid panels (`0.8782`). This
does not mean the timing estimator is numerically broken: those panels contain
the observable timing gap. It means the estimand cannot distinguish delayed
feedback from selection or measurement mimicry. The empirical claim ceiling
must remain observational timing, never mechanism.

## Disposition

This candidate is neither passed nor failed for admissibility, and it is not
selected. Its rules will not be retuned on this opened challenge. A fresh blind
challenge is not yet the next efficient step: it would merely confirm already
visible near-tie and field-heterogeneity limitations.

The next prospective design should specify two objects before any new blind
assessment: a practical-equivalence peak-set rule for near ties, and a
field-vector or partial-identification aggregation for staggered fields. Any
choice among their variants must use theory, analytic fixtures, or a separate
development generator—not success on this opened challenge. Empirical rows,
confirmatory execution, and estimator/support selection remain unauthorized.

The canonical artifact root is
`/ssd/pivot-feedback-lean-h3-peak-set-development-v1-opened-diagnostic-20260806`;
its diagnostic receipt SHA-256 is
`797b4cb41f52c39dddfbc88d66fa032c28ab1516b4ca8af5def0c0dea8b2dc9f`.
The adjacent independent verification SHA-256 is
`b5dd9ffb842a9f9d796856fd53b118de02494424c55be28436073c86fa42ae9b`.

This is opened synthetic estimator development only. It is not blind
confirmation, empirical H3, a null or association, support validation, a
mechanism or causal result, predictive validation, NABM necessity, or a
scientific effect.
