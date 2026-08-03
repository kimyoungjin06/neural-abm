# Pivot-feedback G0 risk-set, weight, support, and stability precommitment v1

Date: 2026-07-31

Status:
`PASS_SIX_OUTCOME_BLIND_RISK_SET_WEIGHT_SUPPORT_CHOICES_AND_CLOSED_FIXTURE`

## Result

This contract freezes the six choices authorized by gate v59 without reading
production, confirmatory, or holdout rows. It binds the 3,256,271-row D0
predecision measurement count only as repository metadata and does not open the
operator-local measurement table.

The application tool and an implementation-independent verifier agree on a
closed synthetic fixture. The fixture covers deterministic downsampling,
full-control retention below the target, weight winsorization, all cohort and
reporting boundaries, capacity-cell support failure, and VIF failure despite
adequate raw cell counts.

## Frozen policies

### Risk-set sampling

Every eligible entry case is retained. Within the exact analysis-partition,
field, certified-episode, and decision-year risk set, at most ten controls per
case are sampled without replacement. A fixed component seed and SHA-256
priority over a private stable opportunity ID make selection reproducible.

If fewer controls exist than the target, all are retained. The exact control
inclusion probability and inverse sampling weight are recorded. Full-risk-set
sensitivity remains mandatory; ten controls is a compute policy, not an
efficiency guarantee.

Nested case-control simulations show that the common four-to-five-control
practice can lose efficiency under few cases, strong relations, or skewed
exposures. That supports choosing more than four here, but does not uniquely
imply ten (`doi:10.1515/em-2013-0007`).

### Analytic weights

Positive finite combined stabilized weights are winsorized within analysis
partition and field at nearest-rank P01 and P99, then divided by their group
mean. Raw and truncated ranges, Kish effective sample sizes, balance
diagnostics, and the untruncated sensitivity must all be reported.

Weight construction requires explicit positivity and bias-variance
diagnostics. P01/P99 is a precommitted policy rather than a universally optimal
cutoff (`doi:10.1093/aje/kwn164`). Any estimand shift caused by truncation must
be stated.

### Minimum cohort and cell reporting

Before negative sampling, each partition-by-field-by-certified-episode unit
must have at least 500 eligible opportunity rows, 100 entry events, and 100
non-entry opportunities. Failure excludes that unit and fails the registered
coverage requirement; fields or partitions cannot be pooled to rescue it.

This is only a minimum screen. A model-specific sample-size calculation remains
required because adequate size depends on event frequency, parameter count,
anticipated fit, shrinkage, and precision (`doi:10.1002/sim.7992`). The fixed
500/100/100 floor is not a power guarantee.

Published counts from 1 through 10 are suppressed, as are percentages and
complements that would recover them. Eleven is the smallest directly
reportable nonzero count. This conservatively borrows the CMS RIF disclosure
boundary; CMS does not govern this OpenAlex study. Exact zeros can be shown
only when they cannot be recovered as the complement of a suppressed count.

### Capacity support and collinearity

For each of `NEITHER`, `PUBLIC_SUCCESS_ONLY`, `CAPACITY_BRIDGE_ONLY`, and
`BOTH`, every partition-by-field-by-episode table must contain at least 50
opportunities, 11 events, and 11 non-events. Every condition must pass in every
cell.

The weighted centered VIF is then computed for public success, capacity, and
their interaction. All three values must be strictly below 5, with full rank
and nonzero variance. VIF is interpreted alongside cell support and interval
precision; it is not used as a universal truth or as permission for selective
post-hoc variable deletion. The literature explicitly cautions against
mechanical VIF thresholds (`doi:10.1007/s11135-006-9018-6`).

If either support or VIF fails, the registered action is to collapse the block
to `ANY_PEER_SIGNAL_OR_CAPACITY` and abandon the separate information-versus-
capacity channel claim. No single variable may be deleted selectively after
results are inspected.

## Closed-fixture results

The fixture contains two sampling strata. One samples 20 of 25 controls for
two cases, yielding inclusion probability 0.8 and weight 1.25; the other keeps
all five controls for one case. A 100-observation weight histogram clips its
single extreme upper value under the exact P99 algorithm.

Four cohort groups exercise one pass and each of the three independent failure
conditions. Counts 0, 1, 10, 11, and 100 exercise the reporting boundary. The
three capacity tables produce exactly one full pass, one raw-support failure,
and one VIF failure while all raw cells meet support.

## Current boundary

The contract has not been applied to discovery production data and has not
modified G0 v3. A future separately authorized binding successor would reduce
the unresolved quantitative-choice count from thirteen to seven. All fifteen
required sections remain incomplete, the holdout receipt remains absent, and
G0, model fitting, and G4 remain closed.
