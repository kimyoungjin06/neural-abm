# Lean H3 prospective result-space v1

- Completed: `2026-08-06T06:21:37Z`
- Status: `PASS_INDEPENDENT_SYNTHETIC_RESULT_SPACE_COUNTEREXAMPLES_FOUND_KERNEL_NOT_ADMISSIBLE_RAW_NOT_YET_SELECTED_NO_EMPIRICAL_ROWS`
- Scope: source-independent synthetic H3 estimator and interpretation stress test
- Empirical, current G1, model-selection, and holdout/private G4 rows read: `0`
- Scientific effect: `NOT_ESTABLISHED`

## What was tested

The design crossed two candidate estimators with seven data-generating or
observation families: delayed feedback, synchronous opportunity, fast
correction, selective composition, capacity depletion, recruitment mixture,
and measurement artifact. The corrected `design_pilot` contained 9 canonical
and 96 deterministic adversarial scenarios. Across 6,912 replicate-estimator
rows, it found 4,012 rows with at least one registered counterexample flag and
retained the top 200 for inspection.

An independent verifier reproduced all stored aggregate counts, flag
assignments, paired-estimator disagreements, and counterexample ranking. It did
not independently regenerate the synthetic panels.

## Results that changed the design

The simple positive, null, and negative canonical cases behaved as expected.
The useful findings came from cases where those expectations broke:

- For a broad true gap of `+2`, the raw estimator had mean bias `0.0503` and
  `0.9609` interval coverage. The kernel estimator had mean bias `0.4779`, only
  `0.0391` coverage, and a `0.9609` confident-wrong rate.
- In the bimodal recruitment case, the observable truth was `0`. The kernel
  estimator nevertheless passed all 128 repetitions, with mean estimate
  `1.7109`, zero coverage, and false-positive, confident-wrong, and
  paired-estimator-disagreement rates all equal to `1.0`.
- The raw estimator did not pass that bimodal null, but its coverage was only
  `0.6016` and its aggregation-paradox rate was `0.9922`. It therefore remains
  a candidate, not a selected estimator.
- Selective composition and a measurement shift reproduced a positive observed
  timing gap without delayed feedback. The raw estimator passed those cases at
  rates `0.9609` and `1.0`, respectively.

The result is not “the raw estimator wins.” The kernel estimator is not
admissible as the v1 primary, while the raw estimator still requires a stronger
identifiability/aggregation treatment and an outcome-blind synthetic challenge.
No support rule is selected.

## Scientific consequence

Even a precisely recovered positive timing gap cannot identify delayed feedback
by itself. The same observable order can arise from composition, measurement,
or recruitment processes. A later empirical H3 success could therefore support
only the bounded timing statement already defined in the successor design. It
could not, without a separate design, establish feedback, social learning,
belief updating, causal influence, or NABM necessity.

The next design step is a generator-held-out synthetic challenge with scenario
labels concealed from the estimator review. Hierarchical partial pooling may
enter that comparison only as a preregistered third candidate; it is not a
post-hoc rescue for either current estimator.

## Lineage and correction

The canonical artifact root is
`/ssd/pivot-feedback-lean-h3-result-space-v1-design-pilot-corrected-20260806`.
Its receipt SHA-256 is
`dfe734ae74b14181e23b5485c66c208b028a3e7b11483f22a46007a17c94f20c`.
The adjacent independent verification SHA-256 is
`bfd43ae94066623a9307fe04a69eb3100c07cd5f60464d61994d8a954f974f84`.

The earlier root
`/ssd/pivot-feedback-lean-h3-result-space-v1-design-pilot-20260806` is preserved
as `SUPERSEDED_DIAGNOSTIC_FLAG_SEMANTICS_ONLY_DO_NOT_CITE`. Its false-negative
flag failed to apply the registered minimum meaningful gap of `1.0`; it is not
the result authority.

This synthetic audit is not empirical H3, a mechanism result, a causal result,
predictive validation, a scientific effect, or permission to decode any source
row.
