# Lean H3 blind estimator challenge v1

- Completed: `2026-08-06T09:31:40Z`
- Precommit: `5b5f7efd2e9264fe4b9db566b123ffb2a1dda210`
- Status: `BLOCKED_BLIND_CHALLENGE_NO_ESTIMATOR_ADMISSIBLE_NO_FREEZE`
- Eligible estimators: none
- Empirical and sealed/private research rows read: `0`
- Scientific effect: `NOT_ESTABLISHED`

## Procedural blind

The generator, estimators, one admissibility rubric, evaluator, independent
verifier, and tests were committed before the assessment seed or any panel
existed. A fresh 256-bit OS-random seed then produced 64 opaque scenarios from
8 held-out generator families, 4,096 panel rows, and 8,192 estimator rows. The
blind estimator completed and wrote its receipt before the seed, generator
family, truth, feedback status, or identifiability labels were opened. Its
private-label access count was `0`.

This was a single-operator procedural blind, not an independent-team blind. The
revealed reproducibility seed is
`5d56e995d1f268828105e7b922b9d07dfe2ef17500200537c80993770777fb06`.

## Frozen adjudication

Neither estimator passed all seven criteria.

| Metric | Frozen criterion | Raw | Kernel |
|---|---:|---:|---:|
| Valid rate | at least 0.98 | 1.0000 | 1.0000 |
| Meaningful-positive detection | at least 0.80 | 0.9620 | 0.9502 |
| Nonpositive false positive | at most 0.05 | 0.0000 | 0.0062 |
| Identifiable 95% interval coverage | at least 0.90 | 0.8133 | 0.5320 |
| Worst-family identifiable coverage | at least 0.80 | 0.5352 | 0.2461 |
| Identifiable sign reversal | at most 0.05 | 0.0223 | 0.0631 |
| Nonidentifiable pass | at most 0.05 | 0.0833 | 0.0814 |

Raw passed validity, detection, false-positive, and sign criteria. It failed
overall coverage, worst-family coverage, and nonidentifiable-case passing.
Kernel additionally failed the sign-reversal criterion. The independent replay
matched all hashes, metrics, denominators, and dispositions.

## Why this is not an excessive-gate failure

The raw result was not a marginal miss caused by an ornate gate. Its overall
coverage was `0.0867` below the tolerant `0.90` threshold, and its asymmetric
family coverage was `0.2648` below the `0.80` worst-family floor. It also passed
128 of 1,536 explicitly nonidentifiable panels, compared with a maximum allowed
rate of `0.05`. Three distinct criteria failed.

At the same time, raw detected 1,216 of 1,264 meaningful positives, generated
zero false positives among 802 identifiable nonpositive cases, and reversed
sign in only 54 of 2,423 identifiable nonzero cases. The result therefore
locates the defect: directional discrimination is useful, but uncertainty and
identifiability handling are not calibrated.

## What failed and why

The raw estimator's coverage in the asymmetric-peak-and-tail family was
`0.5352`, with mean bias `0.1985`. Its bootstrap resamples already reduced
episode-level argmax gaps but does not represent within-episode uncertainty in
which event time is actually the peak. Asymmetric curves make that omission
systematic. Kernel smoothing amplified this effect: asymmetric-family mean bias
rose to `0.7598` and coverage fell to `0.2461`.

Raw passed 12.5% of both plateau/near-tie and informative-missingness panels,
despite their registered nonidentifiability. This is not fixed by a larger
support floor: the estimator needs an explicit peak-set, flatness/tie, and
missing-time identifiability rule that can return `INVALID` rather than forcing
an argmax.

The no-feedback observational mimic passed at rates `0.9141` for raw and
`0.9980` for kernel. That is not an estimator-admissibility criterion because
the estimand is observational, but it independently reinforces the claim
ceiling: a positive gap cannot identify feedback.

## Disposition

No estimator or support rule is selected. Thresholds will not be relaxed and
the opened challenge will not be reused as blind confirmation. It may now serve
only as a discovery diagnostic for an uncertainty-aware peak-set estimator with
explicit tie, flatness, boundary, and missingness nonidentifiability.

Any redesigned estimator must be fully specified and committed before a fresh,
disjoint synthetic blind challenge with a new seed. A hierarchical model is not
automatically the answer: it must propagate peak-location uncertainty and fail
closed on structural nonidentifiability, not merely shrink episode gaps.

The canonical artifact root is
`/ssd/pivot-feedback-lean-h3-blind-challenge-v1-assessment-20260806`; its
unblinding receipt SHA-256 is
`9522b2a7879069f1f98b6a3130d7efaf54cbdcac79bc0074a39e087bed0df0a4`.
The adjacent independent verification SHA-256 is
`f9a29e53aa837820055b38835e69f3509da5eac11154a403ae81a112554419ae`.

This result is synthetic estimator validation only. It is not empirical H3, a
null or association, support validation, a mechanism or causal result,
predictive validation, NABM necessity, or a scientific effect.
