# Section 6 Draft: Calibration and Analysis

Status: draft prose candidate.

Source tables:

- `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 2
- `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 3
- `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 4
- `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 5

## 6.1 Calibration Principle

The evidence package is calibrated around bounded claims rather than a single
global pass/fail result. This is necessary because a neural ABM can fail for
different reasons that require different interpretations. A final-epoch flip
after reaching ceiling is not the same failure as never entering the target
basin. Slow time-to-ceiling is not the same claim as incorrect direction.
Likewise, a classical baseline that is faster in a clean environment does not
invalidate a targeted stress where that baseline's information channel is
fragile.

For that reason, the current analysis reports final ceiling hits, ever ceiling
hits, mean time-to-ceiling, terminal ceiling rate, and failure-mode tags
together. These diagnostics prevent a speed race against a hand-coded baseline
from being mistaken for the whole research question.

## 6.2 Toy5 Threshold-Aware Readiness

The cleanest bounded holdout result is Toy5. In the tested grid, the
threshold-aware readiness adapter preserves no-seed safety and reaches full
cascades in sparse-seed spread cases where output averaging stalls. The grid
includes lattice `k=4`, lattice `k=6`, and rewired `k=6, p=0.10` topologies at
heterogeneous high thresholds `0.85` and `0.95`. The output-average baseline
reaches `0/5` final hits in all six spread cases, while the threshold-aware
main path reaches `5/5` in every spread case and preserves `5/5` no-seed
safety.

This result supports a specific holdout claim: the shared NABM lifecycle can
carry a threshold-aware readiness adapter that preserves safety and recovers
spread where output averaging stalls. It does not prove that threshold-aware
direction is uniquely required for seeded spread. The exposure-anchor negative
control also reaches `5/5` in every seeded spread case, so the stronger
necessity claim remains open.

## 6.3 Toy2/Toy4 Failure-Mode Triage

Toy2 and Toy4 require a different framing. Several clean runs remain
baseline-favored because reputation imitation is a strong hand-coded rule in
these environments. The useful contribution is therefore diagnostic before it
is comparative: the evidence gate separates stochastic final-epoch hazard, slow
time-to-ceiling, baseline-favored success, and targeted stress success before
adding another loss or sampler path.

The triage shows that the failed `revision_operator_quick` cases are not clean
mechanism-impossibility results. Both Toy2 and Toy4 reach the target basin but
lose final hits through final-epoch hazard. The Toy2
`basin_credit_objective_blend_quick` miss is also not a direction failure: the
best main path reaches `3/3` final hits but misses the speed gate at mean
time-to-ceiling `12.00`. Precommitment plus peer evidence removes the late
hazard in the checked control and stress artifacts, reaching `3/3` or `5/5`
final hits depending on the manifest.

This supports a failure-mode taxonomy, not a universal speed claim. Toy2/Toy4
results should be cited with their classification tags, especially when a
passing main variant is still baseline-favored.

## 6.4 Targeted Toy2/Toy4 Reputation Fragility

The strongest Toy2/Toy4 contrast is the targeted reputation-fragility stress.
With sparse initial action seeds, open boundaries, and noisy peer ranking, the
reputation-imitation baseline reaches `0/5` final and ever ceiling hits in both
Toy2 and Toy4. Under the same stress, the precommitment plus peer-evidence
candidate reaches `5/5` final and ever hits, with mean time-to-ceiling `9.4`
for Toy2 and `9.0` for Toy4.

This is useful evidence because it demonstrates a condition where the neural
candidate remains stable while the clean reputation rule's information channel
is fragile. The limitation must remain attached to the result: the stress
directly weakens the reputation baseline's own ranking signal. It is therefore
targeted baseline-fragility evidence, not a general demonstration that neural
ABMs dominate hand-coded rules.

## 6.5 Toy4 Local Resource Robustness

Toy4 adds resource-specific structure. In the heterogeneous local-resource
stress, resource extraction varies in a checkerboard pattern and reputation
ranking is noisy. Under this condition, the noisy reputation diagnostic reaches
`3/5` final ceiling hits, and the population-threshold negative control reaches
`0/5`. The local resource-threshold variants remain stable at `5/5`, with the
local-sustain main variant reaching mean time-to-ceiling `31.8`.

This supports a Toy4-specific robustness claim: local resource thresholding
plus precommitment and peer evidence remains stable when resource damage differs
across space and reputation ordering is noisy. It does not establish that
local-sustain observation is necessary. Hidden and global resource observation
variants are close at mean time-to-ceiling `32.2` and `33.0`, and clean
reputation imitation remains faster in clean ranking conditions with `5/5`
final hits and mean time-to-ceiling `15.0`.

## 6.6 Manuscript Insertion Notes

Use Table 2 for the Toy5 holdout claim, Table 3 for the failure-mode taxonomy,
Table 4 for the targeted Toy2/Toy4 reputation-fragility contrast, and Table 5
for Toy4 local resource robustness.

The safest high-level wording is:

> The current evidence supports a reusable NABM unit contract and several
> bounded robustness cases. It does not yet support general neural dominance
> over clean hand-coded ABM baselines.
