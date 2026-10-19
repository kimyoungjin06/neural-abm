# N14 parity: the Pivot ladder through the shared core

Date: 2026-10-19

Status: `N14_PARITY_HOLDS_BYTE_FOR_BYTE`

Specification: [N14 parity](../../../docs/research/n14-pivot-parity-v1.md),
fixed at `c361a48` before the run. Host: the workstation where the Pivot
result was recorded (CPU only), as the user chose on 2026-10-19.

## What ran

**Target.** The Pivot LLM-prior pilot v3 (precommit `87c90eb`), with
1,152 recorded rows (`runs.jsonl`, sha256 `1f532d98…`).

**Re-expressed through the core.** For its three field-public learner arms,
these came from `scripts/policy_ladder_core_v1.py`:

- the learners' prior (`LinearPrior`);
- the learners (`fit_residual` linear and mlp, `fit_prior_boosted_trees`);
- the refit seeds (`keyed_seed`);
- the outcome record (`OutcomeRecord`);
- the exploration floor (`ExplorationFloor` on `keyed_uniform`).

**Kept in Pivot's code.**
- the world;
- the fixed-rule arm;
- the agents' own policy parameters.

## Result

| Run | Tasks | Rows identical to the record | Whole file | Wall time |
|---|---:|---:|---|---:|
| Baseline: Pivot's own code, replicate 0 | 18 | 72 of 72 | (a subset) | 79 s |
| Parity: re-expressed through the core | 288 | 1,152 of 1,152 | **Identical**, sha256 `1f532d98…` | 451 s |

Every arm matches in every row: the fixed rule, and the linear, tree, and MLP
learners. The quoted estimate was 20–35 minutes; the run took 7.5.

## What the re-expression found

Three findings bear on extraction.

1. **The prior lives in two places in Pivot.**
   - It is the learners' anchor.
   - It is also each agent's own policy parameters, which the fixed rule
     uses and every arm reports as `mean_attention_weight`.
   - The core holds only the anchor.
   - Before the freeze, the field arms ran without Pivot's prior slot. Every
     decision, trajectory, and probe score was identical; only
     `mean_attention_weight` differed.
2. **A fixed rule is not numerically portable.**
   - Pivot's rule is computed in torch, one agent at a time. The core's numpy
     prior works on the stacked matrix.
   - The two differ in the last bits: 5,452 of 20,000 uniform rows, by at
     most 2.2 × 10⁻¹⁶.
   - So the fixed-rule arm stays in the domain's code.
3. **The row order under delays is not yet tested.**
   - The core returns outcome rows in insertion order; Pivot buffers them in
     release order.
   - v3 has immediate visibility, so this run does not test the difference.

## Reading

**What holds.** The core expresses the Pivot ladder's learner, prior-anchor,
outcome-record, exploration, and keyed-seed semantics exactly, at immediate
visibility.

**Together with the information-spread study,** two consumers now use these
semantics without domain-specific branches. That meets the bridge rule for
them.

**Boundary.** One recorded configuration, at immediate visibility, on one
host. No Pivot result is reinterpreted, and no package API is changed.

## Next

Decision 0017 proposes what to extract, using these findings.
