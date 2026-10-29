# Pivot LLM-prior entry-matched control v1: result

Date: 2026-10-29

Status: `C22_PRECISION_IS_SELECTIVITY`

Design: [pivot-llm-prior-entry-matched-v1.md](../../../docs/research/pivot-llm-prior-entry-matched-v1.md),
frozen at `243dad757f69254f666ec92865ddccf162901f2b`.
Machine-readable record:
[pivot_llm_prior_entry_matched_v1_20261029.summary.json](pivot_llm_prior_entry_matched_v1_20261029.summary.json).

## Execution

- **Host:** the workstation, on CPU with 32 workers.
- **Wall time:** 628 s for the development run and 572 s for the reproduction run.
- **Reproduction:** all five artifacts are byte identical.
- **Scale:** 1,152 arm runs. No LLM calls were made.
- **Nesting checks:** every registered check passed exactly. These cover:
  - the hand fixed arm against the persistence predecessor;
  - the hand prior through the slot at zero shift, on every calibration
    replicate;
  - slot restoration;
  - the hash of the v2 prior; and
  - the matched priors' weights.

## Entry calibration

The fixed-rule entry does not depend on the world, so each matched level got
the same intercept shift in all three worlds.

| Matched condition | Target entry | Intercept shift | In sample | Study replicates, matched over source |
|---|---:|---:|---:|---:|
| hand direction at the LLM's entry | 0.3228 | −0.280 | 0.3230 | 0.959 |
| LLM direction at the hand rule's entry | 0.8171 | +1.349 | 0.8171 | 0.988 |

Both out-of-sample ratios lie within the registered bounds [0.95, 1.05]. The
matched hand rule admitted 4.1% fewer agents than the LLM prior. Along the
hand direction that is worth about +0.002 to +0.003 of fixed-rule precision to
the hand rule, far below the 0.02 threshold (post hoc).

## Registered decisions

**Ranking at the LLM's selectivity, without learning (primary).** This
compares `llm_selective` with `hand_at_llm_entry` under the fixed rule.

| World | Outcome | LLM minus hand precision [95% CI] |
|---|---|---|
| linear | `FIRST_BETTER` | +0.059 [+0.035, +0.082] |
| smooth interaction | `NO_CLEAR_DIFFERENCE` | +0.015 [−0.014, +0.044] |
| axis interaction | `NO_CLEAR_DIFFERENCE` | −0.001 [−0.025, +0.023] |

**Ranking at the LLM's selectivity, with pooled learners.** The LLM direction
is better in 1 of 9 cells (linear world, linear learner). It is worse in 1
cell (linear world, MLP learner). The other 7 cells show no clear difference.

**Ranking at the hand rule's selectivity.** The LLM direction is better in no
cell of 12 and worse in 7. All 7 are learner cells, and there are 5 cells
without a clear difference.

**C22 against the entry-matched hand rule:** `C22_PRECISION_IS_SELECTIVITY`.
This compares `hybrid_20` with `hand_at_llm_entry_20`.
- The hybrid's precision is better in no cell of 9. It is worse in 1 cell
  (linear world, MLP learner), at −0.022 [−0.034, −0.011].
- The paired means in the other 8 cells range from −0.016 to +0.012.
- Learning is non-inferior by the interval rule in 4 cells, so those are
  `LEARNING_ONLY`.
- No cell is `LLM_DOMINATES`.

**C22 replication against the hand rule on fresh seeds.** The hybrid's precision
is better in all 9 cells, from +0.036 to +0.090. It dominates in 2 cells by the
interval rule and in 4 cells by C22's mean-only rule.

| Seed set | Dominating cells, mean-only rule | Interval rule |
|---|---:|---:|
| v3 (C22) | 7 of 9 | 2 of 9, recomputed post hoc |
| v4 replication (C29) | 6 of 9 | not recorded |
| this study | 4 of 9 | 2 of 9 |

## Expectations

- **E1 held.** At the LLM's selectivity, the fixed-rule comparison is
  `FIRST_BETTER` in the linear world.
- **E2 held.** It is `NO_CLEAR_DIFFERENCE` in both interaction worlds.
- **E3 failed.** C22's precision survived in no cell, not in a minority of
  linear-world cells. The LLM direction's fixed-rule advantage in the linear
  world did not carry into the learners once the exploration floor was added.

## What this means

- **C22's precision gain is selectivity.** Shifting the hand rule's intercept by
  −0.28 logits, one number, reproduces it. Against that equally selective hand
  rule, the hybrid's precision gain disappears in every cell.
- **The LLM's ranking adds information only where its AUC said it would,** in
  the linear world and under the fixed rule. Pooled learning absorbs that
  advantage. At the hand rule's selectivity, the LLM direction is never better
  and is often worse inside the learners.
- **C22's dominance count depends on the seed set and the rule.** It fell from
  7 to 6 to 4 of 9 across three seed sets under its own mean-only rule, and it
  is 2 of 9 under the interval rule in both sets where that rule was computed.

## Boundary

The worlds are synthetic, and the comparison covers one prior from one model
family elicited once. The result says what this prior adds over this hand rule
in these worlds. It says nothing about LLM knowledge of real pivots. The
consolidation ledgers are frozen, so C22 and the README sentence that cites it
are narrowed only by a later consolidation.
