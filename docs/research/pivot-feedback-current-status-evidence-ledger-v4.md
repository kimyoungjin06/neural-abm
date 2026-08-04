# Pivot-feedback current-status and evidence ledger v4

Date: 2026-08-04

Status: `CURRENT_HUMAN_FACING_INDEX_F_FINAL_PANEL_BLOCKED_NO_F_PASS`

## Current authority

The machine-readable authority remains
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
The v1–v3 ledgers remain immutable provenance. This v4 ledger records the
executed final-panel component and its fail-closed result; it does not promote
F, pass G0, or replace the active authority.

## Final-panel production replay

The
[`F final-panel conformance receipt`](../../experiments/evidence/results/pivot_feedback_f_final_panel_conformance_v1_20260804.summary.md)
replayed the frozen primary/reserve field order over all 194
aggregate-screen-eligible topic-year attempts. It used 647,855,727
author-token risk opportunities, the registered work-level pre-signal
incumbent population, exact focal-author leaveout, and exact preexisting
target-active-peer leaveout. No team cap, outcome label, model score, or raw-ID
output was introduced.

All 194 attempts pass the label-independent total-support floor of 500. The
final-panel block is instead a shock-family conformance failure:

| Slot | Frozen attempts | Final field | Leaveout-surge families | Required | State |
| --- | --- | ---: | ---: | ---: | --- |
| G1 physical | 17 | — | 1 | 3 | BLOCKED |
| G1 social | 32 | — | 1 | 3 | BLOCKED |
| G1 life | 11 | — | 2 | 3 | BLOCKED |
| G4 author-cold | 36 → 27 | 27 | 6 | 3 | PASS structural only |

Fields 16, 18, 20, and 24 cannot be activated because they already fail the
frozen aggregate three-family screen; nominal fields 25 and 28 fail it as
well. Health field 29 has no eligible aggregate candidate, while field 27 is
the next eligible frozen G4 reserve and passes the leaveout replay. The G4
result cannot rescue any failed G1 slot.

The independent verifier rehashed the production artifact, reconstructed all
194 attempts from the 1,744-row assignment parent, replayed reserve order,
verified row algebra and leaveout monotonicity, confirmed raw-ID absence, and
returned
`PASS_INDEPENDENT_BLOCKED_FINAL_PANEL_LEAVEOUT_REPLAY_NO_F_PASS`.

## Gate proportionality correction

Outcome-conditioned support requirements are not treated as F prerequisites.
The 100-entry/100-non-entry and four-cell outcome-class floors require the
first confirmatory labels, but the active contract authorizes that read only
after F. Making them block F would therefore create a circular gate. Their
numeric rules remain frozen and fail-closed at the authorized G1 runtime; no
threshold was relaxed.

This leaves one substantive F block rather than a chain of micro-gates: the
registered primary attention universe does not retain three independent shock
families in any G1 slot after the required leaveouts. Consequently the private
holdout materialization, preflight, and byte seal are not authorized now.

## Next bounded action

The next action is an outcome-blind definition-and-theory audit, not another
release gate and not continued generic auditing. It must separate at least:

1. the work-level interpretation of “exclude same-window new entrants” that
   requires every linked author to be a pre-signal incumbent;
2. aggregate-screen versus author-specific normalization;
3. current-snapshot topic/author-linkage observation-universe loss; and
4. focal versus preexisting-peer leaveout contributions.

Any successor must be substantively justified before inspecting entry,
persistence, effect, or G4 outcome rows. It may not relax the z threshold,
select fields outside the frozen reserve order, or reinterpret this structural
failure as a null scientific effect.

## Hard stops and claim boundary

`pass_f=false` and `pass_g0=false`. Final-panel conformance is blocked, the
private holdout byte seal does not exist, confirmatory entry or persistence
labels have not been computed, outcome-conditioned support has not been
evaluated, realized citation effects have not been read, no model has been
fitted, and G4 remains analytically unopened.

This replay establishes a measurement-definition and observation-universe
failure in the frozen G1 panel construction. It establishes neither absence
nor presence of an attention-entry association, an H3 timing gap, peer
learning, predictive added value, NABM necessity, holdout transport,
causality, nor a scientific effect.
