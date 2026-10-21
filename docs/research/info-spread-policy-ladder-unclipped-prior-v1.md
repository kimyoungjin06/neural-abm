# Information-spread policy ladder: unclipped-prior study v1 (O24)

Date: 2026-10-21

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`info_spread_policy_ladder_unclipped_prior_v1.yaml`](../../experiments/configs/info_spread_policy_ladder_unclipped_prior_v1.yaml)

This is the third of the four follow-ups the user chose on 2026-10-21 to run
in order (N17).

## Question

D5 distilled the once-elicited LLM prior after clipping each answer to a
probability in [10⁻⁴, 1 − 10⁻⁴], a rule inherited from the pivot ladder.
In Regime A, 99% of the model's answers fell below the clip, with a median
log-odds of −16.3. The distillation target was nearly constant
(R² = 0.026), so P0/A was nearly constant too. Consolidation v6 records
this as a readout property and as open item O24.

With the readout no longer saturating:

1. **Do the model's answers carry usable information?**
2. **Does such a prior help the learner?**

## Design

**The prior P0u.**
- **No new LLM call.** The elicitation is D5's first registered elicitation,
  whose hash the D5 record names, and its prompts are checked against D3b.
- **The readout.** The raw log-odds (the logit of Yes minus the logit of
  No), with no clip.
- **The rest is D3b's.** The same design cohort and features, the same
  ridge distillation (λ = 0.001 per profile), and the same single intercept
  shift to the stated base rate 0.4.
- **Not inspected before the freeze.** Neither the unclipped prior's weights
  nor its fit are looked at; they are first read in the record.

**Arms, in the study v2 cells** (8 worlds × 20 cells × 16 replicates):

| Arm | What it is |
|---|---|
| P0u | The unclipped prior, as a fixed rule |
| L1u | The study learner (each family) started from P0u, with study v2's learner seeds |

**Read, not rerun.**
- F1 and L0 come from study v2's development cells.
- P0 and L1 come from D5's development cells.
- Both are bound by their records.
- **Parity.** At replicate 0 of every L1u task, L0 is recomputed. It must
  equal study v2's value, and any failure blocks reading.

## Comparisons and rule

The group rule of study v2 applies, with its own bootstrap keys (tag 12), on
share precision on unverified items:

| Comparison | Pair | Question |
|---|---|---|
| `unclipped_prior_vs_hand` | P0u − F1 | Does the unclipped prior carry usable information? |
| `unclipped_prior_in_learner` | L1u − L0 | Does it help the learner? |
| `closed_loop_on_unclipped_prior` | L1u − P0u | Does the closed loop add to it? |
| Descriptive | P0u − P0, L1u − L1 | What the readout changed |

**Disposition per regime and comparison**, over 6 study groups:

| Disposition | Condition |
|---|---|
| `MOSTLY_POSITIVE` | Material and positive in at least 4 |
| `MOSTLY_NEGATIVE` | Material and negative in at least 4 |
| `MIXED_OR_NONE` | Otherwise |

**Design check.** In the null geometry, L1u − P0u must not be material and
positive in any group. A violation is `DESIGN_DEFECT_IN_NULL_WORLD`.

**Expected before the run.** Not predicted. If the unclipped Regime A
answers carry information, P0u − F1 should rise from D5's P0 − F1. Whether
the learner then gains is the open question.

## Runs and record

1. **Pilot.** A full-load pilot (2 cells per group) for the time estimate.
   Its summary, which contains the unclipped prior's fit, is not opened.
2. **Development and reproduction runs.**
   - On GPU-Server, on GPU 2 with 24 workers.
   - `cells.jsonl` and `summary.json` must match byte for byte.
3. **Record.** A summary and a result test.

## Boundary

- **The model and prompts.** One small local model (Qwen3-4B-Instruct-2507)
  and one prompt wording, read once.
- **The readout.** Removing the clip is one readout choice among others.
- **The worlds.** Synthetic only. The model's knowledge of real information
  spread is not established.
