# Information-spread policy ladder: the LLM arms, stage D5

Date: 2026-10-19

Status: `D5_NULL_PASSES_NABM_BEATS_ARCHETYPES_LLM_PRIOR_ADDS_NOTHING_DETECTABLE_TO_THE_LEARNER`

Design: [the LLM arms (D3b)](../../../docs/research/info-spread-policy-ladder-llm-arms-v1.md),
frozen at `dcc6e74` together with study v2, before D4 v2 ran.

## What ran

**Elicitation.** Qwen3-4B-Instruct-2507 (snapshot `cdbee75`) was elicited
twice on GPU-Server, at batch 1.
- **Prompts:** 17,600, namely 2,048 for P0 and 15,552 for the X0 table.
- **Time:** about 8 minutes of forward passes each time.
- **Validity:** the median answer-token mass is 0.99999, and the two
  elicitations match byte for byte.

**Arms.** P0, X0, and L1 ran in the study v2 cells and were paired with D4
v2's arms.
- **Where:** a checkout of the freeze commit, with GPU 2 and 24 workers.
- **Run times:** 90 minutes for the development run and 90 for the
  reproduction. The quoted estimate was 90–140 minutes.
- **Reproduction:** `cells.jsonl` and `summary.json` match byte for byte.
- **Parity:** L0 was recomputed at replicate 0 of all 320 L1 tasks, and
  equals D4 v2's value in every one.

**Cost of X0.** Querying during the run without a cache would take 480
queries per run in Regime A and 1,440 in Regime B. The table took 15,552
prompts once.

## Registered result

The primary measure is share precision on unverified items. **The design
check passes.** In the null groups, L1 − P0 is not material and positive in
any group.

| Comparison | Regime A (6 study groups) | Regime B (6 study groups) |
|---|---|---|
| L1 − L0, the LLM prior inside NABM | `MIXED_OR_NONE`: 2 positive (axis, +0.010 and +0.020), 4 indistinguishable | `MIXED_OR_NONE`: 6 indistinguishable (−0.001 to +0.010) |
| L0 − X0, NABM against archetypes | `MOSTLY_POSITIVE`: 6 of 6 (+0.162 to +0.229) | `MOSTLY_POSITIVE`: 6 of 6 (+0.155 to +0.227) |
| L1 − P0, the closed loop on the LLM prior | `MOSTLY_POSITIVE`: 6 of 6 (+0.204 to +0.316) | `MIXED_OR_NONE`: 2 positive (axis), 3 negative (linear and smooth, −0.012 to −0.041), 1 indistinguishable |

**Descriptive, under the same rule.**
- P0 − F1: −0.151 to −0.238 in Regime A, and +0.080 to +0.107 in Regime B.
- X0 − F1: −0.114 to −0.162 in Regime A, and −0.089 to −0.098 in Regime B.

**Mean precision on unverified items, study groups.**

| Arm | Regime A | Regime B |
|---|---|---|
| F1 | 0.55–0.64 | 0.53–0.57 |
| P0 | 0.40 | 0.63–0.65 |
| X0 | 0.44–0.48 | 0.44–0.48 |
| L0 | 0.61–0.71 | 0.60–0.70 |
| L1 | 0.61–0.72 | 0.62–0.71 |
| T0-4 | 0.65–0.74 | 0.65–0.75 |

## Description after the run (not registered)

**P0 in Regime A is a near-constant prior, because its readout saturated.**
- Qwen answered "No" with great confidence to most Regime A profiles: the
  median log-odds is −16.3.
- 99% of the answers lie below the clip inherited from Pivot (probability
  10⁻⁴).
- So the distillation target is nearly constant (R² = 0.026), although the
  raw answers vary (sd 2.8).
- As a result, P0/A almost never shares (about 470 unverified shares per run,
  against about 343,000 for F1). Its precision is the base rate.
- In Regime B, 70% of answers lie below the clip. The distilled prior still
  carries information (R² = 0.48) and beats the hand rule.
- The X0 table is not clipped.

**Precision rewards selectivity.**
- **P0 in Regime B.** It shares 2–15% fewer unverified items than L1,
  depending on the world. Where it beats L1 (the linear and smooth worlds),
  part of its gain in precision may come from that selectivity. The
  registered measure is not the agents' utility.
- **X0.** It shares 1.4–1.7 times as many items as F1, at a precision
  near the base rate.

## Reading

Within this design:

- **NABM against archetypes.** The adaptive learner beats AgentTorch-style
  archetypes, given the same public information, in all 12 study groups by
  0.16–0.23. The archetypes also fall below the hand rule.
- **An LLM prior inside the learner adds no detectable gain.** This holds
  even in Regime B, where the prior itself beats the hand rule. Learning
  from the run's outcomes overrides the difference in starting point.
- **The closed loop on the LLM prior.**
  - In Regime A, the learner recovers from an uninformative P0.
  - In Regime B, the comparison is mixed. Its sign may depend partly on the
    prior's selectivity, as described above.

**Boundaries.**
- **P0/A.** It reflects a readout that saturated under the inherited clip.
  It is not evidence that the model lacks knowledge. A readout without the
  clip would be a new design.
- **The model.** One small local model (4B) with one prompt wording.
- **The worlds.** Synthetic, stationary worlds. No claim is made about real
  information spread or real people.
