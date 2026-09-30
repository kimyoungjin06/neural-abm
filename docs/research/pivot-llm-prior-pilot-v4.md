# Pivot local-LLM prior pilot v4: model-size ladder

Date: 2026-09-30

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_llm_prior_pilot_v4.yaml`](../../experiments/configs/pivot_llm_prior_pilot_v4.yaml)

## Why v4 exists

In NABM the LLM is consulted once and distilled into a nine-weight logistic
prior. Its size therefore changes the quality of that prior, not the cost of
running the ABM. The question behind v4 is whether a lighter, faster LLM can do
the job.

[v3](../../experiments/evidence/results/pivot_llm_prior_pilot_v3_20260929.summary.md)
found that the calibrated 4B prior plus a 20% exploration floor dominates the
hand rule in 7 of 9 cells. v4 asks whether smaller models of the same family
still carry the ranking that makes this work, and how each compares with the 4B
model used in v1 to v3.

## Models

| Id | Model | Role |
|---|---|---|
| `qwen3_0_6b` | Qwen3-0.6B, non-thinking mode | Same-release ladder |
| `qwen3_1_7b` | Qwen3-1.7B, non-thinking mode | Same-release ladder |
| `qwen3_4b` | Qwen3-4B, non-thinking mode | Same-release ladder |
| `qwen3_4b_2507` | Qwen3-4B-Instruct-2507 | Anchor: the v1 to v3 model |

The three same-release models separate size from post-training. Comparing the
original 4B with the 2507 4B shows how much the newer instruction tuning
matters. All weights are open, pinned by snapshot, and run locally. An 8B model
does not fit the 16 GB GPU in bf16, so it is not included.

## Readout

Every model sees the v1 prompts byte for byte, checked by hash, on the v1
design cohort of 1,024 profiles.

The v1 readout does not work for every model. It reads the first answer token
directly. A format-only dry run on 64 profiles from a disjoint seed showed that
Qwen3-1.7B opens with "To determine" or "Based on", leaving a median answer mass
of 0.0002. The dry run inspected answer mass, the identity of non-answer top
tokens, determinism, and timing. It inspected no probabilities or rankings.

v4 therefore prefills the assistant turn with `Answer: **` and reads
`logit(Yes) − logit(No)` at the next token. This was the first prefill tried
that was valid for all four models. In the dry run, answer mass was above 0.999
and the top token was always an answer, for every model. Repeat passes were
identical. The same readout is used for all four models, so the 2507 anchor is
re-elicited. The v1 first-token readout is still recorded for every model, as a
validity diagnostic.

Each readout is distilled and calibrated exactly as in v2. A ridge fit on
log-odds is followed by an intercept shift to the stated 0.30 base rate, using
aggregate information only. The canonical batch size is 1, and a batch-32 pass
is reported as a sensitivity check.

## Conditions

Each model contributes two conditions: `selective`, with no exploration, and
`hybrid`, with v3's exploration floor at a 20% share. The share is fixed from v3
and not tuned per model. A `v3_reference` pair uses the exact v3 prior, which
is rebuilt from the v1 elicitation and checked equal to the v2 file. It tests
whether v3's result replicates on fresh seeds, and how much the readout change
matters at 4B. The `hand` and `flat` conditions complete the set.

That gives 12 conditions, 3 fit worlds, 4 arms (the fixed rule and the field
linear, tree, and MLP learners), and 16 replicates of 256 agents: 2,304 runs.

## Registered decisions

In each of the nine combinations of world and field arm:

- **Hybrid against the hand rule**, as in v3. Precision must improve by at
  least 0.02 in at least 12 of 16 replicates. Mean probe Brier may be at most
  0.005 worse. Each cell is classified as dominating, selection only, learning
  only, or neither.
- **Selective prior against the hand rule**, using the same precision rule.
  This measures the model's ranking knowledge in behavior.
- **Same-release models against the anchor.** A model matches the anchor when
  its hybrid's mean precision is no more than 0.02 lower and its mean probe
  Brier no more than 0.005 higher.

Each is summarized over the nine cells as holding in all cells, in a majority,
in a minority, or nowhere. The size frontier reports the smallest same-release
model whose hybrid dominates the hand rule in a majority of cells. It is
descriptive and selects no model.

Phase A also reports each model's readout validity, determinism, distillation
fidelity, and rank agreement with the anchor and with the v1 readout. It
reports the design-cohort AUC of each calibrated prior in each world, next to
the hand rule's, and the cost of each model: seconds per query at batch 1 and
32, peak GPU memory, and parameter count. The cost figures feed the N06
LLM-agent reference design.

## Boundary

The executable must be committed before any elicitation. Each model is elicited
twice, and the elicitation outputs must match byte for byte. The ABM then runs
twice and must reproduce every artifact byte for byte. No post-result change to
the readout, share, priors, or thresholds is authorized, and no model is
selected.

The worlds are synthetic, so the results show how each model's intuitions line
up with the designer's worlds, not knowledge of real pivots. The pilot makes no
paid API calls, reads no empirical rows, adds no LLM dependency to the package,
and changes no generic NABM API.
