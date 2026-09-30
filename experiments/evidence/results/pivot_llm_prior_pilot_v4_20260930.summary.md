# Pivot local-LLM prior pilot v4 result: model-size ladder

Date: 2026-09-30

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED_WITH_REPRODUCED_LOCAL_LLM_ELICITATIONS`

## Result

**Qwen3-4B is the smallest model in the ladder that works.** Within one Qwen3
release, it is the smallest model whose calibrated prior, with the v3
exploration floor, beats the hand rule in a majority of cells (7 of 9). Its
selective prior picks better entrants than the hand rule in all nine cells. It
matches the 4B-2507 anchor in eight.

**Below 4B, quality is erratic rather than gradually lower.** Qwen3-0.6B ranks
the linear world almost perfectly but is near chance in the interaction worlds,
and it dominates in 4 of 9 cells. Qwen3-1.7B's readout carries almost no linear
signal, and it dominates in only 1 of 9. The 0.6B beats the 1.7B on dominance
(4 against 1) and on selection (7 against 0), and they tie at 5 of 9 on
matching the anchor. A smaller model is not a drop-in replacement.

**Post-training matters less than size.** The original Qwen3-4B and the
4B-2507 used in v1 to v3 behave alike: 7 and 8 of 9 cells.

**v3 replicates on fresh seeds.** The exact v3 prior dominates the hand rule in
6 of 9 cells, against 7 of 9 in v3.

The run used `15dc4d1`. Each of the four models was elicited twice on the v1
prompts, which were checked byte for byte, and both passes matched exactly. The
ABM ran 2,304 times: 12 conditions, 3 fit worlds, 4 arms, and 16 replicates of
256 agents. A second execution reproduced all four artifacts byte for byte.
Every nesting check passed exactly. That includes rebuilding the v3 prior from
the v1 elicitation and matching the v2 file exactly. No paid API was called.

## Registered decisions

Each prior was evaluated in the nine combinations of world and field arm.
"Dominates" means entrant precision improves by at least 0.02 in at least 12 of
16 replicates while pooled learning stays within 0.005 Brier of the hand rule.
"Selects better" applies the precision rule to the prior without exploration.
"Matches anchor" means the hybrid loses at most 0.02 precision and 0.005 Brier
against the 4B-2507 hybrid.

| Prior | Hybrid dominates hand | Selective prior selects better | Hybrid matches anchor |
|---|---:|---:|---:|
| Qwen3-0.6B | **4 of 9** | 7 of 9 | 5 of 9 |
| Qwen3-1.7B | **1 of 9** | 0 of 9 | 5 of 9 |
| Qwen3-4B | **7 of 9** | 9 of 9 | 8 of 9 |
| Qwen3-4B-2507 (anchor) | **8 of 9** | 9 of 9 | — |
| v3 reference prior | **6 of 9** | 9 of 9 | — |

The registered size frontier, the smallest same-release model dominating in a
majority, is Qwen3-4B. It is descriptive, and no model is selected.

## Phase A: what each model knows

The readout values are the median answer mass: the probability mass on "Yes"
and "No" at the readout token. The v4 readout prefills the answer with
`Answer: **`. AUC is that of each calibrated prior against the productive labels
of the 1,024-profile design cohort in each world. Rank agreement is Spearman's
correlation with the anchor's readout.

| Model | Parameters | v1 readout mass | v4 readout mass | Distillation R² | Rank agreement with anchor | Linear AUC | Smooth AUC | Axis AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.60B | 0.7814 | 0.9998 | 0.503 | 0.379 | 0.947 | 0.500 | 0.525 |
| Qwen3-1.7B | 1.72B | 0.0002 | 1.0000 | 0.045 | 0.446 | 0.443 | 0.532 | 0.543 |
| Qwen3-4B | 4.02B | 0.7772 | 1.0000 | 0.178 | 0.911 | 0.824 | 0.605 | 0.620 |
| Qwen3-4B-2507 (anchor) | 4.02B | 0.9993 | 1.0000 | 0.249 | 1.000 | 0.715 | 0.630 | 0.642 |
| v3 reference prior | — | 0.9993 | — | 0.449 | 0.961 | 0.707 | 0.632 | 0.644 |
| Hand rule | — | — | — | — | — | 0.647 | 0.622 | 0.643 |

In the interaction worlds, the 4B's AUC is slightly below the hand rule's,
yet its prior selects better in the ABM: 0.454 and 0.442 entrant precision,
against 0.342 and 0.341. The hand rule admits 80% of agents, so its entrants
are close to the population rate. A selective prior needs to rank well only at
the top.

The 1.7B model shows why the readout had to change. Under the v1 readout, it
opens its answer with "To determine" or "Based on". Under the prefilled readout,
every model puts a median of more than 0.999 of its mass on an answer.

At 4B-2507, the change of readout barely matters. The two readouts rank
profiles with a Spearman correlation of 0.961. Their hybrids differ by −0.006
to +0.013 in precision and by at most 0.0041 in probe Brier across the nine
cells.

## Condition profiles

Fixed-rule columns show the prior on its own, with no learning. The other
columns are the pooled field learners.

| World | Prior | Fixed-rule entry | Fixed-rule precision | MLP precision | MLP Brier | Tree precision | Linear precision |
|---|---|---:|---:|---:|---:|---:|---:|
| Linear | hand | 0.803 | 0.338 | 0.438 | 0.015 | 0.402 | 0.412 |
| Linear | 0.6B selective | 0.252 | 0.761 | 0.559 | 0.015 | 0.405 | 0.380 |
| Linear | 1.7B selective | 0.063 | 0.303 | 0.388 | 0.101 | 0.284 | 0.320 |
| Linear | 4B selective | 0.311 | 0.634 | 0.734 | 0.018 | 0.528 | 0.548 |
| Linear | 4B-2507 selective | 0.336 | 0.512 | 0.659 | 0.017 | 0.527 | 0.571 |
| Linear | 4B hybrid | 0.462 | 0.479 | 0.571 | 0.016 | 0.474 | 0.482 |
| Linear | 4B-2507 hybrid | 0.479 | 0.415 | 0.533 | 0.016 | 0.468 | 0.489 |
| Smooth | hand | 0.803 | 0.342 | 0.412 | 0.084 | 0.399 | 0.366 |
| Smooth | 0.6B selective | 0.252 | 0.409 | 0.524 | 0.108 | 0.444 | 0.412 |
| Smooth | 1.7B selective | 0.063 | 0.344 | 0.342 | 0.175 | 0.251 | 0.279 |
| Smooth | 4B selective | 0.311 | 0.454 | 0.576 | 0.117 | 0.484 | 0.453 |
| Smooth | 4B-2507 selective | 0.336 | 0.451 | 0.542 | 0.103 | 0.480 | 0.460 |
| Smooth | 4B hybrid | 0.462 | 0.391 | 0.484 | 0.085 | 0.428 | 0.402 |
| Smooth | 4B-2507 hybrid | 0.479 | 0.391 | 0.464 | 0.083 | 0.429 | 0.402 |
| Axis | hand | 0.803 | 0.341 | 0.403 | 0.109 | 0.423 | 0.361 |
| Axis | 0.6B selective | 0.252 | 0.428 | 0.489 | 0.143 | 0.505 | 0.403 |
| Axis | 1.7B selective | 0.063 | 0.324 | 0.344 | 0.180 | 0.290 | 0.297 |
| Axis | 4B selective | 0.311 | 0.442 | 0.529 | 0.170 | 0.555 | 0.461 |
| Axis | 4B-2507 selective | 0.336 | 0.431 | 0.504 | 0.144 | 0.515 | 0.452 |
| Axis | 4B hybrid | 0.462 | 0.387 | 0.463 | 0.116 | 0.474 | 0.394 |
| Axis | 4B-2507 hybrid | 0.479 | 0.382 | 0.448 | 0.110 | 0.467 | 0.393 |

## Cost

Seconds per query are GPU wall-clock time for the first-token forward pass. The
LLM-agent arm assumes one query per agent-step for one arm over this design,
which is 221,184 queries. It is a lower bound, because a real LLM agent would
also need evidence in its prompt.

| Model | s per query, batch 1 | s per query, batch 32 | One-time elicitation | LLM-agent arm, batch 32 | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.0449 | 0.0064 | 46 s | 0.39 h | 4.7 GB |
| Qwen3-1.7B | 0.0285 | 0.0135 | 29 s | 0.83 h | 7.0 GB |
| Qwen3-4B | 0.0467 | 0.0303 | 48 s | 1.86 h | 11.9 GB |
| Qwen3-4B-2507 (anchor) | 0.0477 | 0.0300 | 49 s | 1.84 h | 11.8 GB |

At batch 1, every model takes 0.03 to 0.05 seconds per query, because the pass
is dominated by overhead. At batch 32, the 0.6B model is 4.7 times faster than
the 4B.

## Post-hoc checks

These checks were not registered. They reuse the frozen outputs and run nothing
new.

**Phase A predicts ABM selection.** Across the five priors and three worlds, the
design-cohort AUC and the fixed-rule selective precision in the ABM have a
Spearman correlation of 0.857. In these synthetic worlds, a cheap screen before
any ABM run already tells which priors will select well.

**The 0.6B's strength is alignment with the designer.** Its weights share all
six signs of the designer's linear productive-fit formula. It also puts its
largest weight on field opportunity, which is the formula's largest term. The
original 4B also matches all six signs, but its largest weight is on openness.
The 0.6B's 0.947 linear-world AUC therefore reflects agreement with the
designer's formula, not knowledge of real pivots.

**Pooled learning erodes the 0.6B's linear-world selection.** On its own, the
0.6B prior admits entrants who are productive 76.1% of the time. Once the
field learners update it, entrant precision falls to 55.9% for the MLP, 40.5%
for the tree, and 38.0% for the linear learner. The 4B prior keeps or improves
its selection: 63.4% on its own, and 73.4%, 52.8%, and 54.8% under the three
learners. The 0.6B's weights are small, with a largest absolute weight of 0.56
against 1.19 for the 4B, which may let the learners override it more easily.

**The 1.7B prior is nearly flat.** Its largest absolute weight is 0.13, and its
distillation explains 4.5% of the readout's log-odds variance. Calibrated to a
0.30 mean, a near-constant prior sits just below the 0.34 pivot threshold, so
only 6.3% of agents enter under the fixed rule.

## What this means for NABM

- **For the prior slot, a lighter LLM is not worth it.** Elicitation takes 29
  to 49 GPU seconds for every model in the ladder. The savings are seconds,
  while quality drops sharply below 4B. The 4B remains the right model for the
  prior.
- **For the LLM-agent reference, a lighter LLM works as a cost floor.** Queried
  in batches of 32, the 0.6B would run an LLM-agent arm of this design in about
  0.4 GPU hours, against about 1.9 for the 4B. Its behavior is weaker, so it
  can bound the cost of the LLM tier from below but cannot stand in for its
  behavior.
- **The readout is part of the model choice.** A first-token readout that works
  for one model can fail for another of the same family. A prefilled forced
  choice worked for all four.

## Boundary

The worlds and the stated base rate are synthetic and stylized, so the results
show how each model's intuitions line up with the designer's worlds, not
knowledge of real pivots. The pilot made no paid API calls, read no empirical
rows, selected no model, added no LLM dependency to the package, and changed no
generic NABM API.
