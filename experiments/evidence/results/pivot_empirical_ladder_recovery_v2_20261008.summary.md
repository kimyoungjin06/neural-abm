# Pivot empirical ladder recovery v2 (N11a v2)

Date: 2026-10-08

Status: `IDENTIFICATION_PASSES_LEARNING_SEPARATED_FROM_ANTICIPATION`

Precommit: `0cbe5f2`, amending v1 (`af43b16`). Design:
[`pivot-empirical-ladder-recovery-v2.md`](../../../docs/research/pivot-empirical-ladder-recovery-v2.md)

## What ran

The study ran 8 synthetic worlds × 8 replicates at the S1 scale, on fresh
seeds. Development and reproduction runs were byte identical, about 6
minutes each. No empirical row was read.

## Registered result: identification passes

An NABM win requires beating the chosen non-learning model, B2, and the new
anticipatory competitor B7.

| World | Role | NABM wins | Beats B7 | Median LL vs B7 per episode | Learned coefficient |
|---|---|---:|---:|---:|---:|
| W1 learning | must recover | **8/8** | 8 | +38.1 | 1.00 |
| W2 learning, growth and linkage | must recover | **8/8** | 8 | +34.7 | 1.01 |
| W3 static | must not | **1/8** | 1 | −0.07 | −0.02 |
| W4 static, growth and linkage | must not | **0/8** | 1 | −0.01 | −0.01 |
| W5 static, independent drift | must not | **0/8** | 1 | +0.07 | 0.02 |
| W6 anticipation | must not | **0/8** | 0 | −1.21 | 0.32 |
| W1w weak learning | reported | 8/8 | 8 | +8.8 | 0.50 |
| W7 learning with anticipation | reported | 8/8 | 8 | +45.4 | 1.26 |

## Reading

- **v1's limit is resolved.** In the anticipation world, the NABM model still
  beats B2 in every replicate. It loses to B7 in all of them, so no win is
  recorded. Learning and anticipation are separated by evidence timing.
- **Learning beats anticipation** in every learning replicate. That holds for
  weak learning and for learning combined with anticipation.
- **The three-way rule earned its place.** In W4, the NABM model beat the
  chosen, overfit B2t in 5 of 8 replicates. Requiring wins over B2 and B7 as
  well prevented every false win.
- **W3** has one marginal win, as in v1, within the registered limit.

## What a win in data would mean

Entry selectivity followed the success relation revealed by visible
evidence, at the time it became visible. That goes beyond what the
following can produce:

- static selection;
- outcome-independent drift;
- topic growth;
- improving linkage;
- anticipation of the eventual relation.

It would still not identify which evidence channel was used, or establish a
causal effect.

## Next

**N11b.** Build risk sets and pre-entry features from the S1 extract. This
needs authorization and reads no outcomes.
