# Pivot empirical ladder recovery v3 (N11a v3)

Date: 2026-10-12

Status: `UNDERPOWERED_AT_THE_EMPIRICAL_SHAPE_NO_FALSE_RECOVERY`

Precommit: `288062a`, amending v2 (`0cbe5f2`). Design:
[`pivot-empirical-ladder-recovery-v3.md`](../../../docs/research/pivot-empirical-ladder-recovery-v3.md)

## What ran

The study ran 10 synthetic worlds × 8 replicates at the N11b shape:

- 63 episodes, each with its measured risk set and entrant count;
- sparse count features;
- 42.5% of dates known only to the year;
- an analyst who sees only dated, linked records.

Development and reproduction runs were byte identical, taking 160 and 178
minutes on 48 workers. No empirical row was read. The only empirical inputs
were N11b's per-episode counts.

## Registered result: underpowered, with no false recovery

An NABM win requires beating three rivals on confirmatory episodes:

- the chosen non-learning model;
- B2;
- the anticipatory competitor B7.

Each comparison needs a positive mean and a positive 5% bootstrap bound.
Margins are log-likelihood per confirmatory episode.

| World | Role | NABM wins | Beats B2 | Beats B7 | Median margin vs B7 | Its median 5% bound | Learned coef. | Anticipatory coef. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W1 learning | must recover | **1/8** | 5 | 1 | +1.9 | −5.5 | 0.40 | 0.38 |
| W2 learning, confounded | must recover | **3/8** | 5 | 4 | +8.2 | −0.2 | 0.40 | 0.39 |
| W3 static | must not | **0/8** | 1 | 3 | +1.2 | −0.6 | 0.03 | 0.07 |
| W4 static, confounded | must not | **0/8** | 0 | 1 | +0.3 | −0.5 | −0.00 | 0.01 |
| W5 independent drift | must not | **0/8** | 0 | 3 | +0.1 | −4.4 | 0.12 | 0.13 |
| W6 anticipation | must not | **0/8** | 4 | 0 | +0.7 | −13.0 | 0.33 | 0.43 |
| W8 recruitment | must not | **0/8** | 1 | 2 | +0.2 | −0.6 | −0.00 | 0.01 |
| W1w weak learning | reported | 0/8 | 4 | 3 | +2.8 | −2.3 | 0.24 | 0.19 |
| W1s strong learning | reported | 1/8 | 2 | 1 | −1.5 | −36.9 | 0.51 | 0.76 |
| W7 learning with anticipation | reported | 0/8 | 5 | 0 | +9.8 | −14.4 | 0.45 | 0.73 |

The registered action applies: N11c and N11d are not proposed in this form.
The analysis is redesigned before any outcome row is read.

## Reading

- **No false recovery.** Every no-learning world won 0 of 8. That includes
  anticipation (W6) and recruitment (W8). W8 also won 0 of 8 when scored
  without the competing-risk censoring. At ρ = 0.3, with about a quarter of
  entrants recruited, the censoring was not needed to prevent a false win.
- **Power fails against B7.** W1 beat B2 in 5 of 8 replicates but B7 in only
  1. Its median margin over B7 fell from +38 per episode in v2 to +1.9.
- **Learning and anticipation are nearly collinear at this shape.** In W1
  the fitted learned and anticipatory coefficients are 0.40 and 0.38. The
  analyst's evidence pool in a median confirmatory episode holds 6, 40, and
  58 precise entrants at quarters 5, 8, and 11. Evidence that small arrives
  late and differs little from the eventual success relation, which B7
  already encodes.
- **The bounds are wide.** Confirmatory episodes range from 10 to 829
  entrants, and per-episode margins vary with them. Even where the mean
  margin is positive (W2 +8.2, W7 +9.8), the 5% bound is usually negative.
- **Stronger learning does not help.** W1s won 1 of 8. Its anticipatory
  coefficient (0.76) exceeds its learned one (0.51), and its bounds are the
  widest of any world.

## What this does and does not say

The synthetic test says the registered analysis would rarely detect
learning in data of the N11b shape, even when learning is present. It does
not say learning is absent from the data. No data outcome was read, and no
model was selected for data.

## Next

A redesign of the analysis, tested again on synthetic worlds at the N11b
shape, is required before N11c. Its form needs the user's decision.
