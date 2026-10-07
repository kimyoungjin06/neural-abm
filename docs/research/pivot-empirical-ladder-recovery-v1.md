# Pivot empirical ladder recovery v1 (N11a)

Date: 2026-10-08

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_empirical_ladder_recovery_v1.yaml`](../../experiments/configs/pivot_empirical_ladder_recovery_v1.yaml)

## Question

The [N11 design brief](pivot-empirical-model-ladder-design-v1.md) compares
a ladder of entry models on held-out empirical episodes. Before any outcome
row is read, this study asks whether the comparison can tell learning from
the confounds that defeat a sign test.

## Synthetic worlds

The worlds mirror the empirical structure that S1 measured:

- 63 episodes per replicate, one third used for discovery;
- twelve response quarters, with an attention surge peaking in quarter 3;
- an at-risk population with four pre-entry features, sized to give about
  200 entrants per episode.

Each entrant succeeds with an episode-specific relation to the features.
Successful entrants publish follow-ups. The **visible evidence** is whether
an entrant published a follow-up within four quarters; it becomes visible in
the fifth quarter after entry.

| World | Learning | Added mechanism | Role |
|---|---|---|---|
| W1 | κ = 1 | none | must recover |
| W2 | κ = 1 | growth and linkage | must recover |
| W3 | none | none | must not falsely recover |
| W4 | none | growth and linkage | must not falsely recover |
| W5 | none | outcome-independent selectivity drift | must not falsely recover |
| W1w | κ = 0.5 | none | reported: power |
| W6 | none | drift toward the episode's success relation from the start | reported: the identification limit |

What each mechanism does:

- **Learning.** Entry selectivity follows a success predictor learned from
  visible evidence. This changes who enters, not how many.
- **Growth.** Later entrants are more likely to succeed.
- **Linkage.** Follow-ups are observed more completely over time.

## Ladder and decision

The ladder is the brief's: B1, B2, B2t, B3, and the NABM models B4–B6.

- **Fitting.** Every model is fitted on discovery episodes only.
- **Selection.** The best NABM model and the best non-learning model are
  chosen by discovery log-likelihood.
- **A replicate counts as an NABM win** only when the chosen NABM model
  beats both the chosen non-learning model and B2 on confirmatory episodes.
  Each difference needs a positive mean and a positive 5% episode-bootstrap
  bound.
- **Why B2 as well.** The NABM models nest B2, so beating it means the
  learned term adds out-of-sample value. An overfit flexible competitor
  cannot then hand NABM a win.

| Disposition | Condition |
|---|---|
| `IDENTIFICATION_PASSES` | W1 and W2 at least 7 of 8 wins; W3, W4, and W5 at most 1 of 8 |
| `IDENTIFICATION_FAILS` | Otherwise. The empirical ladder stops before N11b |

The N10 v2 sign test is computed in every world and reported. It shows where
a sign test is fooled.

## Boundary

The executable must be committed before the development profile runs, and a
deterministic rerun must reproduce every artifact byte for byte. The study is
synthetic, reads no empirical row, selects no model for data, and changes no
generic NABM API.
