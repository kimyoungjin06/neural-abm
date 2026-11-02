# Pivot crossover prediction v1 (experiment 3, stage 2): result

Date: 2026-11-02

Status: `E3S2_THE_LEARNER_WINS_WHERE_DETERMINED_AND_THE_RULE_AGREES`

Design: [pivot-crossover-prediction-v1.md](../../../docs/research/pivot-crossover-prediction-v1.md),
frozen at `015eea5fa5720406491b1504433dbd6dea4d257f`.
Machine-readable record:
[pivot_crossover_prediction_v1_20261102.summary.json](pivot_crossover_prediction_v1_20261102.summary.json).

## Execution

- **Host:** the workstation, one process, with 8 DuckDB threads.
- **Wall time:** 42 s for the development run and 44 s for the reproduction run,
  below the quoted 2 to 15 minutes. Peak memory was 0.2 GiB.
- **Reproduction:** `summary.json`, `receipt.json`, and
  `private/episode_quarters.parquet` are byte identical.
- **Data read:** the 6,291 precise entrants of the 24 discovery episodes and their
  episode-topic works. No confirmatory, imprecise-entrant, coauthor, or G4
  row was read.

**Checks.**
- **Before any label:** the N11b preflight passed, and the 24 episodes and
  6,291 precise entrants equal experiment 4's record. Every episode has at
  least two precise entrants.
- **After the labels:** 920 follow-ups, equal to experiment 4's record.

## The generator check

Synthetic labels at the empirical shape, 40 replicates per τ:

| τ | Determined groups | Agreement | Label match | Reversals | Replicates with a learner win | Passes |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 348 | 0.96 | 0.91 | 3 | 35 of 40 | Yes |
| 0.25 | 309 | 0.92 | 0.83 | 9 | 40 of 40 | Yes |
| 0.5 | 413 | 0.99 | 0.95 | 0 | 40 of 40 | Yes |
| 1 | 561 | 1.00 | 0.99 | 0 | 40 of 40 | Yes |

**It passed at every τ, so the labels were read.**

## The registered decision

| Expectation | Result |
|---|---|
| X1: the generator check passes at every τ | Held |
| X2: at τ = 0.5, at least half the replicates show a learner win | Held: 40 of 40 |
| X3: every table beats the learner below 15 visible labels | **Failed**: the means are negative (−0.0095 to −0.0041), but every interval covers zero |
| X4: the learner beats the 16-episode table at 150 or more visible labels | Held: +0.0163 [+0.0027, +0.0411] |
| X5: no reversal, and agreement with at least 80% of determined groups | Held: 7 of 7, no reversal |

- **Every determined group is a learner win.** There are 7 such groups,
  and the rule agrees with all of them.
- **No group was determined for the table.** The baseline that the learner
  always wins therefore agrees as well as the rule does. **The data test the
  rule on the learner's side only.**

## The comparison

The gap is the table's Brier score minus the learner's; it is positive when
the learner wins. The label says which side won materially; "—" means
neither did, or the rule predicted neither.

| Visible labels | k | Forecasts (episodes) | Observed gap | 95% interval | Winner | Rule | Rule's winner |
|---|---:|---:|---:|---|---|---:|---|
| 1 to 14 | 1 | 248 (17) | −0.0041 | [−0.0211, +0.0068] | — | −0.0007 | — |
| 1 to 14 | 2 | 248 (17) | −0.0070 | [−0.0240, +0.0039] | — | −0.0035 | table |
| 1 to 14 | 4 | 248 (17) | −0.0084 | [−0.0254, +0.0025] | — | −0.0049 | table |
| 1 to 14 | 8 | 248 (17) | −0.0091 | [−0.0261, +0.0018] | — | −0.0056 | table |
| 1 to 14 | 16 | 248 (17) | −0.0095 | [−0.0265, +0.0014] | — | −0.0060 | table |
| 15 to 49 | 1 | 558 (16) | +0.0051 | [+0.0006, +0.0094] | learner | +0.0079 | learner |
| 15 to 49 | 2 | 558 (16) | +0.0023 | [−0.0022, +0.0065] | — | +0.0050 | learner |
| 15 to 49 | 4 | 558 (16) | +0.0008 | [−0.0037, +0.0051] | — | +0.0036 | learner |
| 15 to 49 | 8 | 558 (16) | +0.0001 | [−0.0044, +0.0044] | — | +0.0029 | learner |
| 15 to 49 | 16 | 558 (16) | −0.0002 | [−0.0048, +0.0041] | — | +0.0025 | learner |
| 50 to 149 | 1 | 1,045 (12) | +0.0102 | [+0.0018, +0.0200] | learner | +0.0089 | learner |
| 50 to 149 | 2 | 1,045 (12) | +0.0075 | [−0.0009, +0.0175] | — | +0.0062 | learner |
| 50 to 149 | 4 | 1,045 (12) | +0.0062 | [−0.0023, +0.0163] | — | +0.0049 | learner |
| 50 to 149 | 8 | 1,045 (12) | +0.0055 | [−0.0030, +0.0157] | — | +0.0042 | learner |
| 50 to 149 | 16 | 1,045 (12) | +0.0052 | [−0.0034, +0.0154] | — | +0.0038 | learner |
| 150 or more | 1 | 1,984 (7) | +0.0211 | [+0.0079, +0.0451] | learner | +0.0090 | learner |
| 150 or more | 2 | 1,984 (7) | +0.0185 | [+0.0051, +0.0430] | learner | +0.0064 | learner |
| 150 or more | 4 | 1,984 (7) | +0.0172 | [+0.0037, +0.0419] | learner | +0.0051 | learner |
| 150 or more | 8 | 1,984 (7) | +0.0166 | [+0.0030, +0.0414] | learner | +0.0045 | learner |
| 150 or more | 16 | 1,984 (7) | +0.0163 | [+0.0027, +0.0411] | learner | +0.0041 | learner |

**From which evidence bin on every group is a learner win:** 15 to 49
visible labels for the one-episode table; 150 or more for every larger
table.

## The rule's quantities

These come from all 24 episodes and decide nothing.

- **Within-episode evidence:** V = 0.0683, the per-label variance of an
  episode's own share.
- **Between-episode heterogeneity:** the standard deviation of true shares is
  0.065, corrected for sampling, around an unweighted mean share of
  0.079. Pooled over entrants, the share is 0.146.

| k | H_k | Crossing V/H_k (visible labels) | Forecasts with at least that many |
|---:|---:|---:|---:|
| 1 | 0.01008 | 6.8 | 97.9% |
| 2 | 0.00729 | 9.4 | 96.6% |
| 4 | 0.00590 | 11.6 | 95.6% |
| 8 | 0.00520 | 13.1 | 94.6% |
| 16 | 0.00485 | 14.1 | 93.5% |

**The rule places the crossing at 6.8 to 14.1 visible labels.** That many
are visible for 93.5% to 97.9% of the forecasts.

## Post hoc, descriptive, not registered

- **The observed gap rises more steeply with evidence than the rule says.**
  - Below 50 visible labels, the observed gap lies below the rule's.
  - From 50 on it lies above. At 150 or more, it is +0.0163 to +0.0211
    against the rule's +0.0041 to +0.0090.
  - Every predicted mean still lies inside its observed interval. The
    intervals are wide: the top bin rests on 7 episodes.
- **Larger episodes follow up more often.** The unweighted mean share is
  0.079, and the share pooled over entrants is 0.146.
  - A table weights episodes equally, so it sits near the unweighted mean
    and misses a large episode by more than the average heterogeneity the
    rule uses.
  - That is consistent with the steeper gap; it was not tested.

## What this means

- **For this policy, the closed loop wins in the discovery episodes.**
  - An episode's own visible follow-up share beats a table of other
    episodes' shares.
  - By the rule, it does so once about 7 to 14 outcomes are visible, which
    most forecasts have.
  - The data determine the win at 150 or more visible labels for every table
    size, and from 15 for the one-episode table.
- **The rule chose the side correctly wherever the data decided, with no
  reversal.** But the data decided only the learner's side, so the rule was
  not tested against a table win.
- **Topics differ a lot in how often entrants follow up.** That puts the
  episodes in the high-heterogeneity regime, where the rule says the closed
  loop wins early.
  - In the information-spread ladder, by contrast, a table from other worlds
    sufficed (O23). Those worlds share one veracity relation.
- **The crossing is a textbook quantity.** For a large table, V/H_k
  approaches V over the between-episode variance of shares, 16.0 here.
  - That ratio is Bühlmann's credibility constant: the number of
    observations at which a group's own mean earns half the weight.
  - This stage locates it in these data; it is not a new rule.

## Boundary

- **Partition.** Discovery only. The confirmatory outcomes stay sealed.
- **Policy.**
  - The policy is the topic's follow-up share, with no entrant features.
  - The tables come from other episodes regardless of calendar time.
  - A partially pooled forecast, which could beat both, is not tested.
- **No learning identified.** The learner is a forecaster. Nothing here says
  how researchers decide.
- **Scope.** No LLM call and no change to a generic NABM API.
