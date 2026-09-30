# Pivot observable discriminant v1 (N05)

Date: 2026-09-30

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_observable_discriminant_v1.yaml`](../../experiments/configs/pivot_observable_discriminant_v1.yaml)

## Why this study exists

The empirical target T01 asks whether learned policy can change who enters a
real pivot field. The relative-delay study
[N01](../../experiments/evidence/results/pivot_relative_delay_cost_v1_20260929.summary.md)
found a post-hoc discriminant for this: the **post-failure entry share**, the
fraction of entry that happens after the first failure becomes public. It
separated all twelve N01 cells, but data cannot see when the first failure
became public.

Before this freeze, four candidates that data *can* compute were screened on
the recorded N01 rows. None separated the cells:

- entry after a cohort-persistence decline becomes observable;
- entry after the first cohort's persistence becomes observable;
- the drop from first-cohort persistence to later cohorts; and
- entry after the entry peak.

This study checks whether that conclusion holds when the entry peak sits
inside the window rather than at the first open step. It also adds two
candidates based on who enters. The result decides what the empirical stage
should measure.

## Attention-surge clock

A ready agent enters only if a uniform draw falls below the current admission
probability. That probability is a Gaussian surge in the open step above a floor
of 0.05. The design crosses three peaks (open steps 3, 6, and 9) with two widths
(1.5 and 3), giving six clocks. Each is crossed with the four N01 failure delays
(0, 1, 2, and 4 steps), for 24 cells. Each draw is keyed to
(seed, replicate, step, agent). A flat clock reproduces the N01 `q = 1` arms
exactly, and those nest the persistence predecessor and the v2 MLP arm. Both are
registered nesting checks.

Everything else is the N01 design: the fixed rule and the cross-fitted field
linear, tree, and MLP learners, three fit worlds, and 16 replicates of 256
agents over 24 steps, with fresh seeds. That is 4,608 runs.

## Registered tests

The entry-channel status of each cell uses N01's rule exactly. A cell is active
when the tree or MLP raises entrant precision over the linear learner by 0.02
more in an interaction world than in the linear world, in 12 of 16 replicates.
Otherwise it is saturated or absent.

- **Post-failure entry share, out of sample.** A cell is predicted active when
  its share is at least 0.151284, the midpoint of the N01 separation. The test is
  confirmed if every cell is classified correctly. It holds if at least 75% are
  correct, and fails otherwise.
- **Feature-arm relative delay, as a comparator.** A cell is predicted active
  when the delay divided by the learning arms' entry-wave width is at most
  0.731083, the midpoint of the N01 separation.
- **Observable screen.** Six candidates are computed in every run from entry
  timing, cohort persistence, or entrants' attributes and persistence:

  | Candidate | What data it needs |
  |---|---|
  | Entry after the persistence decline is observable | Entry timing, cohort persistence |
  | Entry after the first cohort is observable | Entry timing |
  | First-to-later cohort persistence drop | Cohort persistence |
  | Entry after the entry peak | Entry timing |
  | Correlation of entry time with attribute-based fit | Entry timing, attribute proxies |
  | Correlation of entry time with persistence | Entry timing, individual persistence |

  No threshold is registered. A candidate separates when the active and
  inactive cell ranges do not overlap. Its rank correlation with the
  post-failure entry share is also reported. A separating candidate is only a
  candidate for a third, registered test.
- **Design check.** The surge clock must put the earliest entry peak after the
  first entry step in at least half of the learning-arm runs. The tests are
  informative only if both active and inactive cells occur.

## Boundary

The executable must be committed before the development profile runs, and a
deterministic rerun must reproduce every artifact byte for byte. No post-result
change to clocks, thresholds, or candidates is authorized.

The surge clock is a stylized attention surge, not a claim about real attention.
The study reads no empirical rows, fits nothing to data, selects no model, and
changes no generic NABM API.
