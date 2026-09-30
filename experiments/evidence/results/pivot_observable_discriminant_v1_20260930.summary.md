# Pivot observable discriminant v1 result (N05)

Date: 2026-09-30

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED`

## Result

**Two data-observable candidates separated every cell.** Both ask whether
people who enter later persist better than people who enter early:

- the **cohort persistence drop**: first-cohort persistence minus that of later
  cohorts; and
- the **outcome selection trend**: the rank correlation between an entrant's
  entry time and whether they persist.

In every cell where learned policy changed who entered, later cohorts persisted
at least as well as the first, and later entrants persisted more. In every cell
where it did not, later cohorts persisted less. Both need only entry timing and
persistence. Neither needs the time of the first public failure, which data
cannot see.

**The model-internal post-failure entry share only mostly holds.** At the
threshold fixed from N01, it classified 18 of 24 cells, exactly the registered
75% line for "holds in most cells". All six errors called a cell active where
learning did not change entry. Every active cell had a share of at least 0.231,
but inactive cells reached 0.740. The share is close to necessary, not
sufficient. The feature-arm relative delay did slightly better, at 19 of 24
with five false actives.

**The remaining four candidates did not separate the cells.** These are entry
after the persistence decline is observable, entry after the first cohort is
observable, entry after the peak, and the attribute-based trend. The first
three had failed on the N01 rows too.

The run used `5b73c19`: 4,608 runs over 6 surge clocks, 4 failure delays,
3 fit worlds, and 4 arms, with 16 replicates of 256 agents over 24 steps. A
flat clock reproduced the N01 `q = 1` arms exactly, and a second execution
reproduced all four artifacts byte for byte. The surge clock put the entry peak
after the first entry step in 79% of learning-arm runs. The cells split 10
active, 5 absent, and 9 saturated, so the tests were informative.

## Observable screen

A candidate separates when the active and inactive cell ranges do not overlap.
No threshold was registered.

| Candidate | Active cells | Inactive cells | Separates | Rank correlation with PFES |
|---|---|---|---|---:|
| First-to-later cohort persistence drop | −0.352 to 0.000 | +0.046 to +0.585 | **yes** | −0.58 |
| Entry time and persistence | +0.148 to +0.335 | −0.649 to +0.049 | **yes** | +0.69 |
| Entry after the persistence decline is observable | +0.001 to +0.480 | 0.000 to +0.686 | no | +0.80 |
| Entry after the first cohort is observable | +0.009 to +0.740 | 0.000 to +0.813 | no | +0.80 |
| Entry after the entry peak | +0.181 to +0.587 | +0.142 to +0.379 | no | +0.04 |
| Entry time and attribute-based fit | +0.036 to +0.138 | −0.096 to +0.076 | no | +0.79 |

The margins are 0.046 for the persistence drop and 0.099 for the outcome
selection trend.

## Cells

| Surge peak, width | Delay | Status | Post-failure entry share | Cohort persistence drop | Outcome selection trend | Attribute selection trend |
|---|---:|---|---:|---:|---:|---:|
| 3, 1.5 | 0 | **active** | 0.487 | −0.346 | +0.335 | +0.109 |
| 3, 1.5 | 1 | saturated | 0.142 | +0.071 | −0.070 | −0.045 |
| 3, 1.5 | 2 | saturated | 0.000 | +0.089 | −0.087 | −0.048 |
| 3, 1.5 | 4 | saturated | 0.000 | +0.510 | −0.502 | −0.048 |
| 3, 3.0 | 0 | **active** | 0.231 | −0.352 | +0.290 | +0.074 |
| 3, 3.0 | 1 | saturated | 0.026 | +0.102 | −0.107 | −0.094 |
| 3, 3.0 | 2 | saturated | 0.000 | +0.108 | −0.113 | −0.096 |
| 3, 3.0 | 4 | saturated | 0.000 | +0.585 | −0.649 | −0.096 |
| 6, 1.5 | 0 | **active** | 0.898 | −0.294 | +0.207 | +0.036 |
| 6, 1.5 | 1 | **active** | 0.869 | −0.021 | +0.163 | +0.105 |
| 6, 1.5 | 2 | absent | 0.720 | +0.248 | −0.036 | +0.017 |
| 6, 1.5 | 4 | saturated | 0.081 | +0.494 | −0.333 | −0.027 |
| 6, 3.0 | 0 | **active** | 0.639 | −0.330 | +0.301 | +0.080 |
| 6, 3.0 | 1 | absent | 0.430 | +0.046 | +0.013 | +0.012 |
| 6, 3.0 | 2 | saturated | 0.154 | +0.110 | −0.072 | −0.031 |
| 6, 3.0 | 4 | saturated | 0.002 | +0.526 | −0.455 | −0.036 |
| 9, 1.5 | 0 | **active** | 0.843 | −0.283 | +0.187 | +0.044 |
| 9, 1.5 | 1 | **active** | 0.871 | −0.176 | +0.201 | +0.108 |
| 9, 1.5 | 2 | **active** | 0.815 | 0.000 | +0.148 | +0.138 |
| 9, 1.5 | 4 | absent | 0.740 | +0.504 | −0.162 | +0.076 |
| 9, 3.0 | 0 | **active** | 0.872 | −0.335 | +0.206 | +0.044 |
| 9, 3.0 | 1 | **active** | 0.790 | −0.085 | +0.232 | +0.127 |
| 9, 3.0 | 2 | absent | 0.646 | +0.112 | +0.049 | +0.066 |
| 9, 3.0 | 4 | absent | 0.271 | +0.524 | −0.315 | −0.016 |

## Post-hoc checks

These checks were not registered, and nothing new was run.

**The cohort version is regime-dependent.** On the recorded N01 rows, the
persistence drop did not separate the cells. The hazard cell with `q = 0.25`
and a four-step delay was active with a drop of 0.547. The outcome selection
trend could not be computed there, because N01 kept no per-entrant records.

**The H3 gap does not discriminate.** The model H3 gap, from the entry peak to
the persistence-decline onset, overlapped across active, absent, and saturated
cells.

## What this means for the empirical stage

- **The observable should be the persistence trend across entry time, not a
  delay ratio.** In these worlds, a decline in later-cohort persistence marks
  an entry channel that is *not* active. Rising or flat persistence marks one
  that is.
- **The data need is concrete.** It is entry year and persistence, per
  researcher or per cohort, within a pivot episode. The lean H3 core already
  builds cohort persistence.
- **Neither candidate is confirmed yet.** One separated these cells and failed
  on N01's. A third registered test should cover both hazard-style and
  surge-style entry on fresh seeds before either is used on data.

## Boundary

The surge clock and the worlds are synthetic and stylized. The study read no
empirical rows, fitted nothing to data, selected no model, and changed no
generic NABM API.
