# Pivot observable confirmation v1 result (N08)

Date: 2026-10-01

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED`

## Result

**Both observables mostly held, but neither was confirmed in every cell.** The
thresholds were fixed from N05, and the entry schedules and seeds were new:

- The **cohort persistence drop** classified 23 of 24 cells.
- The **outcome selection trend** classified 22 of 24.
- The joint rule, active only when both say so, also got 22.

The registered rule carries an observable to the empirical stage only if it is
confirmed in every cell. So neither goes forward as a two-sided discriminant.

**Every error missed an active channel.** Neither observable called an inactive
cell active. Both misses came from cells where learning changed entry only
weakly: one of the four family-geometry pairs passed.

**The no-learning control held.** The fixed rule does not learn, and its
observables read inactive in all 24 cells. Later cohorts persisted less, and
later entrants persisted less. The observables track learning, not entry timing
alone.

The post-failure entry share, reported as a comparator, classified 20 of 24
cells, with four false actives.

The run used `b270e51`: 4,608 runs over 6 new entry schedules, 4 failure delays,
3 fit worlds, and 4 arms, with 16 replicates of 256 agents over 24 steps. A flat
schedule reproduced the N01 arms exactly, and a second execution reproduced all
four artifacts byte for byte. The cells split 10 active, 6 absent, and 8
saturated.

## Cells

✗ marks a missed active cell. Thresholds: the drop predicts active below
0.023175, the trend above 0.098417. "Passing pairs" counts the family-geometry
pairs that passed N01's entry-channel rule.

| Schedule | Delay | Status | Passing pairs | Persistence drop | Outcome trend | Fixed-rule drop | Fixed-rule trend |
|---|---:|---|---:|---:|---:|---:|---:|
| hazard 0.35 | 0 | **active** | 2 | −0.372 | +0.361 | +0.039 | −0.072 |
| hazard 0.35 | 1 | **active** | 3 | −0.017 | +0.137 | +0.039 | −0.072 |
| hazard 0.35 | 2 | **active** | 1 | +0.052 ✗ | +0.037 ✗ | +0.039 | −0.072 |
| hazard 0.35 | 4 | absent | 0 | +0.557 | −0.426 | +0.039 | −0.072 |
| hazard 0.7 | 0 | **active** | 2 | −0.355 | +0.313 | +0.107 | −0.130 |
| hazard 0.7 | 1 | absent | 0 | +0.083 | −0.060 | +0.107 | −0.130 |
| hazard 0.7 | 2 | saturated | 0 | +0.103 | −0.090 | +0.107 | −0.130 |
| hazard 0.7 | 4 | saturated | 0 | +0.592 | −0.619 | +0.107 | −0.130 |
| surge 4, 2 | 0 | **active** | 3 | −0.351 | +0.343 | +0.035 | −0.073 |
| surge 4, 2 | 1 | absent | 0 | +0.054 | −0.009 | +0.035 | −0.073 |
| surge 4, 2 | 2 | saturated | 0 | +0.084 | −0.046 | +0.035 | −0.073 |
| surge 4, 2 | 4 | saturated | 0 | +0.511 | −0.456 | +0.035 | −0.073 |
| surge 4, 4 | 0 | **active** | 2 | −0.356 | +0.302 | +0.118 | −0.136 |
| surge 4, 4 | 1 | saturated | 0 | +0.099 | −0.095 | +0.118 | −0.136 |
| surge 4, 4 | 2 | saturated | 0 | +0.108 | −0.106 | +0.118 | −0.136 |
| surge 4, 4 | 4 | saturated | 0 | +0.584 | −0.631 | +0.118 | −0.136 |
| surge 8, 2 | 0 | **active** | 2 | −0.289 | +0.222 | +0.038 | −0.063 |
| surge 8, 2 | 1 | **active** | 2 | −0.102 | +0.258 | +0.038 | −0.063 |
| surge 8, 2 | 2 | absent | 0 | +0.104 | +0.097 | +0.038 | −0.063 |
| surge 8, 2 | 4 | absent | 0 | +0.517 | −0.260 | +0.038 | −0.063 |
| surge 8, 4 | 0 | **active** | 2 | −0.337 | +0.312 | +0.056 | −0.098 |
| surge 8, 4 | 1 | **active** | 1 | +0.022 | +0.088 ✗ | +0.056 | −0.098 |
| surge 8, 4 | 2 | absent | 0 | +0.102 | −0.034 | +0.056 | −0.098 |
| surge 8, 4 | 4 | saturated | 0 | +0.535 | −0.421 | +0.056 | −0.098 |

## Post-hoc checks

These checks were not registered, and nothing new was run.

**The observables miss only weak channels.** Across N05 and N08 there are 20
active cells. The 17 with two or more passing pairs were all classified
correctly by both observables. Three cells had one passing pair: one in N05,
classified correctly, and the two misses here. Across the 96 predictions in the
two studies, neither observable called an inactive cell active.

**The persistence drop still separated this study, but only just.** Its largest
active value was 0.0524 and its smallest inactive value 0.0536. The N05 midpoint
threshold fell below both.

## What this means for the empirical stage

- **A rising persistence trend across entry time is a candidate sign that
  learning acts on entry.** In these worlds, whenever the observables said
  active, the channel was active.
- **Its absence does not rule out a weak channel.** A flat or falling trend is
  also what a weak channel can look like.
- **The later-cohort persistence decline is the no-learning default.** The lean
  H3 pattern, if found in data, is what these worlds produce when learning does
  not change who enters.
- **The one-sided reading is post hoc.** A registered one-sided test, that a
  trend above the threshold implies an active channel, is the natural successor
  before relying on it.

## Boundary

The schedules and worlds are synthetic and stylized. The study read no empirical
rows, fitted nothing to data, selected no model, and changed no generic NABM API.
