# Pivot empirical ladder risk sets v1 (N11b)

Date: 2026-10-12

Status: `N11B_PASS_RISK_SETS_REPRODUCED_NO_OUTCOME`

Builder `8a375d8`. Authorization `d6b7f98`. Design:
[`pivot-empirical-ladder-risk-sets-v1.md`](../../../docs/research/pivot-empirical-ladder-risk-sets-v1.md)

## What ran

N11b read the private S1 extract and the G4 guard's author-id column twice.
Each run took 2.2 minutes and peaked at 21 GiB of memory. Every identity
artifact was byte identical across the two runs, including the private
panel. Every invariant held with zero violations. No persistence label,
follow-up count, or visible evidence was computed.

## Risk sets

| | Count |
|---|---|
| Eligible author-years (field work in the prior three years) | 111,568,918 |
| Removed for a prior topic work | 672,388 |
| Author-episodes removed by the G4 guard | 11,755,110 |
| Author-episodes removed as discovery entrants | 89,230 |
| Authors at risk, summed over episodes | 42,812,639 |
| Retained S1 entrants, each with exactly one event | 19,132 |

Authors at risk per episode range from 272,830 to 1,444,999, with a median
of 573,693.

## Case-cohort sample

| | Count |
|---|---|
| Subcohort authors | 403,053 (452 of them entrants) |
| Panel rows | 3,340,784 |
| Subcohort rows | 3,225,564 |
| Sampling probability per episode | 0.0021 to 0.074, median 0.0064 |

No episode was small enough to be kept whole.

## Events

- 11,004 entrants have a precise entry quarter. Between 766 and 1,096 of
  them enter in each quarter.
- 8,128 entrants (42.5%) have only an entry year: 2,501, 2,723, and 2,904 in
  response years 0, 1, and 2.

## Pooled subcohort features

These come from the first row of each subcohort author-year.

| Feature | Median | Upper quartile | Maximum |
|---|---:|---:|---:|
| Field works in the window | 1 | 2 | 1,534 |
| Subfield share | 0 | 0.36 | 1 |
| Career age (years) | 3 | 9 | 114 |
| Incumbent coauthors | 0 | 0 | 169 |

7.3% of author-years have an incumbent coauthor. 0.84% of subcohort rows
have a visible coauthor entry.

## Structural diagnostics

These are post hoc and use counts only. They contain no outcome and no
contrast between entrants and other authors.

| | Confirmatory | Discovery |
|---|---|---|
| Episodes (meeting the S1 floors) | 39 (28) | 24 (20) |
| Entrants per episode, median (range) | 126 (10–829) | 242 (11–1,922) |
| Authors at risk per episode, median | 678,736 | 538,722 |
| Subcohort authors per episode, median | 2,518 | 4,818 |

- **Entry hazard.** The precise-entry hazard is flat across the 12 quarters,
  between 2.9 and 3.6 per 100,000 authors at risk.
- **Evidence pool.** Evidence about an entrant can become visible five
  quarters after entry. Counting precise entrants old enough for that, the
  median confirmatory episode has:

  | Quarter | Entrants |
  |---|---:|
  | 5 | 3 |
  | 8 | 16 |
  | 11 | 32 |

N11a's synthetic worlds differ from this shape in several ways:

| | N11a worlds | N11b |
|---|---|---|
| Authors at risk per episode | 600 | about 680,000 |
| Entry probability | about a third per episode | about 3 per 100,000 per quarter |
| Entry timing | a surge peaking in quarter 3 | flat |
| Entry dates | all precise | 42.5% known only to the year |
| Features | continuous Gaussians | sparse, heavy-tailed counts |
| Evidence pool | — | small, and only late in the episode |

Before N11c, the separate decision registered in the design should use
this shape to settle whether the recovery test is repeated.

Outcomes, visible evidence, model fitting, N11c, and N11d remain
unauthorized.
