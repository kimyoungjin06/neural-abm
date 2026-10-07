# Pivot empirical stage S1: structural build

Date: 2026-10-07

Status: `S1_PASS_STRUCTURAL_BUILD_REPRODUCED_NO_PERSISTENCE_LABEL`

Builder `14c93eb`, revised `52dcf76`. Authorization `13fcf87`. Design:
[`pivot-empirical-outcome-selection-trend-s1-build-v1.md`](../../../docs/research/pivot-empirical-outcome-selection-trend-s1-build-v1.md)

## What ran

S1 read column-limited Works rows of the complete 2026-06-26 snapshot twice:
a development run of 13.5 minutes and a reproduction run of 12.8. Every
identity artifact was byte identical across the two runs, including the
private extract. No persistence label was computed.

## Source accounting

All 510,372,821 rows passed every registered check. There were no
malformed identifiers, no duplicate work ids, and no publication year–date
disagreements. The row count read equals the footer total.

| | Count |
|---|---|
| Works in fields 16, 32, 11, eligible, through 2017 | 20,063,050 |
| Distinct work–author pairs | 40,527,738 |
| Authorships without an author id | 10,464,779 |
| `is_xpac` works in the extract | 3,462,282 (17%) |

## Episodes and entrants

The surge rule found 63 episodes. The indexing screen excluded none.

| Field | Discovery episodes (meeting floors) | Confirmatory episodes (meeting floors) |
|---|---|---|
| 11 Agricultural and Biological Sciences | 9 (8) | 19 (12) |
| 16 Chemistry | 10 (8) | 7 (7) |
| 32 Psychology | 5 (4) | 13 (9) |

Every field has at least four confirmatory episodes meeting the floors, so
the registered disposition is **`S1_PASS`**.

The entrant counts narrow at each step:

| Step | Entrants |
|---|---|
| First-time topic authors in the response years | 118,119 |
| With a field work in the prior three years | 32,970 |
| Removed by the G4 author guard | 13,529 |
| Removed as discovery authors | 309 |
| **Retained** | **19,132** |

Across retained entrants, 57.5% have a precise (non–January 1) entry date.

## Structural diagnostics

These are post hoc and use no persistence.

| Response year | Entrants | G4-guarded | Precise date |
|---|---|---|---|
| 0 | 9,962 | 40.8% | 58.9% |
| 1 | 11,076 | 41.3% | 60.3% |
| 2 | 11,932 | 41.0% | 60.4% |

- **G4 guard.** It removes about 41% of entrants, evenly across response
  years. It therefore barely changes the within-episode clock. It does
  narrow the population to authors who were not in the 2017–2021 Medicine
  risk set, a condition on later careers.
- **Date precision.** It rises across calendar years, from about 50% before
  2011 to 78% in 2014, but is nearly flat within episodes.
- **`is_xpac` works.** None is the entry work of an eligible entrant.

Persistence labels, S2, S3, and model fitting remain unauthorized.
