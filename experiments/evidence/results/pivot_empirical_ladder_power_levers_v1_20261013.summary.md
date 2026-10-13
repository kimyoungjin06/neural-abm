# Pivot empirical ladder power levers v1 (exploratory)

Date: 2026-10-13

Status: `EXPLORATORY_NOT_REGISTERED_NO_LEVER_ADOPTABLE`

Script: `scripts/explore_pivot_empirical_ladder_power_levers_v1.py` at
`d29752c`. It imports the frozen v3 script and contract by hash. Nothing here
is registered, and no threshold was fixed before the run.

## What ran

N11a v3 was underpowered at the N11b shape: W1 won 1 of 8 against a required
7 of 8, losing to the anticipatory rival B7. This screen asked whether any of
three levers could restore power without false recovery.

- **Worlds.** W1 (learning), W3 (static), and W6 (anticipation), 8
  replicates each, at v3's N11b shape.
- **Seed.** 20261105, unused by any registered run.
- **Scale.** 72 tasks on 48 workers, 3 h 35 min.
- **Data.** No empirical row was read. A single run, not reproduced.

| Lever | What it changes |
|---|---|
| A0 | v3's three-way held-out rule, with B4 linear as the only learner |
| A1 | One joint model with attention, x, B7 anticipatory, and B4 learned. One-sided Wald test on the learned coefficient with an episode-clustered sandwich |
| A2 | A1 plus an anticipatory × quarter/11 ramp |
| B | Agents and the analyst also learn from earlier episodes of the same field |
| C | The G4 guard is lifted: entrants ×1.707 and at-risk ×1.275 |

A1 and A2 were fitted on confirmatory episodes, and for reference on all
episodes. Each test was run under every configuration (base, B, and C).

## Result

Each cell gives wins out of 8. "z" is the median learned coefficient over its
clustered standard error, from the confirmatory A2 fit.

| Config | World | A0 | A1 | A2 | A1 all | A2 all | A2 z | A0 margin vs B7 (5% bound) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | W1 learning | 3 | 8 | 8 | 8 | 8 | 5.2 | +7.9 (−3.3) |
| base | W3 static | 0 | 0 | 0 | **2** | **2** | 0.6 | +0.8 (−1.6) |
| base | W6 anticipation | 0 | **8** | **7** | **8** | **6** | 2.7 | −2.5 (−12.9) |
| B | W1 learning | 3 | 5 | 5 | 7 | 7 | 3.1 | +4.4 (+0.3) |
| B | W3 static | 0 | **2** | **2** | **2** | **2** | −0.4 | +2.1 (−0.3) |
| B | W6 anticipation | 0 | **1** | **1** | **2** | **2** | −0.8 | +4.0 (−13.0) |
| C | W1 learning | 2 | 8 | 8 | 8 | 8 | 5.1 | +11.8 (−1.0) |
| C | W3 static | 0 | **1** | **1** | **1** | **1** | 1.1 | +0.1 (−2.1) |
| C | W6 anticipation | 0 | **8** | **5** | **8** | **6** | 2.4 | −9.4 (−28.1) |

Bold marks a false win in a world without learning.

## Reading

- **The joint tests are not specific.** A1 and A2 detect W1 in 8 of 8, but
  they also flag anticipation.
  - In W6, A2 wins 7 of 8 under the base configuration and 5 of 8 under C.
  - The ramp lowers W6's median z from 4.9 to 2.7, but not to zero.
  - The learned term picks up time-varying, success-aligned selection that
    the B7 terms do not fully absorb. A positive learned coefficient
    therefore does not distinguish learning from anticipation.
- **Field history adds false wins.**
  - Under B, the joint tests win 2 of 8 in the static world W3, and A0
    stays at 3 of 8 in W1.
  - The synthetic static world uses one β for every field. A field-specific
    static rule, which B would also confuse with learning, is still untested.
- **Scale does not lift A0.**
  - C raises confirmatory events about 1.7 times (4,895 to 8,370 in W1),
    but A0 wins 2 of 8.
  - Its median margin over B7 rises from +7.9 to +11.8, yet the 5% bound
    stays negative.
  - The constraint is identification of learning against anticipation,
    not sample size.
- **A0 stays specific.** It wins 0 of 8 in W3 and W6 under every
  configuration, as v3 did.

## What this does and does not say

No lever screened here yields an analysis that recovers learning in at least
7 of 8 replicates while staying silent on static and anticipatory worlds at
the N11b shape. The screen does not test other identification strategies. It
says nothing about the data, since no outcome row was read.

## Next

v4 is not frozen from these levers. N11c and N11d remain not proposed. The
next step needs the user's decision.
