# Pivot empirical ladder recovery v3 (N11a v3)

Date: 2026-10-12

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_empirical_ladder_recovery_v3.yaml`](../../experiments/configs/pivot_empirical_ladder_recovery_v3.yaml)

Amends: [v2](pivot-empirical-ladder-recovery-v2.md) (`0cbe5f2`, recorded
`886be72`). Its contract, script, and result are bound by hash.

## Why v3

v1 and v2 passed identification in worlds of 600 authors at risk per
episode, a third of whom entered. The
[N11b result](../../experiments/evidence/results/pivot_empirical_ladder_risk_sets_v1_20261012.summary.md)
measured a very different empirical shape. Confirmatory outcomes can be read
only once. v3 therefore asks, before any outcome row is read, whether the
ladder still separates learning from its confounds at that shape.

| | v1 and v2 worlds | v3 worlds, from N11b |
|---|---|---|
| Episodes | 21 discovery, 42 confirmatory | 24 and 39, each with its measured risk set and entrant count |
| Authors at risk per episode | 600 | 272,830 to 1,444,999 |
| Entry timing | A surge peaking in quarter 3 | Flat, with a slight rise |
| Features | Continuous Gaussians | Sparse counts matched to the N11b pooled quantiles |
| Entry and follow-up dates | All exact to the quarter | 42.5% known only to the year |
| Who sees outcomes | Agents and analyst see the same records | Agents see true outcomes; the analyst sees only dated, linked records |

## The analysis under test

v3 registers the analysis that N11c and N11d would apply to data, so that
what passes here is what runs there.

1. **Case-cohort sample.** Every entrant is kept, plus a subcohort drawn
   with N11b's probability, min(1, max(2000, 20 × entrants) / at risk).
   - Rows of entrants have weight 1.
   - Rows of other subcohort authors have weight 1 / probability.
   - The weighted likelihood then estimates the full-cohort likelihood.
2. **Events and censoring.**
   - An entry is an event only if its quarter is known and it was not
     recruited.
   - An imprecise entrant leaves the panel at the start of its entry year.
   - A recruited entry is censored at its quarter, as a competing risk.
     An entry counts as recruited when its entry work has an earlier
     entrant as a coauthor.
3. **Evidence the analyst can build.**

   | Entrant | Label | Visible from |
   |---|---|---|
   | Precise | A seen follow-up in the four quarters after entry, dated to the quarter or to a year lying wholly in that window | Five quarters after entry |
   | Imprecise | A seen follow-up dated in the year after the entry year | The first quarter two years after the entry year |

4. **Centring.** Learned and anticipatory terms are centred in each
   episode-quarter by their weighted mean.
5. **Fit, selection, and the win rule.** These are v2's:
   - weighted unpenalized maximum likelihood on discovery episodes, solved
     by Newton steps;
   - the best NABM and the best non-learning model are chosen by discovery
     likelihood;
   - a win requires beating the chosen non-learning model, B2, and B7 on
     confirmatory episodes, each with a positive mean and a positive 5%
     bootstrap bound.

## The worlds

Learning and drift change who enters, not how many. Each quarter their
terms are shifted so that the expected number of entries matches the static
world, and every world keeps the measured shape.

| World | κ | Confounds | Role |
|---|---:|---|---|
| W1 learning | 1 | — | Must recover |
| W2 learning, confounded | 1 | Growth, linkage, recruitment | Must recover |
| W3 static | 0 | — | Must not |
| W4 static, confounded | 0 | Growth, linkage | Must not |
| W5 independent drift | 0 | Drift unrelated to success | Must not |
| W6 anticipation | 0 | Drift toward the eventual success relation | Must not |
| W8 recruitment | 0 | Recruitment | Must not |
| W1w weak learning | 0.5 | — | Reported power |
| W1s strong learning | 2 | — | Reported power |
| W7 learning with anticipation | 1 | Aligned drift | Reported |

**Recruitment is new.** A successful entrant publishes again, often with a
new coauthor who resembles them. That coauthor then enters the topic.
Without any learning, later entrants come to resemble the successful ones.
- In the simulation, each true follow-up in a response quarter recruits,
  with probability 0.3, one at-risk author chosen by feature similarity.
- ρ was set before any registered seed so that about a quarter of entrants
  are recruited.
- W8 is also scored without the competing-risk censoring. That score is
  reported, to show what the censoring buys.

## Decision

| Disposition | Condition | Action |
|---|---|---|
| `FALSE_RECOVERY_AT_THE_EMPIRICAL_SHAPE` | Some must-not world wins more than 1 of 8 | N11c is not proposed in this form; the analysis is redesigned before any outcome row |
| `UNDERPOWERED_AT_THE_EMPIRICAL_SHAPE` | No false recovery, but W1 or W2 wins fewer than 7 of 8 | N11c and N11d are not proposed in this form; the reported power worlds guide a redesign |
| `RECOVERY_PASSES_AT_THE_EMPIRICAL_SHAPE` | Otherwise | N11c may be proposed for authorization, with this analysis unchanged |

## Boundary

The study is synthetic. Its only empirical inputs are N11b's per-episode
counts, which carry no identifier and no outcome. The script must be
committed before the development run, and a rerun must be byte identical.
No model is selected for data.
