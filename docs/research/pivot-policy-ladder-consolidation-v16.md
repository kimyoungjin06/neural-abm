# Pivot policy-layer ladder consolidation v16

Date: 2026-11-02

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v16.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v16.yaml)

Extends: [consolidation v15](pivot-policy-ladder-consolidation-v15.md), and
through it v1 to v14, which remain the records for everything before.

## Purpose

The user directed stage 2 of recommended experiment 3 on 2026-11-02. Stage 1
registered a rule for where a table written before the run loses to the
in-run learner, from between-run heterogeneity and within-run evidence. It
was tested in the information-spread generator (v15).

Stage 2 asks whether the rule holds in a generator we did not build. In the
OpenAlex discovery episodes, does it tell from other episodes alone whether
a table pooled from them or the episode's own learner forecasts an
entrant's follow-up better? That is O42.

Every claim below cites a hash-bound result and a field that must hold, and
a test checks all of them. This document executes nothing.

| Study | Freeze / record | Outcome | Runs (development / reproduction) |
|---|---|---|---|
| Crossover prediction on OpenAlex v1 | `015eea5` / `39e1aec` | `E3S2_THE_LEARNER_WINS_WHERE_DETERMINED_AND_THE_RULE_AGREES` | 42 / 44 s, byte identical |

## The design in one paragraph

**The policy** is an entrant's chance of a follow-up topic work in four
quarters, experiment 4's label, visible five quarters after entry.
- **The table T-k** forecasts the mean follow-up share of k other discovery
  episodes. Its score is the exact expectation over all k-subsets.
- **The learner** forecasts the episode's own share among visible labels.
- **The score** is the Brier score of each precise entrant of quarters 5 to
  11.

**The rule** predicts the table-minus-learner gap as H_k − V/n, from the
other 23 episodes only:
- H_k is the between-episode heterogeneity as a k-table sees it;
- V is the per-label variance of an episode's own share;
- n is the number of visible labels.

**The scoring.** Twenty groups (5 table sizes by 4 evidence bins) were
labelled with an episode-cluster bootstrap. A generator check on synthetic
labels at the same shape had to pass before any label was read.

## What is established

- **The rule passed its generator check (C107).** On synthetic labels at
  the episodes' shape:
  - its predicted gap had the sign of 92% to 99.6% of the determined groups;
  - it put at most 2.9% of them on the wrong side.
- **The learner wins wherever the data decide, and the rule agrees
  (C108).**
  - All 7 determined groups are learner wins: every table at 150 or more
    visible labels, and the one-episode table from 15.
  - The rule agrees with all 7 and makes no reversal.
  - **No group was decided for the table.** The baseline that the learner
    always wins agrees as well, so the data test the rule on one side only.
- **The rule puts the crossing early (C109).**
  - The between-episode standard deviation of follow-up shares is 0.065,
    around an unweighted mean of 0.079.
  - The per-label variance of an episode's own share is 0.068.
  - The crossing is at 6.8 visible labels for a one-episode table and 14.1
    for a 16-episode table. 93.5% to 97.9% of the forecasts have at least
    that many.
  - For a large table, the crossing approaches V over the between-episode
    variance, which is Bühlmann's credibility constant. The rule here is
    the textbook trade-off, located in these data.
- **Post hoc: the observed gap rises more steeply than the rule (C110).**
  - At 150 or more visible labels the gap is +0.0163 to +0.0211, against the
    rule's +0.0041 to +0.0090. Every predicted mean still lies inside its
    interval.
  - The unweighted mean share, 0.079, is about half the pooled share, 0.146,
    so larger episodes follow up more often. An equally weighted table then
    misses a large episode by more than the average heterogeneity. That is
    not tested.
- **The study reproduced byte for byte and read the discovery partition only
  (C111).**

## What is rejected

- **X3 (R28).** Below 15 visible labels, the observed gap favoured every
  table, but every interval covered zero.

The other four expectations held. No earlier claim is narrowed.

## What is answered in part

**O42.** The rule agrees with every determined comparison on the discovery
episodes, and every one is a learner win. A comparison decided for the table
is still missing.

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | Do the Regime B attributions hold with the steepness also matched in the world? (O34, carried) | Synthetic |
| 2 | Does a policy that starts from the table and weights the episode's share by n / (n + V/H_k) beat both? (O43) | Empirical, discovery partition, after a synthetic check |
| 2 | Does a rule that weights errors near the share cutoff place the crossing within the bound where evidence is scarce? (O41, carried) | Synthetic |
| 2 | Does the onset move with information that shifts within a run? (O29, carried) | Synthetic |
| 2 | Is there a coauthor-outcome design that leaves Study 2's frozen arm flat and has power at the empirical shape? (O38, carried) | Synthetic, then empirical |
| 2 | Does the LLM direction still add nothing inside tuned learners? (O37, carried) | Synthetic |
| 2 | Is it the location or the steepness that carries the cost of D5's prior? (O32, carried) | Synthetic |
| 3 | Does the ordering replicate on the sealed confirmatory episodes, and do they decide a comparison for the table? (O44) | Empirical, confirmatory partition, sealed |
| 3 | Does R5's narrowing hold under a calibration that does not put agent steps and topic quarters on one scale? (O40, carried) | Synthetic |
| 3 | Does a table suffice across veracity relations or domains? (O30, carried) | Synthetic |
| 3 | Do the comparisons hold when an arm is charged for the reach of its false shares? (O31, carried) | Synthetic |

**O43** is NABM's own prior-plus-learner arm in this setting: the table as
the prior, and the episode's outcomes as the closed-loop update.

**Data state.**
- The discovery labels have now been read by experiment 4 and by this
  study.
- The confirmatory and G4 outcomes stay sealed. O44 would need its own
  precommit and the user's authorization.

## Reporting rules

These carry v15's rules and add one.

- **One-sided real-data crossover.** State the OpenAlex result with its
  policy and its one side.
  - With a topic's follow-up share as the policy, an episode's own visible
    share beat a table of other episodes' shares wherever the discovery data
    decided, as the rule predicted.
  - No comparison was decided for the table (C108).

## Paper-1

**The central claim is unchanged.** NABM is a reusable framework that
separates policy learning from social interaction, so fixed policies,
adaptive policies, and external priors can be compared and traced within one
simulation lifecycle.

**What this consolidation changes:**
- **The crossover rule now has one real-data check.** In OpenAlex surge
  topics, the closed loop wins for a follow-up forecast once a handful of
  outcomes are visible, as the rule predicts.
- **The table side is untested there.**
- **The contrast with the synthetic ladder is heterogeneity.**
  - There, worlds shared one veracity relation, and a table from other worlds
    sufficed (O23).
  - Here topics differ widely in how often entrants follow up.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N50 | Paper | Add C107–C111 to the manuscript kept outside the repository | After this consolidation |
| N51 | Repository | Merge the OpenAlex crossover branch into `main` through a pull request | The user's confirmation |
| N52 | Comparison design | Choose between O43, O44, O41, the other open items, and the remaining manuscript sections, each study as a separate precommit | The user's choice |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical row, and changes no generic NABM API. It selects no
model and establishes no neural necessity or scientific effect.

The study's own limits carry over:
- **Discovery partition only.** The confirmatory outcomes stay sealed.
- **One policy.** The topic's follow-up share, with no entrant features.
- **Donors from any calendar time.** The tables come from other episodes
  regardless of when they ran.
- **A forecaster, not a researcher.** The learner is a forecast. Nothing here
  says how researchers decide.
