# Pivot crossover prediction v1 (experiment 3, stage 2): OpenAlex discovery episodes

Date: 2026-11-02

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_crossover_prediction_v1.yaml`](../../experiments/configs/pivot_crossover_prediction_v1.yaml)

Authorization:
[`pivot_crossover_prediction_authorization_v1.json`](../../experiments/configs/pivot_crossover_prediction_authorization_v1.json)

The user directed recommended experiment 3 on 2026-10-30 and chose to run it
in two stages: synthetic first, then OpenAlex discovery episodes. On
2026-11-02 the user directed this second stage.

## Question

[Stage 1](info-spread-crossover-prediction-v1.md) registered a rule for
where a table written before the run loses to the in-run learner. The rule
uses two quantities measured outside the comparison:
- **between-run heterogeneity:** how far a run's truth lies from what other
  runs learned;
- **within-run evidence:** how much the run's own outcomes tell its learner.

In the information-spread generator, the rule beat two baselines but missed
its accuracy bound. It was late where evidence was scarce
([consolidation v15](pivot-policy-ladder-consolidation-v15.md), C101 and
C104).

**This stage asks whether the rule holds in a generator we did not build.**
In the OpenAlex discovery episodes, does it tell from other episodes alone
whether a table pooled from them or the episode's own learner forecasts an
entrant's follow-up better? That is O42.

## The policies

**The run is a discovery episode.** Each of the 24 is a surge topic of the
S1 extract. Its precise entrants are its decision makers, and each has the
label of [experiment 4](pivot-peer-outcome-contrast-v1.md): a follow-up topic
work in the four quarters after entry. The label becomes visible five
quarters after entry.

**The policy is an entrant's chance of a follow-up.**

| Policy | Forecast | What it uses |
|---|---|---|
| Table T-k | The mean follow-up share of k other discovery episodes | Other episodes' outcomes, written before this episode |
| Learner L | This episode's follow-up share among entrants whose label is visible | This episode's own outcomes, as they arrive: a closed loop |

- **Forecasts.** Every precise entrant of quarters 5 to 11 whose learner has
  at least one visible label: 3,835 entrants, by the structural count.
- **Table sizes.** k = 1, 2, 4, 8, and 16. The table's score is its exact
  expected Brier score over all k-subsets of the other 23 episodes, so the
  choice of donors adds no noise.
- **Score.** The Brier score on the entrant's own label. The gap is the
  table's score minus the learner's; it is positive when the learner wins.

**Why no entrant features.**
- The 24 episodes hold 920 follow-ups, so an entrant-level model inside one
  episode is barely identified.
- The follow-up share is what an entry decision rests on in Study 1.
- With one share per episode, the rule is exact in expectation for
  exchangeable entrants, so this stage tests whether the rule transports,
  not how well it approximates.

## The rule

The Brier score of a forecast f, on an episode whose entrants follow up with
probability π, is (f − π)² + π(1 − π). The label noise cancels in the gap,
which is the stage 1 rule's squared-error difference against the truth.

For episode e at a forecast quarter with n visible labels, the predicted gap
is

    H_k(−e) − V(−e) / n

Both terms come from the other 23 episodes only, so no label of e enters its
own prediction.
- **Heterogeneity H_k(−e).** Each other episode d serves as a world. A table
  of k is built from the remaining 22, and its squared error against d's
  share is corrected for d's own sampling noise. H_k(−e) averages this over
  the worlds. It is the between-episode heterogeneity as a k-table sees it.
- **Evidence V(−e).** The mean per-label variance of an episode's own
  share. A learner with n labels has squared error V/n.
- **No fitted parameter.**

**Simple reading: the learner wins once n > V / H_k.**
- H_k is close to D(1 + 1/k) + W/k, where D is the between-episode variance
  of true shares and W a donor share's sampling variance.
- A larger table moves the crossing later; more heterogeneity moves it
  earlier.
- In stage 1, a donor table had the learner's own evidence. Here it has
  every donor outcome.

## Groups and scoring

- **Groups.** Five table sizes by four evidence bins of visible labels:
  [1, 15), [15, 50), [50, 150), and 150 or more. That makes 20 groups.
  - By the structural count, the bins hold 248, 558, 1,045, and 1,984
    forecasts from 17, 16, 12, and 7 episodes.
- **Observed label.** An episode-cluster bootstrap with 2,000 draws gives
  each group's interval.
  - POSITIVE: the learner wins. The interval lies above zero and the mean is
    at least the minimum effect.
  - NEGATIVE: the table wins, by the mirror condition.
  - Otherwise INDISTINGUISHABLE.
- **Minimum effect.** 0.0012485 in Brier score: one percent of the base-rate
  Brier score at experiment 4's recorded share, 920/6,291.
- **Predicted label.** The same minimum on the rule's mean prediction:
  POSITIVE, NEGATIVE, or NEUTRAL.
- **Scoring.** These count over the determined groups, those labelled
  POSITIVE or NEGATIVE.
  - An *agreement* is a group whose predicted mean has the sign of its
    observed label.
  - A *reversal* is a group the rule decisively assigns to the other side.
  - The label match, in which a NEUTRAL prediction counts as a miss, is
    reported.
  - Agreement decides rather than the label match, because in the pilots
    most label misses were gaps realized just above the minimum effect.

**Two baselines are reported.** One says the table always wins, the other
that the learner always wins. If every determined group has one sign, a
baseline agrees as well as the rule. The data then test the rule only
against that side.

## The generator check comes first

**Before any label is read, the same pipeline runs on synthetic labels at
the empirical shape.**
- **Shape.** The real counts of precise entrants by episode and entry
  quarter.
- **Labels.** Entrants follow up independently with probability
  expit(logit(p0) + τ z_e), where p0 = 920/6,291 and z_e is standard normal.
- **Design.** τ = 0, 0.25, 0.5, and 1, each with 40 replicates on keyed
  streams.
- **Pass.** At every τ, agreement in at least 80% of determined groups and
  reversals in at most 5%.
- **Failure.** The run stops before any label is read.

**The pilots set the bins and thresholds.** They used scratch seeds and the
same shape.
- **Agreement:** 0.91 to 1.00.
- **Label match:** 0.84 to 1.00.
- **Reversals:** 0% to 3.5%.
- **Coverage:** 62% to 83% of groups had the predicted mean inside the
  observed interval.
  - The high-evidence bins rest on a few large episodes.
  - The rule predicts the expected gap, not the realized one.
  - Coverage is therefore reported and decides nothing.

## Expectations, stated before the run

| Id | Statement |
|---|---|
| X1 | The generator check passes at every τ |
| X2 | At τ = 0.5, at least half the replicates show a determined learner win |
| X3 | In the data, every table beats the learner in the lowest evidence bin |
| X4 | In the data, the learner beats the 16-episode table in the highest evidence bin |
| X5 | In the data, the rule makes no reversal and agrees with at least 80% of the determined groups |

## Decision

**Checks that read no label come first:**
- the N11b preflight: input hashes and the G4 guard's identity;
- discovery episodes and precise entrants equal experiment 4's record (24
  and 6,291);
- every episode has at least two precise entrants.

**After the labels are read,** the number of follow-ups must equal experiment
4's record of 920. A failed check is an invalid-implementation stop with no
disposition.

The first condition that holds sets the status:

| Condition | Status |
|---|---|
| The generator check fails at some τ; no label is read | `E3S2_STOPPED_THE_RULE_FAILS_ON_KNOWN_TRUTH` |
| Fewer than 2 determined groups | `E3S2_UNDETERMINED_TOO_FEW_DETERMINED_GROUPS` |
| A reversal | `E3S2_THE_RULE_REVERSES_A_DETERMINED_COMPARISON` |
| Agreement below 80% | `E3S2_THE_RULE_MISSES_DETERMINED_COMPARISONS` |
| Determined groups of both signs | `E3S2_THE_RULE_ORDERS_THE_TABLE_AND_THE_LEARNER` |
| Only NEGATIVE groups | `E3S2_THE_TABLE_WINS_WHERE_DETERMINED_AND_THE_RULE_AGREES` |
| Only POSITIVE groups | `E3S2_THE_LEARNER_WINS_WHERE_DETERMINED_AND_THE_RULE_AGREES` |

**Reported beside the status, deciding nothing:**
- H_k, V, and the crossing V/H_k from all 24 episodes;
- the between-episode variance of shares, corrected for sampling;
- the share of forecasts with at least that much evidence;
- for each k, the first evidence bin from which every label is POSITIVE.

## Known risks

- **Stage 1's miss.** Stage 1 was late where evidence was scarce, possibly
  because its squared-error proxy was scored against share precision.
  - Here the score is a squared error, so that cause cannot arise, and the
    rule is exact in expectation.
  - This stage therefore does not test the stage 1 explanation.
- **Transport.** The rule averages heterogeneity over other episodes. The
  high-evidence bins rest on a few large episodes, whose shares may lie
  nearer to or farther from the rest than the average episode's.
- **Real time.** The learner sees outcomes in entry order, five quarters
  late. A share that drifts within an episode would make it worse than the
  rule's exchangeable learner. Experiment 4 found no entry-order gradient in
  the label: +0.010 [−0.224, +0.287].
- **Scarcity.** 920 follow-ups in 24 episodes. Many groups may be
  indistinguishable.

## What was seen before this freeze

- **Experiment 4's recorded aggregates.** 920 of 6,291 discovery precise
  entrants follow up. Experiment 4 read the discovery labels once and
  reported pooled aggregates only. No share by episode was computed or seen.
- **A structural count.** Discovery precise entrants by episode and entry
  quarter. It read no label.
- **Synthetic pilots** at that shape with scratch seeds, which set the
  evidence bins and the thresholds.

## Boundary

- **Data.**
  - The run reads the discovery partition only: S1 episodes and entrants,
    and the episode-topic works of discovery precise entrants.
  - It reads no confirmatory follow-up, no imprecise entrant's follow-up,
    no coauthor row, no G4 outcome, and not the N11b panel.
  - A fixture test adds confirmatory and imprecise follow-ups and checks
    that every output is unchanged.
- **Donors.** The table's donors are other episodes, some later in calendar
  time. This is a comparison of policies, not a real-time forecast for any
  decision year.
- **Policy.** The policy is the topic's follow-up share. Entrant features
  are not used.
- **Scope.** No LLM call and no change to a generic NABM API. The study
  identifies no learning and selects no model.
- **Execution.**
  - The run happens on the workstation, where the private extract is.
  - It needs this precommit and the committed authorization file.
  - Development and reproduction runs must be byte identical.
