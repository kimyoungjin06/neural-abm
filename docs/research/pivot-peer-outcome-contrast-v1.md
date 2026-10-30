# Pivot peer-outcome contrast v1: Study 2 against OpenAlex

Date: 2026-10-30

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_peer_outcome_contrast_v1.yaml`](../../experiments/configs/pivot_peer_outcome_contrast_v1.yaml)

This is the fourth recommended experiment: Study 2 against real data. The
user directed it on 2026-10-30.

## Question

[Study 2](../case-studies/researcher-pivot/README.md#6-study-2--can-learning-agents-self-correct-the-hype-paradox)
postulates that researchers learn from the visible pivot outcomes of their
network neighbors. The direction of that learning decides whether failed
pivots grow:
- **imitative** learners update on every neighbor outcome, success or
  failure;
- **cautionary** learners update on failures only;
- **frozen** agents never update, though their readiness still spreads
  through the network.

Study 2's learning rules are assumptions. This study puts one observable of
them to data. When a coauthor enters a surge topic, does an author's own
entry depend on the coauthor's outcome there once it becomes visible? Does
that dependence go beyond the association it already has before the outcome
is visible?

## Why this is not the N11 ladder

N11 asked whether entry selectivity rotates toward the revealed success
relation across a whole episode. With episode-level variation it could not
separate that rotation from anticipation, and the user closed that design on
2026-10-13 ([consolidation v5](pivot-policy-ladder-consolidation-v5.md)).

This study uses variation between authors inside one episode and quarter:
- one author's coauthor succeeded and another's failed, in the same topic and
  at the same time;
- the episode-by-quarter fixed effects absorb anything the topic does over
  time, anticipation included.

It does not claim to identify learning; see [Reading](#reading).

## Design

### One estimator, two sources

The same discrete-time logistic hazard is fitted to two kinds of rows.

| | Study 2 (synthetic) | OpenAlex (discovery partition) |
|---|---|---|
| Unit | Agent × step, steps 3–14 | Author at risk × quarter, 12 response quarters |
| Neighbor | An agent in the focal agent's neighbor list | A coauthor of an extract work dated in the three years before the response year |
| Neighbor entry | A pivot | A first work in the episode topic in the response years |
| Entry visible | The step after the pivot | N11b's rule: the next quarter, or the next year for a date known only to the year |
| Outcome | The productive flag | N11a v3's analyst label: a follow-up topic work in the four quarters after entry |
| Outcome visible | 1 + delay steps after the pivot, delay 0 or 3 | Five quarters after entry, or two years after an imprecise entry year |
| Fixed effects | Replicate × step | Episode × quarter |
| Controls | Career stage | Binned productivity, subfield share, career age, incumbent ties |

### Neighbor states

At each row, each visible neighbor entrant is in one state:

| State | Meaning |
|---|---|
| `S` | Outcome visible, a success |
| `F` | Outcome visible, a failure |
| `P_S` | Not yet visible; it will be a success |
| `P_F` | Not yet visible; it will be a failure (the reference) |
| `U` | A coauthor entrant who is not an S1 entrant; its outcome is never read |

The state model has an indicator for each state, plus the number of visible
neighbor entrants and the quarters since the first one.

### Contrasts

| Contrast | Formula | Reading |
|---|---|---|
| Outcome gap, visible | `S − F` | Dependence on a visible outcome |
| Outcome gap, before visible | `P_S − P_F` | The same dependence before anyone can see it: homophily |
| Visibility DiD | the first minus the second | Dependence that appears when the outcome becomes visible |
| Success-following | `S − P_S` | Entry rises when a success becomes visible |
| Failure-avoidance | `F − P_F` | Entry falls when a failure becomes visible |

What each Study 2 arm predicts:
- **Imitative:** both success-following and failure-avoidance.
- **Cautionary:** failure-avoidance only.
- **Frozen:** neither. Any outcome gap it shows is homophily or contagion,
  and the placebo should remove it.

### Study 2 as the first source

Study 2 runs unchanged through two slots:
- an environment that keeps every step's events, so agents can learn from a
  neighbor's outcome after a delay;
- an agent class that registers itself so its pivot step can be logged.

Two checks come before anything else:
- At delay 0, the rerun must reproduce all 270 tracked replicate runs
  exactly.
- The frozen arm, which never learns, must be identical at delays 0 and 3.

With delay 0, outcomes are visible at once, so only `S − F` exists. Delay 3
creates the pending states, and with them the placebo.

### OpenAlex as the second source

- **Episodes.** The 24 S1 discovery episodes only. Confirmatory follow-ups
  stay sealed.
- **Who is kept.** Exposed authors are rare, so every author with a
  coauthor entrant is kept, with weight 1. Everyone else comes from N11b's
  case-cohort sample: entrants with weight 1, the subcohort with weight
  1/π.
- **Parity with N11b.** The rebuilt rows of exposed authors must equal
  N11b's row for row: quarters, events, features, and coauthor entry counts.
  Every other N11b row must have no visible coauthor entry.
- **When parity runs.** The check runs before any label, entry work, or
  follow-up is read, and a failure stops the build.
- **Dry run.** A structure-only run on the real extract, before this freeze,
  read no label and passed every check:
  - 503,879 dyads and 177,254 exposed authors;
  - 634,492 rows with a visible coauthor entry;
  - 40 seconds and 7 GB.
- **Events and censoring.** These are N11a v3's rules:
  - an event is a precise, unrecruited entry;
  - an imprecise entrant is censored from the start of its entry year;
  - a recruited entry is censored as a competing risk. It is recruited when
    its entry work has an earlier entrant as a coauthor.

### Inference

- **Bootstrap.** Percentile intervals from 2,000 cluster draws: episodes in
  the data, replicates in Study 2.
- **Joint draws.** Every arm of a scenario uses the same draws, so an arm's
  contrast minus the frozen arm's is a learning-attributable effect with its
  own interval.
- **Ridge.** A ridge of 1 on the non-fixed-effect coefficients keeps the
  estimates finite when a state is rare.
- **Undefined contrasts.** A contrast is undefined when any of its states has
  fewer than 10 events.

## Expectations, stated before the run

| Id | Statement |
|---|---|
| ES2 | In Study 2 as tracked, the imitative arm's visible outcome gap exceeds the frozen arm's in the baseline and grant scenarios |
| ES3 | At delay 3, the frozen arm's visibility DiD covers zero wherever it is defined, in at least two scenarios. Otherwise the estimator flags learning where there is none |
| EE1 | In the data, a visible coauthor entry is associated with a higher entry hazard |
| EE2 | In the data, the outcome gap before visibility is positive (homophily) |

No expectation is stated on the data's visibility DiD.

## Decision

| Condition | Status |
|---|---|
| ES3 failed | `E4_NOT_INTERPRETABLE_THE_ESTIMATOR_FLAGS_STUDY2_FROZEN_ARM` |
| DiD undefined | `E4_UNDEFINED_TOO_FEW_EVENTS` |
| DiD > 0, success-following > 0, failure-avoidance < 0 | `E4_CONSISTENT_WITH_IMITATIVE` |
| DiD > 0, failure-avoidance < 0 only | `E4_CONSISTENT_WITH_CAUTIONARY` |
| DiD > 0, success-following > 0 only | `E4_SUCCESS_FOLLOWING_ONLY_NO_STUDY2_ARM` |
| DiD > 0, neither direction | `E4_OUTCOME_DEPENDENCE_EMERGES_WITHOUT_A_DIRECTION` |
| DiD < 0 | `E4_OUTCOME_ASSOCIATION_WEAKENS_ONCE_VISIBLE` |
| Otherwise | `E4_NO_VISIBILITY_EFFECT_DETECTED` |

A sign here means the 95% interval excludes zero on that side.

The full sign map is reported beside the data:
- every contrast for every scenario, arm, and delay;
- the entry-order gradient;
- the exposure term, also with recruited entries counted as events.

A failed parity check, or a full-sample fit that does not converge, is an
invalid-implementation stop with no disposition.

## Reading

**A disposition says which Study 2 arm the data pattern resembles. It does
not identify learning.** Homophily and contagion are generically confounded
in observational network data (Shalizi and Thomas, 2011). The placebo
removes only the part of the association that already exists before the
outcome is visible.

Two kinds of association remain:
- associations that change over time, such as a group moving into the topic
  together;
- reactions to an outcome that coauthors learn of before it is published.

The second shifts reactions into the pending states, which biases against
finding a visibility effect.

## Boundary

- **Data.** Discovery partition only. No confirmatory follow-up, no G4
  outcome, and no follow-up of a coauthor who is not an S1 entrant is read.
  A fixture test adds such follow-ups and checks that every output is
  unchanged.
- **Study 2.** Its code is unchanged; only the two slots are added.
- **Scope.** No LLM call and no change to a generic NABM API. No model is
  selected, and no causal claim is made.
- **Execution.** A run needs this precommit and the committed authorization
  file. Development and reproduction runs must be byte identical.
