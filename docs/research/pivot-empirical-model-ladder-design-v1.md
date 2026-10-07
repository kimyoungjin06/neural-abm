# Pivot empirical model ladder: design brief v1 (N11)

Date: 2026-10-07

Status: `DRAFT_DESIGN_BRIEF_NOT_A_PRECOMMIT`

## Why this replaces the sign test as the empirical goal

The project goal is NABM: small learned policies, with a rule or an elicited
LLM judgment as their prior, learning from pooled outcome records. It sits
between rule-based ABMs and LLM-agent simulation. The
[case-study bridge](pivot-nabm-case-study-bridge-v1.md) gives the empirical
layer one job. It supplies targets on which a fair ladder of models is
compared. If a simpler model does as well, that is the result.

[N10 v2](pivot-empirical-outcome-selection-trend-v2.md) instead registered a
one-sided sign test on one indicator. Real data contain mechanisms the
synthetic worlds lacked, and each can make later entrants persist more
without any learning:

- topic growth after a surge;
- author linkage that improves over time (ORCID after 2012);
- drift in retrospective topic classification;
- changing team composition.

A positive sign would therefore not separate learning from these, and a
non-positive one is registered as no inference.

## The discriminating prediction

Learning predicts more than "later entrants persist more". It predicts the
direction in which entry selectivity changes. As outcomes become visible,
the features that predict who enters drift toward the features that
predicted success. Suppose early entrants from nearby subfields stayed, and
distant ones did not. Under learning, later entry should become more
proximity-selective.

None of the four mechanisms above predicts that alignment. Growth and linkage
change everyone's measured persistence, and they do not rotate entry toward
the revealed success relation. The ladder below tests the prediction by
asking which policy class forecasts held-out entry best.

## Data unit and inputs

| Element | Definition | Source |
|---|---|---|
| Unit | Episode × at-risk author × quarter, over the 12 response quarters | S1 episodes and extract |
| At risk | Field-active in the three prior years, no prior topic work, not G4-guarded | S1 extract |
| Pre-entry features | Prior productivity, career age, subfield proximity to the topic, coauthor ties to topic incumbents | Works dated before the decision year |
| Social signals | Coauthors who entered the topic earlier in the episode, by quarter | Entry dates only |
| Visible evidence | For earlier entrants: whether they published in the topic again by the current quarter | Dates up to the current quarter; never later |
| Shared environment | Attention clock, topic volume, risk-set size, date precision | S1 extract; given to every model |

The visible evidence is what an author could have seen at that time. The
12-quarter persistence label is never used as a model input.

## The ladder

| Id | Model | Entry depends on |
|---|---|---|
| B1 | Statistical hazard | Attention, time, topic volume. No individual features |
| B2 | Fixed-rule agents | Pre-entry features with fixed weights, plus B1's terms |
| B3 | Explicit adaptive rule | B2, plus a delta-rule update of the entry level from visible peer evidence |
| B4 | NABM, pooled linear | A success predictor learned online from visible evidence, with B2 as its prior |
| B5 | NABM, pooled tree | As B4, with a tree learner |
| B6 | NABM, pooled MLP | As B4, with an MLP learner |

Every model gets the same risk sets, environment inputs, partitions, and
calibration budget. Free parameters are fitted on discovery episodes only.

## Scoring and decision (to be registered)

- **Primary:** held-out log-likelihood of entry events in confirmatory episodes,
  per field.
- **Generative, reported:** the entry curve, cohort persistence curve, and N10
  outcome selection trend that each fitted model reproduces.
- **NABM is supported** when B4–B6 beat both B2 and B3 by a registered margin
  in a registered share of fields.
- **Simpler-model results** (B2 or B3 not beaten, or B1 enough) are reportable
  outcomes, not failures.

## Validation before any outcome row: N11a

The method is checked in synthetic worlds whose truth is known before it is
used on data. Agent-level panels are generated with learning on and off, with
topic growth and drifting linkage added to the no-learning worlds. The ladder
is then fitted on discovery replicates and scored on held-out ones.

It must:
- pick the learning arms when learning is on;
- not pick them when learning is off, even with the confounds present.

If it cannot, the empirical stage stops here.

## Stages

| Stage | Reads | Status |
|---|---|---|
| S1 | Structural build: episodes, entrants | Authorized; running |
| N11a | Synthetic recovery test with confounds | Next; no empirical rows |
| N11b | Risk sets and pre-entry features from the S1 extract | Needs authorization; no outcomes |
| N11c | Discovery: visible evidence and calibration | Needs authorization |
| N11d | Confirmatory scoring | Needs authorization |

N10 v2's outcome selection trend stays a reported generative target. It is
no longer the decision.

## Boundary

This is a design brief. It registers nothing, reads no row, and authorizes
nothing. A precommit with exact definitions, margins, and failure actions
must follow N11a.
