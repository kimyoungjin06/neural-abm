# Peer Exposure Identification Design

Design version: 0.1

Status: prospective G0 component; no peer-learning, causal-influence, or
mechanism claim has passed.

Date: 2026-07-21

This design separates network-local information, direct collaboration or
capacity access, and field-wide attention in the prospective pivot-feedback
study. It supplements the
[empirical protocol](../case-studies/researcher-pivot/pivot-feedback-empirical-nabm-protocol.md)
and is governed
by the same G-1, G0, G1, and sealed-holdout boundaries.

## 1. Question and Claim Boundary

The observable question is:

> Does target-topic activity or time-eligible success among a researcher's
> prior collaborators improve prediction of later topic entry beyond
> field-wide attention, pre-entry similarity, and network or institutional
> capacity proxies?

The OpenAlex-only analysis does not observe whether a researcher read a peer's
work, discussed it, revised a belief, or imitated a decision. It therefore
must not begin from the statement that researchers see peer pivots and then
follow them. A qualifying source peer may also be a long-standing incumbent in
the target topic rather than a researcher who recently pivoted.

The permitted interpretation ladder is:

1. **Predictive association:** a time-eligible network-local proxy predicts
   later entry after registered adjustment and placebos.
2. **Mechanism-consistent pattern:** the visible-success increment and an
   independent-entry contrast survive all registered falsification tests.
3. **Causal social learning:** an external design separately identifies
   receipt of the information and the resulting belief or decision change.

Level 1 does not imply Level 2, and OpenAlex alone cannot promote Level 2 to
Level 3.

## 2. Observation Layers

The empirical and simulated pipelines must keep these layers distinct:

```text
publicly available target-topic event
    -> network-local opportunity to receive it
    -> actual receipt or attention                  [not observed in OpenAlex]
    -> belief or policy update                      [not observed in OpenAlex]
    -> target-topic entry or non-entry
    -> chaperoned or independent entry pathway
```

`available_at <= t-1` establishes only public availability. A prior
collaboration tie establishes only a plausible channel. Neither field is an
observation of receipt, learning, persuasion, or imitation. Bibliometric
silence is not an observed failure and must not supervise a failure-learning
path.

## 3. Competing Explanations

Every analysis must compare, rather than silently combine, the following
explanations:

| Explanation | Observable implication | Main threat to interpretation |
| --- | --- | --- |
| Field-wide attention | Neighbors and matched non-neighbors respond similarly to the same topic-year signal. | A network coefficient can proxy a common opportunity shock. |
| Selection or homophily | Similar researchers collaborate and later enter similar topics without peer influence. | Pre-existing latent interest remains incompletely measured. |
| Recruitment or capacity transfer | Entry occurs with the source peer or through a pre-existing active resource bridge. | Direct project access can look like observational learning. |
| Network-local outcome signal | A peer's time-eligible visible success adds prediction beyond peer activity, common attention, and capacity. | Public availability is still not verified receipt. |

The analysis must also consider reverse direction: an already planned focal
entry can generate new ties or shared work that is incorrectly treated as an
upstream exposure.

## 4. Frozen Empirical Constructs

G0 must define each construct using only information available by `t-1`.

### 4.1 Peer-active exposure

A qualifying pre-existing collaborator has at least one target-topic work
whose bibliographic event time is available by `t-1`. This is the base
network-local activity proxy. The G0 registry must freeze tie history, target
membership, source-peer aggregation, recency, and multiple-peer rules.

### 4.2 Visible-success increment

A qualifying source peer has a pre-specified target-topic outcome that was
publicly measurable by `t-1`. The construct must be estimated conditional on
peer-active exposure so that it tests the increment from visible outcome
information rather than the existence of an active peer. Citation-based
variants must use cutoff-aware citation availability and a frozen maturation
rule. Future accumulated citations are forbidden.

### 4.3 Pre-existing capacity bridge

The capacity proxy is a past-only active collaboration, institutional, or
resource relation to a target-experienced peer. It cannot be created by the
focal entry work. `Neither`, `signal only`, `capacity only`, and `both` remain
separate unless the registered cell-support or collinearity gate fails.

### 4.4 Entry pathway

The first eligible target-topic work defines one outcome subtype:

- `chaperoned_entry`: at least one qualifying pre-entry source peer is a
  coauthor of the focal first target-topic work;
- `independent_entry`: no qualifying pre-entry source peer is a coauthor of
  that work; or
- `unresolved_entry_pathway`: the frozen source or authorship contract cannot
  classify the work without ambiguity.

Entry pathway is a post-entry outcome subtype, not an upstream exposure or
capacity variable. Unresolved rows follow a G0-frozen exclusion or missingness
rule and may not be reassigned after results are inspected.

## 5. Required Contrast Ladder

All rows, clocks, weights, and partitions must be identical across the ladder.

1. field-wide attention and researcher-history model;
2. model 1 plus peer-active exposure;
3. model 2 plus the visible-success increment;
4. model 3 plus the pre-existing capacity bridge and registered interaction;
5. model 4 with separate chaperoned- and independent-entry outcomes.

The primary peer-signal question is whether model 3 adds a G0-frozen,
practically meaningful holdout improvement over model 2. A coefficient alone
does not pass this question. The mechanism analysis must report both entry
pathways even if only one is favorable to the proposed explanation.

## 6. Falsification and Negative Controls

The following controls are mandatory:

1. future attention predicting current entry;
2. a tie first observed at or after entry;
3. field-year degree-preserving shuffled-network exposure;
4. matched non-neighbor target-topic exposure;
5. a matched non-surge topic in the same field and calendar period;
6. the same peer's visible success in a G0-frozen unrelated topic; and
7. the source peer's target-work success measured only with citations or other
   outcomes that become available after the focal decision.

Controls 6 and 7 distinguish topic-specific time-eligible information from
general peer prominence and future-information leakage. Passing requires the
registered equivalence interval or true-minus-placebo contrast, not a
non-significant placebo coefficient.

## 7. Decision Rules

| Observed result | Required interpretation |
| --- | --- |
| Peer-active exposure adds no holdout value beyond field attention. | Drop the network-local mechanism claim. |
| Matched non-neighbor exposure performs similarly. | Retain a field-wide attention account; drop network-local specificity. |
| Future-tie or future-success placebo fails. | Treat the pathway as leakage or reverse direction; block the corresponding mechanism claim. |
| Association appears only for chaperoned entry. | Rename the result recruitment, chaperoning, or capacity association; do not call it observational social learning. |
| Capacity explains the peer-success increment or the four-state cells cannot be separated. | Collapse the channels and abandon signal-versus-capacity separation. |
| Time-eligible visible success adds holdout value, survives the independent-entry branch and every registered placebo, and the chaperoned result is reported separately. | Report a pattern consistent with a network-local response to public outcomes, explicitly noncausal and nonunique. |
| An external randomized or quasi-experimental exposure and belief-update axis also passes. | Consider a causal social-learning claim limited to that design and population. |

## 8. NABM Consequence

The adaptive NABM must not assume that every neighbor automatically observes
every public outcome. Its redesign must expose separate components for:

- public outcome availability;
- receipt or attention probability;
- belief or policy update conditional on receipt;
- direct recruitment or capacity transfer; and
- entry and post-entry outcomes.

At minimum, `no_peer`, `field_wide_signal`, `network_local_signal`, and
`capacity_or_chaperoning` variants must be comparable on identical exogenous
shocks. If the empirical contrast ladder does not require a network-local
update, that NABM channel is not promoted merely because the simulator can
express it.

## 9. External Identification Route

A direct social-learning claim requires an additional axis that measures or
assigns information receipt, such as randomized complete versus
success-selective outcome disclosure, exogenous seminar or laboratory
placement, an external disclosure-policy change, or linked survey and
communication records. The design must record pre-exposure beliefs,
post-exposure beliefs, and a consequential research-allocation or project
choice. Publication follow-up can then test whether the decision change
persists, but publication order alone cannot replace the missing receipt and
belief measurements.

## 10. Related Evidence and Identification Limits

- Venturini S, Sikdar S, Rinaldi F, et al. [Collaboration and topic switches in
  science](https://doi.org/10.1038/s41598-024-51606-6). Scientific Reports.
  2024. The observed association does not separate social influence from
  homophilous selection.
- Zeng A, Fan Y, Di Z, et al. [Impactful scientists have higher tendency to
  involve collaborators in new topics](https://doi.org/10.1073/pnas.2207436119).
  PNAS. 2022. Existing collaborators can accompany entry, which motivates the
  chaperoned-entry contrast.
- Catalini C. [Microgeography and the Direction of Inventive
  Activity](https://doi.org/10.1287/mnsc.2017.2798). Management Science. 2018.
  Exogenous proximity supports a contact channel but does not identify
  observation of a peer pivot.
- Manski CF. [Identification of Endogenous Social Effects: The Reflection
  Problem](https://doi.org/10.2307/2298123). Review of Economic Studies. 1993.
  Correlated group behavior does not by itself identify endogenous influence.
