# Pivot-feedback simple researcher-pivot definition v1

## Decision

The primary event is an **observable-history new-topic pivot**:

> A pivot is the first content-confirmed eligible contribution by a researcher
> to a primary topic with zero earlier content-confirmed eligible
> contributions to that topic in the observable history.

The executable contract is
[`pivot_feedback_researcher_pivot_definition_v1.yaml`](../../experiments/configs/pivot_feedback_researcher_pivot_definition_v1.yaml).
The read-only verifier is
[`verify_pivot_feedback_researcher_pivot_definition_v1.py`](../../scripts/verify_pivot_feedback_researcher_pivot_definition_v1.py).

This is deliberately an observable-history claim. It is not a claim that the
database sees the researcher's true lifetime-first contribution.

## Four-state classification

Only a researcher-primary-topic-year with at least one content-confirmed
eligible focal work enters the classifier. All history variables stop at the
end of `t-1`.

| State | Exact rule |
|---|---|
| `INSUFFICIENT_HISTORY` | Fewer than three observable prior years or fewer than three prior eligible works |
| `NEW_TOPIC_PIVOT` | History is sufficient and the prior confirmed topic-work count is zero |
| `RETURN` | The topic occurred earlier, but not during `t-3` through `t-1` |
| `CONTINUATION` | The topic occurred during `t-3` through `t-1` |

There is at most one event per researcher-primary-topic-year. A researcher-topic
pair cannot generate repeated `NEW_TOPIC_PIVOT` events.

History sufficiency is an eligibility condition, not evidence of novelty.
Researchers with insufficient history are not silently treated as pivots.

## Pivot distance is separate

The primary continuous magnitude is

```text
pivot_distance =
    1 - cosine(
        focal reference-source distribution,
        researcher trailing-three-year reference-source distribution
    )
```

It ranges from zero to one when available. It does not determine the four-state
event and has no event threshold. A publication-time text-embedding distance
is a robustness measure only.

This separation keeps the binary event interpretable without discarding how
far the researcher moved.

## Future outcomes are not inputs

Persistence during `t+1` through `t+3`, post-pivot impact, chaperoned entry,
and prior network-local public availability are separate downstream variables.
They cannot change whether the focal event was a pivot.

In particular, a single valid focal work may define entry. Requiring later
works inside the event definition would condition entry on the persistence
outcome.

## API-only content boundary

Human review is `NOT_RUN_BY_DESIGN`. Human accuracy and human Cohen's kappa are
therefore `NOT_AVAILABLE`, and API consensus is not ground truth.

For the frozen 384-title diagnostic sample, 197 dual-high-confidence
`ON_TOPIC` rows are strict API-consensus candidates and the other 187 rows are
unresolved and quarantined. That diagnostic sample is not the complete
candidate universe.

The definition freeze does not authorize a paid full-universe API run. A
versioned successor must first freeze topic-specific inclusion and exclusion
cards, request construction, a dry-run, and a cost cap. Unresolved rows remain
quarantined rather than being forced into a label.

## Claim boundary

This component freezes a definition and an executable decision table. It does
not complete D0, validate the four provisional episodes, screen the full
candidate universe, rebuild the author panel, read persistence or impact
outcomes, identify peer influence, fit a model, or open G4.
