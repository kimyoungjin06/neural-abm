# Decision 0003: Social Update Pipeline

Status: Accepted

Date: 2026-04-29

## Context

The initial Neural ABM Node interface treated `social_update` as one conceptual
operation after social messages and peer selection. Toy 1 made that too coarse.

The parameter-path diagnostics showed:

- Raw independent-init parameter averaging fragments under high raw parameter
  similarity thresholds.
- Hidden-unit alignment before averaging helps when the peer graph is already
  connected.
- Hidden-unit aligned peer selection changes the peer graph itself, keeping
  independent-init agents connected over a wider threshold range.

This means the observed behavior can come from the peer-selection rule, the
alignment map, the mixer, or their interaction. A single `social_update` label
is not enough to describe or test the mechanism.

## Decision

Define the reusable social pipeline as:

```text
social_message
-> compatibility_score
-> peer_select
-> align_or_translate
-> typed social_mix
-> commit_social_update
```

Keep these stages separately configurable and logged.

Minimum required config concepts:

- `peer_rule`: how candidate contacts are scored and filtered.
- `threshold`: selection threshold or rule-specific cutoff.
- `alignment_scope`: none, peer-selection only, mixing only, or both.
- `mixer`: the social mixing operator.
- `channel`: output, latent, parameter, action/policy, or memory/trust.

For the current codebase, simple selectors, aligners, and mixers may remain in
`mixers.py`. Split them into `selectors.py`, `aligners.py`, and `mixers.py` only
when the implementation pressure justifies it.

## Consequences

- Toy and paper results should name peer rule and mixer together, for example
  `parameter_aligned_average + aligned_state_similarity`.
- Parameter-level claims must distinguish averaging alignment from
  peer-selection alignment.
- Learned edge functions should be treated as compatibility or peer-selection
  modules unless they also perform state mixing.
- Toy 2 should reuse the same social pipeline, replacing supervised outputs
  with action or policy channels.
- Logs should include enough fields to reconstruct peer selection, alignment
  scope, mixer channel, and update magnitude.
