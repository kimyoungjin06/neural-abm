# Pivot-feedback D0 adaptive candidate frontier v1

## Purpose

The candidate scope audit showed that eager content screening would touch 994
topics and 22.4 million distinct works, while the frozen router covers only
four topics. The adaptive frontier evaluates only candidates that can still
change the highest-ranked compatible pair.

## State machine

Candidates retain the frozen `field_candidate_rank` order. Content has four
states:

- `CONTENT_PASS`;
- `CONTENT_FAIL`;
- `CONTENT_UNRESOLVED`; and
- `NOT_EVALUATED`.

Neither unresolved nor unevaluated content counts as a pass.

The first selected candidate must be the earliest `CONTENT_PASS`; every earlier
candidate must be `CONTENT_FAIL`. The second must be the earliest later
`CONTENT_PASS` with a distinct subfield and at least nine years of entry-year
separation. Earlier candidates incompatible with the first may remain
unevaluated, but every earlier compatible candidate must be a frozen
`CONTENT_FAIL`. Any unresolved required row blocks the certificate.

## Bounded equivalence evidence

The producer and independent verifier separately enumerated every assignment
for candidate lengths two through six, two subfield values, two entry-year
values, and binary content truth. Across 299,584 fixtures:

- all 118,404 eager-pair cases matched the lazy certificate;
- 331,424 required-state unresolved mutations all blocked; and
- 146,832 structurally incompatible skipped-row checks left the pair
  unchanged.

This validates the executable state machine on the declared finite fixture
space. It is not presented as a general formal proof.

## Initial structural frontier

Under the structural branch “rank 1 eventually passes,” Computer Science needs
ranks 1 through 45 and Medicine needs ranks 1 through 6. Of these 51 rows, 47
are structurally incompatible with the assumed first candidate. Only four rows
require content:

- `F17/T14430/2003`;
- `F17/T10579/2012`;
- `F27/T12505/1991`; and
- `F27/T12421/1953`.

The branch assumption is not a content decision. If a rank-1 candidate fails
or remains unresolved, the frontier must be regenerated from the next rank.

## Claim boundary

All four required content states remain unresolved. No selection certificate,
episode reselection, title scan, new API request, production disposition,
author-panel reconstruction, pivot classification, outcome read, model fit,
G4 opening, or scientific claim is authorized by this component.
