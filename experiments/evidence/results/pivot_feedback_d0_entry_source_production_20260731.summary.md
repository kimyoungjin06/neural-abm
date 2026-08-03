# Pivot-feedback D0 production entry source v1

Status:
`PASS_INDEPENDENT_EXACT_FOUR_THREE_COUNT_SOURCE_REPLAY_NO_STATE_LABELS`

The production feature source retains all 3,256,271 registered exact-four
author-episode rows. Its eight columns are five private numeric keys and the
three frozen pivot-definition counts. It contains no classifier state or pivot
label.

The producer and independent verifier replayed all 492,293,206 work rows and
779,630,258 author-event rows. Every one of the 4,096 work and 4,096 event
partitions was rehashed. The bounded windows contained 36,138 eligible works,
35,057 author-linked works, and 70,008 distinct author-work pairs. After the
registered-population join, 61,030 pairs remained.

The independent PyArrow/NumPy replay found zero pre-entry, trailing, or focal
feature mismatches. The output key is globally sorted and unique, and all
temporal/count invariants pass.

The source contains 46,911 registered trailing author-work pairs and 14,119
registered focal author-work pairs. These are count-source results, not
classifier states or evidence of persistence, impact, association, causality,
or scientific effect.

No API request was created and no post-entry, persistence, model, or G4 row was
opened. A successor gate is required before applying the frozen four-state
classifier.
