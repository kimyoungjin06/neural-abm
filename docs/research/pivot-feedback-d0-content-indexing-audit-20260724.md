# Pivot-feedback D0 content and indexing audit

Date: 2026-07-24

## Why this gate was necessary

The measurement-ABI successor fixed the earlier work-type/linkage mismatch and
reselected four episodes without reading entry or persistence outcomes.
However, a rapid increase in current-snapshot primary-topic counts can reflect
either a scientific transition or a change in what OpenAlex indexed under that
topic. Leave-outs and peer/network measures would make the latter look like a
social mechanism unless content and indexing were checked first.

The v1 audit therefore froze its thresholds before scanning title content. It
selected the 32 lowest SHA-256 work IDs within each episode-window cell,
yielding 384 works across four episodes and three pre-decision windows. It then
scanned only current-snapshot main-work metadata and titles. Entry,
persistence, citation, prediction, and G4 rows remained unopened.

## Automated result

All 384 sampled works were found exactly once. Their publication years,
eligible work types, and current primary-topic IDs matched the measurement-ABI
projection. No title or topic score was missing. The extraction status is:

```text
PASS_D0_CONTENT_INDEXING_EXTRACTION_HUMAN_REVIEW_REQUIRED
```

This status passes data integrity only. It explicitly does not complete D0 or
authorize downstream measurement.

## Episode-level finding

The strongest signal is not a subtle borderline case. In episode 3,
`F27/T12505/1991`:

- the sampled median primary-topic score falls from `0.99695` in the baseline
  window to `0.03435` in the signal window;
- the dominant source share rises from `0.09375` to `0.84375`;
- 27 of 32 sampled signal works come from
  `Digital Kenyon (Kenyon College)`; and
- the dominant titles are identifiers such as `PVN-LOT-033-A-007`, not
  recognizable peptidase-inhibition publications.

The annual author-linked attention also jumps by `4.887x` in 1988 and `9.997x`
in 1990 after previous-year counts of 556 and 592. Taken together, the source,
score, title, and count discontinuities are consistent with archival records
being assigned a low-confidence current primary topic. They do not support
interpreting the episode as a scientific takeoff.

Episode 3 is therefore quarantined. Human double review remains required by the
frozen contract, and the remaining three episodes are not automatically
promoted.

## Next decision

The next outcome-blind successor should:

1. complete and freeze two independent title-review columns plus adjudication;
2. define a versioned candidate-level indexing-contamination exclusion using
   only pre-decision metadata;
3. apply that rule to the complete candidate universe, not just episode 3;
4. rerun deterministic two-field/four-episode selection or fail coverage; and
5. independently replay the new freeze before any leave-out, network, bridge,
   entry, or persistence construction.

The review template is private at:

```text
/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724/content_review_template.csv
```

Its receipt is
`/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724/content_indexing_audit.outputs.json`.

## Claim boundary

This audit demonstrates a current-snapshot indexing threat in one selected
episode. It does not establish that every sampled record is off topic, that the
other episodes are valid, that a historical topic transition occurred, that
people observed peer pivots, or that NABM is necessary. D0 and every downstream
outcome/model gate remain closed.
