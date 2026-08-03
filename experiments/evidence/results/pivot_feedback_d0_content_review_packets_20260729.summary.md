# Pivot-feedback D0 blinded content-review packet summary

Date: 2026-07-29

Status:
`PASS_INDEPENDENT_D0_BLINDED_TWO_HUMAN_REVIEWER_PACKETS_REVIEW_NOT_RUN`

Two title-only review packets were generated from the frozen 384-row content
audit template. Each packet contains all 384 items, uses a reviewer-specific
opaque item ID, and has a different deterministic order. Episode, window,
source, topic score, publication year, and OpenAlex work identifiers are
hidden from reviewers.

The independent verifier reproduced both packet orders and item IDs from the
source template, matched the private adjudication key exactly, confirmed mode
`0700` for the output directory and `0600` for every file, and imported no
producer module.

The output root is:

```text
/ssd/pivot-feedback-d0-content-review-packets-v1-20260729
```

The receipt SHA-256 is
`8938d35c28ae3175a416dd0ef223103376d127bd01ddde7b5e9913299304867a`;
the artifact-tree SHA-256 is
`36552e421cd80847d41f5787ca3294059f1a7114d87464747f1d9ab95891610d`.

Both reviewer label columns remain empty for all 384 items. Human review,
adjudication, the contamination-rule successor, D0 completion, downstream
measurement, outcomes, G0, model fitting, G4, and scientific effects remain
not run, closed, or unestablished.
