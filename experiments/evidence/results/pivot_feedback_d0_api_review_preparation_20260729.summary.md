# Pivot-feedback D0 API-assisted review preparation summary

Date: 2026-07-29

Status:
`PASS_INDEPENDENT_D0_API_ASSISTED_REVIEW_INPUT_VERIFICATION_NOT_SUBMITTED`

Two outcome-blind Batch API input files were generated from the frozen
384-title template: 384 requests for each of two pinned model snapshots, for
768 requests total. Each request exposes only target topic, title, language,
and work type. The independent verifier reconstructed every request and
confirmed that the forbidden source and outcome fields are absent.

Two separately ordered human calibration packets were also generated. Each
contains the same deterministic 96-title base sample: eight records from every
one of the 12 episode-by-window strata. The packets contain no labels and hide
episode, window, OpenAlex ID, and API output.

The prepared output root is:

```text
/ssd/pivot-feedback-d0-api-review-v2-prepared-20260729
```

The receipt SHA-256 is
`01186d0d4804a35ff32f63b55f5ac9ef5c17d473cf4bdd2cd8a35aeb78d27373`;
the artifact-tree SHA-256 is
`3dd28d0bfe752515f09b0d171da7abfd4827c9136fe36379bb2199395bcaeb96`.
The external independent-verification report is
`/ssd/pivot-feedback-d0-api-review-v2-prepared-20260729.verification-v2.json`
with SHA-256
`f216da2a1c6bccdf70588c1227d850c0f2e6d62fb0f38b8e1574208698f33fe3`.

No API batch was submitted, no response was received, and no human calibration
was run. Episode 3 remains quarantined; the remaining episodes are not
promoted; and D0, the complete-candidate-universe screen, outcomes, G0, model
fitting, G4, and scientific effects remain closed or unestablished.
