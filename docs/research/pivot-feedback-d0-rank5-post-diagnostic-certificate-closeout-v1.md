# Pivot-feedback D0 rank-5 post-diagnostic certificate closeout

Medicine rank 5 is T10151, `Hepatitis C virus research`, at entry year 1993.
The preceding external-evidence closeout had frozen 30 accept, 25 reject, and
9 unresolved rows among the 64 recent works. Because neither frozen 32-row
lower-bound threshold was reached, rank 5 was `CONTENT_UNRESOLVED`.

## Why a deterministic amendment was used

The provider disagreements were concentrated in rows for which exact external
titles and PubMed controlled subjects were more specific than either provider's
short rationale. Human review remained `NOT_RUN_BY_DESIGN`. Instead, gate v34
authorized one transparent accept-only replay over all nine unresolved rows.

This amendment was defined after inspecting the disagreement pattern. It is
therefore explicitly:

- retrospective and post-diagnostic;
- not an out-of-sample validation;
- not evidence that API labels are ground truth;
- not evidence that PubMed indexing is semantic ground truth.

The amendment can add an accept only. A failed condition leaves the row
unresolved; it never creates a reject.

## Frozen accept-only rule

A row is accepted only when all four conditions hold:

1. a nonempty exact PubMed PMID link exists;
2. the normalized OpenAlex and PubMed titles agree;
3. the external title explicitly contains `HCV`, `HCV-Ab`, `hepatitis C`, or
   `hepacivirus`;
4. PubMed has a `Hepatitis C` or `Hepacivirus` descriptor for which either the
   descriptor itself or at least one attached qualifier has
   `MajorTopicYN=Y`.

Title normalization is NFKC Unicode normalization, case folding, whitespace
collapse, and terminal-punctuation stripping. The independent verifier parsed
the frozen raw PubMed XML itself rather than importing the producer.

## Exact nine-row replay

The same rule was applied to the six cross-provider unresolved rows and the
three no-new-evidence holdouts.

| OpenAlex work | Short title | Result | Decisive reason |
| --- | --- | --- | --- |
| `W2320892240` | Interferon for Chronic Hepatitis | unresolved | no explicit HCV title marker; HCV subject not major |
| `W2417302102` | Preventing posttransfusion hepatitis | unresolved | HBV controlled subject only |
| `W7132643734` | Viral hepatitis, Lithuanian title | unresolved | no exact PubMed link or explicit target marker |
| `W4236630938` | Viral hepatitis | unresolved | no exact PubMed link or explicit target marker |
| `W7124514286` | HCV-Ab ELISA false-positivity | accept | exact title link, explicit HCV marker, major HCV/Hepacivirus qualifiers |
| `W2402692453` | Viral hepatitis in children | unresolved | HAV/HBV controlled subjects only |
| `W2479898326` | HIV epidemiology and transfusion | unresolved | no exact PubMed link or explicit target marker |
| `W3206943743` | Current antiviral drugs | unresolved | no exact PubMed link or explicit target marker |
| `W2417500009` | HBV/HCV serostatus in drug users | accept | exact title link, explicit HCV marker, major Hepatitis C qualifier |

The exact additions are therefore two accept, zero reject, and seven rows that
remain unresolved.

## Production disposition and certificates

The new conservative lower bounds are:

- accept: 32/64;
- reject: 25/64;
- unresolved: 7/64.

The accept lower bound reaches the frozen 32-row pass threshold, so rank 5 is
`CONTENT_PASS`. The seven unresolved rows are not counted as passes; they may
remain unresolved because the decisive accept lower bound is already met.

Frontier v10 contains four passes, four fails, zero unresolved required rows,
and 43 structurally not-evaluated rows. Replaying the frozen selection contract
produces:

- Computer Science: ranks `[1, 45]`;
- Medicine: ranks `[5, 6]`.

Both field certificates and the global selection certificate are issued.
At this closeout stage, gate v36 authorized but did not execute a deterministic
successor for exactly those four certified episodes. The later independently
verified reselection now materializes those four candidate rows; see
`docs/research/pivot-feedback-d0-certified-episode-reselection-v1.md`.
Author-panel rebuilding, pivot classification, outcome access, model fitting,
G4, and scientific-effect claims remain closed at this certificate boundary.

## Frozen artifacts

- rule replay:
  `/ssd/pivot-feedback-d0-t10151-rank5-external-controlled-subject-rule-v1-20260730`
- rule verification:
  `/ssd/pivot-feedback-d0-t10151-rank5-external-controlled-subject-rule-v1-20260730.verification-v1.json`
- certificate closeout:
  `/ssd/pivot-feedback-d0-rank5-post-diagnostic-certificate-closeout-v1-20260730`
- closeout verification:
  `/ssd/pivot-feedback-d0-rank5-post-diagnostic-certificate-closeout-v1-20260730.verification-v1.json`
