# Pivot-feedback D0 rank-5 external-evidence closeout

Medicine rank 5 remains T10151, `Hepatitis C virus research`, at entry year
1993. The local-abstract stage had established conservative lower bounds of 28
accept, 22 reject, and 14 unresolved among 64 recent works.

## Exact external collection

Gate v32 froze the fourteen unresolved OpenAlex IDs before network access. The
collector queried only three public scholarly-data hosts:

- current OpenAlex work records for all 14 IDs;
- exact Crossref DOI records for 7 works;
- exact PubMed records for 8 works.

Raw responses were preserved. The normalized packet contained no OpenAlex ID,
DOI, PMID value, author, year, citation, candidate rank, treatment, outcome, or
downstream-effect field.

Eleven works received new decision-relevant evidence: an external abstract,
PubMed controlled subjects, a Crossref-corrected title, or an exact cross-source
title with target-relevant terms. Three works were held out because external
sources added no new decision-relevant text or controlled subjects:

- `Principais drogas antivirais na atualidade`;
- `Viral hepatitis`;
- `Kai kurie virusinio hepatito išplitimo ir vėlyvųjų baigčių aspektai`.

Absence of a PubMed controlled subject was never treated as proof of exclusion.

## Cross-provider result

The eleven-row packet was submitted once to pinned OpenAI and Claude models
under the same dual-`HIGH` rule. No retry or human review occurred.

The two providers added:

- 2 accept: the non-A/non-B antigen-isolation study and the Scandinavian
  non-A/non-B transfusion-risk study;
- 3 reject: post-transfusional AIDS, fulminant-hepatic-failure transplant
  management, and liver disease after renal transplantation;
- 6 reviewed but unresolved.

The unresolved reviewed rows include three especially informative provider
disagreements:

- `Interferon for Chronic Hepatitis`: `OFF_TOPIC/HIGH` versus
  `AMBIGUOUS/MEDIUM`, despite PubMed subjects including hepatitis C;
- the HCV-antibody false-positivity study: `OFF_TOPIC/MEDIUM` versus
  `ON_TOPIC/HIGH`, despite HCV-specific title and controlled subjects;
- the HBV/HCV serostatus study: `OFF_TOPIC/HIGH` versus `ON_TOPIC/HIGH`,
  despite HCV-specific title and controlled subjects.

These disagreements show that external metadata reduced missingness but did not
make provider labels ground truth.

## Preserved-response transport normalization

Both provider calls completed. The initial reducer then stopped because ten
Claude rationales exceeded the frozen 180-character persistence limit. Raw
responses were preserved and no retry was made. A deterministic transport-only
finalizer retained every `item_id`, label, and confidence unchanged and stored
only the first 180 Unicode characters of overlong rationales. The independent
verifier reconstructed all labels, confidence values, truncation flags, and
routes from the raw responses.

## Final production state

The independently verified conservative lower bounds are:

- accept: 30/64;
- reject: 25/64;
- unresolved: 9/64.

Neither 32-row threshold is reached, so rank 5 remains `CONTENT_UNRESOLVED`.
Frontier v9 still contains three passes, four fails, one unresolved row, and 43
not-evaluated rows. Computer Science retains its certificate for ranks 1 and
45. Medicine rank 5 remains an earlier required blocker, so the later rank-6
pass cannot issue a Medicine or global certificate.

The external provider pair cost an estimated USD `0.01450475` from
provider-reported usage. Adding the prior conservative bound gives USD
`0.07059325`, below the authorized USD `0.12` cap.

Human review was `NOT_RUN_BY_DESIGN`. No deterministic reselection, pivot
classification, outcome read, model fit, G4 opening, or scientific-effect claim
occurred.
