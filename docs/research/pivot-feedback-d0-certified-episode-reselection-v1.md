# Pivot-feedback D0 certified episode reselection

The independently verified global selection certificate names two candidates
per field:

- Computer Science: ranks `[1, 45]`;
- Medicine: ranks `[5, 6]`.

Gate v36 authorized deterministic materialization of exactly those four
measurement-ABI candidate rows. No candidate substitution, content
reinterpretation, outcome access, or panel reconstruction was allowed.

## Source-bound selection

The producer joined the certificate and frontier v10 to the frozen 70,277-row
measurement-ABI candidate table by `(field_id, field_candidate_rank)`. Every
selected row had to match the certificate-bound topic, entry year, and
subfield and had to be `CONTENT_PASS` in frontier v10.

| Episode | Field | Rank | Topic | Entry year | Selection change |
| --- | --- | ---: | --- | ---: | --- |
| 1 | Computer Science | 1 | T14430, Higher Education and Teaching Methods | 2003 | retained |
| 2 | Computer Science | 45 | T10579, Cognitive Radio Networks and Spectrum Sensing | 2012 | retained |
| 3 | Medicine | 5 | T10151, Hepatitis C virus research | 1993 | newly selected |
| 4 | Medicine | 6 | T12421, Adrenal Hormones and Disorders | 1953 | retained |

The prior provisional Medicine rank-1 episode, T12505 at entry year 1991, is
deselected. Overall, three episodes are retained, one is newly selected, and
one prior episode is removed.

All original integer ranking metrics, count metrics, link-rate fields, and
calendar-window fields are preserved in the four-row selection artifact.
The two within-field pairs retain distinct subfields and at least nine years
of entry-year separation.

## Independent verification

The independent verifier did not import the producer. It:

1. reloaded all 70,277 measurement-ABI candidate rows;
2. replayed the fail-closed compatible-pair selection over all 51 frontier
   rows;
3. recovered CS `[1,45]` and Medicine `[5,6]`;
4. compared every source candidate field to the four output rows;
5. confirmed the three-retained/one-new/one-deselected transition;
6. confirmed that source-work rescanning, panel rebuilding, pivot
   classification, outcome reads, and G4 access did not occur.

## Current boundary

Gate v37 records the exact episode reselection as complete and independently
verified. It authorizes the next outcome-blind step only: rebuild the exact
predecision selected-work manifest and author-opportunity panel for these four
episodes under the unchanged measurement ABI.

The previous author panel is superseded provisional evidence and cannot be
reused. Leave-out, network, and bridge construction remain paused until the
rebuilt panel passes independent verification. Production pivot
classification, entry and persistence outcomes, API calls, fitting, G4, and
scientific-effect claims remain closed.

## Frozen artifacts

- reselection root:
  `/ssd/pivot-feedback-d0-certified-episode-reselection-v1-20260730`
- independent verification:
  `/ssd/pivot-feedback-d0-certified-episode-reselection-v1-20260730.verification-v1.json`
