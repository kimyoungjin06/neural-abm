# Pivot-feedback D0 rank-5 local-abstract closeout

Medicine rank 5 remains T10151, `Hepatitis C virus research`, at entry year
1993. The previous correct-HCV title-only stage left 16 of 64 recent works
unresolved, with conservative lower bounds of 27 accept and 21 reject.

## Local source recovery

The canonical OpenAlex repair-replay snapshot contains a separate
`openalex_works_20260330__excepted__abstract_inverted_index` table on the HDD
backup. The `excepted` name records extraction of the nested field from the
main flat table; it does not mean that every payload is unusable.

The exact 16 unresolved OpenAlex IDs were frozen before the production scan.
The recovery read only the `id` column across all 493 parquet files and read
`id,value` only from row groups containing those IDs. All 16 IDs were found
exactly once:

- reconstructable abstracts: 6;
- mechanically eligible abstracts with at least 20 whitespace-delimited words
  and zero position collisions: 5;
- null abstract payloads: 10;
- unusable three-word fragment: 1.

The prior exploratory inspection had already exposed previews of the six
reconstructable abstracts. Gate v30 therefore explicitly records that this
stage is not content-blind. Eligibility is mechanical only, and no local
semantic label was generated from the previews.

## Bounded provider review

Only the five eligible abstracts were sent to the pinned OpenAI and Claude
models. Each request exposed an opaque item ID, the frozen HCV target
definition, title, language, work type, and abstract. OpenAlex IDs, source
metadata, topic scores, authors, citations, candidate rank, treatment,
outcomes, and downstream effects were excluded.

The precommitted rule was unchanged:

- add accept only for exact `ON_TOPIC/HIGH` agreement;
- add reject only for exact `OFF_TOPIC/HIGH` agreement;
- leave every disagreement or lower-confidence decision unresolved.

The provider labels agreed for three of five abstracts. Exact dual-`HIGH`
consensus added one accept and one reject. Three reviewed abstracts remained
unresolved:

- a historical non-A/non-B blood-safety abstract split
  `AMBIGUOUS/HIGH` versus `ON_TOPIC/HIGH`;
- a broad viral-hepatitis review split only on confidence,
  `ON_TOPIC/HIGH` versus `ON_TOPIC/MEDIUM`;
- a corrupted-title work split `OFF_TOPIC/LOW` versus
  `UNREVIEWABLE/LOW`.

The dual-`HIGH` accept was a multi-infection prevalence study that explicitly
measured HCV. The dual-`HIGH` reject was an HIV/cytogenetics study with no HCV
role in the abstract.

## Final state

The independently verified lower bounds are now:

- accept: 28/64 (`0.4375`);
- reject: 22/64 (`0.34375`);
- unresolved: 14/64.

Neither frozen 32-row threshold is reached. The rank-5 production disposition
therefore remains `CONTENT_UNRESOLVED`. Frontier v8 contains the same eight
production dispositions: three pass, four fail, and one unresolved. Computer
Science retains its certificate for ranks 1 and 45. Medicine rank 5 remains an
earlier required blocker, so the later rank-6 pass cannot issue a Medicine or
global certificate.

The local-abstract provider pair cost an estimated USD `0.0074385` from
provider-reported usage. Adding the prior conservative title-stage bound gives
USD `0.0560885`, below the frozen USD `0.10` cumulative cap.

Human review was `NOT_RUN_BY_DESIGN`. Provider labels are bounded evidence,
not ground truth. No deterministic reselection, pivot classification, outcome
read, model fit, G4 opening, or scientific-effect claim occurred.
