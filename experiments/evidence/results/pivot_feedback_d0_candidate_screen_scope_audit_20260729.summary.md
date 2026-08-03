# Pivot-feedback D0 candidate-screen scope audit

Status:
`PASS_INDEPENDENT_ZERO_COST_CANDIDATE_SCREEN_SCOPE_AND_ROUTER_COVERAGE_AUDIT`

The 70,277 frozen rows are field-topic-entry-year candidates spanning 994
topics. Expanding each candidate into its three baseline and three recent
years produces 421,662 candidate-window rows and 99,433 unique
field-topic-publication-year cells.

Those cells represent 22,437,851 distinct eligible author-linked works.
Because adjacent entry-year candidates overlap, copying work rows once per
candidate window would instead produce 102,900,039 work memberships.

The frozen router has four topic cards. They cover:

- 4 of 994 candidate topics (0.4024%);
- 236 of 70,277 candidate rows (0.3358%);
- 362 unique topic-year cells; and
- 195,708 distinct linked works.

Therefore, the four-card router cannot screen the complete candidate universe.
Materializing a 22.4-million-row title-bearing copy before fixing the screening
design would create a large artifact that the current router cannot use.

The preferred successor is a deterministic adaptive frontier: inspect
candidates in frozen rank order and retain a certificate showing that every
candidate able to alter the selected compatible pair has either a frozen
content disposition or a structural incompatibility. Local rules run first;
cross-provider API review is reserved for ambiguous sampled titles. The
equivalence certificate must be implemented and proven before this strategy
can drive reselection.

This audit read no raw OpenAlex main row, title, work ID, credential, API, or
outcome. It made no content-validity, accuracy, reselection, or scientific
claim.
