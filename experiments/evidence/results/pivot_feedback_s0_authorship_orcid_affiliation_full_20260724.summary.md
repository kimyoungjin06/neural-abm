# S0 authorship ORCID and affiliation projection

Status:
`PASS_S0_FULL_AUTHORSHIP_ORCID_AND_AFFILIATION_PROJECTION_WITH_TOKEN_AND_CURRENT_SNAPSHOT_LIMITATIONS`

The full CPU/I/O run scanned all 1,363 source files and 1,361,551,038 raw
authorship rows. It preserved the single declared `W-1` quarantine, retained
1,361,551,037 rows, and matched every grammar-valid work to the receipt-bound
492,293,206-work lookup. A separate verifier rehashed the compact output and
replayed the global, field-year, and token-multiplicity conservation rules.

Of 458,308,219 non-null ORCID rows, 458,276,860 passed the frozen syntax rule
and 458,275,464 also passed ISO 7064 MOD 11-2. Exactly 8,324,011 current
OpenAlex author tokens had one distinct checksum-valid ORCID and none had
multiple valid ORCIDs. This is not person-identity validation: 35,765 valid
ORCIDs were shared across 78,963 stable tokens, with a maximum of 37 tokens
on one ORCID.

The current snapshot contains 487,064,370 grammar-valid author-work rows with
a nonempty raw affiliation and 442,695,661 with at least one grammar-valid
OpenAlex institution token. There are 103,574 distinct institution tokens
globally and 16,099,316 distinct institution-by-field-year pairs. These
quantities do not establish historical affiliation observability.

This is one S0 component, not an S0 gate pass. The independently verified
coauthor graph component and a three-component additive completion receipt are
still required before D0 can open. G0, final-panel materialization, model
fitting, G4, empirical effects, causal effects, and scientific effects remain
closed or unestablished.
