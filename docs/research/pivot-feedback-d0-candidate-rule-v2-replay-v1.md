# Candidate rule v2 replay

Candidate rule v1 missed a specific data-quality failure: the baseline and
signal windows can both be dominated by opaque archival identifiers, leaving
too little score drop for the existing dynamic anomaly test.

Rule v2 adds a simpler first branch:

> If at least 32 of the 64 recent sampled titles are quarantined as
> `OPAQUE_ARCHIVAL_IDENTIFIER`, classify the candidate dry run as an indexing
> anomaly.

This is a sample-usability rule, not a semantic rejection. It does not inspect
the source name, topic score, API label, or human judgment. The retained v1
branches run only when the new majority-opaque branch does not fire.

The five-candidate retrospective replay changes only T10543. Higher Education,
Cognitive Radio, and Adrenal Hormones remain pass; Peptidase remains an
indexing anomaly under the original joint score/source branch; T10543 becomes
an indexing anomaly under the new majority-opaque branch.

Because this amendment was designed after the T10543 diagnostic, the replay is
explicitly not out-of-sample validation. A separate successor gate is required
before production freeze or any new content disposition.
