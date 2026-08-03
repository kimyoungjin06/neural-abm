# Pivot-feedback D0 rank-5 API-augmented closeout

Medicine rank 5 is T10151, `Hepatitis C virus research`, at entry year
1993. The pre-sample topic card accepts explicit hepatitis C, HCV,
hepacivirus, or historically specific non-A/non-B hepatitis anchors; generic
or mixed hepatitis remains ambiguous.

## Evidence path

The local rule resolved 39 of the 64 recent titles: 27 accept and 12 reject.
The remaining 25 titles were sent to pinned OpenAI and Claude models using
only an opaque item ID, the target topic and definition, title, language, and
work type. Human review was `NOT_RUN_BY_DESIGN`.

The precommitted API rule was deliberately strict:

- add an accept only when both providers return `ON_TOPIC/HIGH`;
- add a reject only when both return `OFF_TOPIC/HIGH`;
- leave every disagreement, lower-confidence response, `AMBIGUOUS`, or
  `UNREVIEWABLE` response unresolved.

The first Claude call failed before inference because its structured-output
compiler does not support `minItems/maxItems=25`. A transport-only successor
removed those two schema keywords while retaining exact post-response
cardinality verification.

A later provenance check found that the v1/v2 API contracts had mistakenly
named hepatitis B rather than the frozen T10151 hepatitis-C target. That
response pair and its 29/13/22 reduction are preserved for audit but
invalidated completely. They did not create a production disposition or
certificate. The corrected v3 request used a new opaque-ID seed, the exact
v23 hepatitis-C definition, the same 25 source rows, models, labels, visible
fields, and dual-`HIGH` rule.

## Corrected result

The corrected providers returned 25 complete structured decisions each.
Their labels agreed for 17 titles. Exact dual-`HIGH` consensus contributed:

- accept: 0;
- reject: 9;
- unresolved: 16.

The resulting recent-title lower bounds are:

- accept: 27/64 (`0.421875`);
- reject: 21/64 (`0.328125`);
- unresolved remainder: 16/64.

Neither the 32-row accept threshold nor the 32-row reject threshold is met.
The independently verified production state therefore remains
`CONTENT_UNRESOLVED`.

## Why the providers differed

Eight titles had different labels. The disagreements cluster at the intended
title-only boundary:

- broad antiviral, generic viral-hepatitis, post-transfusion-hepatitis, and
  post-transplant liver-disease titles split between `OFF_TOPIC` and
  `AMBIGUOUS`;
- three HIV/transfusion or injection-drug-use titles split because one model
  treated possible HCV co-occurrence as plausible while the other required an
  explicit HCV anchor;
- one explicit HCV-antibody false-positivity assay split
  `OFF_TOPIC`/`ON_TOPIC`, showing a genuine disagreement over whether an
  HCV-focused diagnostic-specificity study is substantively on topic.

Eight further titles received the same non-decisive label. These include
generic hepatitis, mixed HBV/HCV surveillance, blood-safety context, one
corrupted title, and a non-A/non-B hepatitis title. Both models labeled the
last title ambiguous even though the frozen definition treats historically
specific non-A/non-B hepatitis as a positive anchor. The strict consensus
rule correctly prevents such model instruction slippage from becoming a
promotion.

The nine dual-`HIGH` rejects are explicit HBV/HBeAg/anti-HBc, HIV, acyclovir,
general antiviral, bleeding, cirrhosis, or other non-HCV titles. API labels
remain bounded title-only evidence, not expert ground truth.

## Terminal state

Frontier v7 still contains eight production dispositions: three pass, four
fail, and one unresolved. Computer Science retains its field certificate for
ranks 1 and 45. Medicine rank 5 remains an earlier required unresolved row,
so the later rank-6 pass cannot bypass it.

The terminal status is
`BLOCKED_REQUIRED_RANK5_AUGMENTED_CONTENT_UNRESOLVED`. Medicine and global
certificates are not issued, deterministic reselection is not executed, and
outcomes, pivot classification, fitting, G4, and scientific-effect claims
remain closed.

The two successful corrected calls cost an estimated USD `0.019424` from
provider-reported usage. Including the successful wrong-target calls and the
worst-case bound for the unobserved first OpenAI call gives a conservative
cumulative upper bound of USD `0.048650`, below the frozen USD `0.10` cap.
