# Pivot-feedback post-primitives critical review and next-stage design

Date: 2026-07-23

Status: `S0_COMPLETE_D0A_STRUCTURAL_FREEZE_PASS_D0_PARTIAL_D0B_ONLY_AUTHORIZED`

Implementation update, 2026-07-23: the first ordered action below is complete.
The additive v2 source-replay verifier independently compared all eight source
projection fields for all 492,293,206 retained works, rehashed the cutoff and
canonical payloads, and observed an unchanged production output tree. Its
tracked compact evidence is
`experiments/evidence/results/pivot_feedback_cutoff_primitives_source_replay_full_20260723.summary.json`.
The gate-order, author-identity, field-structure, and discovery tasks remain.

Second implementation update, 2026-07-23: the gate-order repair is now a
validated versioned amendment at
`experiments/configs/pivot_feedback_gate_sequence_v1.yaml`. Its validator
returns `PASS_ACYCLIC_GATE_SEQUENCE_CONTRACT_ONLY` with 10 nodes, 9 forward
edges, no cycle, and G0.5 as the sole final-panel owner. This validates the
order only; at that point G-1A, S0, and D0 receipts remained unissued or not
run.

Third implementation update, 2026-07-23: the bounded G-1A successor receipt now
passes and authorizes S0 only. The full S0 core scan then rehashed 8,192
normalized input partitions and aggregated all 492,293,206 work rows and
779,630,258 distinct author-work event rows into a verified 26-field-by-year
inventory. S0 itself remains incomplete because ORCID stability,
author-work-affiliation coverage, and distinct coauthor graph topology are
still `NOT_RUN`; D0 and model fitting remain unauthorized. See
`docs/research/pivot-feedback-g-minus-1a-and-s0-core-20260723.md`.

Fourth implementation update, 2026-07-24: the full authorship
ORCID/affiliation projection and all three exact non-ranked graph profiles now
pass independent verification. A separate completion verifier rehashed the
three current output trees and verification reports, passed S0, and opened
only the pre-2013 D0 discovery-calibration gate. The additive v3 ledger keeps
G0, final-panel materialization, model fitting, and G4 fail-closed. See
`docs/research/pivot-feedback-s0-completion-20260724.md`.

Fifth implementation update, 2026-07-24: D0-A froze Computer Science and
Medicine using only eight pre-2013 S0 structural metrics, then selected two
calendar-disjoint current-snapshot primary-topic episodes per field. The
independent verifier rehashed all 4,096 cutoff artifacts, replayed
492,293,206 rows, and reproduced the exact field and episode selection. The
additive v4 ledger records D0 as partial and opens only D0-B outcome-blind
author-opportunity and leave-out construction. Entry and persistence
outcomes, G0, model fitting, and G4 remain closed. See
`docs/research/pivot-feedback-d0a-structural-pilot-freeze-20260724.md`.

Sixth implementation update, 2026-07-24: the first D0-B component constructed
83,565 exact pre-decision target-topic works and 5,849,627 author-episode
opportunity rows under four non-ranked team-cap profiles. The independent
verifier used a separate Arrow `group_by` path, replayed 1,271,923,464 source
rows, and matched all 16,384 author-output row groups. The additive v5 ledger
therefore completes only the author-opportunity foundation and authorizes only
the remaining outcome-blind D0-B measurements. It does not open entry or
persistence. See
`docs/research/pivot-feedback-d0b-author-opportunity-foundation-20260724.md`.

Seventh implementation update, 2026-07-24: the selected-work author-linkage
audit found only 48,245 linked works among 83,565 frozen works. The gap is
decisively heterogeneous: F27 T11361's signal window has 23,073 works but
1,790 linked, while type `other` supplies 22,451 pre-decision works with just
14 linked. The independent verifier replayed both 4,096-partition inputs,
106,148 incidences, and all 1,200 profile cells. The additive v6 ledger now
pauses leave-out/network/bridge work and authorizes only outcome-blind
measurement-ABI redesign plus deterministic episode reselection. See
`docs/research/pivot-feedback-d0b-selected-work-author-linkage-audit-20260724.md`.

## Decision

The production foundations are strong enough to begin a bounded discovery
program, but they are not yet sufficient for a confirmatory empirical run or
NABM fitting.

The next executable work remains CPU/I/O measurement construction, not GPU
training. The ordered program and current state are:

1. independently replay the cutoff-metadata projection from the 493 source
   files;
2. split the circular G-1/G0 dependency into pre-freeze readiness, discovery,
   measurement freeze, and post-freeze panel conformance;
3. build threshold-free field, author-identity, collaboration, and affiliation
   inventories;
4. **D0 measurement review required; leave-outs paused.** The author foundation
   and linkage audit are exact, but the all-work attention universe is not
   stable against author linkage. Freeze a scholarly work-type/linkage ABI,
   recompute the complete candidate universe, and rerun deterministic episode
   selection before incumbent, network, peer, or bridge construction;
5. freeze the final measurements and sealed partitions in a versioned G0
   successor; and
6. materialize and verify the final panel before empirical G1.

The main scientific question should be:

> Does target-specific success among a researcher's prior collaborators predict
> independent entry and later persistence beyond field-wide attention, peer
> activity, and pre-existing access to collaboration or institutional
> resources?

This question has informative failure branches. An increment confined to
coauthored first entries supports recruitment or capacity transfer rather than
an information-channel interpretation. An association explained by field-wide
attention supports common opportunity rather than a network-local mechanism.

## What the current production result establishes

The normalized cutoff-primitives production run passed its bounded contract:

- 493 main-work files and 492,293,207 raw rows;
- one frozen `W-1` quarantine and 492,293,206 retained works;
- exact work-ID closure to the canonical time spine in 4,096 partitions;
- 4,101 receipt artifacts;
- work-metadata tree SHA-256
  `a37448b4df39c00e97b540b3ac7d06de6f2e9ce12186838120d59c820aa89d77`;
- output receipt SHA-256
  `91e3f689e574d524eabbe7daa58f5e678da37f851ba9b7af0561dab57b70846e`;
  and
- read-only output verification report SHA-256
  `703d98eeddb5266cef507ef00400c5cb3700958bf3a1744fc16b6ade2121d485`.

The live repository registry validation returns
`PASS_FAIL_CLOSED_REGISTRY_REPOSITORY_EVIDENCE_ONLY`. It remains
`G_MINUS_1_PENDING`, with `pass_g_minus_1=false` and three missing components:

1. a G0-resolved final cutoff panel;
2. an outcome-blind field-structure audit; and
3. a representative discovery pilot.

The G0 overlay is structurally valid but returns
`VALID_PENDING_G0_MEASUREMENT_FREEZE`, `sealable=false`, and
`execution_authorized=false`.

These passes establish data accounting and deterministic current-snapshot
proxies. They do not establish person identity, historical OpenAlex
observation time, information receipt, belief updating, a final risk set, a
scientific association, or a causal effect.

## Critical findings

### P0. The present gate order is circular

The pending G0 overlay requires a receipt with
`PASS_G_MINUS_1_DATA_READINESS` before G0 can freeze. The current G-1 registry,
however, keeps the final `cutoff_panel` missing until there is a
`G0_RESOLVED_FINAL_PANEL_SUCCESSOR`.

Taken literally, G0 waits for G-1 while G-1 waits for a G0-resolved panel.
This cannot be repaired by promoting either frozen v0 artifact in place.

The outcome-blind field audit has a related dependency problem. Its current
selection metrics include eligible-author count, matured opportunities,
fixed-horizon coverage, and independent-shock count. Those quantities require
eligibility, maturation, and shock definitions that are still unresolved in
G0. Threshold-free structural coverage and post-freeze analytic support must
therefore be separate products.

### P0. “Independent verification” does not replay source metadata

The cutoff verifier rehashes all output artifacts, scans all output rows,
replays the work-ID and canonical-time join, validates type and null margins,
and checks the bound topic and citation parents. It does not accept the
original main-work source root and does not independently compare the source
values for:

- work type;
- `is_paratext`;
- `is_retracted`;
- declared author count; or
- current primary topic, field, and domain.

Consequently, a producer defect that attached valid source metadata to the
wrong valid work IDs could preserve global margins and canonical-time closure.
The current verification is correctly described as full output and
canonical-time verification, not an independent source-projection replay.

Resolution update: this gap is now closed by
`scripts/verify_pivot_feedback_cutoff_primitives_source_replay_v2.py`. The
finding remains here because it explains why the additive verifier was
required. The v2 full result compared all retained rows and fields exactly.
It used the content-freeze SHA checkpoints plus unchanged rich file identities;
it did not repeat the optional full-byte source rehash.

### P0. Author tokens are not yet a stable scientific unit

The author-history component establishes current-snapshot OpenAlex author-token
accounting:

- 779,630,258 distinct author-work pairs;
- 776,803,175 temporally usable pairs; and
- 112,897,930 tokens with usable history.

It deliberately does not establish natural-person identity or split/merge
truth. This is substantively important: a token split can look like entry by a
new researcher, while a merge can look like a long-distance pivot.

The source can support a bounded robustness audit. The 1,361,551,038-row
authorship table includes author ID, ORCID, raw name, author position,
corresponding-author status, countries, affiliations, and institution IDs.
The completed source audit counted 458,308,219 non-null ORCID authorship rows;
this is a row count, not a distinct-person count or ground truth.

The primary estimand should remain explicitly
`CURRENT_SNAPSHOT_OPENALEX_AUTHOR_TOKEN_TRAJECTORY`. A separately frozen
ORCID-stable subset and split/merge-risk sensitivity must be required before
promoting a researcher-level interpretation.

### P0. Signal and capacity are not yet structurally separable

The peer registry defines `peer_visible_success` as a subset of `peer_active`;
both use pre-existing collaborators. The capacity bridge allows a pre-existing
active collaboration or local resource relation with a target-experienced
peer. If “active collaboration” is operationalized as the same relation that
defines the peer set, `PUBLIC_SUCCESS_ONLY` can become impossible and the
four-cell contrast can be mechanically collinear.

G0 must create independent rules:

- `prior_peer_set`: any qualifying collaborator in the frozen pre-entry graph
  window;
- `recent_collaboration_bridge`: a stricter, recent pre-cutoff collaboration
  relation, defined without the focal entry work or success outcome; and
- `same_affiliation_bridge`: a work-time affiliation relation known at the
  cutoff.

The recent-collaboration bridge should be the primary high-coverage proxy.
Same-affiliation should remain a separately reported secondary proxy until its
coverage and temporal semantics pass. Their union must not be used as the
primary bridge unless all four signal-by-bridge cells have adequate support.

The raw source makes this extension feasible. Besides nested authorship
affiliations, it contains 204,773,216 rows in the corresponding-institution
table and 272,766,845 rows in the corresponding-author table. Those surfaces
are not present in the current normalized author-event output, so a new
receipt-bound author-work-affiliation product is required.

### P1. Entry mode is an outcome type, not a baseline stratum

Whether a first target work is chaperoned is observed only when entry occurs.
Conditioning an entry-hazard model on chaperoned versus independent entry would
condition on a post-decision outcome.

For entry, use competing outcomes:

1. no qualifying entry;
2. chaperoned first entry; and
3. independent first entry.

For persistence among entrants, entry mode may define descriptive
heterogeneity, but the resulting contrast is selected on entry and is not a
causal subgroup effect. This distinction must be explicit in the estimand and
reporting language.

### P1. Historical observability remains unidentifiable

Current topic membership is a retrospective annotation. Citation
`available_at` is the citing work's publication time, not the date OpenAlex or
the researcher first observed the citation. These surfaces can code a
bibliographically available proxy at a cutoff, but cannot establish message
receipt or belief updating.

The empirical ladder must therefore remain:

```text
field-wide attention
        |
        v
prior peer target activity
        |
        v
prior peer bibliographically visible success proxy
        |
        +--------------------+
        v                    v
recent collaboration   same-affiliation
capacity proxy         capacity proxy
        |                    |
        +---------+----------+
                  v
      chaperoned vs independent entry
                  |
                  v
        fixed-horizon persistence
```

A surviving independent-entry increment can be called compatible with a
network-local information channel. It still cannot be called observed social
learning or a causal influence effect without an external causal axis.

## Revised gate sequence

The immutable v0 contracts should remain unchanged. A versioned successor
should adopt the following acyclic sequence:

```text
G-1A  primitive readiness
      canonical time + author tokens + topics + citation proxy
      + cutoff primitives + source-projection replay
                         |
                         v
S0    threshold-free structural inventory
      field-year coverage + author-token/ORCID coverage
      + graph support + affiliation coverage
                         |
                         v
D0    discovery-only pilot, entry year <= 2012
      provisional measurements, 2 fields x 2 episodes
      no final field choice and no G1 claim
                         |
                         v
G0    versioned measurement freeze
      quantitative definitions + final structural selection rule
      + atomic G1/model-selection/G4 seals
                         |
                         v
G0.5  frozen-panel conformance
      materialize final panel + leakage/source replay
      + eligibility/episode/cell support audit
                         |
                         v
G1    empirical replication, entry years 2013-2016
                         |
                         v
G2    explicit baseline and model selection, 2017-2019
                         |
                         v
G3    NABM only if the baseline ladder leaves a predictive gap
                         |
                         v
G4    sealed temporal/domain/author-cold holdout, 2020-2022
                         |
                         v
G5    external causal design, if available
```

Under this sequence, the current normalized cutoff product satisfies part of
G-1A. The final analytic cutoff panel moves to G0.5. D0 precedes G0 because its
only legitimate purpose is measurement calibration and feasibility; it cannot
select fields by effect size, change confirmatory years, or pass an empirical
claim.

## Source-replay verifier design

The next engineering component should be a v2 verifier, leaving the v1
verifier and production output immutable.

Required properties:

1. accept the exact 493-file source root, the cutoff output root, the canonical
   root, and the content receipt;
2. import no producer module;
3. validate the source task list and source stat/content bindings before
   reading data pages;
4. project the same eight source columns independently;
5. parse IDs and null states with separately implemented code;
6. spool records by `work_id mod 4096`, sort them, and compare every projected
   field and null mask to every output row;
7. re-check the frozen `W-1` quarantine and global uniqueness;
8. change no byte under the production output root; and
9. emit a small external receipt with source, output, config, and result
   fingerprints.

A full pass would close the principal remaining engineering gap in the cutoff
foundation. It would not resolve author identity or historical observability.

## Threshold-free structural inventory

S0 should avoid any quantity that requires a G0 threshold. For every one of the
26 fields and publication year, record:

- total canonical works and usable-date coverage;
- primary-topic and all-topic relation coverage;
- usable author-token and author-work-event counts;
- ORCID-linked authorship-row coverage and candidate ORCID-stable token
  coverage;
- coauthor edge counts, degree quantiles, and connected-component support;
- citation follow-up calendar support, without labeling success;
- author-work affiliation and institution coverage; and
- counts of raw attention trajectories at multiple declared, non-ranked
  resolutions.

Do not compute or use entry effects, persistence effects, overshoot gaps,
predictive scores, G4 outcomes, or a selected “best” shock threshold.

After G0, G0.5 may add the threshold-dependent quantities currently mixed into
the field-selection contract: eligible authors, matured opportunities,
fixed-horizon coverage, independent episodes, and signal-by-bridge cell
support.

## Discovery-only empirical pilot

### Population and period

- Use entry years through 2012 only.
- Use two fields and two structurally independent attention episodes per field.
- Choose pilot fields by threshold-free coverage and deterministic numeric
  tie-breaks, never by effect size.
- Freeze a pilot-only measurement file before inspecting the pilot outcomes.
- Do not use G1, model-selection, or G4 rows to tune any choice.

### Observable constructs

For candidate author token \(i\), target topic \(q\), and entry year \(t\):

- field attention uses incumbent works available by \(t-1\), excluding the
  focal candidate and registered local-network leave-outs;
- peer activity asks whether a qualifying prior peer had target-topic activity
  by \(t-1\);
- peer success uses only citation edges bibliographically available by \(t-1\)
  and a fixed maturation convention;
- recent-collaboration and same-affiliation bridges use only relations existing
  by \(t-1\); and
- entry is the first qualifying target-topic work, represented as no entry,
  chaperoned entry, or independent entry.

The primary persistence outcome uses target-topic activity in \(t+1\) through
\(t+3\), after the horizon and work-type rules are frozen.

### Nested empirical ladder

Estimate the same scoring rows under:

1. field attention and pre-entry author history;
2. plus peer activity;
3. plus peer visible-success proxy conditional on peer activity;
4. plus the two capacity proxies and their pre-specified interactions; and
5. competing chaperoned and independent entry outcomes, followed by
   fixed-horizon persistence among entrants.

The main discovery contrast is the visible-success increment over peer
activity, not the raw exposed-versus-unexposed difference. The scientifically
most discriminating readout is whether that increment appears for independent
entry and persists after capacity-proxy adjustment.

The pilot should also produce cohort timing curves for entry and three-year
persistence. It is evidence for continuing only if the data can distinguish
an early entry response from later sustained participation; it is not a
confirmatory H3 test.

### Required controls

Retain all seven registered controls:

- lead attention;
- the same peer's unrelated-topic success;
- post-decision future citation success;
- future ties;
- degree-preserving shuffled networks;
- matched non-neighbors; and
- matched non-surge topics.

Add two measurement sensitivities:

- ORCID-stable author-token subset; and
- recent-collaboration versus same-affiliation bridges reported separately.

## Decision branches and stop rules

| Discovery result | Next interpretation or action |
| --- | --- |
| Source-replay mismatch | Stop. Repair the projection or verifier before any field audit. |
| No two fields support two independent episodes each | Stop the cross-field route; retain engineering evidence only. |
| Independent-entry cells are empty or suppressed | Do not claim an information channel; redesign the population or study recruitment only. |
| Signal and capacity proxies are structurally collinear | Abandon the four-cell mechanism contrast or redefine the bridge prospectively. |
| ORCID-stable sensitivity materially reverses the discovery pattern | Treat author identity as blocking, not a footnote. |
| Field attention explains the association | Interpret as common opportunity; do not claim a network-local mechanism. |
| Peer activity adds signal but visible success does not | Interpret as collaborator activity, recruitment, or capacity association. |
| Visible success appears only for chaperoned entry | Interpret as recruitment, co-production, or capacity transfer. |
| Future citation or future-tie controls predict as strongly as valid past proxies | Treat the feature construction as temporally invalid. |
| Independent visible-success increment survives controls | Proceed to G0 with the bounded phrase “network-local information-channel compatible.” |
| No reproducible entry-before-persistence timing gap | Stop the feedback-overshoot route even if a local peer association remains. |
| Explicit baselines match the later NABM | Use the simpler model and drop neural-necessity language. |

Quantitative margins for “material,” “adequate,” “collinear,” and “match” must
be resolved from D0 and external literature in the G0 successor. They must not
be selected on G1 or G4 performance.

## Compute and deliverables

The immediate stages are CPU/I/O jobs:

| Stage | Main compute | Durable output |
| --- | --- | --- |
| Source replay | HDD read + CPU parsing + SSD spool | v2 verification receipt |
| Author/ORCID audit | CPU/I/O | token-ORCID multiplicity and sensitivity receipt |
| Affiliation audit | CPU/I/O | author-work-institution surface and coverage receipt |
| S0 field inventory | CPU/I/O | threshold-free 26-field cube |
| D0 pilot | CPU, with optional parallel fitting | discovery report and G0 decision ledger |
| G0.5 final panel | CPU/I/O | frozen panel, exclusions, weights, leakage receipt |
| G1 baselines | CPU first | empirical replication and baseline scores |
| G3 NABM | GPU only after promotion | fitted model and paired simulation receipts |

GPU-server work is premature until G1 establishes a reproducible signature and
the explicit baseline ladder leaves a pre-specified predictive gap.

## Immediate implementation order

1. **Completed 2026-07-23.** Add the independently implemented cutoff
   source-replay verifier and adversarial tests that permute one metadata field
   across valid work IDs.
2. **Completed 2026-07-23 as a prospective contract.** Add a versioned
   gate-sequence amendment implementing G-1A, S0, D0, G0, and G0.5 without
   mutating the v0 contracts. Execution receipts remain future work.
3. Build a footer/schema/coverage audit for ORCID, authorship affiliations,
   corresponding authors, and corresponding institutions.
4. Materialize only the minimum new normalized surfaces needed for S0:
   author-token-to-ORCID evidence and author-work-institution events.
5. Build the threshold-free field-year inventory and verify that it never
   reads registered outcome columns or sealed partitions.
6. Freeze and run the two-field discovery pilot.
7. Use the pilot only to resolve the quantitative G0 successor, then seal the
   final field/episode/author partitions before G0.5 and G1.

The first implementation target is item 1. It has a bounded acceptance
criterion, strengthens an already completed production component, and does not
require choosing a scientific threshold.
