# Pivot-Feedback Cascade Study Protocol

Protocol version: 0.4

Status: prospective science-first study charter; no substantive empirical
claim has passed yet. This is not an executable preregistration until G0
freezes the quantitative decision rules listed below.

Date: 2026-07-21

This protocol governs the prospective high-selectivity route built around
researcher topic mobility. It does not replace the current method-first paper
outline or turn the existing hand-set researcher-pivot case study into
empirical evidence. The existing case study is a hypothesis generator until
the data, measurement, baseline, and holdout gates below pass.

The peer-mechanism component is specified in
`docs/research/peer-exposure-identification-design.md` and the prospective
machine-readable registry at
`experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml`. Both are
`PENDING_G0_MEASUREMENT_FREEZE`; neither authorizes execution or establishes a
peer effect.

## 1. Plain-Language Research Question

When a scientific area becomes highly visible, successful early entrants can
become visible before unsuccessful attempts are observed. Do researchers then
continue entering the area after later entrants' probability of persistence or
success has already begun to fall?

The proposed feedback sequence is:

```text
attention surge
    -> prepared early entrants
    -> publicly available early success
    -> network-local opportunity for receipt and later entry
    -> entry continues while later-cohort persistence declines
    -> abandonment or return becomes visible only after a delay
    -> entry finally adjusts
```

The study asks whether this delayed adjustment exists in observed scientific
mobility, whether pre-entry capacity-bridge proxies distinguish researchers
with comparable network-local public-availability proxies, and whether an
adaptive model adds predictive value beyond simpler explanations. OpenAlex can
construct bounded proxies for public availability and published entry, but not
whether a researcher received a signal, believed it, or changed a decision rule
because of it.

The cross-field transport question is whether the same timing gap survives in
scientific fields with different pre-existing publication, collaboration, and
outcome-revelation structures, and whether a pre-surge ratio of bibliographic
outcome-availability lag to prior-collaborator exposure-opportunity timescale
predicts the size of that gap. Field is a transport and effect-modification
axis, not an assigned treatment. No field contrast by itself identifies a
causal field effect.

Final empirical field selection remains `PENDING`. The protocol must not name
final selected fields until the completed current-snapshot taxonomy-reference
component is coupled to parent content identity and frozen as an authoritative
binding, and the author-entity, temporal-availability, and outcome-blind
structural coverage audit have passed. Neither the work-source profile nor the
later bounded reference component establishes broader field-entity validity or
an eligible researcher panel.

## 2. Science-First Claim Boundary

The primary scientific object is the timing and composition of researcher
entry into a newly attractive topic. The framework is an instrument for
testing the object, not the headline result.

The study may eventually support the following bounded claim:

> Publicly available early outcomes, delayed feedback, and collaboration
> structure jointly predict when topic entry overshoots later-cohort
> persistence, and pre-existing capacity-bridge proxies condition that
> boundary.

It must not begin from any of the following claims:

- that topic switching is generally good or bad;
- that publication growth is equivalent to hype;
- that every one-off topic entry is a failed research attempt;
- that public availability proves receipt, attention, belief revision, or
  learning by a researcher;
- that collaboration exposure is causal social influence;
- that a grant acknowledgement identifies a causal funding effect;
- that a neural policy weight is a causal mechanism;
- that only a Neural ABM can generate an information cascade; or
- that a simulation counterfactual is an estimated policy effect.

Classical rule-based, Bayesian, evolutionary, and reinforcement-learning ABMs
can all represent adaptive entry and delayed feedback. NABM earns a place only
if its adaptive heterogeneous micro-rule improves frozen holdout prediction or
robustly changes an intervention boundary after fair comparison with simpler
models. Even then, predictive improvement does not establish that real
researchers learned through the represented channel.

## 3. What Is New Relative to Existing Findings

Prior work already establishes several adjacent facts:

- larger research pivots are associated with lower subsequent impact and
  publication success;
- prior collaborators active in a topic predict later topic entry;
- switching patterns vary by career stage and have changed over time; and
- selective disclosure can change social learning and experimentation.

Those results do not by themselves establish the proposed sequence. This study
targets four additions:

1. **Temporal ordering.** Test whether entry continues to rise after
   later-cohort persistence has begun to decline.
2. **Endogenous composition.** Test whether publicly available early success
   predicts who enters later, rather than treating pivot cost as fixed across
   cohorts or assuming that availability was received as a signal.
3. **Activity, success, and capacity proxies.** Test whether a prior
   collaborator's target-topic activity, the incremental public availability of
   that collaborator's target-topic success, and an active pre-entry tie or
   shared resource carry separable predictive information. Distinguish entry
   with the source peer from entry without that peer, without claiming that
   bibliometric traces identify receipt, belief revision, recruitment, or
   learning.
4. **Intervention timing.** Map when faster outcome disclosure, structural
   bridge support, or reduced social amplification changes the overshoot
   boundary without assuming that more disclosure or less exploration is
   uniformly beneficial.

## 4. Prospective Hypotheses

The hypotheses are directional but do not assume that the mechanism exists.
They must be frozen before the confirmatory corpus is inspected.

### H1: Attention and Entry

An incumbent-only, leave-out measure of target-topic attention acceleration
predicts subsequent entry after controlling for researcher history, topic-year
conditions, and pre-existing network position.

### H2: Network-Local Activity and Public Success

A pre-existing collaborator's target-topic activity is the required baseline.
The registered H2 contrast tests whether that collaborator's time-eligible,
publicly available target-topic success adds entry prediction beyond peer
activity, the incumbent-only topic-attention signal, and matched non-neighbor
proxies. This is a public-availability contrast, not evidence that the focal
researcher received the information or updated a belief. Entry with the source
peer and entry without that peer must be reported separately.

### H3: Cohort Overshoot

After an attention surge, entry reaches its peak after later-cohort persistence
or other pre-specified outcome measures have begun to decline. The primary
signature is therefore a timing gap, not merely a high entry count.

### H4: Pre-Existing Capacity-Bridge Proxy

Among researchers with comparable network-local public-success proxies, a
documented active pre-entry tie or institutional resource connection to a
target-experienced peer is associated with higher later-cohort persistence than
no such pre-entry proxy. This is an associational contrast. A tie first observed
after entry is a post-treatment descriptive outcome, not the primary comparator.

### H5: Feedback Timescale

The overshoot gap increases as bibliographic outcome availability becomes slow
relative to prior-collaborator exposure opportunity. A simulation phase
boundary may be summarized by

```text
R = bibliographic outcome-availability lag
    / prior-collaborator exposure-opportunity timescale
```

only if the empirical and simulation definitions of both timescales are fixed
in advance and the boundary transports across held-out domains.

The empirical diffusion denominator measures how rapidly public outcome
availability becomes linked to eligible researchers through pre-existing ties;
it does not observe messages being received. A simulator may represent message
receipt and belief updates explicitly, but those latent stages cannot be renamed
as empirical measurements.

The shock-level ratio `R` is a secondary transport endpoint, not a co-primary
way to pass H3. It tests whether pre-surge timing information orders the
subsequent overshoot gap across fields and held-out episodes. It does not turn
field differences into causal effects, and it cannot rescue a failed primary
timing-gap test.

### H6: Added Value of Adaptive NABM

An adaptive NABM improves temporal and domain holdout prediction over a fixed
hazard model, a fixed-rule ABM, and an explicit adaptive baseline. If it does
not, the neural-necessity claim is rejected and the simpler model becomes the
scientific model.

## 5. Observable Constructs

### 5.1 Unit of Analysis

The primary unit is a researcher-topic entry opportunity at time `t`, nested
within researcher, topic, field, and calendar time. Inclusion eligibility must
use information available strictly before `t`.

### 5.2 Pivot or Topic Entry

The primary definition must use past-only representations. Candidate primary
and robustness definitions are:

1. distance between a new work and the researcher's trailing three-year topic
   or text portfolio;
2. distance between the focal work's reference-source distribution and the
   researcher's prior distribution;
3. entry into a primary topic previously absent or negligible in the trailing
   portfolio; and
4. a publication-time text-embedding displacement.

OpenAlex topic labels derived from a later full snapshot can leak future
citation structure. They may be used for a robustness construction, but the
primary temporal measure must not depend on post-entry information.

### 5.3 Attention

The OpenAlex-only construct is `topic attention acceleration`, not `hype`. The
primary index must use information available by `t-1` and growth generated by
incumbent target-topic researchers whose entry predates the signal window. It
must exclude the focal researcher, the focal researcher's pre-existing
neighborhood, and publications by researchers whose entry is an outcome in the
same window. Focal or follower entry therefore cannot mechanically create its
own upstream attention signal. Lagged citation or publication variants may be
reported separately, with lead-placebo and alternative-window checks.

A claim specifically about hype requires an external attention series such as
news, search, social, preprint-view, or announcement data.

### 5.4 Network-Local Public-Availability Proxies

Every network-local construct must be based on ties observed before the focal
entry decision. The empirical contract separates four stages:

```text
publicly available outcome
    -> receipt or access by the focal researcher
    -> belief or decision-rule update
    -> published topic entry
```

OpenAlex can support a bounded time-eligibility proxy for the first stage and
observes the last stage only when a work is published. A prior tie is an
opportunity-for-contact proxy; it does not reveal receipt, attention, trust, or
belief revision. Accordingly, artifacts may retain a legacy `peer_exposure`
field name for compatibility, but all registries, tables, and claims must label
it as a network-local public-availability proxy rather than observed exposure or
learning.

Two nested predictors are required. `Peer active` records whether a prior
collaborator was already active in the target topic. `Public success increment`
records only the additional time-eligible target-topic outcome information that
was publicly available by `t-1`. H2 tests the second predictor conditional on
the first, incumbent-only field attention, researcher history, network position,
and matched non-neighbor proxies. Matched non-neighbor and shuffled-network
controls plus pre-tie similarity diagnostics remain required.

The focal researcher's first target-topic work is also classified as
`chaperoned` when it includes a collaborator who supplies the registered
peer-active or public-success proxy, and `independent` when it includes none of
those source peers. These are observed entry modes, not latent motives. An
association confined to chaperoned entry is labeled as compatible with
recruitment, co-production, or capacity transfer rather than peer learning. An
association that survives for independent entry is consistent with a
network-local response to public information, but still does not identify
receipt, belief updating, or causal influence.

The design treats field-wide attention or common opportunity,
selection/homophily into prior collaboration, and recruitment or capacity
transfer as competing explanations. No peer-feedback interpretation is
promoted unless the registered contrasts separate the network-local proxy from
these alternatives; observational separation is not causal identification.

### 5.5 Capacity Bridge

A pre-existing capacity-bridge proxy requires an active collaboration or local
institutional/resource relation with a target-experienced peer, all observed
before the focal researcher's target entry. It is not defined merely by the
peer's visible success or by a focal target-topic work that would itself define
entry. No focal work used to construct this proxy may satisfy any primary or
robustness entry definition. The primary design codes `neither`,
`public-success proxy only`, `capacity proxy only`, and `both`, includes their
interaction, and audits cell eligibility before estimation. Peer activity
remains a nested baseline, and chaperoned versus independent entry remains a
separate observed entry-mode split. If sparse cells, collinearity, or entry
leakage prevent separation, the channels must be collapsed and the separate
public-success-versus-capacity claim abandoned.

### 5.6 Outcomes

OpenAlex cannot observe an attempted pivot that produces no public work.
Accordingly, the empirical study must not label all unobserved or discontinued
activity as failure. Primary outcomes remain separate rather than being
collapsed into one hand-set productive threshold:

- target-topic persistence at a pre-specified horizon;
- one-off entry;
- return to the prior portfolio;
- publication interruption;
- change from the researcher's pre-entry publication trajectory;
- field-year-normalized citation or placement measures;
- subsequent collaboration in the target topic; and
- novelty or atypical knowledge combination, where available.

The terms `post-entry abandonment` and `non-persistence` are allowed for these
observables. `Failed pivot` is reserved for a design that observes the attempt
and defines failure independently of publication visibility.

### 5.7 Cross-Field and Shock Panel

The prospective, machine-readable cross-field contract is
`experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml`. Until the G-1
entity and temporal audits pass and G0 writes and hashes that contract, it is a
prospective path rather than evidence that a field panel exists.

The field universe must come from a versioned authoritative taxonomy reference
bound to the analytical snapshot. Selection then uses only information from
the discovery period and only the following outcome-blind structural axes:

- eligible opportunity and time-usable author coverage;
- publication-date and fixed-horizon outcome maturation coverage;
- team-size and pre-entry collaboration-network structure;
- topic concentration and turnover; and
- pre-surge outcome-revelation and network-local-signal-diffusion timing.

The G0 implementation standardizes the frozen axes, applies an outcome-blind
maximin rule, and selects four structurally contrasting fields. Ties are broken
by stable numeric field ID. The remaining eligible fields form a deterministic
reserve list by the same maximin score and tie-break. A selected field may be
replaced only for a pre-registered coverage or integrity failure observed
before any confirmatory entry, persistence, or timing-gap result is inspected.
The final field IDs, reference hash, structural inputs, standardized matrix,
selection trace, holdout assignment, and reserve order must be written to the
cross-field contract. No final field names may appear in a result or manuscript
before that freeze.

The full matrix contains three G1 fields and one sealed G4 domain field. The G4
field is assigned by the frozen outcome-blind rule, not by observed model or
scientific performance. Each field must contain at least three independent
attention-surge episodes. G0 fixes episode independence using non-overlapping
topic families, a maximum eligible-author overlap, and a shock-family rule. If
a field cannot supply three eligible episodes, it fails coverage and the next
frozen reserve is considered before outcomes open. Relabeling one event into
several topics does not create independent episodes.

### 5.8 Cohort Clocks and Atomic Splits

The primary clock is annual. For an entry opportunity at year `t`, the default
candidate windows to be frozen or replaced at G0 are:

- researcher portfolio: `t-3` through `t-1`;
- active collaboration graph: `t-5` through `t-1`;
- incumbent-only attention acceleration: `t-3` through `t-1` relative to
  `t-6` through `t-4`;
- entry decision: year `t`;
- target-topic persistence: at least one eligible target-topic work from
  `t+1` through `t+3`;
- shock-response cohorts: event time `k = 0, 1, 2`; and
- event-study pretrend: event time `k = -3, -2, -1`.

Every feature must satisfy its own `available_at <= t-1` rule. A label may
mature after the decision year, but it must never become a feature, threshold,
normalizer, selector, or tuning signal for an earlier decision or another
partition.

Authors, related topic families, and all response cohorts belonging to one
shock episode are assigned atomically. An episode whose response cohorts would
cross a frozen discovery, G1, model-selection, or G4 boundary is wholly assigned
under the registered episode rule or excluded; it is never split to enlarge a
sample. The provisional calendar partitions in Section 7 are decision-cohort
partitions and do not override this atomicity rule.

### 5.9 Confirmatory and Negative-Control Endpoints

H3 has one confirmatory primary endpoint. For each shock episode `j`, define

```text
delta_j =
    earliest argmax over k of adjusted entry hazard
    - earliest argmax over k of adjusted three-year persistence probability,
    where k is 0, 1, or 2.
```

The earliest maximum resolves a frozen flat peak without analyst discretion.
Fields receive equal weight, and episodes within a field receive equal weight.
The G0 registry freezes adjustment, uncertainty, missing-cohort, flat-peak, and
minimum-gap rules. No alternative endpoint may substitute for `delta_j` after
confirmatory labels open.

`R`, the pre-surge bibliographic outcome-availability lag divided by the
prior-collaborator exposure-opportunity timescale, is the pre-specified
secondary transport endpoint. Its estimators use only pre-surge, time-eligible
information. It is evaluated as an ordering or predictive-modification test
across episodes and then on the held-out domain; it is not interpreted as a
causal field coefficient or a measured cognitive diffusion rate.

The required negative-control endpoints are:

1. lead-attention exposure at `t+1` or `t+2` predicting entry at `t`;
2. a future tie first observed at or after entry used only as a time-reversal
   placebo, never as a primary exposure;
3. field-year degree-preserving shuffled-network exposure;
4. matched non-neighbor exposure; and
5. a matched non-surge topic from the same field and calendar period;
6. the same prior collaborator's time-eligible success in a registered
   unrelated topic, matched on calendar time; and
7. the same peer work's citation success realized only after the focal decision,
   used only as an intentionally unavailable future-outcome placebo.

G0 fixes a smallest effect of interest and an equivalence margin for every
negative control using discovery-only measurement resolution and power, not
the confirmatory effect estimate. Passing requires the registered equivalence
interval or true-minus-placebo contrast, not merely a non-significant null
test. Failure of a temporal lead, network placebo, matched non-surge,
unrelated-topic-success, or future-citation-success gate blocks the
corresponding network-local feedback claim even if a descriptive H3 timing gap
remains. An unrelated-topic result is relabeled as a collaborator-status or
general-prestige association unless a registered target-specific contrast
passes.

## 6. Data Readiness Gate (G-1)

The historical local data profile records a post-repair OpenAlex 20260330 works
snapshot with 492,293,207 distinct raw work keys, zero duplicate raw keys after
repair, and works-side authorship, topic, reference, award, and funder tables.
That convention counted the later-confirmed `W-1` sentinel-like key as a raw
distinct value; it did not establish 492,293,207 grammar-valid works. The
historical audit is provenance evidence, not a substitute for verifying the
live analytical input.

The dated audit chain from the 2026-07-16 inventory and recounts through the
2026-07-19 current-content and main-work temporal runs found:

- a mounted backup contains the full derived OpenAlex 20260330 merged-works
  parquet root, including 493 main-work files, 1,363 authorship files, 930 topic
  files, and 3,044 referenced-work files;
- the tracked inventory auditor reproduced 23 table directories, 21,019 readable
  parquet footers, 988,404,537,807 bytes, 20,996,232,817 footer rows, and
  650,939 row groups without reading data pages; the run finished in about 15
  seconds on a warm local cache and returned `INVENTORY_ONLY`, not
  `PASS_METADATA` or `PASS_G1`;
- the zero-invalid main-work body recount scanned all 492,293,207 rows and
  found zero null IDs but one malformed `https://openalex.org/W-1` row. The
  untouched raw source is therefore `BLOCKED`; exact distinct and duplicate
  fields were deliberately left unavailable under that failed contract;
- a separate, frozen analysis-view amendment quarantined exactly that `W-1`
  row and passed the grammar-valid recount: all 492,293,206 retained IDs are
  distinct, duplicate row excess and duplicate groups are both zero, and
  maximum multiplicity is one. `https://openalex.org/W0` is retained because
  it satisfies the frozen decimal grammar, but its entity validity remains
  unresolved under `WARN_ZERO_ID_UNRESOLVED`;
- the companion targeted incidence audit found `W-1` in 21 source rows across
  six files and `W0` in six source rows across six files. Neither value occurs
  in the declared referenced-work or related-work target columns. These are
  incidence counts on their own, not a row-conserved relation view or complete
  join audit;
- the 2026-07-17 full audit streamed all 20,996,232,817 declared data-page rows
  in all 21,019 files across the 23-table scope. Its accounting view
  quarantined exactly the registered 21 `W-1` source rows, retained
  20,996,232,796 rows, and found zero null, other-malformed, or orphan source
  work IDs among the retained rows. This is a passed source-membership and
  row-conservation component, not a materialized filtered dataset;
- the completed 2026-07-19 full-root content freeze hashed all 21,019 files and
  988,404,537,807 bytes and established
  `PASS_CURRENT_CONTENT_BASELINE_COMPONENT`. Its outputs-receipt SHA-256 is
  `082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12`.
  This is a baseline for the current bytes only; historical inventory-era byte
  identity remains `UNVERIFIED`;
- the separate v3 full main-work temporal audit scanned all 493 files and
  492,293,207 raw rows, quarantined the frozen `W-1`, and retained 492,293,206
  rows. Every retained `created_date` and `updated_date` is non-null, exact
  UTC-millisecond text before the exclusive snapshot boundary, and all
  464,727,674 nonnull publication-date/year pairs agree. The three
  `created_date > updated_date` pairs are a fingerprint-bound descriptive
  profile because OpenAlex does not define their ordering as a validity
  invariant. This is a bounded main-work temporal-field pass, not citation-edge
  time or record-event-order evidence;
- declared targets remained an open-world coverage measurement: referenced-work
  targets had 2,904,520,313 in-snapshot rows and 138,015,847
  snapshot-external grammar-valid rows, while related-work targets had
  2,366,002,635 in-snapshot rows and 127,158,784 snapshot-external
  grammar-valid rows. Snapshot-external targets are coverage observations, not
  source orphans. Duplicate source-target edges were not measured, and the
  target space was not declared closed;
- the raw authorship table has 1,361,551,038 footer rows, of which 580,116,420
  have null author IDs. The accounting view removes the one registered `W-1`
  source row and therefore has 1,361,551,037 retained authorship rows,
  580,116,419 null author-ID rows, 781,434,618 grammar-valid non-null author-ID
  rows, and zero malformed non-null author-ID rows. At the work level it records
  263,750,998 `complete`, 2,109,042 `partial`, 177,217,069 `none_usable`, and
  49,216,097 `zero_authorship_row` works. These are descriptive missingness and
  grammar results, not evidence of author-entity validity or researcher-panel
  eligibility;
- the mesh filename anomaly is a historically verified two-file repair: pooled
  deduplication replaced `base-0` and `base-213` with two repartitioned repair
  files, removed 45 duplicate rows, and produced 1,212,666,618 rows. The
  historical repair records lack per-file cryptographic checksums, so this
  remains a provenance warning rather than a full lineage closure;
- the awards table has two physical column orders but one order-insensitive
  names/types/nullability schema, so order drift is recorded as a warning rather
  than semantic schema failure;
- the abstract exception table contains raw exception payload fields, not a
  validated abstract-text source, and is prohibited from the primary text
  branch; and
- the mounted root does not preserve the raw full snapshot or carry a
  self-contained `post_validation.json`, `schema_manifest.json`, and
  `artifact_contract.json` alongside the parquet;
- the available 20260330 nonworks export is a delta lineage, not an
  automatically valid current-state author master; and
- older text and reference sidecars belong to a 202602 lineage and must not be
  mixed into the 20260330 panel without an explicit compatibility audit.

The compact, path-redacted outputs are preserved in the
[source inventory summary](../../../experiments/evidence/results/openalex_g1_source_inventory_20260716.summary.md),
[blocked raw recount](../../../experiments/evidence/results/openalex_g1_main_id_recount_20260716.blocked.summary.md),
[passed grammar-valid recount](../../../experiments/evidence/results/openalex_g1_grammar_valid_main_id_recount_20260716.summary.md),
[targeted sentinel-incidence audit](../../../experiments/evidence/results/openalex_g1_sentinel_incidence_20260716.summary.md),
[full work-source membership/profile audit](../../../experiments/evidence/results/openalex_g1_work_source_membership_20260717.summary.md),
[full taxonomy-reference JSON](../../../experiments/evidence/results/openalex_g1_field_taxonomy_reference_20260718.summary.json),
[full taxonomy-reference Markdown](../../../experiments/evidence/results/openalex_g1_field_taxonomy_reference_20260718.summary.md),
[main temporal v3 JSON](../../../experiments/evidence/results/openalex_g1_main_temporal_validity_v3_full_20260719.summary.json),
[main temporal v3 Markdown](../../../experiments/evidence/results/openalex_g1_main_temporal_validity_v3_full_20260719.summary.md),
and [quarantine amendment](openalex-main-id-quarantine-amendment.md).

The data are present, and the frozen work-source membership/profile component
and later current-snapshot taxonomy-reference component have passed, but the
corpus is not yet analysis-ready. This repository now contains deterministic
commands and frozen contracts for the
[footer-only inventory](../../../scripts/audit_openalex_source_inventory.py),
[raw main-ID recount](../../../scripts/recount_openalex_g1.py),
[grammar-valid recount](../../../scripts/recount_openalex_g1_canonical_quarantine.py),
[sentinel-incidence audit](../../../scripts/audit_openalex_sentinel_incidence.py),
[full work-source membership/profile audit](../../../scripts/audit_openalex_work_source_membership.py),
[full taxonomy-reference audit](../../../scripts/audit_openalex_field_taxonomy_reference.py),
[current-content freeze](../../../scripts/audit_openalex_content_freeze.py),
and [main temporal v3 audit](../../../scripts/audit_openalex_main_temporal_validity_v3.py).
The work-source component proves row accounting and retained source membership
without writing a filtered analysis view. The later taxonomy component proves
current-snapshot projected-column row conservation, hierarchy-reference
integrity, and frozen field/null margin equality only. Neither component
measures duplicate source-target edges or establishes `W0`, author, candidate,
or broader field-entity validity. The compact feasibility artifact's historical
`field_entity_validity=UNVERIFIED` label therefore remains unchanged. The
full-root content scan separately establishes a current-content baseline, while
historical inventory-era byte identity remains `UNVERIFIED`. The bounded v3
main-work temporal component separately passes publication-date/year
concordance and exact non-null metadata timestamps before the snapshot
boundary. Primary-topic child inclusion, citation-edge chronology and
observation time, cutoff-aware `available_at` rules, outcome maturation and
right censoring, `t-1` graph construction, leakage tests, candidate-author
eligibility, a deterministic representative pilot, pivot detection, attention
construction, hazard models, and a temporal-holdout pipeline remain pending.
Neither G-1 nor empirical G1 has passed: G-1 data readiness is `PENDING`,
empirical G1 is `NOT_RUN`, and `pass_g1=false` is not a failed scientific
hypothesis. G-1 is a verification and empirical-pipeline construction gate,
not a corpus reacquisition gate.

The local-versus-remote compute, transfer, GPU-promotion, and sealed-label
boundary is governed by the
[compute execution contract](compute-execution-contract.md). GPU availability
does not waive any G-1 or G0 requirement.

No full substantive analysis or model fitting is authorized by this protocol
until G-1 passes.

The machine-readable readiness ledger at
`experiments/configs/pivot_feedback_g_minus_1_readiness_v0.yaml` binds completed
component evidence separately from missing production receipts. Its validator
must continue to report `G_MINUS_1_PENDING`, `pass_g_minus_1=false`, and
`execution_authorized=false` while any canonical, relation, author, citation,
cutoff-aware panel, outcome-blind field-structure, or representative-pilot
prerequisite is absent.

G-1 requires:

1. preserve and exactly revalidate the completed current-content baseline and
   record its source date, coverage window, lineage, file inventory, row counts,
   schema fingerprints, and checksums in a tracked, path-redacted data manifest;
   do not reinterpret the current baseline as historical inventory-era byte
   identity;
2. preserve the 492,293,207-row raw-source failure, use only the frozen
   one-row quarantine for the exact 492,293,206-row grammar-valid view, and
   require every downstream view to reproduce the completed work-source row
   conservation, source-membership, declared-target coverage, authorship
   profile, current-snapshot taxonomy-reference checks, and exact 26-field plus
   null margins before candidate selection;
3. inherit the completed bounded main-work publication-date/year and metadata
   timestamp validity, then establish citation-edge chronology and observation
   time, cutoff-aware `available_at`, outcome maturation, right censoring, and
   field-specific coverage;
4. identify a defensible author-entity source and audit OpenAlex author IDs with
   an ORCID-linked replication, split/merge diagnostics, implausible career
   trajectories, hyper-prolific authors, and consortium-authorship sensitivity;
5. build a deterministic cutoff-aware author-work-topic panel in which every
   feature has an `available_at` rule, collaboration edges stop at `t-1`, and
   citation outcomes use a fixed post-publication horizon;
6. forbid current `cited_by_count`, `fwci`, current author summary/topic fields,
   future coauthor or affiliation edges, post-entry readiness, and any scaler or
   threshold fit using holdout-period information from model inputs;
7. create a checksummed pilot cohort through a deterministic full-root scan and
   prove that it reconstructs author timelines without future-information or
   treatment-dependent leakage; and
8. freeze distinct discovery, G1 empirical-replication, model-selection, and G4
   sealed final temporal/domain/shock-family partitions before confirmatory
   analysis.

Reading isolated `part-*` or `b000000` files is permitted for schema and
pipeline smoke tests, but not as a representative scientific pilot or for
effect estimation. Because the mounted parquet is not year- or topic-partitioned,
the pilot requires a deterministic full-root scan or a predeclared index built
from one.

### G-1 completed CPU/I/O audit execution boundary

At the inventory checkpoint, the frozen boundary stated: **The next executable
step is a resumable CPU/I/O body recount, not model fitting and not a GPU job.**
The main body recount and the broader work-source component are now complete.
The original zero-invalid contract remains blocked on the one raw `W-1` row,
while the separate frozen-quarantine contract passes exact equality counting
for the 492,293,206-row grammar-valid view with `W0` retained.

The 2026-07-17 full run completed the following bounded CPU/I/O scope:

1. streamed and explicitly counted all 20,996,232,817 data-page rows in all
   21,019 declared files rather than relying on footer-only `COUNT(*)` results;
2. reproduced all 21 registered `W-1` source rows and six registered `W0`
   source rows at their frozen file locators, applied the `W-1` quarantine only
   in the accounting view, and proved exact global and per-table row
   conservation;
3. checked every retained child source work ID against the 492,293,206-ID main
   set, finding zero retained source orphans, and separately measured open-world
   coverage for referenced-work and related-work targets;
4. produced a descriptive, row-conserved author-ID/work profile by source
   lineage, year, primary field, and declared/team-size bands without promoting
   grammar-valid author IDs to valid entities or eligible researchers; and
5. checkpointed by input file, rebound checkpoints and final artifacts to the
   frozen plan and parent metadata, and recorded elapsed time, peak memory,
   runtime versions, and output checksums.

The component status is
`PASS_WORK_SOURCE_MEMBERSHIP_AND_PROFILE_AUDIT_WITH_FROZEN_QUARANTINE`.

A later 2026-07-18 full CPU/I/O run established the separate status
`PASS_CURRENT_SNAPSHOT_TAXONOMY_REFERENCE_COMPONENT`. It scanned 493 main-work
and 930 exploded work-topic files: 1,423 files, 1,420,855,753 projected rows,
and 166,699,471,247 source bytes in total. Exact conservation comprises one
frozen `W-1` quarantine row, 138,147,958 allowed all-null retained main-work
rows, and 1,282,707,794 complete valid hierarchy rows. Partial and invalid
hierarchy counts are both zero; the one retained `W0` occurrence remains an
unresolved warning. All 26 non-null field margins and the null margin match the
frozen compact profile exactly.

The tracked JSON SHA-256 is
`f8a0d01ee5b04133c1472d4ed7c07f7a578343004c859645f174f1a0f6021f62`,
the tracked Markdown SHA-256 is
`1e9db873d97aa9035fde472d587d91a9c1f3eb7592816ee1df1f149ce9ac77e9`,
and the result fingerprint is
`36209d119cc32802d0e7391b92425dcef931e9513841afa19e8be7b0bc06b30b`.
This later pass is limited to current-snapshot projected-column row conservation
and taxonomy-reference integrity. It does not retroactively alter the compact
artifact's historical `field_entity_validity=UNVERIFIED` label or establish
parent content identity, duplicate work-topic edges, primary-topic child
inclusion, citation-edge time or cutoff-aware availability, author/candidate
eligibility, final field selection, or an effect.

The completed 2026-07-19 current-content run separately established
`PASS_CURRENT_CONTENT_BASELINE_COMPONENT` across the full 21,019-file,
988,404,537,807-byte root. Its exact outputs-receipt SHA-256 is
`082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12`.
That pass freezes only the bytes read in the current run; historical
inventory-era byte identity remains `UNVERIFIED`.

The separate v3 full temporal run established
`PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE`
for the bounded 493-file main-work projection. It scanned 492,293,207 raw rows,
retained 492,293,206 after the frozen `W-1` quarantine, matched all 464,727,674
nonnull publication-date/year pairs, and validated non-null exact
UTC-millisecond metadata timestamps before the snapshot boundary. The three
`created_date > updated_date` rows remain descriptive. The report SHA-256 is
`de98be73fb0365672cec7ee770c40de090ae012d45a0422faebd34c0b9d19200`,
the receipt SHA-256 is
`2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe`,
and all 497 receipt artifacts were independently rehashed exactly. This pass
does not establish citation-edge chronology or `available_at`, record-event
order, maturation, censoring, `t-1` construction, or leakage validity.

A later bounded positive-ID sensitivity component excludes the single retained
`W0` row and changes only the total and null-primary-field margins by one; all
26 non-null field margins remain unchanged. This component passes without
deciding `W0` entity validity.

Production update, 2026-07-23: the canonical time spine, author-token history,
primary-score-adjudicated work-topic relation, and citation chronology proxy now
have full production receipts and independent verification. The normalized
cutoff-primitives successor also passed over all 493 main files. It retained
492,293,206 works in 4,096 strict work-modulo partitions, closed every row to
the canonical spine, projected no current citation-count or FWCI outcome field,
and bound the four production parents. Its receipt SHA-256 is
`91e3f689e574d524eabbe7daa58f5e678da37f851ba9b7af0561dab57b70846e`;
the compact tracked evidence is
`experiments/evidence/results/pivot_feedback_cutoff_primitives_production_full_20260723.summary.json`.

That pass does not convert current topic labels or bibliographic citation event
time into historically observed information. It also deliberately leaves
candidate eligibility, included work types, team caps, same-day treatment,
fixed maturation horizon, sampling, and weights to G0. It is therefore a
completed `cutoff_primitives` foundation, not the final cutoff-panel pass
required by item 5 above.

The fail-closed registry now has ten exact repository-evidence passes and still
has three missing required components:

1. a G0-resolved final cutoff-panel successor with fixed maturation,
   eligibility, inclusion, sampling, and weighting rules;
2. an outcome-blind field-structure audit; and
3. a checksummed deterministic representative discovery pilot.

These remaining gates do not authorize model fitting. Author IDs remain
current-snapshot tokens rather than verified persons, and OpenAlex
first-observation time remains unidentifiable from the current snapshot.

The earlier raw body recount measured 5,137,957,807 compressed ID-column bytes
and 300.8 seconds for the complete 493-file scan. The separate grammar-valid
run completed in 122.72 seconds and wrote two 3,938,345,648-byte equality
spools. The broader one-worker work-source run completed in 8,410.30 seconds
with peak RSS of 51,093,992 KiB. The completed content freeze scanned the full
920.523 GiB root; downstream jobs should revalidate its exact receipt and
source bridge rather than repeat that scan by default. It is a current baseline,
not a reconstruction of the missing earlier content-hash baseline. These are
all CPU/I/O evidence jobs; neither the completed components nor the current
content baseline alone makes the data analysis-ready. Raw source validity
remains `BLOCKED`, `W0` remains `UNRESOLVED`, and historical content identity
remains `UNVERIFIED`. G-1 remains
`PENDING`, empirical G1 remains `NOT_RUN` with `pass_g1=false`, G0 remains
`PENDING`, and no GPU job is yet authorized by this protocol.

## 7. Measurement Freeze Gate (G0)

Before inspecting the confirmatory sample, create a versioned measurement
registry containing:

- author eligibility and publication-history requirements;
- the primary pivot definition and all robustness variants;
- topic-attention construction and leave-out rules;
- network edge, peer-active, network-local public-success, chaperoned versus
  independent entry, and capacity-bridge definitions;
- outcome horizons and citation-maturation rules;
- treatment of multi-topic works, team-authored works, multiple affiliations,
  career gaps, and author merges or splits;
- minimum cohort sizes and small-cell suppression;
- distinct discovery, G1 empirical-replication, model-selection, and G4 sealed
  final temporal/domain/shock-family partitions; and
- exclusion reasons recorded per author-entry opportunity.

The cross-field portion of that registry is frozen at
`experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml`. It must record
the taxonomy-reference hash, the fact that field selection remained `PENDING`
until the author, edge-availability/leakage, and outcome-blind structural
audits passed, every
eligible and excluded field, the standardized maximin inputs, stable numeric-ID
tie-break, reserve order, episode-independence decisions, and atomic partition
assignments.

The peer-mechanism portion is prospectively structured at
`experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml`. G0 must
replace its `PENDING` thresholds with quantitative definitions, bind it to the
frozen cross-field matrix hash, and preserve every registered claim ceiling and
downgrade rule before confirmatory outcomes are inspected.

The additive pre-G1 resolution overlay is
`experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml`. It preserves
both v0 parents byte-for-byte, records the choices already supported by this
protocol, and lists every unresolved quantitative section. A structurally valid
pending overlay is not a G0 pass: while any resolution item or prerequisite
receipt remains pending, `pass_g0=false` and execution remains unauthorized. A
fully quantitative freeze requires a separately versioned successor rather
than promotion of this pending artifact in place.

G0 cannot pass until the registry also fixes:

- the eligible researcher-topic-year universe, entry-event construction,
  risk-set generation, negative-sampling ratio, sampling strata, and inverse
  probability weights;
- the public-availability `available_at` rule, the nested peer-active baseline,
  the incremental public-success contrast, source-peer assignment,
  chaperoned-versus-independent entry estimand, and all field-wide,
  homophily/selection, recruitment/capacity, and network placebo contrasts;
- attention-surge detection, minimum cohort size, outcome-decline onset, the
  minimum overshoot timing gap, interval criterion, and the replication
  tolerance used by G1;
- the unit, estimator, censoring rule, and uncertainty interval for empirical
  bibliographic outcome-availability lag and prior-collaborator
  exposure-opportunity timescale, including how their central values map to
  the G3 factorial's explicitly modeled message-diffusion factor;
- the practically meaningful G2 score margin, maximum permitted calibration
  degradation, macro-trajectory tolerance, probability clipping, and missing
  prediction rule;
- the simulation fit objective, number of predictive draws or Monte Carlo
  standard-error stopping rule, parameter and tuning budgets, and identical
  compute-comparison rule across B2-B4; and
- the rule that all works belonging to one author remain in a single fold when
  author-level prediction or identity-sensitive features are evaluated.

G0 must also freeze one confirmatory H3 timing-gap endpoint, the field-balanced
and episode-clustered uncertainty procedure, the one-cohort-year minimum gap,
the negative-control equivalence margins, and a hierarchical multiplicity
rule. H3 is tested first at family-wise alpha 0.05. Only if it passes may H1,
H2, and H4 be tested as a three-hypothesis Holm family. H2 contributes one
registered confirmatory contrast: public success incremental to peer activity
and the other frozen controls. The peer-active coefficient and the chaperoned
versus independent split are required mechanism diagnostics, not extra ways to
pass H2. Field-specific results are transport diagnostics, not additional
confirmatory discoveries; any exploratory field-wise testing must be labeled
separately and use its own registered false-discovery-rate family. H5's `R`
endpoint is secondary and H6 belongs to the later predictive-model gate, so
neither can rescue H3 or reuse a failed primary path.

Before the full matrix opens, a minimum engineering pilot uses two fields by
two independent shocks from the discovery period. The pilot is produced by the
same deterministic full-root extraction path, executes the clocks, primary
endpoint code, placebos, weights, and leakage checks, and is labeled
`DISCOVERY_ONLY` in every artifact. It cannot pass G1, finalize a field because
of its outcome, tune on a confirmatory period, or support a substantive claim.
The full run is four fields with at least three independent episodes per field:
three fields for G1 and one sealed field for G4.

A provisional partition is discovery/calibration through 2012, a G1 empirical
replication period from 2013 through 2016, model selection from 2017 through
2019, and a sealed G4 temporal holdout for 2020-2022 entry cohorts with outcome
follow-up through 2025. The G1 replication partition is distinct from every G4
final test partition. These years may change only through a documented G0
coverage audit, not after seeing the target pattern.

## 8. Empirical Signature Gate (G1)

G1 tests whether the proposed phenomenon exists before expanding the model.

The single confirmatory primary is H3's field-balanced timing gap `delta_j`.
G1 passes its primary branch only when the registered point criterion is at
least one cohort-year, the registered one-sided 95% episode-clustered interval
has lower bound greater than zero, the leave-one-field-out estimates retain the
registered positive direction, and the direction is stable under the one
pre-specified alternative pivot construction. A larger effect in one field
cannot compensate for a reversed or missing transport pattern elsewhere under
the registered rule.

The minimum analyses are:

1. cohort-resolved event studies around multiple independently defined
   attention surges;
2. matched risk-set or discrete-time hazard models using only pre-entry
   covariates;
3. researcher, topic, field, and calendar controls appropriate to the chosen
   estimand;
4. leave-out attention, peer-active, and incremental public-success measures;
5. pre-tie topic similarity and trajectory diagnostics for
   selection/homophily;
6. pretrend and negative-control outcomes;
7. shuffled-network, non-neighbor, unrelated-topic-success, and unavailable
   future-citation-success placebos;
8. chaperoned and independent first-target-work entry modes;
9. alternative pivot definitions; and
10. at least three independent domains or shock families, not a COVID-only
   result.

All registered lead-attention, future-tie, shuffled-network, non-neighbor,
matched non-surge, unrelated-topic-success, and unavailable
future-citation-success controls must satisfy their equivalence or contrast
gates. Only after the H3 primary and its measurement controls pass are H1, H2,
and H4 tested with Holm family-wise error control at 0.05. The secondary `R`
analysis is reported with its frozen uncertainty and held-out prediction rule,
but it does not determine G1 passage.

G1 passes only if:

- the incumbent-only attention signal in H1 predicts later entry in the
  registered direction and survives the designated G1 replication partition;
- the H2 public-success increment predicts entry beyond the peer-active
  baseline, field-wide attention, and registered non-neighbor controls;
- the direction of H2-H4 is stable across the primary and at least one
  substantively different pivot construction;
- entry continues after the pre-specified persistence or outcome decline in a
  reproducible cohort pattern;
- the result is not driven entirely by one field, one shock, one career stage,
  or the focal researcher's contribution to the attention measure; and
- the effect survives the designated G1 empirical-replication partition in
  direction and timing within the G0-registered tolerance.

No G4 final temporal, domain, or shock-family test partition may be opened to
decide whether G1 passes.

If G1 fails, stop the cascade mechanism route. Do not tune the simulator to
manufacture the missing pattern. The method-first framework paper remains a
separate fallback.

The independent-entry branch governs interpretation, not a hidden alternative
endpoint. If the H2 association is confined to chaperoned entry, it must be
reported as a recruitment, co-production, or capacity-transfer association and
must not be called peer learning or social transmission. If peer activity
predicts entry but its public-success increment does not, retain only a
prior-collaborator-activity association. Even a surviving independent-entry
increment is labeled `consistent with a network-local response to publicly
available outcomes`; OpenAlex alone does not promote it to receipt, belief
updating, learning, or causal influence.

## 9. Baseline Model Ladder (G2)

Models must be compared in increasing order of flexibility on identical
partitions and observables.

| Level | Model | Question answered |
| --- | --- | --- |
| B0 | Cohort and field trend nulls | Is the pattern distinguishable from common time and field movement? |
| B1 | Discrete-time fixed/logistic hazard | Can observed pre-entry covariates predict entry and persistence without endogenous simulation? |
| B2 | Fixed-rule ABM | Can a compact hand-specified feedback rule reproduce the held-out micro and macro moments? |
| B3 | Explicit adaptive baseline | Can a Bayesian, bandit, or similarly interpretable learner explain the adaptation? |
| B4 | Adaptive NABM | Does learned heterogeneous, nonlinear, history-dependent adaptation add transportable predictive value? |

Primary predictive scores are out-of-sample log score, Brier score, and
calibration for individual entry or persistence. Macro checks include cohort
entry curves, persistence curves, overshoot timing and area, recovery timing,
and network/career-stage heterogeneity. Parameter count, tuning budget, and
training data must be reported for every level.

For B2-B4, an individual probability is the mean event indicator across a
G0-fixed number of predictive simulation draws conditional only on information
available at the decision time. The alternative G0 rule may stop draws at a
fixed Monte Carlo standard error. Every model receives the same scoring rows;
probability clipping, simulation budget, fit objective, and tuning budget are
fixed before the G4 partitions open. This connects stochastic ABM output to the
proper scores rather than scoring a hand-selected representative trajectory.

NABM earns a scientific role only if it:

1. improves frozen micro-level prediction by a pre-specified practically
   meaningful margin;
2. improves at least one pre-specified macro trajectory without degrading
   calibration materially;
3. transports across a held-out time period and domain; and
4. yields an intervention ordering or phase boundary that remains stable under
   plausible measurement and parameter uncertainty.

If B1-B3 match B4, adopt the simpler model and drop any neural-necessity claim.

## 10. NABM Redesign Gate (G3)

The current researcher-pivot study has hand-set coefficients, static productive
fit, immediate and complete outcome visibility, and a one-time absorbing pivot.
Those choices are useful for a lifecycle demonstration but cannot identify the
proposed feedback mechanism.

The new simulator must add:

- stochastic outcomes separated from the entry decision;
- delayed, noisy, partial, and selectively disclosed outcomes;
- repeated or reversible research moves;
- evolving skill, resource, or collaboration capacity after entry;
- separate field-wide attention, network-local public-availability,
  recruitment/capacity, and entry channels;
- distinct simulated states for public availability, message receipt, belief or
  policy update, and action;
- a pre-existing capacity-bridge channel separate from the public-success
  increment;
- a latent potential-outcome ledger shared across disclosure arms;
- component-keyed random streams shared across paired counterfactuals; and
- full micro traces for public availability, message delivery and receipt,
  belief or policy update, transition, outcome revelation, and network change.

Changing a peer rule must not silently change the local learner's target or the
domain transition target. Channel-isolation tests must hold all non-targeted
learner supervision and transition inputs fixed.

### 10.1 Confirmatory Core Factorial

The minimum confirmatory design is:

| Factor | Core levels |
| --- | --- |
| Outcome disclosure | full, success-biased, volume-matched random |
| Revelation delay | immediate, empirical central delay, long-delay stress |
| Network-local message diffusion speed | slow, empirical proxy-mapped speed, fast |
| Learner | frozen, explicit adaptive baseline, NABM |
| Environment | attention-only, structural support with matched direct pressure |
| Capacity bridge | absent, pre-existing |

Paired cells use the same exogenous shocks and unit-level random quantiles.
Disclosure-only arms also share one latent potential-outcome ledger. Structural
support and capacity arms may change the pre-specified potential-outcome
mapping; the shared quantile is then passed through each arm's mapping rather
than forcing identical realized outcomes. Failure-only, extreme-only,
misclassification, alternative graph families, and policy withdrawal are
extension analyses rather than unbounded additions to the core grid.

### 10.2 Simulation Outcomes

- entry and non-persistence incidence;
- later-cohort outcome degradation;
- overshoot timing and area;
- time to correction;
- learner calibration and parameter stability;
- retained exploration and topic diversity;
- career-stage and resource-position disparity; and
- intervention-ranking stability.

No learned coefficient is interpreted causally without an isolated
intervention on the corresponding input channel.

Receipt and belief-update states are simulated latent variables unless an
external data source observes them; matching an OpenAlex entry pattern does not
validate those internal states.

## 11. Frozen Holdout Gate (G4)

The final test partitions are frozen at G0 and remain unopened through G1-G3.
G4 opens them only after model classes, fitting objectives, hyperparameter
budgets, score margins, factorial definitions, and stopping rules are locked.
The G1 empirical-replication set cannot be reused as a G4 final test.

The sealed G4 package contains:

- one temporal holdout;
- one entire scientific domain holdout;
- one attention-shock family holdout; and
- the primary score and acceptable prediction interval for each key empirical
  moment.

The primary cross-field test is domain-and-author-cold. The entire selected G4
field and every episode assigned to it are absent from model fitting and model
selection. Training labels for any researcher who appears in the sealed domain
evaluation rows are also withheld. Past information that would genuinely have
been observable at prediction time may enter only through the frozen feature
contract; author identity, field-specific fitted effects, held-out outcomes,
and held-out normalization statistics may not leak into training. A separate
domain-cold but author-warm score may be reported only as a labeled robustness
analysis and cannot replace the primary cold test.

Topic families, shock episodes, and response cohorts remain atomic in G4. The
domain and shock-family labels are sealed locally, and all remote prediction
bundles remain features-only until every registered model has returned a
complete bundle.

G4 fails if:

- the adaptive model fits the training period but misses the timing or
  direction of H2-H4 in holdout;
- performance depends on one topic representation;
- the bridge-proxy association disappears under pretrend matching or a
  past-only network definition;
- the phase boundary changes qualitatively across plausible seeds,
  architectures, or outcome definitions; or
- the adaptive NABM does not improve on the simpler baseline ladder.

Training fit alone cannot promote a claim into the manuscript.

## 12. Causal Validation Gate (G5)

OpenAlex-only analyses are observational. They can support a predictive and
generative-adequacy claim that a specified process reproduces held-out
patterns. That is neither causal identification nor proof that the process is
the unique explanation, and it cannot support a policy-effect claim.

A stronger route requires at least one external causal axis, such as:

- a randomized or cutoff-based seed-grant allocation with the full eligible
  applicant pool;
- an exogenous information-disclosure change;
- a randomized networked decision experiment comparing full and selective
  outcome visibility; or
- another design that separates exposure from common opportunity and
  homophilous selection.

Grant acknowledgements among published works are not an untreated applicant
pool. News or publication growth is not a randomized attention shock. G5 must
therefore name the estimand and counterfactual population explicitly.

## 13. Promotion and Stop Rules (G6)

| Evidence state | Decision |
| --- | --- |
| Data lineage, author-panel validity, or measurement leakage unresolved | Block substantive execution; permit engineering feasibility work only. |
| Taxonomy reference, field entity, author entity, temporal availability, or outcome-blind structural selection unresolved | Keep final field selection `PENDING`; do not name a final field panel. |
| A selected field has fewer than three independent eligible episodes and the frozen reserve rule cannot replace it | Stop the four-field confirmatory expansion; retain only `DISCOVERY_ONLY` engineering evidence. |
| No reproducible overshoot signature | Stop the feedback-cascade route. |
| H3 timing gap appears but a registered lead, network, or matched non-surge equivalence gate fails | Retain at most a descriptive timing pattern; make no peer-feedback or delayed-information mechanism claim. |
| Peer activity predicts entry but the registered public-success increment does not | Drop the visible-success feedback claim; retain at most a prior-collaborator-activity association. |
| The public-success increment is confined to chaperoned first target works | Rename the result as recruitment, co-production, or capacity-transfer association; do not call it peer learning. |
| The network-local proxy does not outperform matched non-neighbors, or unrelated-topic success yields an equivalent association | Rename the result as field-wide common opportunity or general collaborator-status association; drop the target-specific network-local mechanism claim. |
| Unavailable future citation success passes as if it were a current predictor | Treat the feature pipeline as temporally invalid; return to G-1/G0 and do not test H2 on that construction. |
| The independent-entry increment survives all observational controls but receipt and belief update remain unobserved | Use only `consistent with a network-local response to publicly available outcomes`; do not claim observed learning or causal social influence. |
| Signature appears only in one shock or topic definition | Keep a narrow descriptive result; no general cascade claim. |
| Direction or prediction does not transport to the domain-and-author-cold G4 field | Drop the cross-field transport claim; report the bounded in-domain result if its own gates passed. |
| Simpler model matches NABM holdout performance | Use the simpler model; retain NABM only as implementation support if needed. |
| NABM predicts better but intervention ranking is unstable | Report predictive value; do not recommend policy. |
| Empirical signature, baseline separation, and holdouts pass | Promote a bounded predictive and generative-adequacy claim, explicitly noncausal and nonunique. |
| External causal axis also passes | Consider a policy-effect claim limited to that design and population. |

No venue is guaranteed by passing these gates. For a broad methods-and-science
submission, the minimum target is G1-G4. A stronger general-science route should
include G5 or comparably decisive independent validation.

## 14. Planned Paper Logic

The science-first manuscript should follow this order:

1. empirical attention, entry, and later-cohort outcome pattern;
2. peer-active versus incremental public-success proxies, chaperoned versus
   independent entry, and the pre-existing capacity-bridge contrast;
3. baseline model ladder and frozen holdout prediction;
4. NABM lifecycle only to the extent required by the winning model;
5. disclosure-delay and support phase diagram;
6. external causal validation, if available;
7. limits, non-identifiability, fairness, and policy trade-offs.

The current Route B method-first manuscript remains a companion route. Its toy
cases and architecture should not be appended wholesale to this paper.

Candidate main displays are:

1. cohort-resolved empirical overshoot and adjustment;
2. network-local public-success versus pre-existing bridge-proxy heterogeneity,
   split by chaperoned and independent entry;
3. baseline-ladder holdout calibration and trajectory prediction; and
4. disclosure-delay by modeled message-diffusion-speed phase map with
   uncertainty.

## 15. Ethics and Misuse Boundary

- Do not publish individual researcher rankings or a model-generated label of
  researcher quality.
- Report career-stage, institution, geography, and resource-position disparity
  where coverage permits, with small-cell protection.
- Do not equate citation impact with scientific or social value.
- Treat exploration suppression as a potential intervention cost, not an
  automatic benefit.
- Audit whether bridge-targeting policies concentrate resources in already
  elite networks.
- Distinguish prediction of observed behavior from a normative recommendation
  about what researchers should study.

## 16. Required Artifacts

The route is not ready for full execution until the following artifacts exist:

1. authoritative empirical data manifest and availability audit;
2. measurement registry with frozen definitions and partitions;
3. empirical extraction and cohort-construction tests;
4. baseline-ladder evaluation specification;
5. redesigned outcome-revelation and reversible-pivot simulator contract;
6. registered hypotheses, scores, and stop rules;
7. holdout results with failed as well as passed predictions;
8. claim matrix linking every promoted statement to code, data, result,
   uncertainty, and limitation; and
9. remote preflight, transfer manifest, run bundle, and sealed-label evaluation
   artifacts required by the compute execution contract.
10. a hashed cross-field matrix containing the authoritative taxonomy-reference
    binding, outcome-blind maximin trace, tie-break, reserve order, four-field
    assignment, and independent-episode ledger;
11. a `DISCOVERY_ONLY` two-field-by-two-shock pilot report with all leakage and
    negative-control checks; and
12. a confirmatory endpoint, equivalence-margin, multiplicity, and atomic-split
    registry linked to the domain-and-author-cold G4 package.
13. a hash-bound peer-mechanism registry with peer-active and public-success
    increments, entry-mode strata, placebos, and claim-downgrade rules.
14. a receipt-bound normalized cutoff-primitives component plus a distinct
    G0-resolved final-panel successor that freezes eligibility, inclusion,
    maturation, sampling, and weighting without changing the primitive rows.

## 17. Conceptual References

- Hill R, Yin Y, Stein C, et al. [The pivot penalty in
  research](https://doi.org/10.1038/s41586-025-09048-1). Nature. 2025.
- Venturini S, Sikdar S, Rinaldi F, et al. [Collaboration and topic switches in
  science](https://doi.org/10.1038/s41598-024-51606-6). Scientific Reports.
  2024.
- Zeng A, Shen Z, Zhou J, et al. [Increasing trend of scientists to switch
  between topics](https://doi.org/10.1038/s41467-019-11401-8). Nature
  Communications. 2019.
- Zeng A, Fan Y, Di Z, et al. [Impactful scientists have higher tendency to
  involve collaborators in new
  topics](https://doi.org/10.1073/pnas.2207436119). Proceedings of the National
  Academy of Sciences. 2022.
- Manski CF. [Identification of endogenous social effects: The reflection
  problem](https://doi.org/10.2307/2298123). Review of Economic Studies. 1993.
- Catalini C. [Microgeography and the direction of inventive
  activity](https://doi.org/10.1287/mnsc.2017.2798). Management Science. 2018.
- Bikhchandani S, Hirshleifer D, Welch I. [A theory of fads, fashion, custom,
  and cultural change as informational
  cascades](https://doi.org/10.1086/261849). Journal of Political Economy.
  1992.
- Bénabou R, Vellodi N. [(Pro-)Social Learning and Strategic
  Disclosure](https://doi.org/10.1257/mic.20240190). American Economic Journal:
  Microeconomics. 2025.
- Ale Ebrahim Dehkordi M, Lechner JM, Ghorbani A, et al. [Using machine
  learning for agent specifications in agent-based models and
  simulations](https://doi.org/10.18564/jasss.5016). JASSS. 2023.
