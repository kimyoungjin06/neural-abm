# Package Release Boundary

This release boundary fixes the product-facing entry points for the current v0
package shape. It is a packaging contract, not a new simulation claim.

## Released Tag: `v0.1.0a5`

Default Git tag install:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

The default profile is the lightweight `api_lite` floor. It supports package
metadata, toy feature-taxonomy lookup, compatible runner/result helpers,
readiness utilities, NumPy-only scalar/bounded-scalar social helpers, and
torch-free lifecycle reports/local-step helpers. It must not import or require
`torch`.

Torch-backed API:

```bash
uv pip install "neural-abm[torch] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

Use this profile for `neural_abm.api`, `NABMUnit`, `NABMStep`, `SocialBlock`,
tensor/state-dict social messages, and the full tensor-backed lifecycle.

Research workflows:

```bash
uv pip install "neural-abm[research] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
uv pip install "neural-abm[full] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

Use these profiles for toy runners, evidence matrices, paper workflows,
plotting, and the full development/research stack.

## Main / Next-Alpha Candidate

The unreleased `main` / next-alpha candidate extends the default torch-free
`api_lite` floor with `scenario_lite` bounded-scalar scenario comparison
helpers, including seed-paired replicated runs with outcome distributions.
Its package metadata uses `0.1.0a6.dev0` so artifacts from `main` cannot be
mistaken for the verified `v0.1.0a5` release.
The `v0.1.0a5` tag predates `scenario_lite`; do not use that tag when testing
these researcher-facing exports or examples.
Install that candidate explicitly when evaluating this surface:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@main"
```

This candidate does not change the dependency boundary: `api_lite` and the
scenario helpers must not import or require `torch`.

## Public Entry Points

Use `neural_abm.api_lite` for no-torch metadata and lightweight package
operations available from the released tag:

```python
from neural_abm.api_lite import toy_catalog, toys_by_taxonomy

binary_toys = toys_by_taxonomy("output_family", "binary_probability")
catalog = toy_catalog()
```

On `main` / the next-alpha candidate, bounded-scalar researcher scenarios use
the same no-torch facade:

```python
from neural_abm.api_lite import (
    BoundedScalarScenarioSpec,
    ScenarioDefinition,
    run_bounded_scalar_scenarios,
)
```

The scenario spec's `success_direction` and `success_min_delta` are
user-provided comparison metadata. Returned `success`, `success_fraction`,
paired-mean confidence intervals, and separate empirical replicate intervals
are mechanical simulation reports, not framework judgment that a scientific
claim is supported. Domain semantics, calibration, identification assumptions,
and claim interpretation remain caller-owned; see
[Decision 0015](decisions/0015-researcher-scenario-lite-contract.md).

Use `neural_abm.api` for the stable torch-backed v0 lifecycle:

```python
from neural_abm.api import NABMUnit, SocialBlock, SocialChannel
```

Keep `neural_abm.__init__` as a lazy compatibility layer for existing
module-path imports. Product-facing examples and docs should prefer `api_lite`
or `api`.

## Toy Catalog Boundary

Stable model IDs such as `toy1` are reproducibility IDs for configs, artifacts,
tests, and paper references. User-facing selection should use feature taxonomy
fields:

```text
display_name
domain_family
state_family
output_family
topology_family
coordination_family
unit_surface
evidence_role
```

Run artifacts include these fields in `metadata.json` and `summary.json`. CSV
and sweep field contracts keep established NABM fields for compatibility.

## Source Distribution Boundary

The package-facing sdist includes the replicated stylized researcher-pivot case
study. The prospective empirical protocol, peer-exposure identification
design, compute contract, OpenAlex quarantine amendment, audit scripts/configs,
and compact audit evidence remain repository-only research artifacts.
Package-facing Markdown references to that stack use explicit repository paths
rather than clickable links to files that are not embedded in the sdist. In
particular, `experiments/` remains excluded from release artifacts.
The package-facing case-study subtree is a hash-pinned stylized-study snapshot,
not the live operator ledger for the prospective empirical route. Current
OpenAlex gate state is reported in the root README and repository-only compact
evidence; changing the pinned subtree requires a deliberate public-surface and
downstream-attestation provenance refresh.

The Pivot-to-NABM flagship bridge is repository-only while it remains a design
contract. It repositions empirical timing work as one constraint on a future
NABM model comparison and authorizes no package API change, empirical access,
or model selection. Only a later executable, reproduced case-study successor
may propose a deliberate refresh of the hash-pinned public subtree.

The first delayed-feedback executable remains repository-only as well. Its
Pivot-owned event queue separates privately realized outcomes from publicly
available messages and varies success/failure visibility without changing
`neural_abm.api`, `api_lite`, or the generic lifecycle core. The precommitted
contract, runner, tests, and generated phase-map evidence are development
artifacts, not packaged examples or empirical claims. Promotion requires a
separate public-surface decision after reproduction and comparison with fair
simpler models.

The nested persistence-and-exit successor also remains repository-only. It
holds entry, learning, topology, and public availability exactly fixed while a
Pivot-owned state adapter observes active productive, active failed, and
exited states. Persistence and exit are domain meanings rather than generic
unit semantics. Its precommit, runner, tests, and evidence cannot enter the
package without a later API and public-case promotion decision.

The fixed-versus-delta-versus-neural adaptive baseline ladder remains on the
same repository-only surface. Its explicit scalar delta rule and common-target
scorecard test case-level neural necessity without changing package policy
classes or selecting a model. The contract, runner, tests, and evidence are
excluded until empirical or held-out predictive validation justifies a
separate public-case decision.

The matched-marginal policy discriminator is repository-only as well. It holds
synthetic success prevalence fixed while reversing conditional feature-outcome
structure and compares scalar, explicit linear, classical nonlinear, and
residual-neural policies on disjoint synthetic cohorts. Its contract, runner,
tests, and evidence remain outside the package; a result can distinguish policy
capacity only on the frozen generator and cannot select an empirical model or
change the generic API.
Its v2 successor, which corrects only the scalar learner's reduction after the
v1 run stopped at its invariance check, has the same repository-only boundary.
The result-informed v3 successor, with a matched linear baseline, tuned
baselines, and an axis-aligned control, has the same boundary.

The cross-field expansion follows the same boundary. The no-rescan feasibility
summarizer, engineering-only field-matrix planner, and current-snapshot taxonomy
reference auditor live at
`scripts/summarize_openalex_cross_field_feasibility.py`,
`scripts/plan_research_pivot_field_matrix.py`, and
`scripts/audit_openalex_field_taxonomy_reference.py`. Their frozen contracts
live at `experiments/configs/openalex_g1_cross_field_feasibility_v0.yaml`,
`experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml`, and
`experiments/configs/openalex_g1_field_taxonomy_reference_v0.yaml`. The
separate peer-mechanism registry at
`experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml` freezes a
prospective G0 identification component without authorizing execution. The
additive readiness and resolution ledgers at
`experiments/configs/pivot_feedback_g_minus_1_readiness_v0.yaml` and
`experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml`, together
with their read-only validators, preserve the same fail-closed boundary: a
valid pending ledger is not a G-1 or G0 pass. Compact and full evidence remains
under `experiments/evidence/`. None of these files is a package API, package
example, or sdist payload. The artifact inspector forbids the repository-only
scripts explicitly and rejects every `experiments/` path by prefix.

The later G-1A, S0, and additive gate-state successors remain on that same
repository-only surface. Their contracts, full compact evidence, completion
receipt, and v2/v3 ledgers live under `experiments/`; their source-replay,
threshold-free core, ORCID/affiliation, exact coauthor-graph, completion, and
gate validators live under `scripts/` and `docs/research/`. The v3 ledger
records an S0 structural pass and authorizes only pre-2013 D0 discovery. It is
followed by the repository-only D0-A producer, independent verifier, compact
summary, and additive v4 ledger. The v4 ledger records an outcome-blind
two-field/four-episode structural freeze and authorizes only D0-B
author-opportunity and leave-out measurement construction. The repository-only
D0-B author-foundation producer, independent verifier, compact summary, and
additive v5 ledger then bind exact pre-decision works and the focal-field-active
author-token opportunity surface while leaving D0-B and D0 partial. The
repository-only selected-work linkage producer, independent verifier, compact
summary, and additive v6 ledger then record the severe heterogeneous
work-type/linkage gap and pause leave-out/network/bridge construction pending
an outcome-blind ABI and episode-reselection successor. The later D0 review
contracts, cross-provider submission/collection tools, valid result reducer,
independent result verifier, compact evidence, and additive v7-v11 ledgers are
repository-only too. The v10 ledger records 384 valid responses per provider
and two API-blinded 233-row human packets, but no completed human review. These
packets remain historical artifacts under the API-only v11 successor. V11
records human review as `NOT_RUN_BY_DESIGN`, freezes the simple past-only
observable-history pivot definition, and keeps the 197 strict API-consensus
rows separate from 187 unresolved quarantined rows. The 384 titles remain a
diagnostic sample. These artifacts are not part of the package API, the
hash-pinned stylized case-study subtree, or the sdist, and they do not authorize
a paid full-universe screen, production pivot classification, entry or
persistence outcomes, G0, final-panel materialization, model fitting, G4, or a
scientific-effect claim.

The resumable current-content freeze is also repository-only. Its sequential
per-file SHA-256 auditor and frozen parent-inventory contract live at
`scripts/audit_openalex_content_freeze.py` and
`experiments/configs/openalex_g1_content_freeze_v0.yaml`. Its full component
pass created the first baseline for the bytes read in that run. It does not
prove earlier inventory-era byte identity, pass G-1, or establish an effect.
Its tool SHA-256 is a same-run tool-file snapshot, not loaded-code attestation;
the stronger latter claim would require a trusted launcher or signed artifact.
The repository-only auditor also treats a result as published only when a
validated outputs manifest and the final atomic `COMPLETE` state coexist. It
re-reads all checkpoint files before report construction and final validation,
removes PASS-facing files on caught publication failure, and enforces an
operator-owned, non-shared-writable output directory plus a protected ancestor
chain. These operational artifacts remain outside the distribution package.
The artifact inspector forbids both files explicitly.

The bounded W0 sensitivity, blocked production work-topic parent audit, and
primary-exact-score successor follow the same repository-only boundary. Their
tools and contracts live at `scripts/audit_openalex_w0_positive_id_sensitivity.py`,
`experiments/configs/openalex_g1_w0_positive_id_sensitivity_v0.yaml`,
`scripts/audit_openalex_work_topic_relation.py`, and
`experiments/configs/openalex_g1_work_topic_relation_v0.yaml`, plus
`scripts/adjudicate_openalex_work_topic_primary_scores.py`,
`scripts/verify_openalex_work_topic_primary_adjudication.py`, and
`experiments/configs/openalex_g1_work_topic_relation_primary_adjudication_v1.yaml`.
The tracked W0
summary is component-only evidence. The full relation run counted 714
duplicate edge groups and passed primary-child inclusion, but two duplicate
groups had different scores; it therefore blocked without materializing the
exact-score dedup view. The immutable compact blocked result remains under
`experiments/evidence/`. The successor rehashed that parent, used exact frozen
main-row primary scores to resolve only those two groups, and independently
verified the 928,561,783-record canonical relation. This passes only that
bounded relation component and does not pass G-1. Neither surface nor its
outputs is a package API or
sdist payload, and the artifact inspector forbids the tools/configs explicitly
and every evidence path by the `experiments/` prefix.

The main temporal, canonical work-time-spine, and citation `available_at`
implementations, plus the author-history designs, are also repository-only.
Their documents and frozen contracts live under
`docs/case-studies/researcher-pivot/`, `docs/research/`, and
`experiments/configs/`; their audit tools live at the versioned
`scripts/audit_openalex_main_temporal_validity*.py` surfaces,
`scripts/audit_openalex_canonical_work_time_spine.py`, its v3-parent-only
`scripts/audit_openalex_canonical_work_time_spine_v2.py` synthetic successor,
the external-memory production-capable
`scripts/audit_openalex_canonical_work_time_spine_v3.py` successor, and
`scripts/audit_openalex_citation_available_at.py`, plus its temporal-v3-only
`scripts/audit_openalex_citation_available_at_v2.py` successor and the
canonical-receipt-bound production successor
`scripts/audit_openalex_citation_available_at_v3.py`. Temporal v1 has a terminal
blocked full output; temporal v2 has a terminal dirty one-file parser pilot;
temporal v3 now has a content-bound full component PASS. Its runtime output
remains outside the distribution package, while only a compact claim-bounded
summary lives under `experiments/evidence/results/`. The first two spine tools
and first two citation tools are synthetic-fixture-only. The third spine tool
and third citation tool are production-capable CPU/I/O implementations. After a nonpromotable one-file
pilot on 2026-07-21, a separately authorized 2026-07-22 full run completed all
493 source files and its 4,096-partition output passed the independent
verifier's default full-Parquet mode. The operator tree and receipt remain
outside the distribution; only the compact component-bounded summary lives
under `experiments/evidence/results/`. The frozen citation v2 successor executes
only closed synthetic direct temporal reconstruction. Its output retains
`BLOCKED_CANONICAL_PRODUCTION_RECEIPT_NOT_AVAILABLE`; the later receipt does
not retroactively rewrite that ABI. The separate v3 citation successor binds
the real canonical receipt explicitly and writes its detailed deduplicated and
chronology-safe edge products only to a private operator tree. The original author design and
its temporal-v3-bound v1 successor remain `DESIGN_ONLY / PRODUCTION_NOT_RUN`.
A repository-only v2 overlay, closed fixture, and stdout-only validator now
exercise the bounded token-accounting core without opening production data;
they are not a production runner or receipt. These author-history surfaces are
plain repository paths, not package links. Their frozen core DAG is content →
temporal v3 → canonical spine → author-token history; work-topic relation
remains a parallel sibling, while citation chronology is a downstream bounded
component rather than an author-history parent. Its v3 production successor
now passes only as a current-snapshot bibliographic-time proxy. The v2 validator
creates no durable result or receipt. The later author-history v3 production
successor separately completed its receipt-bound token-accounting component;
its event/history tree remains operator-local and does not validate people or
past observability. In particular, neither the temporal v3
created/updated ordering profile nor the citation convention claims an
OpenAlex first-observation timestamp. The artifact inspector forbids these
paths explicitly, rejects author-history operator-local output basenames and
data directories, rejects citation control, manifest, aggregate, temporary,
deduplicated-edge, and chronology-safe-edge basenames, and the package sdist
excludes all repository-only documents and audit tools.

Post-freeze operational readiness material for the canonical spine follows the
same repository-only boundary. The six notes are
`docs/research/openalex-canonical-work-time-spine-ops-readiness-20260720.md`,
`docs/research/openalex-canonical-work-time-spine-pilot-benchmark-tool.md`,
`docs/research/openalex-canonical-work-time-spine-capacity-review-v0.md`,
`docs/research/openalex-canonical-work-time-spine-downstream-benchmark-v0.md`,
`docs/research/openalex-canonical-work-time-spine-downstream-stress-v1.md`, and
`docs/research/openalex-canonical-work-time-spine-independent-verifier-v1.md`.
Their executable surfaces are
`scripts/benchmark_openalex_canonical_work_time_spine_downstream.py`,
`scripts/benchmark_openalex_canonical_work_time_spine_downstream_stress_v1.py`,
`scripts/verify_openalex_canonical_work_time_spine_downstream_benchmark_v1.py`,
`scripts/verify_openalex_canonical_work_time_spine_downstream_stress_v1.py`,
`scripts/summarize_openalex_canonical_work_time_spine_pilot_benchmark.py`,
`scripts/review_openalex_canonical_work_time_spine_capacity.py`, and
`scripts/verify_openalex_canonical_work_time_spine_v1.py`, with the analytic
template at
`experiments/configs/openalex_g1_canonical_work_time_spine_capacity_review_v0.yaml`.
None is a package API. The bounded benchmark artifacts remain nonpromotable;
the dated operations note additionally records the separately authorized full
run without embedding its operator-local tree in the package. The actual
`openalex_g1_canonical_work_time_spine_pilot_time.txt` and
`openalex_g1_canonical_work_time_spine_pilot_benchmark.json` files are
operator-local and forbidden from release artifacts. The downstream result
`openalex_g1_canonical_work_time_spine_downstream_benchmark.json` is likewise
operator-local and forbidden, as are the full spine's receipt,
Parquet/checkpoint tree, and core operator files `audit.lock`, `audit_plan.json`,
and `run_state.json`. The downstream verifier
is stdout-only and introduces no additional operator-local artifact basename.
The bounded stress result
`openalex_g1_canonical_work_time_spine_downstream_stress_v1.json` remains
operator-local and forbidden from release artifacts. External read-only
verification returned
`PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_STRESS_VERIFICATION_NONPRODUCTION`;
the verifier SHA-256 is
`4ade14fd53f80a964ff8668b7f4e2391ea220402327fcc0fca9c0dd88e19373f`.
The hardened verifier narrows its acceptance set with the exact result pin,
nested invariants, and row-group plus no-dictionary validation. That status is
still nonproduction and noncapacity: it does not pass the capacity gate or
authorize a full run.

The pivot-feedback API-only topic-routing successor is also repository-only.
Its frozen four-card contract, current OpenAlex topic-reference snapshot,
producer, independent verifier, v12 gate ledger, and compact dry-run summaries
live under `experiments/`, `scripts/`, and `docs/research/`; none is a package
API or sdist payload. The independently replayed 384-title dry-run made no API
request and does not establish accuracy. Its private decisions, topic summary,
Markdown report, receipt, and verification report remain operator-local and
are rejected by basename if copied into a release artifact. The v12 gate
authorizes only complete title-bearing candidate-work manifest construction
and local routing. Paid API submission, production dispositions, episode
reselection, pivot classification, outcomes, fitting, and G4 remain closed.

The lean H3 synthetic result-space model, procedural blind challenge, and
uncertainty-aware peak-set development candidate remain repository-only too.
The subsequent set-identified analytic prospectus and its closed fixtures are
inside the same boundary and are definition-conformance artifacts, not a
package estimator or empirical result. Its public nonblind stochastic
development successor, generated panels, truths, estimates, reports, and
receipts are likewise repository-only or operator-local and create no package
API.
Their configs, compact summaries, research note, producers, estimators,
evaluators, and independent verifiers are forbidden from the sdist. Full
synthetic panels, private labels, estimates, diagnostic reports, receipts, and
verification outputs remain operator-local. The opened challenge is
discovery-only: it cannot provide fresh blind confirmation, estimator or
support selection, empirical row access, confirmatory execution, or a
scientific-effect claim.

The later candidate-screen scope correction and adaptive-frontier proof follow
the same repository-only boundary. Their v13/v14 ledgers, configs, compact
summaries, research notes, producers, and independent verifiers are forbidden
from the sdist. The 421,662-row candidate-window map, 994-row topic profile,
51-row initial frontier, bounded proof report, receipts, Markdown reports, and
verification reports remain operator-local and are rejected by basename if
copied into a release artifact. The bounded fixture result does not claim a
general formal proof, and all four content-required frontier rows remain
unresolved. No selection certificate or production reselection is packaged or
authorized.

The subsequent candidate-content dry-run remains repository-only as well. Its
v15 ledger, frozen aggregation contract, compact summaries, research note,
producer, and independent verifier are forbidden from the sdist. The private
four-row candidate table, twelve-row sensitivity table, counterfactual cost
estimate, Markdown report, receipt, and verification report remain
operator-local and are rejected by basename if copied into a release artifact.
Three dry-run pass states and the T12505 indexing-anomaly state are not
production dispositions; all four production states remain unresolved. No raw
scan, new API request, certificate, or reselection is packaged or authorized.

The v16-v29 candidate-successor chain is also repository-only. It includes
both exact T10543 96-title samplers, local routers, the disclosed
post-diagnostic candidate-rule-v2 replay, both production-rule freezes, the
rank-3 through rank-5 cumulative dispositions, the exact T12505 rank-4 and
T10151 rank-5 samples and routers, the bounded rank-5 API exception, compact
summaries, research notes, and independent verifiers. The first API contract
mistakenly targeted hepatitis B and is preserved as invalid audit-only
evidence. A corrected successor re-bound T10151 to its pre-sample hepatitis-C
card, used new opaque IDs, and completed two pinned-provider calls for the
same 25 locally ambiguous titles. Its dual-`HIGH` result added zero accept and
nine reject rows, leaving recent lower bounds at 27/64 accept and 21/64
reject. Detailed title samples, requests, raw provider responses, local and
API routes, disposition ledgers, frontier Parquet files, certificate-status
JSON, receipts, and verification reports remain operator-local. Frontier v7
has eight production dispositions: 3 pass, 4 fail, and 1 unresolved. Only
Computer Science has a field certificate. The required Medicine rank-5
T10151 row remains unresolved and cannot be bypassed by the later rank-6 pass,
so Medicine and global certificates and deterministic reselection remain
blocked. Human review was `NOT_RUN_BY_DESIGN`; API labels are not ground truth.
No outcome, fit, or G4 result is packaged or authorized.

The v30-v31 local-abstract successor remains inside the same repository-only
boundary. It recovered the exact sixteen unresolved OpenAlex IDs from the
operator-local abstract-inverted-index table without scanning the full value
column. Six payloads were reconstructable, five met the frozen mechanical
eligibility rule, ten were null, and one was an unusable fragment. The five
eligible abstracts were sent to one pinned OpenAI model and one pinned Claude
model under the same dual-`HIGH` rule. One accept and one reject were added,
leaving 28/64 accept, 22/64 reject, and 14 unresolved. Detailed recovered
abstracts, request keys, provider requests and responses, decisions,
disposition v7, frontier v8, certificate-status v3, receipts, and verification
reports remain operator-local. Medicine and global certificates remain
blocked. Human review was `NOT_RUN_BY_DESIGN`; provider labels are not ground
truth, and no outcome, fit, or G4 result is packaged or authorized.

The v32-v33 external-evidence successor is repository-only as well. Its exact
current OpenAlex, Crossref, and PubMed responses, identifier mapping, evidence
packet, provider request and response bodies, preserved transport-failure
receipt, decisions, disposition v8, frontier v9, certificate-status v4, and
verification reports remain operator-local. Repository-facing contracts and
bounded summaries record only that two accepts and three rejects were added,
leaving 30/64 accept, 25/64 reject, and 9 unresolved. Medicine and global
certificates remain blocked; human review was `NOT_RUN_BY_DESIGN`, provider
labels are not ground truth, and no outcome, fit, or G4 result is packaged or
authorized.

The preliminary temporal/author gate inventory at
`docs/case-studies/researcher-pivot/openalex-temporal-author-gate-inventory.md`
is also repository-only. It records data limitations and execution ordering;
it is not a package guide or a passed evidence artifact.

The G0 holdout-access seal contract, its independent validator, compact
summaries, research notes, creation and receipt-verification tools, and
v54/v55 gate ledgers are repository-only as well. The
contract freezes the 2020--2022 entry partition, follow-up through 2025, and
domain/author-cold rule while keeping seal creation, independent verification,
and opening separate. No holdout path or byte is included in the package. Any
future private operator tree under `pivot-feedback/holdout-seal-v1/` is
forbidden from the sdist, as is the uniquely named operator preflight. The
creator and verifier pass a 39-byte closed fixture, and the explicit-list-only
preflight author passes a separate 27-byte end-to-end closed fixture. Gate v56
initially kept production at
`NOT_RUN_MISSING_SOURCE_ROOT_AND_EXPLICIT_ARTIFACT_LIST`. The metadata-only
input resolver and gate v57 establish the more precise current boundary: the
G4 partition is logically reserved, but its selected domain, selected field,
and materialized artifact set do not exist yet. Production is therefore
`NOT_RUN_NO_MATERIALIZED_G4_ARTIFACT_SET`, not merely waiting for an operator
path.

The four-choice measurement/proxy precommitment, its closed synthetic fixture,
application and independent-verification tools, G0 v3 binding successor, and
gates v58-v59 are repository-only too. G0 v3 serializes the robust
attention-surge rule, age-3 field-year top-decile citation proxy, three-year
citation maturation lag, and one-percentage-point TOST practical-null policy.
These are outcome-blind design policies, not production measurements or
literature-mandated truths. Six choices are resolved in total; thirteen remain
null, all fifteen required sections remain not freeze-ready, and the holdout
receipt remains absent. Gate v59 authorizes only the next six-choice
discovery-only precommitment and closed fixture.

The resulting six-choice risk-set/weight/support contract, its closed fixture
and independent verifier, G0 v4 binding successor, and gates v60-v61 are also
repository-only. G0 v4 freezes ten controls per case, P01/P99 weight
winsorization, the 500/100/100 cohort floor, counts-one-through-ten
suppression, the 50/11/11 four-cell capacity floor, and a strict VIF-below-five
gate. These are conservative outcome-blind policies, not efficiency, power,
precision, or effect guarantees; the CMS suppression boundary is used only by
analogy. Twelve choices are resolved, seven remain null, all fifteen required
sections remain not freeze-ready, and the holdout receipt remains absent. Gate
v61 authorizes only the single G1 replication-tolerance precommitment and
closed fixture. The validated contracts and gates do not themselves create or
open a production seal, read a holdout row, pass G0, authorize model fitting,
or open G4.

## Release Checklist

Run this checklist before tagging a release candidate:

```bash
uv run ruff check src tests scripts
uv run pytest -q
git diff --check
uv run python scripts/inspect_release_artifacts.py --build
uv run python scripts/smoke_package_profiles.py
```

For a narrower default-profile check during iteration:

```bash
uv run python scripts/smoke_package_profiles.py --profiles default
uv run --no-dev python examples/toy_catalog.py
```

For the `main` / next-alpha candidate, also run:

```bash
uv run --no-dev python examples/research_pivot_scenario_lite.py
```

The artifact inspector should show that default wheel metadata does not require
`torch`, and the default-profile smoke should show `torch_loaded=false` while
blocking torch imports.

See [pre-release-artifact-flow.md](pre-release-artifact-flow.md) for the
metadata, version, Python-floor, wheel/sdist, and install-command review.

Use [git-distribution-flow.md](git-distribution-flow.md) for commit/tag based
installation from `https://github.com/kimyoungjin06/neural-abm`.

## Non-Goals

- Do not market `neural_abm.api` as no-torch while `NABMUnit`, `NABMStep`, and
  `SocialBlock` remain torch-backed.
- Do not move toy-owned payoff, resource, event, market, or evidence semantics
  into stable API metadata.
- Do not rename stable model IDs in artifacts or configs; feature names are the
  display layer.
- Do not treat a scenario comparison's `success=true` as scientific claim
  adjudication.
- Do not document the `v0.1.0a5` tag as containing the unreleased
  `scenario_lite` surface.
