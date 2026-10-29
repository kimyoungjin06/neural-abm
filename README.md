# Neural ABM

[![CI](https://github.com/kimyoungjin06/neural-abm/actions/workflows/ci.yml/badge.svg)](https://github.com/kimyoungjin06/neural-abm/actions/workflows/ci.yml)

Neural ABM is a pre-release Python package for Neural Agent-Based Models
(NABM): simulations where neural local updates, explicit social exchange, and
ABM-style state logging are kept separate and inspectable.

NABM aims at the space between rule-based ABMs and LLM-agent simulations.
Agents carry small learned policies that start from a hand rule, or from an
LLM judgment elicited once, and learn from pooled outcome records. That makes
them heavier than a fixed rule but far lighter than querying a language model
at every decision. See [Where NABM Sits](#where-nabm-sits).

The current alpha is clone-first. Start from a Git clone, use `uv`, and stay on
the default torch-free `neural_abm.api_lite` surface unless you explicitly need
the torch-backed lifecycle API. The current distribution path is the repository
and Git tags.

## Quick Start

Clone the current verified alpha tag and run the torch-free first-run check:

```bash
git clone --depth 1 --branch v0.1.0a6 https://github.com/kimyoungjin06/neural-abm.git neural-abm
cd neural-abm
uv run --no-dev python examples/first_run.py
```

The first-run output should report `status=ok`, `toy_count=10`, and
`torch_loaded=false`. Then inspect the full toy capability catalog:

```bash
uv run --no-dev python examples/toy_catalog.py
```

Query the lightweight API from the clone:

```bash
uv run --no-dev python - <<'PY'
from neural_abm.api_lite import toy_catalog, toys_by_taxonomy

catalog = toy_catalog()
binary_probability_toys = toys_by_taxonomy("output_family", "binary_probability")
print(len(catalog), binary_probability_toys)
PY
```

Use the default branch only when you intentionally want unreleased changes from
`main`. For early-user reports, reproduce against `v0.1.0a6` first.

Install from the current Git alpha tag only when you want to consume it as a
dependency:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a6"
```

Then query the torch-free toy catalog from another project:

```python
from neural_abm.api_lite import toy_catalog, toys_by_taxonomy

catalog = toy_catalog()
binary_probability_toys = toys_by_taxonomy("output_family", "binary_probability")
```

Install the torch-backed lifecycle API only when needed:

```bash
uv pip install "neural-abm[torch] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a6"
```

```python
from neural_abm.api import NABMUnit, SocialBlock, SocialChannel
```

The same tag carries a no-torch researcher scenario example. `v0.1.0a6` is
the first tag with it; it is not present in `v0.1.0a5`. From the clone:

```bash
uv run --no-dev python examples/research_pivot_scenario_lite.py
```

The example compares a science-of-science PIVOT question across baseline and
counterfactual environments through local adaptation, typed peer exchange,
domain-owned transition, and audit evidence.

## Main / Next-Alpha Candidate

`main` currently matches `v0.1.0a6` in its package surface. Changes added to
`main` after that tag are candidates for the next alpha. Test them in a
separate checkout so that their results are not confused with the verified
tag path:

```bash
git clone --depth 1 --branch main https://github.com/kimyoungjin06/neural-abm.git neural-abm-next-alpha
cd neural-abm-next-alpha
uv run --no-dev python examples/first_run.py
```

## What You Just Ran

- `examples/first_run.py` imports only from `neural_abm.api_lite`.
- The default clone-first path should report `torch_loaded=false`.
- The package currently exposes 10 toy capability entries.
- The first toy families to inspect are binary-probability social-learning
  models; the full taxonomy is printed by `examples/toy_catalog.py`.
- From the same tag, `examples/research_pivot_scenario_lite.py` shows a
  no-torch
  science-of-science PIVOT scenario: baseline vs interdisciplinary seed grants
  vs hot-field hype, with productive-pivot outcomes and aggregate/micro audit
  rows.
- Python 3.11 or newer is supported.

## Troubleshooting

If the clone-first path fails, capture the command output before changing the
environment or installing optional extras. Use the same command order as the
Quick Start:

```bash
uv --version
python --version
git rev-parse --short HEAD
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

Open an issue with the failed command, full output, operating system, Python
version, `uv --version`, the attempted Git ref, and whether you expected
torch-backed behavior. The early-user checklist is in
[docs/early-git-user-handoff.md](docs/early-git-user-handoff.md).

## Project Map

- [docs/release-readiness.md](docs/release-readiness.md): clone-first readiness
  and remaining gates before final `0.1.0`.
- [docs/git-distribution-flow.md](docs/git-distribution-flow.md): Git clone
  and commit/tag distribution flow.
- [docs/early-git-user-handoff.md](docs/early-git-user-handoff.md): stable,
  torch-backed, and experimental surfaces for early Git users.
- [docs/package-release-boundary.md](docs/package-release-boundary.md):
  product-facing entry points, install profiles, and package checklist.
- [examples/README.md](examples/README.md): lightweight package-facing examples.
- [docs/case-studies/researcher-pivot/README.md](docs/case-studies/researcher-pivot/README.md):
  end-to-end stylized predecessor with a torch-free fixed-rule study and a
  torch-backed learning-agent study, seed-paired replication, figures, and
  explicit limitations.
- Repository-only prospective pivot-feedback empirical/NABM protocol at
  `docs/case-studies/researcher-pivot/pivot-feedback-empirical-nabm-protocol.md`:
  prospective empirical/NABM study gates for delayed outcome visibility,
  cohort-level entry overshoot, capacity bridges, baseline comparison, and
  frozen holdout validation. The companion
  `docs/research/peer-exposure-identification-design.md`
  treats prior-collaborator activity and outcomes as time-eligible availability
  proxies rather than observed receipt or belief change, and prospectively
  separates field-wide attention, homophily, chaperoned entry or recruitment,
  capacity, and network-local signals. The fail-closed state ledgers at
  `experiments/configs/pivot_feedback_g_minus_1_readiness_v0.yaml` and
  `experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml` distinguish
  a structurally valid pending contract from an actual gate pass; neither can
  authorize execution. Its current G-1 evidence separates a raw
  zero-invalid recount blocked by one `W-1` row from a passed exact recount of
  the 492,293,206-row grammar-valid view. A subsequent 23-table full audit
  passed frozen work-source membership and row conservation, and measured
  declared-target coverage plus a descriptive author/work profile. A later
  current-snapshot taxonomy-reference component also passed across 1,423 files
  and 1,420,855,753 projected rows, with 1,282,707,794 complete valid hierarchy
  rows, 138,147,958 allowed all-null main-work rows, one frozen `W-1`
  quarantine, and zero partial or invalid hierarchy rows. A separate bounded
  positive-ID sensitivity pass then removed the retained `W0` row and found
  zero change in every one of the 26 non-null field margins; only the null
  margin and total work count fell by one. `W0` entity validity nevertheless
  remains unresolved. The resumable global work-topic relation auditor then
  completed the full 1,423-file production-source scan. It found 714 duplicate
  edge groups and 763 excess rows, with zero missing primary-child edges, but
  two duplicate groups carried different scores. The frozen exact-score dedup
  view was therefore not materialized and that parent remains immutable
  `BLOCKED_WORK_TOPIC_RELATION_COMPONENT` evidence. A versioned successor then
  rehashed all 1,430 parent artifacts and selected `0.0806` and `0.7347` only
  because the frozen main rows identified the same primary topics with those
  exact child scores. It materialized a 928,561,783-edge canonical relation;
  an independent read-only verifier replayed all parent event groups and every
  output record. The narrow relation component now passes, without rewriting
  the raw source or treating maximum score as a general rule. The separately
  authorized canonical work-time-spine run then scanned all 493 source files
  and retained 492,293,206 distinct grammar-valid works after quarantining the
  frozen `W-1` row. Its 4,096-partition, 9,185-artifact output passed a full
  read-only independent Parquet verification. The later full author-history
  successor closed 1,361,551,038 source rows into 779,630,258 distinct
  author-work events, and the full citation successor established a bounded
  current-snapshot bibliographic-time proxy. A completed full-root SHA-256 run
  separately establishes the current-content baseline; historical
  inventory-era byte identity remains `UNVERIFIED`. The versioned gate-order
  successor then passed bounded G-1A primitive/source fidelity. On 2026-07-24,
  all three threshold-free S0 components passed: the 18,544-cell core,
  full ORCID/affiliation projection, and 55,632-row exact non-ranked
  cap-10/50/100 coauthor-graph surface. The additive v3 ledger now authorizes
  only D0 discovery through 2012. D0-A subsequently selected Computer Science
  and Medicine from eight threshold-free structural metrics, froze two
  calendar-disjoint current-snapshot primary-topic episodes per field, and
  passed an independent replay of all 4,096 cutoff partitions and
  492,293,206 rows. The additive v4 ledger records D0 as partial and opens
  only outcome-blind D0-B author-opportunity and leave-out construction. The
  first D0-B component has now materialized 83,565 exact pre-decision works
  and 5,849,627 author-episode opportunity rows; a producer-independent Arrow
  replay matched all 16,384 output row groups across 1,271,923,464 replayed
  source rows. The additive v5 ledger completes only that author foundation
  and historically authorized only the remaining outcome-blind D0-B
  measurements. A successor linkage audit then found only 48,245 of 83,565
  selected works linked to valid author tokens. For Medicine T11361, only
  1,790 of 23,073 signal-window works link; type `other` supplies 22,451
  pre-decision works with only 14 linked. The additive v6 ledger therefore
  pauses leave-out/network/bridge work and authorizes only outcome-blind
  work-type/linkage ABI redesign plus deterministic episode reselection. It
  does not validate people, historical
  collaboration or affiliation observability, candidate eligibility,
  maturation/censoring, `t-1` leakage, final fields or thresholds, G0,
  final-panel conformance, model fitting, G4, empirical G1, or an effect.
  The later outcome-blind D0 title review executed two pinned API judges over
  the same frozen 384-title set. Both returned 384 valid structured decisions,
  with 306 label agreements and 197 dual-`HIGH` `ON_TOPIC` candidates.
  Human review is now explicitly `NOT_RUN_BY_DESIGN`; API consensus is not
  ground truth. The additive v11 ledger freezes a simple past-only
  observable-history new-topic pivot definition while keeping production
  classification closed. The v12 successor freezes four conservative topic
  cards and an independently replayed zero-cost routing dry-run: 229 titles
  route to local accept, 24 to local reject, 94 to an API exception queue, and
  37 to unresolved quarantine. This avoids 290 of 384 counterfactual API
  requests (75.5208%) but does not estimate accuracy. The existing 70,277 rows
  are topic-year candidates, not title-bearing candidate works. Only complete
  work-manifest materialization and local routing are now authorized; paid API
  submission, episode reselection, pivot classification, outcomes, fitting,
  and G4 remain closed. A subsequent scope audit corrected that immediate
  execution plan: 70,277 candidates span 994 topics, 421,662 six-year window
  rows, and 22,437,851 distinct linked works, while the four cards cover only
  236 candidates. The v13 successor therefore withdraws the eager full-title
  copy and replaces it with a deterministic adaptive frontier. Its bounded
  state machine passed independent replay across 299,584 exhaustive small
  fixtures, and v14 freezes a 51-row initial structural frontier with only four
  content-required rows. A v15 zero-cost aggregation of the existing 384
  titles passed T14430, T10579, and T12421 under the simple conservative rule
  and flagged T12505 as a joint score/source indexing anomaly. Candidate rule
  v1 then froze those four production dispositions. Two exact, entry-relative
  96-title T10543 samples subsequently failed the indexing-quality screen for
  different reasons: rank 2 had 52 opaque identifiers among 64 recent titles,
  while rank 3 showed a `0.93230` median topic-score drop and a `0.53125`
  dominant-source-share increase. The majority-opaque candidate-rule-v2
  amendment is explicitly post-diagnostic and not out-of-sample validation.
  Rank 4 T12505 then failed the retained majority-opaque indexing branch. Its
  exact 96-title sample contained 38 opaque identifiers among 64 recent
  titles, producing the seventh production disposition and frontier v5. The
  final required rank-5 T10151 card was frozen before target-title access; the
  full normalized replay selected 96 exact titles with no misses. Of 64 recent
  titles, 27 were locally accepted, 12 rejected, and 25 remained ambiguous.
  A bounded two-provider exception then evaluated only those 25 ambiguous
  titles under a dual-`HIGH` consensus rule. The first response pair was
  invalidated because its contract named hepatitis B instead of the frozen
  T10151 hepatitis-C target. The corrected pair added zero accept and nine
  reject rows, leaving lower bounds at 27/64 accept and 21/64 reject. Neither
  frozen content threshold fired. The exact 16 unresolved OpenAlex IDs were
  then recovered from the local abstract-inverted-index table: six had
  reconstructable text, five met the mechanical long-abstract eligibility
  rule, ten were null, and one was a three-word fragment. A second bounded
  provider pair reviewed only those five abstracts and added one dual-`HIGH`
  accept and one dual-`HIGH` reject. The remaining fourteen IDs were then
  matched against current OpenAlex, seven exact Crossref DOI records, and
  eight exact PubMed records. Eleven rows received new decision-relevant
  evidence and a no-retry provider pair added two accept and three reject
  rows; three rows with no new evidence were held out. Final lower bounds are
  30/64 accept, 25/64 reject, and 9 unresolved. Rank 5 therefore remains
  `CONTENT_UNRESOLVED`, not pass or fail. Frontier v9 has eight
  production dispositions (3 pass, 4 fail, 1 unresolved).
  Computer Science is the only field with a certificate. The earlier required
  Medicine rank-5 row blocks the later rank-6 pass, so Medicine and global
  certificates and deterministic reselection are not issued. No human or API
  label was used as ground truth; human review was `NOT_RUN_BY_DESIGN`.
  No outcome, model, or G4 gate is open.
- Repository-only prospective cross-field feasibility and planning tranche at
  `experiments/configs/openalex_g1_cross_field_feasibility_v0.yaml`,
  `experiments/evidence/results/openalex_g1_cross_field_feasibility_20260718.summary.json`,
  `experiments/evidence/results/openalex_g1_cross_field_feasibility_20260718.summary.md`,
  `experiments/evidence/results/openalex_g1_field_taxonomy_reference_20260718.summary.json`,
  `experiments/evidence/results/openalex_g1_field_taxonomy_reference_20260718.summary.md`,
  `experiments/configs/openalex_g1_w0_positive_id_sensitivity_v0.yaml`,
  `experiments/evidence/results/openalex_g1_w0_positive_id_sensitivity_20260718.summary.json`,
  `experiments/evidence/results/openalex_g1_w0_positive_id_sensitivity_20260718.summary.md`,
  `experiments/configs/openalex_g1_work_topic_relation_v0.yaml`,
  `experiments/evidence/results/openalex_g1_work_topic_relation_full_20260722.summary.json`,
  `experiments/evidence/results/openalex_g1_work_topic_relation_full_20260722.summary.md`,
  `experiments/configs/openalex_g1_work_topic_relation_primary_adjudication_v1.yaml`,
  `experiments/evidence/results/openalex_g1_work_topic_relation_primary_adjudication_full_20260722.summary.json`,
  `experiments/evidence/results/openalex_g1_work_topic_relation_primary_adjudication_full_20260722.summary.md`,
  `experiments/evidence/results/openalex_g1_canonical_full_work_time_spine_20260722.summary.json`,
  `experiments/evidence/results/openalex_g1_canonical_full_work_time_spine_20260722.summary.md`,
  `scripts/adjudicate_openalex_work_topic_primary_scores.py`,
  `scripts/audit_openalex_work_topic_relation.py`,
  `scripts/verify_openalex_work_topic_primary_adjudication.py`,
  `experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml`,
  `experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml`, and
  `scripts/plan_research_pivot_field_matrix.py`. The compact feasibility result
  is profile-only: it can revalidate the small taxonomy entity graph and the
  frozen cross-field profile projection. Its historical labels remain
  `field_entity_validity=UNVERIFIED`,
  `candidate_eligibility=NOT_ESTABLISHED`, and
  `final_field_selection=PENDING_G0`. The separate subsequent full component
  has status `PASS_CURRENT_SNAPSHOT_TAXONOMY_REFERENCE_COMPONENT`; it validates
  current-snapshot projected work-to-taxonomy references and exact 26-field
  plus null margins, but does not retroactively alter those compact-artifact
  labels or establish broader field validity. Final empirical field selection
  remains `PENDING`, empirical G1 is `NOT_RUN`, `pass_g1=false`, and no
  statistical or NABM model run or scientific effect is established. The
  planner freezes an engineering candidate universe with scientific selection
  `PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT` and
  `execution_authorized=false`; those remain the immutable planner's
  historical labels. The later S0 structural receipt does not rewrite that
  file. Its additive v3 gate authorized a separate pre-2013 D0 pilot, and the
  additive v4 successor records the outcome-blind two-field/four-episode D0-A
  freeze. The additive v5 successor records only the independently replayed
  pre-entry author-opportunity foundation. The additive v6 successor records
  the severe heterogeneous author-linkage gap and withdraws authorization to
  use those provisional episodes for leave-out construction pending ABI
  redesign and reselection. Those pilot choices and author rows are not final
  G0 fields, final eligible researchers, or scientific results.
- Repository-only policy-ladder ledger at
  `docs/research/pivot-policy-ladder-consolidation-v10.md`, which extends
  `docs/research/pivot-policy-ladder-consolidation-v9.md`,
  `docs/research/pivot-policy-ladder-consolidation-v8.md`,
  `docs/research/pivot-policy-ladder-consolidation-v7.md`, and
  `docs/research/pivot-policy-ladder-consolidation-v6.md` and, through
  them, v1 to v5: the hash-audited record of the researcher-pivot policy ladder,
  including the measured cost and behavior of each tier between rule-based
  ABMs and LLM agents, the closure of the current empirical model-ladder
  design before confirmatory scoring, the second domain (information spread)
  with the N14 reusability test, the N15 package parity, and the four
  information-spread follow-ups (table transfer, flip rate, unclipped prior,
  utility, prior calibration, and world calibration), and the entry-matched
  control of the Pivot LLM prior.
- [docs/classical-reductions.md](docs/classical-reductions.md): exact DeGroot
  and Granovetter special cases, plus explicitly labeled FJ-like anchored and
  self-excluding HK variants, as settings of the same lifecycle.
- [docs/toy-models/README.md](docs/toy-models/README.md): capability-first
  model-family roadmap.
- [docs/toy-models/capability-matrix.md](docs/toy-models/capability-matrix.md):
  current capability taxonomy and package catalog fields.
- [docs/api-surface-audit.md](docs/api-surface-audit.md): stable,
  experimental, internal, and paper-only API boundary.
- [docs/pre-release-artifact-flow.md](docs/pre-release-artifact-flow.md):
  alpha artifact, wheel/sdist, and install-command validation flow.

## Package Outputs

- Reusable simulation modules in `src/neural_abm/`.
- Package-facing examples in `examples/`.
- Release and profile-smoke scripts in `scripts/`.
- Public package docs in `docs/`.

## Public API and Package Status

Reusable code should start from the narrow stable facade:

```python
from neural_abm.api import NABMUnit, SocialBlock, SocialChannel
```

The package root is a lazy compatibility surface for existing module-path
imports. New code should use `neural_abm.api` for the stable torch-backed v0
contract. The first torch-free profile seed is `neural_abm.api_lite`; it exposes
runner, result, diagnostics, readiness utilities, NumPy-only social primitives,
and lightweight lifecycle reports/local-step primitives that can be imported
without loading torch. Its `SocialChannel` metadata is limited to
scalar/bounded scalar mix channels; distribution helpers remain standalone, and
tensor/state mixing requires the torch-backed API. Both `neural_abm.api` and
`neural_abm.api_lite` also expose feature-taxonomy helpers for mapping stable
model IDs to user-facing families.

The default package profile is now a lightweight torch-free install for the
`api_lite` surface. Full `NABMUnit` lifecycle work, `SocialBlock` tensor/state
mixing, and research workflows require explicit extras such as
`neural-abm[torch]`, `neural-abm[research]`, or `neural-abm[full]`.
Decision 0014 records the dependency policy and transition rules.
[Package release boundary](docs/package-release-boundary.md) records the
product-facing entry points and release checklist.

Torch-free catalog lookup:

```python
from neural_abm.api_lite import toy_catalog, toys_by_taxonomy

catalog = toy_catalog()
binary_probability_toys = toys_by_taxonomy("output_family", "binary_probability")
```

## Where NABM Sits

| | Rule-based ABM | NABM | LLM-agent simulation |
|---|---|---|---|
| Agent policy | Fixed hand-written rule | Small learned policy, with the rule or an elicited LLM judgment as its prior | Language model queried at each decision |
| What agents learn from | Nothing | Pooled outcome records | Prompt context |
| Cost per agent-step | 1× | About 2 to 7× for pooled linear, tree, and MLP policies | One model call |

The repository-only researcher-pivot case study measures this range in
synthetic, stylized worlds:

- Each step of the study exactly reproduces its rule-based predecessor before
  a learned arm is interpreted, so the rule ABM is a special case.
- Pooled linear, tree, and MLP policies cost 1.9 to 6.7 times the rule-based
  ABM per agent-step. Separate per-agent learners cost about 60 times, and with only
  neighbor-scale evidence they do worse than the rule they start from.
- The gain comes from learned policies rather than neural ones as such. An MLP
  wins on smooth success boundaries and a tree on axis-aligned ones.
- A local LLM (Qwen3-4B) was consulted once, calibrated to a stated base rate,
  and paired with a small exploration floor. It then selected more productive
  entrants than the hand rule, but it also admitted far fewer agents, and the
  same gain came from one shift of the hand rule's intercept. At equal
  selectivity in the world, the LLM prior ranked better only in the linear
  world under the fixed rule. Eliciting it took under 10 GPU
  minutes. Smaller models of the same family (0.6B, 1.7B) did not do the
  same job.
- The same model was also run as an LLM agent, queried at every agent-step
  with the public outcomes in its prompt. The NABM design matched or beat it
  in all 9 settings. "Matched" means each mean metric lay within a registered
  0.02 margin; it is not a test of statistical equivalence. The LLM agent needed about 110 GPU-seconds per run,
  against about 5 single-core CPU-seconds per NABM learner run, because it
  was queried at every step. Once evidence was in its prompt, its forecasts
  drifted above the population's success rate.
- When the same model instead role-played each researcher and decided for
  itself, it almost never chose to pivot and never stayed after a failed
  pivot. The NABM design matched or beat this agent in all 9 settings as well.

These are synthetic results from two LLM-agent designs and one local 4B model.
They are not claims about real researchers or about every LLM agent. The
learned arms include linear and tree learners, so they do not establish that a
neural policy is necessary. An empirical test of whether real researchers'
entry follows visible outcomes was closed before confirmatory scoring: at the
measured data shape, the registered analysis lacked the power to tell learning
from anticipation. This does not imply the absence of learning. The
repository-only, hash-audited ledger is at
`docs/research/pivot-policy-ladder-consolidation-v5.md`, which extends
`docs/research/pivot-policy-ladder-consolidation-v4.md`.

A second domain, information spread with delayed fact-checks, ran the same
comparison through a shared core. The core's semantics are now the
`neural_abm.policy_ladder` and `neural_abm.policy_learners` modules, first
released in `v0.1.0a6`.

- **Learners against the hand rule.** In the stationary worlds, every
  learner beat it.
- **Tables written before the run.**
  - A table from one donor run in the same world was not detectably
    different from the closed loop in 11 of 12 settings.
  - The average of four donor runs beat the closed loop in all 12 settings.
    So did a four-donor table whose donors ran in other worlds of the same
    generator.
- **When the closed loop wins.** It won when a run carried information the
  donors lacked. It beat the best table once about a quarter of runs did.
- **LLM comparators.**
  - The learner beat AgentTorch-style LLM archetypes in every setting.
  - An LLM prior added nothing detectable inside the learner on share
    precision. Scored by the agents' own utility, it mostly lowered it.
    For the once-elicited prior, with share rates matched in the world,
    that loss traced to how the prior was calibrated in one regime, and to
    both its calibration and the model's answers in the other.
- **Reproduction.** The Pivot ladder was re-expressed through the same core,
  and then through the package. Both times it reproduced its recorded runs
  byte for byte.

These results are synthetic, and they are scored on items whose fact-check
was not yet public. They make no claim about real information spread. The
ledger is `docs/research/pivot-policy-ladder-consolidation-v10.md`, which
extends `docs/research/pivot-policy-ladder-consolidation-v9.md`,
`docs/research/pivot-policy-ladder-consolidation-v8.md`,
`docs/research/pivot-policy-ladder-consolidation-v7.md`, and
`docs/research/pivot-policy-ladder-consolidation-v6.md`.

## Scope

In this repository, a model is inside the NABM claim when it has:

- neural agents or policies whose local update changes the simulated behavior;
- an ABM time loop with explicit scheduling, state transition, and logging;
- a separated local update and social update path;
- social message, peer selection, typed mixing, and commit stages;
- ABM-style aggregate and micro-state outputs for inspection.

The current framing is conservative:

- Position NABM between rule-based ABMs and LLM-agent simulations, as a
  replacement for neither.
- Do not claim a general Transformer replacement.
- Do not claim a universal simulator or general-purpose ABM framework.
- Treat the core unit as a Neural ABM Node with learnable social mixing.
- Position the system as a temporal heterogeneous GNN-style simulator with
  neural agents, explicit social update rules, and ABM logging.

## Development

Use `uv` for all Python work:

```bash
uv sync
uv run pytest
```

`uv sync` installs the dev dependency group used by the full research test
suite. For package-profile checks, use `uv build` plus isolated installs of the
wheel or explicit extras. The release-smoke helper is:

```bash
uv run python scripts/smoke_package_profiles.py
```

The pre-release artifact inspection command is:

```bash
uv run python scripts/inspect_release_artifacts.py --build
```

The no-torch catalog example is:

```bash
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

The primary user path is a Git clone. Direct Git URL installs are also
supported from committed refs or tags:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a6"
```

The package currently supports Python 3.11 or newer.
