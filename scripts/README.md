# Scripts

Use this directory for thin command-line entry points.

Maintainer support scripts:

- `audit_openalex_source_inventory.py`: run the first, footer-only G-1 source
  gate from `experiments/configs/openalex_g1_source_inventory_v0.yaml`. It
  verifies the root/table/file contract, footer totals, canonical and physical
  schemas, selected null statistics, declared repair lineage, and inventory
  stability without scanning data pages. The default result is
  `INVENTORY_ONLY` while content SHA-256 remains pending. Even `PASS_METADATA`
  would pass only this metadata component: G-1 data readiness would still need
  the separate body, join, author, date, panel, and leakage gates, while
  empirical G1 would remain `NOT_RUN` (`pass_g1=false`) until its registered
  analyses execute.
  For the local metadata pass:

  ```bash
  OPENALEX_DERIVED_ROOT=/operator/local/root \
    uv run python scripts/audit_openalex_source_inventory.py \
    --output experiments/runs/openalex-g1/source_manifest.json \
    --markdown-output experiments/runs/openalex-g1/source_manifest.md \
    --require-inventory-pass
  ```

  `--content-sha256` forces one sequential worker because the source is a
  rotational-disk-scale root; it is a multi-hour byte-freeze job, not a
  substitute for G-1 scientific validation. That legacy flag has no per-file
  resume state; use the dedicated content-freeze command below for a production
  current-byte baseline.
- `audit_openalex_content_freeze.py`: run the separate resumable current-byte
  freeze under `experiments/configs/openalex_g1_content_freeze_v0.yaml`. It
  binds the footer-only parent manifest and its exact path-size-mtime
  fingerprints, hashes one regular contained source file at a time, and writes
  an atomic checkpoint after every successful file. Resume revalidates the
  plan, checkpoint digest, parent identity, and current
  device/inode/mode/size/mtime/ctime before reusing a hash; it does not rehash
  completed content. Before report construction and again during final-output
  validation, every checkpoint is re-read from disk and its exact file digest,
  schema, task binding, self-digest, and current source stat are checked; only
  those revalidated objects build the checkpoint and content trees. The
  checkpoint SHA-256 is an unkeyed accidental-corruption check, not
  authentication against a writer who controls the output directory. The tool
  creates new output components with private permissions and requires the leaf
  to be operator-owned and not group/world writable; every ancestor must be
  root/operator-owned, with sticky protection on any shared-writable ancestor.
  A report is complete only when both the valid outputs manifest and the
  atomically written final `COMPLETE` state are present. The state is written
  last, and a caught publication failure removes the report, Markdown, and
  outputs manifest before recording `PAUSED`. The recorded tool SHA-256 binds a
  same-run snapshot of the tool file, not cryptographic attestation of the
  Python code already loaded into memory; a trusted launcher or signed artifact
  would be required for that stronger claim. A `--max-files` invocation is
  always `PILOT_CONTENT_FREEZE_ONLY`, even when the bound is larger than the
  declared file count.

  ```bash
  uv run python scripts/audit_openalex_content_freeze.py \
    --source-root /operator/local/root \
    --output-dir /operator-private/openalex-g1/current-content-v0 \
    --require-pass
  ```

  After a file error or operator interruption, use the same immutable source,
  tool, config, parent manifest, and output directory with `--resume`. A full
  pass would freeze only the current bytes for future comparison. It cannot
  reconstruct the missing earlier content baseline, retroactively bind prior
  audits, pass G-1, or establish an effect. On the current host, resolve the
  recent `/dev/sda` read/reset event and coordinate other HDD readers before
  launching the 920.523 GiB production run.
- [`recount_openalex_g1.py`](recount_openalex_g1.py): run the resumable exact
  main-work body scan under the original zero-null, zero-invalid contract in
  [`openalex_g1_recount_v0.yaml`](../experiments/configs/openalex_g1_recount_v0.yaml).
  The full run is intentionally
  [`BLOCKED`](../experiments/evidence/results/openalex_g1_main_id_recount_20260716.blocked.summary.md):
  all 492,293,207 body rows were scanned, with zero null IDs and one malformed
  `W-1`; exact distinct and duplicate fields are unavailable under this failed
  raw-source contract.

  ```bash
  OPENALEX_DERIVED_ROOT=/operator/local/root \
    uv run python scripts/recount_openalex_g1.py \
    --inventory-manifest experiments/runs/openalex-g1/source_manifest.json \
    --output-dir /local/ssd/openalex-g1/main-id-raw \
    --require-pass
  ```

  `--require-pass` returns nonzero for the frozen raw source and must not be
  weakened or reinterpreted as a clean source-key result.
- [`recount_openalex_g1_canonical_quarantine.py`](recount_openalex_g1_canonical_quarantine.py):
  run the separate frozen-quarantine amendment in
  [`openalex_g1_recount_canonical_quarantine_v1.yaml`](../experiments/configs/openalex_g1_recount_canonical_quarantine_v1.yaml).
  Its [dated evidence](../experiments/evidence/results/openalex_g1_grammar_valid_main_id_recount_20260716.summary.md)
  passes only the 492,293,206-row grammar-valid analysis view: all retained IDs
  are distinct, duplicate excess is zero, and the real numeric `W0` key is
  retained as `WARN_ZERO_ID_UNRESOLVED`.

  ```bash
  OPENALEX_DERIVED_ROOT=/operator/local/root \
    uv run python scripts/recount_openalex_g1_canonical_quarantine.py \
    --inventory-manifest experiments/runs/openalex-g1/source_manifest.json \
    --output-dir /local/ssd/openalex-g1/main-id-grammar-valid \
    --require-pass
  ```
- [`audit_openalex_sentinel_incidence.py`](audit_openalex_sentinel_incidence.py):
  scan the source `id` and declared work-target columns under
  [`openalex_g1_sentinel_incidence_v0.yaml`](../experiments/configs/openalex_g1_sentinel_incidence_v0.yaml).
  The [dated targeted audit](../experiments/evidence/results/openalex_g1_sentinel_incidence_20260716.summary.md)
  found `W-1` in 21 source rows across six files and `W0` in six source rows
  across six files, with zero target rows for both. It does not apply the
  quarantine or prove relation closure.

  ```bash
  OPENALEX_DERIVED_ROOT=/operator/local/root \
    uv run python scripts/audit_openalex_sentinel_incidence.py \
    --inventory-manifest experiments/runs/openalex-g1/source_manifest.json \
    --output-json experiments/runs/openalex-g1/sentinel-incidence.json \
    --output-md experiments/runs/openalex-g1/sentinel-incidence.md
  ```
- [`audit_openalex_work_source_membership.py`](audit_openalex_work_source_membership.py):
  stream all 23 declared source tables under
  [`openalex_g1_work_source_membership_v0.yaml`](../experiments/configs/openalex_g1_work_source_membership_v0.yaml),
  apply the frozen `W-1` quarantine only in the accounting view, validate every
  retained child source ID against the 492,293,206-ID main-work set, measure
  open-world coverage for the declared work-target columns, and build the
  descriptive author-ID/work profile. The
  [2026-07-17 full audit](../experiments/evidence/results/openalex_g1_work_source_membership_20260717.summary.md)
  passed exact membership and row conservation across 21,019 files and
  20,996,232,817 data-page rows. It measured target coverage and author-ID
  missingness; it did not materialize a filtered view, close the target space,
  measure duplicate source-target edges, or establish author eligibility.

  ```bash
  OPENALEX_DERIVED_ROOT=/operator/local/root \
    uv run python scripts/audit_openalex_work_source_membership.py \
    --inventory-manifest experiments/runs/openalex-g1/source_manifest.json \
    --master-ids /local/ssd/openalex-g1/main-id-grammar-valid/main_ids.grammar_valid.sorted.u64 \
    --output-dir /local/ssd/openalex-g1/work-source-membership \
    --require-pass
  ```

  These source audits are CPU/I/O workloads. A current full-root content-hash
  baseline, unresolved earlier-inventory byte equality, author entity and
  split/merge validity, duplicate-edge checks,
  date/censoring/`available_at` and leakage validation, candidate eligibility,
  and a deterministic representative pilot remain pending. G-1 data readiness
  is therefore `PENDING`, and empirical G1 is `NOT_RUN` (`pass_g1=false`). The
  later bounded `W0` positive-ID sensitivity component is documented below; it
  does not resolve `W0` entity validity or promote G-1.
- `summarize_openalex_cross_field_feasibility.py`: validate the seven-file
  OpenAlex domain/field/subfield/topic taxonomy graph and independently
  reproject the frozen full work-source report's 122,472-cell joint cube into
  exact domain, field, authorship-status, and temporal-window margins. This
  command reads no work Parquet data. It uses
  `experiments/configs/openalex_g1_cross_field_feasibility_v0.yaml` and writes
  the path-redacted compact JSON and Markdown evidence.

  ```bash
  OPENALEX_TAXONOMY_ROOT=/operator/local/openalex-taxonomy \
  OPENALEX_G1_WORK_SOURCE_FULL_REPORT=/local/ssd/openalex-g1/work_source_membership_audit.json \
    uv run python scripts/summarize_openalex_cross_field_feasibility.py \
    --output-json /local/ssd/openalex-g1/cross-field-feasibility.summary.json \
    --output-markdown /local/ssd/openalex-g1/cross-field-feasibility.summary.md
  ```

  `PASS_CROSS_FIELD_PROFILE_FEASIBILITY_ONLY` means only that the small
  taxonomy graph and frozen profile projection agree. Field-entity validity
  remains `UNVERIFIED`, candidate eligibility remains `NOT_ESTABLISHED`, final
  field selection remains `PENDING_G0`, G-1 and G0 remain `PENDING`, and
  empirical G1 remains `NOT_RUN` (`pass_g1=false`). These are the compact
  artifact's historical boundary labels; the later full reference component
  does not retroactively rewrite them.
- `plan_research_pivot_field_matrix.py`: validate the tracked
  `experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml` engineering
  manifest or expand it into a deterministic plan JSON. It exposes only
  `validate` and `plan`; it has no simulation, fitting, or launch command.

  ```bash
  uv run python scripts/plan_research_pivot_field_matrix.py validate \
    --manifest experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml

  uv run python scripts/plan_research_pivot_field_matrix.py plan \
    --manifest experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml \
    --output /local/ssd/openalex-g1/pivot-field-matrix-plan.json
  ```

  A written plan remains `ENGINEERING_ONLY`: scientific field selection is
  `PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT`, `execution_authorized=false`, and
  no plan cell is empirical or simulation evidence.
- `verify_pivot_feedback_g_minus_1_readiness.py`: verify the byte-frozen
  repository evidence bindings in the fail-closed G-1 readiness registry. It
  also re-runs the exact bounded author-history v2 fixture through a fixed,
  isolated, shell-free subprocess and checks its pinned evidence class,
  runtime fingerprint, six foundation records, and canonical stdout. The
  CPython 3.14 / Unicode 16 stdout hash is reattested only on that runtime;
  other supported runtimes report `NOT_REATTESTED_RUNTIME_MISMATCH` without a
  cross-runtime byte claim. By default the two operator-local control receipts
  remain not reverified. Explicit receipt paths may be supplied for read-only
  hash and field checks; even when both match, the three missing production,
  structural-audit, or discovery receipts keep G-1 pending.

  ```bash
  uv run python scripts/verify_pivot_feedback_g_minus_1_readiness.py

  uv run python scripts/verify_pivot_feedback_g_minus_1_readiness.py \
    --receipt current_content_baseline=/operator/local/content_freeze_audit.outputs.json \
    --receipt main_temporal_v3=/operator/local/main_temporal_validity_audit.outputs.json \
    --require-operator-receipts
  ```

  Both successful verification modes still emit `G_MINUS_1_PENDING`,
  `pass_g_minus_1=false`, and `execution_authorized=false`. The one bounded
  synthetic subprocess does not expose a source-root, production-scan,
  model-run, or output-file interface.
- `verify_pivot_feedback_g0_measurement_freeze.py`: validate the additive,
  pending G0 overlay and its byte-exact peer-registry and cross-field-matrix
  parents.

  ```bash
  uv run python scripts/verify_pivot_feedback_g0_measurement_freeze.py validate
  uv run python scripts/verify_pivot_feedback_g0_measurement_freeze.py seal
  ```

  `validate` confirms only `VALID_PENDING_G0_MEASUREMENT_FREEZE`; it does not
  pass G0. `seal` is intentionally blocked until a versioned successor binds
  the missing G-1 and holdout receipts and resolves every registered
  quantitative choice. `run`, `execute`, `authorize`, and `promote` are also
  fail-closed commands, and none of these surfaces starts empirical or GPU
  work.
- `audit_openalex_field_taxonomy_reference.py`: run the resumable projected
  main-work plus exploded work-topic audit under
  `experiments/configs/openalex_g1_field_taxonomy_reference_v0.yaml`. It checks
  current-snapshot taxonomy IDs, parent chains, embedded names, scores, row
  conservation, and frozen field/status margins across 493 main and 930 topic
  files.

  ```bash
  OPENALEX_DERIVED_ROOT=/operator/local/openalex-derived \
  OPENALEX_TAXONOMY_ROOT=/operator/local/openalex-taxonomy \
  OPENALEX_G1_WORK_SOURCE_FULL_REPORT=/local/ssd/openalex-g1/work_source_membership_audit.json \
    uv run python scripts/audit_openalex_field_taxonomy_reference.py \
    --inventory-manifest /operator/local/openalex-g1/source_manifest.json \
    --output-dir /local/ssd/openalex-g1/field-taxonomy-reference \
    --resume \
    --require-pass
  ```

  The tracked full run has status
  `PASS_CURRENT_SNAPSHOT_TAXONOMY_REFERENCE_COMPONENT`. It scanned 1,423 files,
  1,420,855,753 projected rows, and 166,699,471,247 source bytes. Exact
  conservation comprises one frozen `W-1` quarantine row, 138,147,958 allowed
  all-null retained main-work rows, and 1,282,707,794 complete valid hierarchy
  rows, with zero partial or invalid hierarchy rows. The one retained `W0`
  occurrence remains an unresolved warning. All 26 non-null field margins and
  the null margin match the frozen compact profile exactly. See the tracked
  [JSON](../experiments/evidence/results/openalex_g1_field_taxonomy_reference_20260718.summary.json)
  and
  [Markdown](../experiments/evidence/results/openalex_g1_field_taxonomy_reference_20260718.summary.md).
  Their SHA-256 values are, respectively,
  `f8a0d01ee5b04133c1472d4ed7c07f7a578343004c859645f174f1a0f6021f62`
  and
  `1e9db873d97aa9035fde472d587d91a9c1f3eb7592816ee1df1f149ce9ac77e9`;
  the result fingerprint is
  `36209d119cc32802d0e7391b92425dcef931e9513841afa19e8be7b0bc06b30b`.

  `--max-files-per-table` is a bounded smoke path and always returns
  `PILOT_REFERENCE_ONLY`; it cannot establish the full component pass. Even
  `PASS_CURRENT_SNAPSHOT_TAXONOMY_REFERENCE_COMPONENT` leaves duplicate
  work-topic edges, primary-topic child inclusion, publication dates,
  cutoff-aware `available_at`, author entities, candidate eligibility, G-1,
  G0, and final field selection pending, with empirical G1 `NOT_RUN` and
  `pass_g1=false`. Raw source validity remains `BLOCKED`, `W0` remains
  `UNRESOLVED`, and parent content identity remains `UNVERIFIED`. The pass
  establishes no model result or scientific effect. This is a local CPU/I/O
  workload, not a GPU job.

  These three cross-field commands, their configs, and all outputs under
  `experiments/evidence/` are repository-only research artifacts and are not
  included in the package sdist.
- `audit_openalex_w0_positive_id_sensitivity.py`: revalidate the two current
  Parquet projections that contain the retained `W0` main-work and authorship
  rows, then propagate the predeclared positive-work-ID exclusion through the
  frozen 26-field profile. The tracked 2026-07-18 run projected 2,000,000 rows
  and passed `PASS_W0_POSITIVE_ID_EXCLUSION_SENSITIVITY_COMPONENT`.

  ```bash
  OPENALEX_DERIVED_ROOT=/operator/local/openalex-derived \
  OPENALEX_G1_WORK_SOURCE_FULL_REPORT=/local/ssd/openalex-g1/work_source_membership_audit.json \
    uv run python scripts/audit_openalex_w0_positive_id_sensitivity.py \
    --output-json /local/ssd/openalex-g1/w0-positive-id.summary.json \
    --output-markdown /local/ssd/openalex-g1/w0-positive-id.summary.md \
    --require-pass
  ```

  The positive-ID view has 492,293,205 works rather than 492,293,206. Only the
  null-primary-field, 2024, one-declared-author, no-usable-author-ID stratum
  changes by one; all 26 non-null field counts are unchanged. The tracked
  [JSON](../experiments/evidence/results/openalex_g1_w0_positive_id_sensitivity_20260718.summary.json)
  and
  [Markdown](../experiments/evidence/results/openalex_g1_w0_positive_id_sensitivity_20260718.summary.md)
  have SHA-256 values
  `999bf3eff8ebb5fbb506ecbb571d6d491e8876f297c7a12163ca24b2a1d30157`
  and
  `05d8e6e84b1e9f31fa446209ae4e242fd5b25a03f7254ebd940308d88eb62216`;
  the result fingerprint is
  `3056af90867d813ef70b9bcf11c67385c3b288d2f21d89de150c4f2ddbc30190`.
  `W0` entity validity remains `UNRESOLVED`. The bounded v3 main temporal-field
  component now passes, while edge time, authors, candidates, G-1, G0, and
  empirical G1 remain pending or not run.
- `audit_openalex_canonical_work_time_spine.py`: exercise the frozen canonical
  four-column work-time-spine ABI against synthetic fixtures. The executable
  validates parent receipts, disposition classification, all 4,096 partition
  surfaces, artifact hashes, terminal re-entry, and failure cleanup without a
  production source-root interface: its CLI accepts only
  `--synthetic-source-root`, rejects the production root path identity, and
  enforces small task/row/byte caps before opening any source payload. Its status is
  `IMPLEMENTED_SYNTHETICALLY / PRODUCTION_NOT_RUN`; it creates no production
  spine or temporal component pass.
- `audit_openalex_canonical_work_time_spine_v2.py`: preserve the same
  synthetic-only safety boundary while validating the versioned canonical
  contract against a temporal-v3-only parent ABI. It rejects temporal v1/v2
  parents, binds the exact v3 full-evidence anchors in the production design,
  and still exposes no production source-root interface. Its status remains
  `IMPLEMENTED_SYNTHETICALLY / PRODUCTION_NOT_RUN`.
- `audit_openalex_canonical_work_time_spine_v3.py`: production-capable,
  repository-only successor with distinct production, unpromotable pilot,
  and closed-fixture schemas. Full execution requires three independent
  acknowledgements (`--execute`, `--authorize-full`, and
  `--acknowledge-exclusive-source-scan`). The engine validates the frozen
  content and temporal-v3 control plane without replaying the parent's
  five-column payload scan, then performs one bounded three-column source
  scan, fixed-width staging, bounded-FD spooling, chunked external merge sort,
  exact adjacent duplicate detection, streaming Parquet publication, and
  receipt-last verification. Its implementation status does not imply a
  result. A separately authorized one-file pilot remains nonpromotable. The
  later 2026-07-22 full run scanned all 493 source files, published 4,096
  partitions and 9,185 receipt artifacts for 492,293,206 retained distinct
  works, and passed the independent verifier's default full-Parquet mode. Its
  runtime tree remains outside the repository; only a compact component-bounded
  summary is retained under `experiments/evidence/results/`.
- `summarize_openalex_canonical_work_time_spine_pilot_benchmark.py`: validate
  an already completed one-file pilot report plus a private four-line GNU
  `time` capture, then publish a new operator-local, nonpromotable planning
  diagnostic. The v1 capture does not prove exit status, command identity,
  fresh execution, or report/capture pairing. The tool never launches the
  runner, passes a capacity gate, creates a receipt parent, or authorizes full
  execution. A 2026-07-21 operator-local pilot benchmark now exists, but it is
  nonpromotable and full execution remains `NOT_RUN`.
- `benchmark_openalex_canonical_work_time_spine_downstream.py`: after an
  independent read-only pilot verification, copy only the receipt-bound 16 MB
  staging payload into fresh private `/ssd` scratch and exercise the frozen
  downstream private kernels. The reviewed bounded path performs 4,096-way
  demultiplex, all 4,096 pilot-occupancy single-run/empty Parquet partitions,
  decoded validation and hashes, and three hash sweeps over an exact 9,185-file
  benchmark topology assembled from 493 benchmark shards. It accepts no source
  root and never calls production finalization or receipt construction. Its
  immutable result is operator-local and nonpromotable; a completed result
  remains partial evidence and cannot pass the capacity gate or authorize a
  full run.
- `benchmark_openalex_canonical_work_time_spine_downstream_stress_v1.py`:
  define a repository-only bounded stress surface for the downstream benchmark
  path. Its immutable result is operator-local, operational-only, and
  nonpromotable.
- `verify_openalex_canonical_work_time_spine_downstream_stress_v1.py`:
  independently verify the bounded stress result in read-only mode. External
  verification returned
  `PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_STRESS_VERIFICATION_NONPRODUCTION`;
  the verifier SHA-256 is
  `4ade14fd53f80a964ff8668b7f4e2391ea220402327fcc0fca9c0dd88e19373f`.
  Hardening narrows the acceptance set with the exact result pin, nested
  invariants, and row-group plus no-dictionary validation. This status remains
  nonproduction and noncapacity; it cannot pass the capacity gate or authorize
  a full run.
- `verify_openalex_canonical_work_time_spine_downstream_benchmark_v1.py`:
  independently verify the bounded downstream result without importing the
  benchmark harness or production runner and without opening any supplied
  path for write. It reconstructs all three exact 9,185-artifact scratch
  surfaces, the 493-shard staging/demultiplex commitments, every partition
  checkpoint and decoded Parquet record digest, controls, H9185 hashes,
  aggregate metrics, and the pilot/result bindings. Its canonical stdout pass
  remains explicitly nonproduction and cannot pass the capacity gate.
- `review_openalex_canonical_work_time_spine_capacity.py`: verify the frozen
  production config/runner hashes and emit a conservative analytic I/O/time
  sensitivity model to stdout without importing the runner or accepting a
  source path. Its frozen v0 input remains an analytic-estimate-only snapshot;
  it does not ingest the later bounded downstream result, pass the capacity
  gate, or record operator acceptance.
- `verify_openalex_canonical_work_time_spine_v1.py`: independently reconstruct
  the terminal output ABI in read-only mode. Its default mode streams all
  published Parquet; `hash-only` emits a distinct weaker status. It neither
  launches a source scan nor attests loaded code, replays the source projection,
  refreshes parent artifacts or creates a production receipt. It independently
  passed the 2026-07-22 production output in default full-Parquet mode across
  all 4,096 partitions and observed the receipt-bound output tree unchanged.
- `audit_openalex_citation_available_at.py`: exercise the frozen directed
  citation chronology ABI against synthetic fixtures in both direct-temporal
  and canonical-spine-reuse modes. It tests exact-pair multiplicity, the
  seven-bucket chronology classifier, accounting closure, receipt-last
  publication, and terminal re-entry. It is
  `IMPLEMENTED_SYNTHETICALLY / PRODUCTION_NOT_RUN`; citing-work publication
  date remains a bibliographic event-time convention rather than an OpenAlex
  first-observation timestamp.
- `audit_openalex_citation_available_at_v2.py`: exercise the versioned
  temporal-v3-parent-only citation chronology contract against closed synthetic
  direct-reconstruction fixtures. It binds the exact temporal-v3 evidence ABI
  and reviewed canonical successor config/tool hashes. Its frozen synthetic ABI
  retains `BLOCKED_CANONICAL_PRODUCTION_RECEIPT_NOT_AVAILABLE`; the later
  production receipt does not retroactively rewrite that historical output or
  make the tool a production runner. A new citation successor must bind the
  receipt explicitly. The v2 tool emits only distinct synthetic schemas/statuses
  and remains `PRODUCTION_NOT_RUN`.
- `audit_openalex_citation_available_at_v3.py`: execute the receipt-bound
  production successor against all current referenced-work rows and the exact
  canonical work-time-spine receipt. It performs exact directed-pair
  deduplication, the frozen seven-bucket classifier, open-world accounting,
  and 4,096-way deduplicated plus chronology-safe private Parquet products.
  Prefix runs are permanently nonpromotable; full execution requires
  `--execute`, `--authorize-full`, and
  `--acknowledge-exclusive-source-scan`. This is CPU/I/O work and uses no GPU.
  The completed 2026-07-22 full run scanned 3,042,536,160 rows and published
  3,042,536,160 unique pairs plus 2,858,679,472 chronology-safe pairs under a
  19,438-artifact receipt.
- `verify_openalex_citation_available_at_v3.py`: independently rehash every
  receipt artifact and decode every final edge row. It recomputes pair
  multiplicity, bucket precedence, nullability, date/lag arithmetic,
  cross-tabs, sensitivity counts, sortedness, and source/target partition
  formulas without importing the producer. Its PASS still cannot establish
  historical edge observation, maturation, G-1, G0, empirical G1, or an
  effect. The full production output passed this verifier's complete artifact
  and row scans.
- `verify_openalex_work_side_author_history_fixture_v2.py`: validate the
  versioned author-token accounting core against a closed JSON fixture and emit
  one canonical JSON report to stdout. It has no production root, output
  directory, or production execution option. A pass covers exact author-work
  deduplication, temporal disposition 2/3 eligibility, yearly/token reductions,
  the `A0` sensitivity view, and team-cap threshold closure only. The v0/v1
  author designs remain byte-exact historical surfaces; canonical production,
  author identity, G-1, and G0 remain blocked or pending.
- `audit_openalex_work_side_author_history_v3.py`: execute the production
  work-side author-token accounting component against exact content,
  temporal-v3, and canonical-spine receipts. It is a CPU/I/O pipeline with
  4,096 work and author partitions, receipt-bound event/year/token Parquet,
  exact source and join closures, and receipt-last publication. Prefix runs are
  permanently nonpromotable; full execution requires `--execute`,
  `--authorize-full`, and `--acknowledge-exclusive-source-scan`.
- `verify_openalex_work_side_author_history_v3.py`: independently rehash the
  production output, reopen all event/year/token partitions, recompute every
  published reduction and sensitivity cell, and verify aggregate cross-surface
  closures without importing the producer or opening source payloads. It does
  not replay the source projection or work join. The completed 2026-07-22 full
  output passed this verification, reducing the G-1 missing-component count to
  four while leaving author entities and past observability unverified.

  ```bash
  uv run python scripts/verify_openalex_work_side_author_history_v3.py \
    --output-root /operator/local/openalex-g1-author-history-v3-full
  ```
- `audit_openalex_work_topic_relation.py`: prepare the resumable global
  work-topic multiplicity, exact-score conflict, deterministic dedup, and
  primary-topic child-inclusion audit under
  `experiments/configs/openalex_g1_work_topic_relation_v0.yaml`. Its synthetic
  tests pass, and a 2026-07-21 one-file-per-table production-source pilot
  completed with `PILOT_WORK_TOPIC_RELATION_ONLY`. That pilot is permanently
  nonpromotable. The 2026-07-22 full 1,423-file run then completed with
  `BLOCKED_WORK_TOPIC_RELATION_COMPONENT`: it found 714 duplicate edge groups,
  763 excess rows, zero missing primary-child edges, and two exact-score
  conflicts. The deduplicated relation was not materialized.

  ```bash
  OPENALEX_DERIVED_ROOT=/operator/local/openalex-derived \
  OPENALEX_TAXONOMY_ROOT=/operator/local/openalex-taxonomy \
    uv run python scripts/audit_openalex_work_topic_relation.py \
    --output-dir /local/ssd/openalex-g1/work-topic-relation-pilot \
    --max-files-per-table 1 \
    --execute
  ```

  Every max-files or hash-bucket execution is
  `PILOT_WORK_TOPIC_RELATION_ONLY`; it cannot promote the full relation gate.
  Any reviewed full rerun additionally requires `--allow-full-scope` and
  currently
  has a 77.35 GiB artifact upper bound and a 31.76 GiB minimum sort-array
  estimate. The frozen safety policy raises the preflight thresholds to 106.69
  GiB free disk and 51.64 GiB available memory; failure blocks before a source
  data-page scan. It is a CPU/I/O job. Parent content identity, temporal and
  author validity, candidate eligibility, G-1, G0, and empirical G1 remain
  explicitly unestablished.
- `adjudicate_openalex_work_topic_primary_scores.py`: derive the versioned
  full-scope successor from the exact blocked 2026-07-22 receipt. It rehashes
  all 1,430 parent artifacts, reads only the two parent-declared conflict keys,
  and selects a conflicting score only when the frozen main row's exact
  primary-topic ID and float64 score match a child row for the same edge. It
  preserves every raw conflict row in a sidecar and streams the parent sorted
  events into the 928,561,783-row canonical relation. It is a CPU/I/O job and
  requires both `--execute` and `--allow-full-scope`.
- `verify_openalex_work_topic_primary_adjudication.py`: independently verify
  the successor without importing its producer. It rehashes the bound output
  and parent controls, replays both frozen main rows, and compares all
  1,420,855,752 parent events with every canonical output record. The completed
  run passed this full read-only equivalence check. This removes only the
  relation component blocker; after the separate canonical-spine and
  author-history and citation-chronology passes, three other G-1 components
  remain missing.

  ```bash
  uv run python scripts/verify_openalex_work_topic_primary_adjudication.py \
    --parent-output-dir /operator/local/work-topic-parent \
    --source-root /operator/local/openalex-derived \
    --output-dir /operator/local/work-topic-successor
  ```
- `build_pivot_feedback_cutoff_primitives.py`: build the normalized
  cutoff-primitives successor from the exact current-content, canonical-time,
  author-event, adjudicated work-topic, and citation-chronology receipts. It
  projects only outcome-blind work metadata, closes all retained work IDs to
  the canonical spine, and publishes 4,096 strict work-modulo Parquet shards.
  Prefix runs are permanently nonpromotable. Full execution is CPU/I/O work,
  requires `--authorize-full`, and rehashes every parent receipt artifact.
- `verify_pivot_feedback_cutoff_primitives.py`: independently rehash the
  cutoff output, scan every work row, replay every canonical join, recompute
  manifest trees and source-profile closures, rehash the adjudicated relation,
  and verify the citation partition tree without changing the output. The
  2026-07-23 full 492,293,206-row output passed. This establishes normalized
  primitives only: a G0-resolved final cutoff-panel successor, outcome-blind
  field structure, and representative discovery pilot remain missing.
- `verify_pivot_feedback_cutoff_primitives_source_replay_v2.py`: additive,
  producer-independent verifier that closes the source-projection boundary
  left explicit by the first verifier. It validates the content-freeze
  checkpoints and current rich source identities, independently parses the
  exact eight-column main-work projection, demultiplexes fixed-width records
  by work ID, and compares every value and null mask with the published 4,096
  cutoff shards. It also rehashes the cutoff output and canonical spine while
  proving that the production tree was unchanged. Source-byte rehashing against
  the content baseline is available as an explicit stronger option. Prefix
  pilots are permanently nonpromotable, and even a full pass cannot establish
  author-person identity, historical observability, a final G0 panel, or a
  scientific effect.
- `verify_pivot_feedback_gate_sequence_v1.py`: validate the versioned acyclic
  gate-order amendment without mutating the legacy G-1, pending G0,
  cross-field, or peer-mechanism v0 parents. It requires the exact forward
  order `G-1A -> S0 -> D0 -> G0 -> G0.5 -> G1 -> G2 -> G3 -> G4 -> G5`,
  assigns the final cutoff panel only to post-freeze G0.5, separates
  threshold-free S0 metrics from threshold-dependent G0.5 support, and binds
  the completed cutoff source replay. A pass validates the prospective DAG
  only; it emits no G-1A/G0 pass and authorizes no model fitting.
- `reproduce_early_git.py`: reproduce the current early Git alpha support paths
  from a tag. It runs the fresh-clone first-run/catalog checks and the direct
  Git install smoke in temporary directories, then verifies matching version
  metadata, `toy_count=10`, and torch-free default behavior. This is a
  maintainer-only triage helper, not a package-facing example.

Expected future scripts:

- `run_toy1.py`: run Neural HK Classification from a YAML config.
- `run_toy2.py`: run Neural Spatial Prisoner's Dilemma from a YAML config.
- `run_toy3.py`: run Neural Opinion Rewiring from a YAML config.
- `run_toy4.py`: run Neural Public Goods and Commons from a YAML config.
- `run_toy5.py`: run Neural Contagion and Adoption from a YAML config.
- `run_toy6.py`: run the multi-action categorical spatial game from a YAML
  config.
- `run_toy7.py`: run the continuous extraction-intensity resource ABM from a
  YAML config.
- `run_toy8.py`: run the asynchronous event-driven adoption/failure ABM from
  a YAML config.
- `run_toy9.py`: run the heterogeneous-agent binary adoption ABM from a YAML
  config.
- `run_toy10.py`: run the dynamic-network market/ecology ABM from a YAML
  config.
- `run_toy2_sweep.py`: run Toy 2 payoff-regime, alpha, and initial-condition
  basin sweeps across neural, Fermi, reputation-imitation, and RD reference
  policy rules, including reputation/mobility diagnostics and optional neural
  reputation observation features. The Toy 2 sweep defaults to the legacy
  `output_average/none` social path, and can opt into
  `output_average/output_similarity` with `--peer-rules output_similarity` plus
  `--coordination-thresholds`.
- `run_toy3_sweep.py`: run Toy 3 update-rule, confidence-threshold,
  rewiring-rate, mixer, peer-rule, threshold, alpha, and seed sweeps with
  summary and grouped-summary CSV outputs. The default social peer rule remains
  `bounded_confidence`.
- `run_toy4_sweep.py`: run Toy 4 public-goods sweeps across imitation,
  neural-policy, reputation-imitation, coordination mixing, peer-rule,
  threshold, alpha, and fixed-cell mobility variants, with optional neural
  reputation observation features.
- `run_toy5_sweep.py`: run Toy 5 contagion and threshold-adoption sweeps
  across policy rules, threshold regimes, repeated-exposure settings,
  coordination mixing, threshold, alpha, and seeds with summary and
  grouped-summary CSV outputs.
- `run_toy6_sweep.py`: run Toy 6 strategy-count, initial-distribution,
  payoff-profile, mixer, threshold, alpha, and seed sweeps with summary and
  grouped-summary CSV outputs.
- `run_toy7_sweep.py`: run Toy 7 recovery-rate, extraction-cost,
  initial-intensity, exploration, mixer, threshold, alpha, and seed sweeps with
  summary and grouped-summary CSV outputs.
- `run_toy8_sweep.py`: run Toy 8 initial-state, activation, failure,
  recovery, mixer, threshold, alpha, and seed sweeps with summary and
  grouped-summary CSV outputs.
- `run_toy9_sweep.py`: run Toy 9 group-composition, coordination-gate,
  threshold, payoff, mixer, threshold, alpha, and seed sweeps with summary and
  grouped-summary CSV outputs.
- `run_toy10_sweep.py`: run Toy 10 recovery, extraction-cost, network-churn,
  initial-market-state, mixer, threshold, alpha, and seed sweeps with summary
  and grouped-summary CSV outputs.
- `run_toy_validation.py`: run the representative Toy 1-10 validation suite
  and write run-level CSV, metric CSV, and Markdown report outputs.
  Presets:
  `--preset quick` runs one short scenario per toy for fast smoke validation;
  `--preset representative` runs the default diagnostic suite;
  `--preset paper-candidate` runs the full suite with more seeds and epochs.
- `run_nabm_effect_matrix.py`: run the small Toy1-5 NABM effect matrix from
  `experiments/evidence/nabm_effect_matrix_quick.yaml`. The manifest declares
  cases, baseline/NABM variants, seeds, primary metrics, and direction
  (`maximize` or `minimize`). The default outputs are preserved under
  `experiments/results/nabm_effect_matrix/` as `<label>_runs.csv`,
  `<label>_effects.csv`, `<label>_pairwise_effects.csv`, and
  `<label>_effects.md`; generated configs are written under
  `experiments/configs/generated/evidence_matrix/<label>/`, and run
  directories are kept in the manifest `runs_dir`.
- `run_basin_credit_evidence_gate.py`: evaluate the Toy2/Toy4 basin-credit
  success gate from `experiments/evidence/toy24_basin_credit_quick.yaml` after
  `run_nabm_effect_matrix.py` has produced `<label>_runs.csv`. The gate writes
  audited JSON and Markdown summaries under `experiments/evidence/results/` and
  only treats non-teacher, non-bootstrap, non-replay NABM variants as main
  success candidates. It rejects run CSVs whose label/case/variant/seed rows do
  not match the manifest.
- `run_adapter_holdout_evidence.py`: run the adapter-only threshold adoption
  holdout from `experiments/evidence/adapter_only_threshold_holdout_quick.yaml`.
  The domain lives in the runner rather than `src/neural_abm` and uses public
  binary policy/readiness unit APIs with baseline, negative-control, and main
  variants.
- `run_adapter_congestion_holdout_evidence.py`: run the adapter-only congestion
  holdout from `experiments/evidence/adapter_only_congestion_holdout_quick.yaml`.
  This non-cascade holdout measures capacity error, overcrowding, and welfare
  while using only public binary policy unit callbacks.
- `run_adapter_stochastic_commons_holdout_evidence.py`: run the adapter-only
  stochastic commons holdout from
  `experiments/evidence/adapter_only_stochastic_commons_quick.yaml`. This
  closed-loop holdout lets binary harvest/conserve actions change future local
  resources and measures collapse avoidance, welfare, harvest rate, and shock
  recovery through public binary policy unit callbacks.
- `run_basin_credit_evidence_workflow.py`: run the basin-credit evidence matrix
  and immediately evaluate the hardened gate. Use `--skip-matrix --runs-path`
  to audit an existing `<label>_runs.csv`, or `--require-pass` in automation to
  fail when the Toy2/Toy4 main-claim criteria are not met. The workflow also
  writes read-only evidence profiles by default as `<label>_profile.json`,
  `<label>_profile.md`, and `<label>_profile_cases.csv`; use
  `--profile-output-dir` to redirect them or `--skip-profile` for gate-only
  audits. Toy-specific profile adapters must interpret generated artifacts only;
  they must not own core toy semantics or change the gate.
- `python -m neural_abm.diagnostics.evidence_profile`: profile an existing
  manifest/run CSV pair without rerunning simulations. This is the preferred
  way to inspect new toy evidence artifacts before adding a toy-specific
  adapter. Generic profiles summarize case/variant hits, time-to-ceiling,
  terminal ceiling rate, and issue codes; adapters can add diagnostic notes and
  structured JSON details for toy-specific mechanisms.
- `profile_evidence_artifacts.py` or
  `python -m neural_abm.diagnostics.profile_index`: build a registry across
  many existing evidence profiles. By default it scans `experiments/evidence/*.yaml`,
  profiles manifests with matching `<label>_runs.csv`, skips missing run CSVs,
  and writes `evidence_profile_index.csv`, `evidence_profile_index.md`, and
  `evidence_profile_index.json`. The index is case-level so repeated issue
  codes such as final-epoch hazard, material+basin collapse, or missing run
  artifacts can be filtered without opening each profile.
- `run_basin_phase_critic_training.py`: train and evaluate the offline learned
  basin phase critic from `basin_transition_samples.parquet` artifacts. The
  default manifest reads the dataset-producing Toy2/Toy4 run rows, trains on
  seed 1-2 samples, evaluates on seed 3, scores candidate actions 0/1 with a
  bootstrap-head uncertainty diagnostic, and writes per-case model, prediction,
  JSON, CSV, and Markdown quality artifacts under
  `experiments/results/basin_critic/`. The candidate-context manifest
  `toy24_basin_phase_critic_candidate_context_quick.yaml` adds features that
  recompute phase action rate, policy rate, and consensus under each candidate
  action before scoring. The pairwise-direction manifest
  `toy24_basin_phase_critic_pairwise_direction_quick.yaml` uses
  `label_mode: prototype_direction` to train an explicit candidate-action
  direction critic from prototype basin-advantage labels. The
  future-outcome-direction manifest
  `toy24_basin_phase_critic_future_outcome_direction_quick.yaml` uses
  `label_mode: future_outcome_direction` to train direction labels from
  observed future basin motion instead of the prototype sign.
- `summarize_basin_learned_diagnostics.py`: summarize read-only learned basin
  runtime diagnostics from a basin-credit evidence run. It reads each run's
  `micro_state.csv` and `aggregate_metrics.csv`, compares prototype and learned
  action-1 advantages, and writes run-level, grouped CSV, and Markdown
  diagnostic summaries.
- `diagnose_time_to_ceiling_bottlenecks.py`: derive early-window
  time-to-ceiling bottleneck diagnostics from an evidence matrix run CSV. It
  reads each run's `aggregate_metrics.csv` and writes epoch-level deltas,
  run-level bottleneck flags, and a grouped Markdown report for credit, local
  policy, social mixing, decision/action, and revision-rate attribution.
- `train_basin_replay_weight_scorer.py`: train frozen per-agent basin replay
  weight scorers from `basin_transition_samples.parquet` and the
  candidate-context learned basin critic. The scorer predicts continuous
  replay loss weights for `learned_credit_replay_mode: learned_weight`.
  Supported supervision modes are the default current-signal `magnitude`
  target, `future_basin_motion`, which derives labels from forward payoff
  movement over `--future-horizon`, and `intervention_pressure`, which combines
  current signal magnitude with future basin motion as a pressure target.
- `experiments/evidence/toy24_basin_learned_credit_replay_quick.yaml`: evidence
  manifest for the first learned-credit replay slice. It keeps the prototype
  escalation replay variant as diagnostic context, then tests a gated learned
  phase-critic credit source with prototype fallback as the main NABM candidate
  and a zero-fallback variant as an isolation diagnostic.
- `experiments/evidence/toy24_basin_learned_credit_candidate_context_replay_quick.yaml`:
  evidence manifest for replay with the candidate-context learned critic model.
  It uses the same prototype escalation diagnostic but points learned credit at
  the candidate-conditioned critic artifacts.
- `experiments/evidence/toy24_basin_learned_credit_replay_selection_quick.yaml`:
  evidence manifest for learned/prototype replay selection. It compares
  candidate-context learned replay over all candidates with confident-agreement
  and confident-disagreement replay subsets.
- `experiments/evidence/toy24_basin_learned_credit_replay_floor_quick.yaml`:
  evidence manifest for starvation-safe learned/prototype replay selection. It
  keeps confident-agreement replay as the selector but fills a minimum replay
  rate from high-magnitude prototype candidates.
- `experiments/evidence/toy24_basin_learned_credit_replay_curriculum_quick.yaml`:
  evidence manifest for temporal learned/prototype replay selection. It starts
  with a broad replay floor and linearly narrows toward the configured floor as
  training progresses.
- `experiments/evidence/toy24_basin_learned_credit_soft_replay_quick.yaml`:
  evidence manifest for soft learned-credit replay weighting. It keeps the
  confident-agreement learned signal but scales each replay loss with a
  per-agent `replay_weight` instead of using only a hard include/drop gate.
- `experiments/evidence/toy24_basin_learned_credit_weight_scorer_quick.yaml`:
  evidence manifest for the first learned replay-weight scorer. It loads a
  frozen scorer artifact to predict `replay_weight` at runtime instead of using
  the fixed soft replay formula as the main mechanism.
- `experiments/evidence/toy24_basin_learned_credit_future_motion_weight_scorer_quick.yaml`:
  evidence manifest for the future-motion replay-weight scorer. It keeps the
  magnitude-supervised scorer as a diagnostic variant and tests a scorer trained
  from `future_basin_score_delta` as the main candidate.
- `experiments/evidence/toy24_basin_direction_pressure_quick.yaml`: evidence
  manifest for the direction/pressure split slice. It keeps the learned
  candidate-context basin critic as the direction module and tests an
  `intervention_pressure` replay scorer as the main pressure module.
- `experiments/evidence/toy24_basin_pairwise_direction_pressure_quick.yaml`:
  evidence manifest for the pairwise direction-pressure slice. It keeps the
  `intervention_pressure` scorer as the pressure module and swaps the direction
  module to the pairwise candidate-action basin critic trained with
  `toy24_basin_phase_critic_pairwise_direction_quick.yaml`.
- `experiments/evidence/toy24_basin_future_outcome_direction_pressure_quick.yaml`:
  evidence manifest for the future-outcome direction-pressure slice. It keeps
  the `intervention_pressure` scorer as the pressure module and swaps the
  direction module to the candidate-action critic trained with
  `toy24_basin_phase_critic_future_outcome_direction_quick.yaml`.
- `run_toy1_ablation.py`: run the first Toy 1 ablation matrix from a base
  config and write CSV/Markdown summaries.
- `run_toy1_sweep.py`: run Toy 1 alpha/threshold sweeps and grouped summaries.
- `benchmark_accelerator_policy_core.py`: benchmark per-agent neural policy
  inference against the batched accelerator MLP core on CPU/GPU devices.
- `benchmark_toy_gpu_core.py`: benchmark Toy2/4/5 neural binary runners
  end-to-end across CPU/GPU devices, agent counts, and binary coordination
  mixers, writing the common GPU-core benchmark CSV fields plus optional
  detailed stage timing CSVs for bottleneck attribution. Toy4/Toy5 also accept
  `--training-backends loop batched auto` to compare the default per-agent
  optimizer loop with the batched gradient backend, and `--warmup-runs`/
  `--repeats` to report mean/std benchmark timings.
- `plot_toy1_output_alpha.py`: plot the output-average alpha sweep as a
  paper-ready figure candidate.
- `plot_toy1_mixer_comparison.py`: plot the 5-seed Toy 1 mixer ablation as a
  paper-ready comparison figure.
- `plot_toy1_parameter_phase.py`: plot parameter-path alpha/threshold phase
  behavior.
- `plot_toy1_parameter_alignment.py`: plot raw and aligned parameter averaging
  diagnostics for independent-init agents.
- `plot_toy1_cluster_dynamics.py`: plot per-agent prediction cluster dynamics
  from probe prediction snapshots.
- `plot_toy1_cluster_comparison.py`: plot several probe-prediction snapshot
  runs side by side to compare connected, partial, and fragmented regimes.
- `plot_toy2_initial_comparison.py`: plot initial Toy 2 no-social versus
  policy-output mixing trajectories.
- `plot_toy2_regime_sweep.py`: plot Toy 2 regime, dynamics-reference,
  alpha-response, basin-sensitivity, reputation, and mobility validation
  figures.
- `plot_toy4_sweep.py`: plot Toy 4 contribution, reputation, mobility rate,
  and mobility gain diagnostics from `run_toy4_sweep.py`.
- `plot_toy_validation.py`: plot validation suite summaries from quick,
  representative, or paper-candidate result CSVs.
- Export plots from experiment runs.
- Build paper figures.
- Create archive snapshots.

Scripts should call reusable code from `src/neural_abm/` instead of containing
core simulation logic.

## Pivot-feedback gate and S0 inventory

- `verify_pivot_feedback_g_minus_1a_receipt_v1.py`: bind the exact
  full-production canonical-time, author-token, work-topic,
  citation-chronology-proxy, cutoff-primitives, and independent source-replay
  summaries into the narrow G-1A receipt. The verifier is metadata-only and
  authorizes S0, not D0, G0, or model fitting.
- `build_pivot_feedback_s0_threshold_free_inventory.py`: rehash and scan the
  receipt-bound normalized work-metadata and distinct author-work-event
  partitions, producing exact 26-field-by-year work, author-event,
  non-ranked team-size, calendar-support, and raw attention-resolution
  profiles. The core component deliberately leaves ORCID, affiliation, and
  distinct graph topology `NOT_RUN`.
- `verify_pivot_feedback_s0_threshold_free_inventory.py`: independently rehash
  the compact S0 output, verify its gate boundary and field-year conservation,
  and replay every annual, trailing-three-year, and trailing-five-year
  attention count without importing the builder.
- `build_pivot_feedback_s0_authorship_orcid_affiliation.py`: scan the exact
  content-checkpoint-bound raw authorship source and report ORCID syntax and
  checksum coverage, one-valid-ORCID-per-author-token stability diagnostics,
  raw-affiliation and institution coverage, and field-year margins. Pilot
  limits are permanently nonpromotable, and the output never establishes
  people or historical affiliation observability.
- `verify_pivot_feedback_s0_authorship_orcid_affiliation.py`: independently
  rehash the compact projection, replay its field-year margins and token
  multiplicity constraints, and reject S0/D0/model-fitting promotion without
  importing the builder. `--report` creates a new private verification JSON
  outside the producer output root and refuses to overwrite an existing file.
- `build_pivot_feedback_s0_coauthor_graph_profile.py`: use receipt-bound
  distinct author-work events, DuckDB external exact pair reduction, and SciPy
  connected components to compute all predeclared non-ranked team-cap
  profiles `10`, `50`, and `100`. Raw author and edge intermediates are
  removed before the compact atomic output is completed. DuckDB is a
  repository-only pinned tool overlay; invoke this script with
  `uv run --with duckdb==1.5.5` rather than adding it to package extras.
- `verify_pivot_feedback_s0_coauthor_graph_profile.py`: independently verify
  every core field-year-by-cap row, degree-sum and component bounds,
  largest-component shares, nested cap margins, materialized-node/edge closure,
  artifact hashes, and the fail-closed gate boundary. `--report` has the same
  external, create-only semantics as the ORCID/affiliation verifier.
- `verify_pivot_feedback_s0_completion_v1.py`: rehash the current core,
  ORCID/affiliation, and graph outputs plus their three external independent
  verification reports, require all component verifier hashes to match the
  current repository, and issue a compact S0 completion receipt. Only that
  three-component receipt can authorize the pre-2013 D0 discovery gate; it
  does not authorize G0, a final panel, model fitting, or G4.
- `verify_pivot_feedback_gate_state_v2.py`: validate the additive post-run
  ledger that records bounded G-1A as complete, S0 as partial with three
  deferred components, and D0/model fitting/G4 as fail-closed.
- `verify_pivot_feedback_gate_state_v3.py`: validate the immutable-v2-parent
  successor after a full S0 completion receipt. The only newly open state is
  `D0=AUTHORIZED_NOT_RUN` with entry year at most 2012; every downstream gate
  remains fail-closed.
- `build_pivot_feedback_d0_structural_pilot_freeze.py`: implement D0-A without
  reading outcomes. It ranks all 26 fields from eight pre-2013 S0 structural
  metrics, chooses a second field outside the first field's domain, rescans
  the receipt-bound cutoff work surface, and freezes two current-snapshot
  primary-topic episodes per field with distinct subfields and disjoint
  `t-6` through `t+2` calendars. Episode-score order uses exact integer
  rational comparison rather than float order.
- `verify_pivot_feedback_d0_structural_pilot_freeze.py`: independently rehash
  and replay the 4,096 cutoff artifacts, field ranks, topic-year counts,
  episode candidates, and exact selection without importing the producer.
  `--report` writes a private create-only receipt outside the output root.
- `verify_pivot_feedback_gate_state_v4.py`: validate the immutable-v3-parent
  successor after D0-A. It binds the compact summary, live producer receipt,
  current independent verification report, and current producer/verifier
  hashes. D0 remains partial; only D0-B outcome-blind author-opportunity and
  leave-out measurement construction is authorized.
- `build_pivot_feedback_d0b_author_opportunity_foundation.py`: bind the four
  D0-A episodes, rescan the 4,096-partition cutoff and author-history
  projections, materialize exact target-topic works through `t-1`, and build
  the focal-field-active author-token opportunity base under four non-ranked
  team-cap profiles. It reads no entry, persistence, or realized-citation
  outcome and does not establish final candidate eligibility.
- `verify_pivot_feedback_d0b_author_opportunity_foundation.py`: independently
  rehash both 4,096-partition inputs and replay the selected works plus every
  author output row group with Arrow `group_by`, without importing the NumPy
  producer. Its private external report is create-only.
- `verify_pivot_feedback_gate_state_v5.py`: validate the immutable-v4-parent
  successor after the D0-B author-opportunity foundation. It binds the compact
  summary, live producer receipt, independent verification report, and current
  code hashes. D0-B and D0 remain partial; only the remaining outcome-blind
  content, incumbent, leave-out, network, bridge, ORCID-subset, and leakage
  measurements are authorized.
- `build_pivot_feedback_d0b_selected_work_author_linkage_audit.py`: join every
  D0-A-selected pre-decision work to valid, distinct author-token events while
  retaining zero-token works. It emits exact work linkage, private
  author-work incidence, and 1,200 work-type-by-window-by-cap profile cells
  without choosing a threshold or excluding an episode.
- `verify_pivot_feedback_d0b_selected_work_author_linkage_audit.py`:
  independently rehash both 4,096-partition inputs, reconstruct all author
  incidences with Python sets and work grouping, and replay every linkage and
  profile cell without importing the producer. It requires measurement review
  when the verified heterogeneous gap is present.
- `verify_pivot_feedback_gate_state_v6.py`: validate the immutable-v5-parent
  successor after the linkage audit. It binds the compact summary, live
  receipt, independent verification, and current code hashes; pauses
  leave-out/network/bridge construction; and authorizes only outcome-blind
  measurement-ABI redesign plus deterministic episode reselection.
- `build_pivot_feedback_d0_measurement_abi_reselection.py`: apply the frozen
  scholarly-communication and valid-author-token ABI to the complete two-field
  pre-decision surface, rebuild all topic-year candidates, and deterministically
  select two calendar-disjoint episodes per field without reading entry or
  persistence outcomes.
- `verify_pivot_feedback_d0_measurement_abi_v1.py`: validate the frozen ABI
  before production reselection. The ABI excludes ineligible work types,
  paratext, retractions, and zero-valid-token works from primary attention
  while retaining explicit missingness accounting.
- `audit_pivot_feedback_d0_content_indexing.py`: bind the outcome-blind
  measurement-ABI reselection, build a deterministic 32-work sample for every
  episode-by-window cell, scan only current-snapshot main-work title and
  metadata columns, and emit an annual indexing profile plus a double-review
  template. Automated success leaves D0, outcomes, leave-outs, networks,
  bridges, model fitting, and G4 closed until semantic review and a versioned
  reselection decision pass.
- `verify_pivot_feedback_gate_state_v7.py`: bind the full ABI reselection and
  automated content-audit compact evidence to the immutable v6 parent, recheck
  the live receipts and blank 384-row review template, quarantine episode 3,
  mark the old 5,849,627-row author panel as superseded, and keep every
  downstream gate fail-closed.
- `build_pivot_feedback_d0_content_review_packets.py`: create two
  reviewer-specific, separately ordered title-only CSV packets and a private
  adjudication key from the frozen 384-row template. It hides episode, window,
  source, topic-score, publication-year, and OpenAlex-ID fields from reviewers.
  It does not classify titles or count as either required human reviewer.
- `verify_pivot_feedback_d0_content_review_packets.py`: independently
  reconstruct both packet orders and opaque item IDs, validate the adjudication
  key and every artifact hash and mode, and reject populated labels, metadata
  leakage, or any downstream authorization.
- `verify_pivot_feedback_gate_state_v8.py`: record independently verified
  packet preparation as the latest additive gate state. It authorizes only two
  distinct human reviews, followed by third-party adjudication after both
  reviewer files are frozen; D0, outcomes, G0, model fitting, and G4 remain
  closed.
- `prepare_pivot_feedback_d0_api_review_v2.py`: create two exact
  `/v1/responses` Batch API JSONL inputs for pinned model snapshots plus
  separately ordered 96-row human-calibration packets. It is create-only,
  exposes only topic/title/language/work type, submits nothing, and leaves
  outcomes and every downstream gate closed.
- `verify_pivot_feedback_d0_api_review_preparation_v2.py`: independently
  reconstruct all 768 API requests, the stratified calibration sample, both
  packet orders, opaque identifiers, private keys, artifact hashes, and file
  modes without importing the producer.
- `verify_pivot_feedback_gate_state_v9.py`: bind the independently verified
  API-input preparation to the immutable v8 parent. It records that API
  execution and human calibration are still not run, allows submission only
  with an operator credential and cost control, and forbids the full candidate
  screen until calibration passes.
- `prepare_pivot_feedback_d0_cross_provider_api_review_v3.py` and
  `prepare_pivot_feedback_d0_cross_provider_api_review_v4.py`: build the
  OpenAI/Claude title-only batch inputs and repair provider-prefixed IDs to the
  shared 64-character IDs accepted by both APIs. Their independent verifiers
  bind request content, blinding, hashes, modes, and the one-dollar cost cap.
- `submit_pivot_feedback_d0_cross_provider_api_review_v4.py` and
  `collect_pivot_feedback_d0_cross_provider_api_review_v4.py`: submit and
  collect both pinned providers without persisting credential values. The
  Claude run completed with 384 valid results; the OpenAI run was retained as
  invalid because 241 rows hit the output-token ceiling.
- `prepare_pivot_feedback_d0_openai_retry_v5.py`,
  `submit_pivot_feedback_d0_openai_retry_v5.py`, and
  `collect_pivot_feedback_d0_openai_retry_v5.py`: retry only OpenAI while
  preserving every ID and visible request field, changing only minimal
  reasoning effort and the output cap. The retry completed with 384 valid
  results; Claude was not resubmitted.
- `build_pivot_feedback_d0_cross_provider_results_v6.py`: validate and join
  both 384-row result sets, materialize 197 dual-high-confidence on-topic
  candidates, and create two separately ordered, API-blinded 233-row human
  packets from the calibration-plus-escalation union.
- `verify_pivot_feedback_d0_cross_provider_results_v6.py`: independently
  replay the response schemas, IDs, decisions, escalation rule, reviewer IDs,
  packet blinding, costs, hashes, and modes without importing the producer.
- `verify_pivot_feedback_gate_state_v10.py`: bind the valid cross-provider
  results and human queue to the immutable v9 parent while keeping human
  calibration, D0, candidate-universe screening, outcomes, G0, fitting, and G4
  fail-closed.
- `verify_pivot_feedback_researcher_pivot_definition_v1.py`: validate and
  execute the simple past-only four-state researcher-topic-year decision table.
  The primary event is observable-history first entry; return and continuation
  are distinct, insufficient history fails closed, pivot distance does not gate
  the event, and persistence remains a future outcome.
- `verify_pivot_feedback_gate_state_v11.py`: bind that definition to the
  immutable v10 parent and record the API-only branch. Human review and
  human-calibrated accuracy are `NOT_RUN_BY_DESIGN` and `NOT_AVAILABLE`;
  full-universe paid screening, production pivot classification, outcomes,
  fitting, and G4 remain unauthorized.
- `build_pivot_feedback_d0_api_only_topic_routing_v1.py`: apply four frozen
  title-level inclusion/exclusion cards to the 384-row diagnostic sample
  without network or credential access. It sends only mixed or unmatched
  titles to a counterfactual API exception queue and keeps opaque titles in
  unresolved quarantine.
- `verify_pivot_feedback_d0_api_only_topic_routing_v1.py`: independently
  reconstruct every local route, source-row join, topic summary, artifact
  hash, and private file mode without importing the producer. The comparison
  to prior API labels is descriptive and cannot establish accuracy.
- `verify_pivot_feedback_gate_state_v12.py`: bind the independently verified
  zero-cost dry-run to the immutable v11 parent. It authorizes only complete
  title-bearing candidate-work manifest construction and local routing; paid
  API submission, production dispositions, outcomes, fitting, and G4 remain
  closed.
- `build_pivot_feedback_d0_candidate_screen_scope_audit_v1.py`: expand the
  70,277 field-topic-entry-year candidates into 421,662 six-year window rows
  using the frozen aggregate artifacts. It measures unique work scope,
  overlapping memberships, and four-card coverage without reading titles,
  work IDs, raw OpenAlex rows, credentials, APIs, or outcomes.
- `verify_pivot_feedback_d0_candidate_screen_scope_audit_v1.py`: independently
  replay the candidate-window and topic-scope Parquet artifacts and establish
  that four cards cover only 4 of 994 topics. The result is a workload and
  coverage audit, not a content-validity or accuracy result.
- `verify_pivot_feedback_gate_state_v13.py`: bind the scope correction and
  withdraw the 22.4-million-work eager title copy as the immediate step. It
  authorizes only an adaptive frontier contract, bounded equivalence fixtures,
  and zero-cost initial structural materialization.
- `build_pivot_feedback_d0_adaptive_candidate_frontier_v1.py`: implement eager
  and fail-closed lazy compatible-pair selection, exhaustively enumerate
  299,584 small fixtures, and materialize the 51-row initial structural
  frontier. Four content-required rows remain unresolved.
- `verify_pivot_feedback_d0_adaptive_candidate_frontier_v1.py`: independently
  replay every bounded fixture, unresolved-state mutation, structural skip,
  and initial frontier row without importing the producer.
- `verify_pivot_feedback_gate_state_v14.py`: bind the frontier proof and keep
  the four content states, certificate, reselection, raw title scans, new API
  requests, outcomes, fitting, and G4 closed. Only content-rule design,
  title-sampling design, and a zero-cost dry-run on the existing 384 titles are
  authorized.
- `build_pivot_feedback_d0_candidate_content_dry_run_v1.py`: aggregate the
  existing 384 local title routes into four fail-closed candidate dry-run
  states, separate a joint score/source indexing anomaly from semantic
  off-topic evidence, and report a three-threshold sensitivity grid plus the
  counterfactual targeted-frontier cost without reading new titles or APIs.
- `verify_pivot_feedback_d0_candidate_content_dry_run_v1.py`: independently
  reconstruct all four candidate aggregates and twelve sensitivity rows from
  the frozen title sample and local routes without importing the producer.
- `verify_pivot_feedback_gate_state_v15.py`: bind the zero-cost candidate
  dry-run while keeping every production state unresolved. It authorizes only
  a rule decision and targeted T10543 successor design; raw scans, new API
  requests, production dispositions, certification, and reselection stay
  closed.
- `build_pivot_feedback_d0_production_candidate_rule_v1.py` and
  `verify_pivot_feedback_d0_production_candidate_rule_v1.py`: freeze and
  independently replay the four original candidate dispositions plus
  frontier v2. Computer Science closes locally; Medicine remains unresolved.
- `build_pivot_feedback_d0_t10543_targeted_title_sample_v1.py` and its
  independent verifier: scan the frozen normalized work and author-event
  surfaces, select exactly 32 linked works in each entry-relative window, and
  look up only those 96 current-snapshot titles.
- `build_pivot_feedback_d0_t10543_local_routing_v1.py`: route the rank-2
  T10543 sample locally. Fifty-two of 64 recent titles are opaque archival
  identifiers; they stay quarantined and are not sent to an API.
- `build_pivot_feedback_d0_candidate_rule_v2_replay_v1.py` and its verifier:
  disclose and replay the post-diagnostic majority-opaque indexing branch.
  The replay changes only rank-2 T10543 from unresolved to indexing fail and
  is not out-of-sample validation.
- `build_pivot_feedback_d0_production_candidate_rule_v2.py` and its verifier:
  freeze candidate rule v2, five dispositions, and frontier v3. Medicine rank
  3 becomes the exact next required candidate.
- `build_pivot_feedback_d0_t10543_rank3_targeted_title_sample_v1.py` and its
  verifier: reuse the exact sampling implementation under the rank-3 entry
  year and independently verify the resulting 96-title sample.
- `build_pivot_feedback_d0_t10543_rank3_local_routing_v1.py` and its verifier:
  establish that rank 3 fires the retained joint score-drop and
  source-concentration indexing anomaly without human or API labels.
- `build_pivot_feedback_d0_rank3_production_disposition_v1.py` and its
  verifier: append the sixth production disposition and regenerate frontier
  v4. Medicine rank 4 T12505 entry 1993 is now required; no Medicine or global
  certificate has been issued.
- `verify_pivot_feedback_gate_state_v16.py` through
  `verify_pivot_feedback_gate_state_v20.py`: independently bind each exact
  sample, local-routing, rule-freeze, and cumulative-disposition transition
  while keeping APIs, outcomes, fitting, and G4 fail-closed.
- `build_pivot_feedback_d0_t12505_rank4_targeted_title_sample_v1.py` and its
  independent verifier: replay the full normalized work and author-event
  surfaces, select an exact 32-per-window rank-4 sample, and read only the 96
  selected current-snapshot title rows.
- `build_pivot_feedback_d0_t12505_rank4_local_routing_v1.py` and its verifier:
  reuse the frozen T12505 card and independently establish the
  majority-opaque rank-4 indexing-failure branch without an API or human
  label.
- `build_pivot_feedback_d0_rank4_production_disposition_v1.py` and its
  verifier: append the seventh production disposition and regenerate frontier
  v5, making Medicine rank 5 T10151 entry 1993 the final required candidate.
- `build_pivot_feedback_d0_t10151_rank5_targeted_title_sample_v1.py` and its
  independent verifier: use the topic card frozen before sample access,
  replay all normalized partitions, and verify the exact 96 selected raw rows.
- `build_pivot_feedback_d0_t10151_rank5_local_routing_v1.py` and its verifier:
  independently reproduce 27 accepts, 12 rejects, 25 ambiguous routes, and
  the fail-closed `CONTENT_UNRESOLVED` rule-v2 branch.
- `build_pivot_feedback_d0_rank5_terminal_disposition_v1.py` and its verifier:
  append the eighth production disposition, freeze frontier v6, and
  independently verify that the earlier unresolved rank-5 row blocks the
  Medicine and global certificates even though rank 6 is a frozen pass.
- `verify_pivot_feedback_gate_state_v21.py` through
  `verify_pivot_feedback_gate_state_v25.py`: bind the rank-4 and rank-5 sample,
  routing, and production transitions while preserving zero API requests,
  zero human labels, no deterministic reselection, and closed outcomes,
  fitting, and G4.
- `verify_pivot_feedback_gate_state_v26.py` through
  `verify_pivot_feedback_gate_state_v29.py`: bind the API-only rank-5
  augmentation, preserve the first transport failure, quarantine the
  wrong-target HBV execution, restore the authoritative HCV definition, and
  authorize only the corrected unresolved production closeout.
- `prepare_pivot_feedback_d0_t10151_rank5_cross_provider_v1.py`,
  `execute_pivot_feedback_d0_t10151_rank5_cross_provider_v1.py`, and the
  independent request verifier: freeze the exact 25-title request, make one
  OpenAI and one Anthropic call, enforce the cumulative cost cap, and retain
  raw provider responses and receipts without using human review.
- `build_pivot_feedback_d0_t10151_rank5_cross_provider_result_v1.py` and its
  verifier: apply the dual-`HIGH` consensus rule to the corrected HCV pair.
  The production v2 result accepts no additional titles, rejects nine, leaves
  sixteen unresolved, and never treats provider labels as ground truth.
- `build_pivot_feedback_d0_rank5_api_augmented_closeout_v1.py` and its
  verifier: recompute the augmented rank-5 disposition and frontier v7,
  independently preserve `CONTENT_UNRESOLVED`, and keep both the Medicine and
  global certificates blocked with human review `NOT_RUN_BY_DESIGN`.
- `verify_pivot_feedback_gate_state_v30.py` and
  `build_pivot_feedback_d0_t10151_rank5_local_abstract_recovery_v1.py` plus its
  verifier: freeze the exact sixteen unresolved IDs, scan only local OpenAlex
  ID columns globally, reconstruct six selected abstract payloads, and
  independently retain exactly five mechanically eligible long abstracts
  without creating semantic labels.
- `prepare_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.py`,
  `execute_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.py`, and
  their verifier: expose only five opaque title/abstract packets to the pinned
  providers, enforce the cumulative cost cap, and preserve raw responses with
  human review `NOT_RUN_BY_DESIGN`.
- `build_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_result_v1.py`
  and its verifier: independently reduce one dual-`HIGH` accept, one
  dual-`HIGH` reject, and three reviewed unresolved rows.
- `verify_pivot_feedback_gate_state_v31.py` and
  `build_pivot_feedback_d0_rank5_local_abstract_closeout_v1.py` plus its
  verifier: freeze disposition v7, frontier v8, and blocked certificate status
  v3 at 28 accepts, 22 rejects, and 14 unresolved.
- `verify_pivot_feedback_gate_state_v32.py`,
  `collect_pivot_feedback_d0_t10151_rank5_external_evidence_v1.py`, and its
  verifier: freeze the fourteen remaining IDs, collect exact current OpenAlex,
  Crossref, and PubMed records, and independently build an eleven-row
  identifier-free external-evidence packet with three no-new-evidence
  holdouts.
- `execute_pivot_feedback_d0_t10151_rank5_external_cross_provider_v1.py`,
  `finalize_pivot_feedback_d0_t10151_rank5_external_cross_provider_v1.py`, and
  their verifier: make one no-retry provider pair, preserve both raw responses,
  deterministically normalize overlong rationale transport without changing
  labels or confidence, and reduce two accept, three reject, and six reviewed
  unresolved rows.
- `verify_pivot_feedback_gate_state_v33.py` and
  `build_pivot_feedback_d0_rank5_external_evidence_closeout_v1.py` plus its
  verifier: freeze disposition v8, frontier v9, and blocked certificate status
  v4 at 30 accepts, 25 rejects, and 9 unresolved.
- `verify_pivot_feedback_gate_state_v34.py`,
  `build_pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_v1.py`,
  and its independent verifier: replay one retrospective post-diagnostic
  accept-only rule over all nine unresolved external-evidence rows, independently
  parse the frozen PubMed XML, add exactly two accepts, create no rejects, and
  preserve seven unresolved rows.
- `verify_pivot_feedback_gate_state_v35.py`,
  `build_pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_v1.py`,
  and its independent verifier: freeze disposition v9 and frontier v10 at
  32 accepts, 25 rejects, and 7 unresolved; independently reproduce Computer
  Science ranks `[1, 45]` and Medicine ranks `[5, 6]`; and issue both field
  certificates plus the global selection certificate.
- `verify_pivot_feedback_gate_state_v36.py`: bind the independently verified
  global certificate and authorize only deterministic materialization of the
  exact four certified episodes. It does not execute reselection, rebuild the
  author panel, classify pivots, read outcomes, fit a model, or open G4.
- `build_pivot_feedback_d0_certified_episode_reselection_v1.py` and its
  independent verifier: join the global certificate to all 70,277 frozen
  measurement-ABI candidates, materialize the exact four selected rows, and
  independently replay the 51-row frontier plus every selected source field.
  Three episodes are retained, Medicine rank 5 is newly selected, and
  provisional Medicine rank 1 is deselected.
- `verify_pivot_feedback_gate_state_v37.py`: record the certified reselection
  as independently complete and authorize only the outcome-blind exact
  selected-work manifest and author-opportunity panel rebuild. The old panel is
  superseded; leave-out/network/bridge construction and all outcome-facing
  work remain closed.
- `verify_pivot_feedback_holdout_access_seal_v1.py`: independently validate
  the metadata-only G0 holdout-seal contract. It freezes the 2020--2022 entry
  partition, follow-up through 2025, domain/author-cold rule, future manifest
  and receipt schemas, receipt-last atomic protocol, and fail-closed behavior
  without enumerating or hashing holdout artifacts. Gate v54 authorizes only a
  later private explicit-allowlist byte-hash seal-creation phase; row parsing,
  seal opening, model fitting, and G4 remain closed.
- `create_pivot_feedback_holdout_access_seal_v1.py` and
  `verify_pivot_feedback_holdout_access_seal_receipt_v1.py`: require an
  operator-local, sorted explicit path-size-SHA256 preflight; copy and hash
  only those bytes into an atomic private sealed tree; write the receipt
  create-only and last; and independently rehash the exact tree without a row
  parser. Gate v55 records closed-fixture success while keeping production
  `NOT_RUN_MISSING_OPERATOR_PREFLIGHT` and seal opening closed.
- `prepare_pivot_feedback_holdout_access_seal_preflight_v1.py`: hash only
  explicitly repeated relative file paths into a sorted create-only operator
  preflight. It rejects globbing, recursive discovery, directories, symlinks,
  duplicates, missing zero-access attestations, and an existing output. Gate
  v56 records the full closed-fixture chain while production remains
  `NOT_RUN_MISSING_SOURCE_ROOT_AND_EXPLICIT_ARTIFACT_LIST`.
- `verify_pivot_feedback_holdout_production_input_resolution_v1.py`: replay the
  repository-only G4 input-resolution evidence without listing or scanning an
  operator directory. It distinguishes a logically reserved 2020--2022
  partition from an actual byte seal, rejects relabelling the D0 source ending
  in 2012, and routes field/episode selection plus artifact materialization
  before any production preflight or seal action. Gate v57 records production
  as `NOT_RUN_NO_MATERIALIZED_G4_ARTIFACT_SET`.
- `apply_pivot_feedback_g0_measurement_proxy_precommitment_v1.py` and
  `verify_pivot_feedback_g0_measurement_proxy_precommitment_v1.py`: apply and
  independently replay four outcome-blind measurement/proxy policies on a
  closed JSON fixture. They cover the robust attention-surge threshold,
  inclusive-tie top-decile public-success proxy, three-year citation maturity,
  and a one-percentage-point TOST practical null. They do not read production,
  confirmatory, or holdout rows; gate v58 authorizes only their exact G0 v3
  binding.
- `verify_pivot_feedback_g0_measurement_freeze_v3.py`: verify that G0 v3 binds
  exactly those four v58-authorized choices while preserving nine unchanged G0
  v2 top-level subtrees. It rejects filling any fifth choice, leaves thirteen
  choices and all fifteen required sections pending, and keeps G0, model
  fitting, and G4 closed. Gate v59 authorizes only a separate next six-choice
  discovery-only precommitment and closed fixture.
- `apply_pivot_feedback_g0_risk_set_weight_support_precommitment_v1.py` and
  `verify_pivot_feedback_g0_risk_set_weight_support_precommitment_v1.py`:
  apply and independently replay the exact six v59-authorized sampling,
  weighting, cohort, reporting, capacity-support, and VIF policies on a closed
  JSON fixture. They test deterministic ten-control sampling, P01/P99
  winsorization, every support boundary, and both required channel-collapse
  paths without reading production, confirmatory, or holdout rows. Gate v60
  authorizes only their exact G0 v4 binding.
- `verify_pivot_feedback_g0_measurement_freeze_v4.py`: verify that G0 v4 binds
  exactly those six policies while preserving ten unchanged G0 v3 top-level
  subtrees. It rejects filling any additional choice, leaves seven choices and
  all fifteen required sections pending, and keeps G0, model fitting, and G4
  closed. Gate v61 authorizes only a separate G1 replication-tolerance
  precommitment and closed fixture.

Toy1-10 sweep scripts use `src/neural_abm/sweep.py` for shared output specs,
result-row field mapping, stable summary CSV writing, grouped-summary
construction, spec-bound compatibility helpers for script-local summary
functions, optional grouped Markdown hooks, selected config/result extraction
for resolved row fields, and final aggregate-metric CSV readers. Toy1 now uses
the shared explicit-case orchestration for base YAML loading, generated config
routing, run execution, row assembly, and summary/grouped output writing while
keeping its case matrix and Markdown readout local. Toy3-10 use shared
coordination/case expansion,
domain parameter-grid iteration, prepared-case config writing with nested
updates and optional Toy-specific mutation hooks, run execution, row assembly,
and main orchestration helpers.
Toy2 and Toy3-10 also share common CLI argument registration with toy-specific
legacy defaults preserved through helper options. Toy2 still owns its
payoff/regime matrix semantics, conditional grid pruning, and RD reference
insertion, but its point-row orchestration, non-overwriting output path
resolution, and summary/grouped/Markdown output writing now run through the
shared sweep helpers.
Toy-specific scripts keep their own domain parameter definitions and aggregation
specs; domain run-name and config-update builders are kept as small
toy-specific functions.
