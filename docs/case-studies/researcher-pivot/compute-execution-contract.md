# Pivot-Feedback Compute Execution Contract

Contract version: 0.4

Status: the full 23-table work-source membership/profile component, the later
current-snapshot taxonomy-reference component, the current-content SHA-256
baseline, and the bounded v3 main-work temporal-field component have passed
their separate contracts. The content pass establishes the current baseline
only; historical inventory-era byte identity remains `UNVERIFIED`. Raw source
ID validity remains `BLOCKED`, `W0` entity validity is `UNRESOLVED`, G-1 data
readiness is `PENDING`, empirical G1 is `NOT_RUN` (`pass_g1=false`), and no
remote empirical job has run. Final empirical field selection is also
`PENDING` until a parent-content-identified authoritative binding and the
remaining author, duplicate/primary-child, citation-edge chronology and
`available_at`, maturation/censoring, `t-1` leakage, candidate-eligibility, and
outcome-blind structural audits pass.

Date: 2026-07-19

This contract assigns each stage of the
[pivot-feedback protocol](pivot-feedback-empirical-nabm-protocol.md) to the
machine class that matches its actual bottleneck. A GPU server is an execution
resource, not a substitute for data lineage, measurement freeze, or sealed
holdout discipline.

The repository-only readiness and measurement overlays at
`experiments/configs/pivot_feedback_g_minus_1_readiness_v0.yaml` and
`experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml` are
read-only, fail-closed state ledgers. Their validators may confirm that a
pending contract is internally consistent, but cannot authorize production
scans, model fitting, GPU work, G0 passage, or a scientific claim.

## 1. Current Boundary

The operator-local authoritative candidate is referenced publicly as:

```text
$OPENALEX_DERIVED_ROOT
```

Its mount-specific value belongs in an operator-local manifest, not this public
contract.

The footer-only inventory records 988,404,537,807 bytes (920.523 GiB). The main
works, authorships, topics, and referenced-works directories alone are
approximately 287 GB. The original exact main-ID contract scanned 492,293,207
rows and remains `BLOCKED` on one malformed `W-1` row. A separate frozen
analysis-view contract passed exact equality counting for 492,293,206 retained
grammar-valid IDs with zero duplicate excess; it retains `W0` as an unresolved
warning. The targeted audit found 21 `W-1` source rows in six files and six
`W0` source rows in six files, with zero declared work-target rows for either.
The subsequent full audit scanned all 21,019 files and 20,996,232,817 source
rows across 23 tables. It accounted for 20,996,232,796 retained rows plus the
exact 21 frozen `W-1` rows, with zero null, unexpected-malformed, or
grammar-valid orphan source IDs. It measured open-world coverage for the two
declared work-target columns and produced a descriptive author-ID/work profile;
it did not materialize a filtered analysis view or establish author entity or
candidate eligibility.

A separate 2026-07-18 full taxonomy-reference audit scanned 1,423 files,
1,420,855,753 projected rows, and 166,699,471,247 source bytes. It accounted for
one frozen `W-1` row, 138,147,958 allowed all-null retained main-work rows, and
1,282,707,794 complete valid hierarchy rows, with zero partial or invalid
hierarchy rows. `W0` occurred once and remains an unresolved retained warning.
All 26 non-null field margins and the null margin matched the frozen compact
profile exactly. Its status is
`PASS_CURRENT_SNAPSHOT_TAXONOMY_REFERENCE_COMPONENT`.

These are bounded CPU/I/O component results, not G-1 or empirical G1 passage.
The taxonomy pass establishes only current-snapshot projected-column row
conservation and taxonomy-reference integrity; it does not establish general
field-entity validity, candidate eligibility, field selection, or an effect.
The later bounded `W0` positive-ID sensitivity pass changes only the total and
null-primary-field margins by one; every non-null field margin is unchanged,
while `W0` entity validity remains unresolved.

The completed current-content freeze scanned the full 21,019-file,
988,404,537,807-byte root and established
`PASS_CURRENT_CONTENT_BASELINE_COMPONENT`. Its exact outputs-receipt SHA-256 is
`082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12`.
This is a baseline for the bytes read in that run, not proof of historical
inventory-era byte identity, which remains `UNVERIFIED`. The separate v3 full
main-work temporal audit scanned all 493 files and 492,293,207 raw rows,
quarantined the frozen `W-1`, and passed exact publication-date/year
concordance plus non-null exact UTC-millisecond `created_date` and
`updated_date` validity for all 492,293,206 retained rows. Its three
`created_date > updated_date` pairs are a nonblocking descriptive profile, not
a record-event-order invariant. Author entity and split/merge validation,
duplicate-edge and primary-topic-child checks, citation-edge chronology and
`available_at`, outcome maturation and censoring, `t-1` graph and leakage
checks, and a representative pilot remain pending.
See the
[blocked raw recount](../../../experiments/evidence/results/openalex_g1_main_id_recount_20260716.blocked.summary.md),
[grammar-valid recount](../../../experiments/evidence/results/openalex_g1_grammar_valid_main_id_recount_20260716.summary.md),
[sentinel audit](../../../experiments/evidence/results/openalex_g1_sentinel_incidence_20260716.summary.md),
[full work-source audit](../../../experiments/evidence/results/openalex_g1_work_source_membership_20260717.summary.md),
and [full taxonomy-reference audit](../../../experiments/evidence/results/openalex_g1_field_taxonomy_reference_20260718.summary.md).
The [W0 sensitivity audit](../../../experiments/evidence/results/openalex_g1_w0_positive_id_sensitivity_20260718.summary.md)
is a separate component-only result. The bounded temporal result is preserved
as [compact Markdown](../../../experiments/evidence/results/openalex_g1_main_temporal_validity_v3_full_20260719.summary.md)
and [JSON](../../../experiments/evidence/results/openalex_g1_main_temporal_validity_v3_full_20260719.summary.json).
The [full relation audit](../../../experiments/evidence/results/openalex_g1_work_topic_relation_full_20260722.summary.md)
has now completed. It passes multiplicity accounting and primary-child
inclusion, but two duplicate edge groups have different scores, so the frozen
exact-score dedup view was not materialized and no production relation result
was promoted.

The default plan is therefore **not** to copy the full root to a GPU server.

The operator-local alias represented here as `$GPU_HOST_ALIAS` authenticates
non-interactively when the dedicated identity represented as
`$GPU_SSH_IDENTITY` is supplied explicitly with `IdentitiesOnly=yes`. The
generic SSH configuration does not select that non-default identity, which is
why a key-unspecified probe failed while the Local_Map_Analysis preflight
succeeded.

The corrected 2026-07-16 read-only probe verified 48 logical CPUs, approximately
251 GiB RAM, and four NVIDIA GeForce RTX 5090 GPUs with 32,607 MiB each. Access
is ready, but launch capacity is not: the observed free memory was 435, 5,458,
2,507, and 2,674 MiB. These resource values are a dated observation and must be
rechecked immediately before every launch.

## 2. Compute Placement

| Gate or task | Primary bottleneck | Default placement | GPU role |
| --- | --- | --- | --- |
| G-1 inventory, recount, work-source, and current-content baseline audits (executed; historical identity, entity, relation, edge-time/leakage, and pilot gates remain) | HDD throughput, CPU, RAM, SSD spill | Local data host | None |
| G-1 taxonomy-reference, main-work temporal, author, and outcome-blind field-structure audits (projected-reference and bounded date-field components executed; authoritative binding, duplicate/primary-child, author, edge-time/leakage, and structural gates remain) | HDD throughput, full-root joins, CPU, RAM, SSD spill | Local data host | None |
| G-1 cutoff-aware panel extraction | Sequential scan, CPU, SSD write | Local data host, materialized to local SSD | None |
| G0 measurement registry and split freeze | Research decisions and versioning | Local tracked repository | None |
| Two-field by two-shock `DISCOVERY_ONLY` pilot | Extraction correctness, RAM, CPU, fast columnar I/O | Local data host and local SSD | None |
| Four-field G1 event studies, hazards, equivalence placebos, bootstrap | RAM, CPU, fast columnar I/O | Remote CPU/RAM/NVMe after panel transfer | None by default |
| Text-embedding robustness | Batched dense inference | Remote GPU, only after lineage-compatible text is selected | Optional |
| G2 B0-B3 | Statistical fitting and seed/cell parallelism | Remote CPU/RAM | None by default |
| G2 B4 adaptive NABM | Batched learned policy fitting and predictive draws | Remote GPU after promotion benchmark | Primary |
| G3 non-NABM factorial cells | Independent simulations | Remote CPU queue | None |
| G3 NABM factorial cells | Batched training and repeated inference | Remote GPU queue | Primary |
| G4 prediction generation | Model-dependent | Remote CPU/GPU with features only | Model-dependent |
| G4 scoring and claim promotion | Sealed labels and registered rules | Local evaluation host | None |

G1 event studies, hazards, placebos, bootstrap, and equivalence checks remain
CPU/RAM tasks even when the four-field panel is large. GPU availability does
not change the endpoint, field-selection, or negative-control contracts.

The current G3 core design has 324 cells per domain and seed: 108 NABM cells
and 216 frozen or explicit-adaptive-baseline cells. This partition must be
recomputed if G0 changes any factor level. A four-domain run would therefore
contain 1,296 cells per seed if every field reaches G3; that arithmetic is a
capacity estimate, not authorization to run before G1 and GPU promotion pass.

The prospective cross-field contract is
`experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml`. Its
engineering-only shell and candidate universe are frozen, but scientific field
selection, outcome-blind maximin inputs, reserve assignment, and the G0 field
assignment remain unfrozen or `PENDING`. Compute availability must not be used
to choose fields, shocks, endpoints, or reserves from observed outcomes.

## 3. Artifact Flow

```text
mounted OpenAlex derived root (local HDD)
    -> footer inventory: INVENTORY_ONLY
    -> raw main-ID recount: BLOCKED on one W-1 row
    -> frozen grammar-valid view: 492,293,206 distinct IDs, zero duplicates, W0 retained WARN
    -> work-source membership + row conservation: PASS under frozen quarantine (audit only)
    -> declared work-target coverage + descriptive author/work profile: MEASURED
    -> current-snapshot taxonomy references + exact 26-field/null margins: PASS (audit only)
    -> current-content full-root SHA-256 baseline: PASS (historical identity UNVERIFIED)
    -> bounded main-work date fields and metadata timestamps v3: PASS (audit only)
    -> W0/author entity, duplicate/primary-child, citation-edge chronology/available_at,
       maturation/censoring, t-1 graph, and leakage gates
    -> outcome-blind field-structure and candidate-eligibility audit
    -> four-field maximin trace, reserve order, atomic shock/split registry
    -> deterministic cutoff-aware extraction
    -> two-field x two-shock DISCOVERY_ONLY pilot (local SSD, no claim gate)
    -> partitioned four-field panel + temporal graph + shock/cohort table (local SSD)
    -> checksummed transfer package without raw identity mapping
    -> remote CPU jobs: G1 and B0-B3
    -> remote GPU jobs: approved embedding robustness and B4/G3 NABM cells
    -> row-level predictions, draw summaries, checkpoints, logs, and manifests
    -> local sealed-label scoring and claim gate
```

The membership/profile and taxonomy-reference passes are audit-only component
results. They do not mean that a filtered child dataset or an eligible
researcher panel now exists, nor that an empirical field panel has been
selected.

The full OpenAlex root, raw author-ID mapping, unnecessary abstracts, and
unselected sidecars stay on the data host. An exception requires more than an
I/O benchmark: a tracked protocol amendment must designate the remote host as
an authorized data-processing replica, prove persistent capacity and access
controls, reproduce full checksums and lineage manifests, and specify teardown
or retention. Without all of those conditions, the root is not transferred.

## 4. Materialized Transfer Package

G0 must freeze the estimand, risk-set sampling, weights, and partitions before
any confirmatory transfer package is built. Releases are staged:

1. **G0 release:** discovery, G1 empirical-replication, and model-selection
   artifacts only. No G4 row or feature is released.
2. **Post-G3 release:** G4 features only, released after model code, fitted
   checkpoints, hyperparameters, seeds, prediction schema, and failure rules
   for every registered model and robustness variant are frozen.

Each release contains only the minimum required artifacts:

- pseudonymous researcher, topic, opportunity, and row IDs;
- cutoff-aware researcher-topic opportunity rows;
- `t-1` temporal graph data in a declared sparse format;
- attention-shock and entry-cohort tables;
- time-eligible peer-active and prior-collaborator outcome-availability
  proxies;
- matched-non-neighbor and shuffled-network placebo exposures;
- frozen chaperoned-versus-independent-entry labels;
- pre-entry capacity-bridge proxies;
- risk-set inclusion probabilities and registered weights;
- permitted partition identifiers for that release;
- immutable field, shock-family, episode, response-cohort, and cold-holdout
  identifiers;
- the component-keyed seed and random-stream ledger; and
- `transfer_manifest.json` plus file checksums.

The transfer manifest records:

- source-manifest hash and authoritative-root identifier;
- measurement-registry hash;
- authoritative taxonomy-reference and cross-field-matrix hashes;
- maximin selection-trace, stable-ID tie-break, reserve-order, and
  independent-episode-ledger hashes;
- extractor Git commit and dirty-tree status;
- schema version, row counts, date range, domains, and partition names;
- every feature's source-event convention, `available_at` rule, graph cutoff,
  and proxy/non-observation interpretation;
- atomic author/topic-family/shock assignment and domain-cold, author-cold,
  or labeled author-warm status;
- pseudonymization method and location of the local-only ID mapping;
- per-file SHA-256, byte size, and compression;
- destination path and transfer completion time; and
- the command and software version used to validate the package.

An isolated parquet bucket is not a representative pilot package. The pilot
must come from a deterministic full-root scan or a predeclared index produced
from that scan.

## 5. G4 Label Isolation

The final G4 package sent to the server is `features_only`. Entry, persistence,
and other scoring labels remain on the local evaluation host. The remote job
returns predictions keyed by immutable `row_id`; it does not return a score.

The primary cross-field bundle is domain-and-author-cold: the entire selected
field and its shock episodes are absent from fitting and model selection, and
no training label belongs to a researcher in the sealed evaluation rows. Only
past-observable features allowed by the frozen contract may describe such a
researcher. A domain-cold but author-warm bundle is a separately named
robustness artifact and cannot replace a missing or failed primary bundle.
Topic families, shock episodes, and response cohorts are atomic; no row may be
moved across a split to repair coverage after labels open.

Before any G4 label opens, complete checksummed prediction bundles must exist
for every registered B0-B4 model and robustness variant. A model that cannot
produce its declared bundle is recorded as a pre-label failure; it is not
silently removed after another model's score becomes visible.

The local evaluator opens each sealed label partition once, joins on `row_id`,
and applies the G0-registered log score, Brier score, calibration, and macro
trajectory rules. Any remote artifact containing G4 labels, any tuning after
labels open, or any second unregistered evaluation invalidates the G4 result.

## 6. GPU Promotion Gate

The existing researcher-pivot learning example uses per-agent Python execution,
CPU `float64` tensors, and no device option. Merely running that script on a
machine with GPUs will not make it a GPU workload. The empirical NABM path must
first implement and verify a tensor-batched device-aware kernel.

GPU execution is promoted only when:

1. CPU and GPU outputs satisfy the G0-registered numerical and stochastic
   parity tolerance;
2. memory use fits the declared device and leaves a recorded safety margin;
3. an end-to-end benchmark at the confirmatory scale, including data movement
   and synchronization, reaches the registered speedup threshold;
4. stage timings show that policy training or inference, rather than Python,
   graph, or I/O overhead, is the dominant cost; and
5. the benchmark writes device, runtime, config, seed, and result provenance.

A provisional operational threshold is at least 2x end-to-end throughput over
the best CPU execution. G0 must either register that threshold or replace it
before confirmatory runs. If parity or throughput fails, use CPU parallelism
even when a GPU is available.

The repository already provides preflight benchmarks:

```bash
uv run python scripts/benchmark_accelerator_policy_core.py \
  --device cuda --agent-counts 128 512 2048

uv run python scripts/benchmark_toy_gpu_core.py \
  --devices cpu cuda --toys toy2 toy4 toy5 \
  --training-backends loop tensor_batched \
  --warmup-runs 1 --repeats 3 --require-backend-parity
```

These scripts validate the existing accelerator surface; they do not validate
the future empirical NABM implementation.

## 7. Remote Preflight Gate

Before code, data, or jobs are sent to the GPU server:

1. establish non-interactive authenticated access suitable for resumable jobs;
2. record hostname, OS, CPU, RAM, mounted storage, free space, and filesystem;
3. record `nvidia-smi`, GPU count and model, driver, CUDA, and per-device memory;
4. record whether Docker, another container runtime, Slurm, or only direct
   process execution is available;
5. clone or check out an exact pushed Git commit; never execute an unrecorded
   dirty worktree;
6. install from the committed `uv.lock` using the research dependency profile;
7. run the package tests and the CPU/GPU parity benchmark; and
8. save the result as a checksummed remote-preflight artifact.

Do not assume Slurm or a particular GPU model from historical server use. The
launcher contract is chosen only after this live preflight.

## 8. Remote Run Bundle

Every remote run has a unique run ID and immutable bundle containing:

- Git commit and `uv.lock` hash;
- source, transfer, and measurement-registry hashes;
- model and experiment config;
- partition name, seed ledger, and paired random-stream map;
- host, device, driver, CUDA, Python, Torch, and dependency metadata;
- start/end time, exit status, stdout/stderr, and stage timings;
- resumable checkpoints and the checkpoint-parent relation;
- row-level predictions or declared sufficient summaries;
- model weights when required for audit; and
- an output manifest with per-file SHA-256.

Retries reuse the same immutable inputs and receive a new attempt ID. A retry
must not silently change a seed, config, model, partition, or numerical mode.
Partial outputs remain quarantined until the output manifest and row-count
checks pass.

## 9. Immediate Execution Order

1. Preserve the immutable evidence checkpoints: the completed `INVENTORY_ONLY`
   footer audit, blocked zero-invalid raw recount, passed grammar-valid recount,
   targeted sentinel-incidence result, passed work-source membership/profile
   and current-snapshot taxonomy-reference components, bounded `W0`
   sensitivity, completed current-content baseline, and passed v3 main-work
   temporal-field component. Do not rewrite the raw failure as a clean
   source-key result or promote any component to G-1 or empirical G1 passage.
2. Preserve the compact, path-redacted work-source successor summary and its
   completed clean semantic-resume check together with the tracked full
   taxonomy-reference JSON and Markdown. Keep the work-source fingerprint and
   output-manifest SHA-256, taxonomy result fingerprint, 1,423-checkpoint tree
   binding, projected-payload rehash rule, temporal v3 report/receipt hashes and
   compact summaries, and each component's non-claim adjacent to every
   component `PASS`.
3. Preserve the completed current-content plan, report, outputs receipt, and
   `COMPLETE` state. Before every downstream full run, revalidate that exact
   receipt and its plan/report/state/source bridge rather than repeating the
   920.523 GiB content scan by default. This current baseline cannot
   retroactively prove inventory-era byte equality and does not by itself pass
   G-1.
4. Preserve the completed bounded `W0` sensitivity without treating it as an
   entity decision. After exact revalidation of the completed content parent,
   run the separately authorized relation audit against the exact content-bound
   493-main plus 930-topic source task set; pilots cannot promote this gate.
   Inherit the bounded v3 main-work date-field pass without treating its three
   descriptive reversals as record-event-order failures. Then resolve author
   entity and split/merge validity, citation-edge chronology and `available_at`,
   outcome maturation and censoring, `t-1` graph rules, leakage, candidate
   eligibility, and an authoritative, parent-content-identified binding of the
   completed current-snapshot taxonomy-reference component.
5. Complete the outcome-blind field-structure audit. Keep final field selection
   `PENDING` until that authoritative binding and the author, edge-time/leakage,
   and structural checks all pass; inspect no confirmatory entry, persistence,
   or timing-gap result while constructing the maximin inputs.
6. Revalidate the fail-closed G-1 readiness ledger and resolve every missing
   canonical, relation, author, citation, cutoff-aware panel, outcome-blind
   field-structure, and representative pilot receipt. Structural validation of
   a pending ledger is not G-1 passage.
7. Freeze and hash
   `experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml`, including
   the four-field selection, stable-ID tie-break, reserve order, at least three
   independent episodes per field, atomic splits, endpoints, equivalence
   margins, and the domain-and-author-cold assignment.
8. Build the cutoff-aware extractor and materialize the checksummed two-field
   by two-shock `DISCOVERY_ONLY` pilot on local SSD. It tests extraction,
   clocks, placebos, weights, and leakage only and cannot pass G1.
9. Resolve every quantitative item in the additive G0 overlay by writing a
   separately versioned successor; do not promote the pending v0 overlay in
   place.
10. Commit and publish the complete protocol and audit chain so later remote
   jobs use an exact Git revision.
11. Run the live preflight with `$GPU_SSH_IDENTITY` selected explicitly and
   wait for or coordinate sufficient free GPU memory.
12. Transfer only the pilot package needed for an execution benchmark; no
    confirmatory label is included.
13. Freeze G0, build the four-field G0 release without G4 rows, and begin the
    three-field G1 CPU/RAM analysis with its Holm and equivalence gates.
14. Expand to B4/G3 GPU jobs only after G1, parity, and throughput gates pass.
15. After G3 freezes every registered predictor, build the separate
    domain-and-author-cold G4
    feature-only release and return all prediction bundles before local scoring.

No current step requires copying the approximately 921 GiB root to the GPU
server.
