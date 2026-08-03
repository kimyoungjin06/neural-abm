# Canonical work-time-spine pilot benchmark tool

Status: `PILOT_BENCHMARK_CAPTURED_OPERATOR_LOCAL_NONPROMOTABLE`

Production status: `PRODUCTION_NOT_RUN`

This repository-only tool converts two already existing operator-local inputs:

1. the exact one-file, nonpromotable canonical-spine pilot report; and
2. a GNU `/usr/bin/time` capture made outside the pilot output root.

It does not launch the production runner, open Parquet, validate a complete
pilot receipt, pass the downstream capacity review, or authorize a full run.
No pilot was executed while implementing or synthetic-testing the converter;
a separately authorized operator-local pilot was executed on 2026-07-21.

## Tool and output contract

The implementation is
`scripts/summarize_openalex_canonical_work_time_spine_pilot_benchmark.py`.
It writes exactly
`openalex_g1_canonical_work_time_spine_pilot_benchmark.json` with schema
`openalex_g1_canonical_work_time_spine_pilot_benchmark.v1` and status
`PILOT_BENCHMARK_CAPTURED_OPERATOR_LOCAL_NONPROMOTABLE`.

The tool requires the pilot report to have the exact canonical-spine pretty
JSON byte ABI and the distinct pilot schema, status, one-file scope,
`PRODUCTION_NOT_RUN`, zero partitions, `production_component_pass=false`, and
path-redacted provenance. It hashes both input files from stable regular-file
snapshots. The inputs and output must be operator-owned, private, outside the
repository, and the benchmark output must also be outside the pilot output
root. The output parent must preexist with mode `0700`; the new artifact is
atomically published with mode `0600` and is never overwritten.

Rates are calculated with decimal arithmetic and rounded half-up to six decimal
places:

```text
rows_per_second = selected_source_rows / pilot_wall_seconds_external
source_bytes_per_second = selected_source_bytes / pilot_wall_seconds_external
projected_full_scan_wall_seconds =
    492293207 * pilot_wall_seconds_external / selected_source_rows
```

The last quantity is only a scan-and-staging projection. The artifact retains
`downstream_pipeline_capacity_review=NOT_RUN` and explicitly excludes:

- the 4,096-handle spool and NumPy demultiplex;
- external sorting and Parquet publication;
- the 9,185-artifact final rehash; and
- storage contention or concurrent I/O.

It records `external_command_exit_status=NOT_CAPTURED_BY_V1_FORMAT`,
`pilot_report_time_pairing=OPERATOR_ASSERTED_NOT_PROGRAMMATICALLY_VERIFIED`,
and `terminal_resume_exclusion=NOT_PROGRAMMATICALLY_VERIFIED`. It is therefore
not a programmatic parent of a future full receipt, is insufficient for full
authorization, and must remain a manual-review-only diagnostic.

## 2026-07-21 operator-local result

The one-file pilot ran once from 10:18:37 to 10:18:48 KST with runner exit
status 0. It selected 1,000,000 rows and 312,250,942 source bytes. Its report
retained `PRODUCTION_NOT_RUN`, `production_component_pass=false`,
`pass_g1=false`, zero partitions, and the exact nonpromotable pilot status.

The private output roots are:

- pilot:
  `/ssd/openalex-g1-canonical-work-time-spine-pilot-20260721`;
- timing and benchmark:
  `/ssd/openalex-g1-canonical-work-time-spine-pilot-benchmark-20260721`.

The evidence hashes are:

| Artifact | SHA-256 |
| --- | --- |
| pilot receipt | `eef08252fc6147084cb50752a456541b408992d599d6d80d2d5806b48310b099` |
| pilot report | `c24e7242a821f43f4d959046fd3ee578b9f47da4fd6357b05b102fc1176e8cba` |
| raw GNU time capture | `3577585673cf9526b5d740d1d35c49ec0160902590ed9878479d0a93975cc7d3` |
| benchmark JSON | `8179b531ae8f70f28d707e0a043ac4aea13c3b5a831744fb683d6ec4962f5666` |

The capture reported 11.69 seconds, maximum RSS 152,388 KiB, 85,543.199316
rows/s, and 26,710,944.568007 source bytes/s. The simple full-row
scan-and-staging projection is 5,754.907590 seconds, approximately 95.9
minutes. The implementation-level independent verifier separately returned
`PASS_READ_ONLY_INDEPENDENT_OUTPUT_VERIFICATION` for the eight receipt-bound
artifacts and observed an unchanged output tree.

The runner exit status and report/time pairing were observed by the operator in
one shell execution but are not encoded by the v1 time-capture format. The
benchmark therefore correctly keeps both limitations explicit. This result
does not pass the downstream capacity gate or authorize full execution.

## Operator procedure

The following command contract was used with private concrete paths and is
retained for future reruns. It may be used only after the source-drive health
and quiet-window gates pass and a separate one-file pilot is explicitly
authorized.

```bash
umask 077
LC_ALL=C /usr/bin/time \
  -f 'elapsed_seconds=%e\nmaximum_rss_kib=%M\nfilesystem_inputs=%I\nfilesystem_outputs=%O' \
  -o /operator/private/benchmark/openalex_g1_canonical_work_time_spine_pilot_time.txt \
  uv run python scripts/audit_openalex_canonical_work_time_spine_v3.py \
  --execute --max-files 1 \
  --content-receipt /operator/exact/content_freeze_audit.outputs.json \
  --temporal-receipt /operator/exact/main_temporal_validity_audit.outputs.json \
  --temporal-config experiments/configs/openalex_g1_main_temporal_validity_v2.yaml \
  --source-root /operator/exact/openalex-root \
  --output-dir /operator/private/pilot-output
```

After the nonpromotable pilot finishes and its report has been reviewed:

```bash
uv run python \
  scripts/summarize_openalex_canonical_work_time_spine_pilot_benchmark.py \
  --pilot-report /operator/private/pilot-output/work_time_spine_pilot.json \
  --time-capture /operator/private/benchmark/openalex_g1_canonical_work_time_spine_pilot_time.txt \
  --output /operator/private/benchmark/openalex_g1_canonical_work_time_spine_pilot_benchmark.json
```

## Remaining trust boundary

The converter verifies the report's exact byte format, bounded claim fields,
result fingerprint, and SHA-256, but it does not independently reconstruct the
complete pilot receipt or rehash every pilot artifact. The v1 four-line GNU
`time` capture also records no exit status, nonce, command identity, or report
fingerprint. It therefore cannot prove that the capture came from a successful
fresh pilot rather than a failed command, another command, or terminal resume,
nor can it programmatically pair the two inputs. Operator review must establish
that pairing and bind the reported SHA-256 to the reviewed pilot output before
using the diagnostic. GNU `time` filesystem counters are platform-level
diagnostics, not exact physical-device byte accounting. The source-drive
health, sustained quiet window, downstream capacity, independent full-output
verifier, fresh output preflight, and explicit full authorization gates remain
separate.

The frozen production config intentionally continues to say
`PENDING_IMPLEMENTATION_AND_REVIEW` until this new tool and its review are
integrated through a deliberate future contract refreeze. Implementation alone
does not rewrite an already frozen readiness claim.
