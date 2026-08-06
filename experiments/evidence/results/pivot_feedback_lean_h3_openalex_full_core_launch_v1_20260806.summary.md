# Lean H3 OpenAlex full-core transfer launch v1

- Observed: `2026-08-06T05:13:36Z`
- Status: `PASS_COMMITTED_FULL_CORE_TRANSFER_LAUNCH_RUNNING_AT_OBSERVATION_NO_ROW_DECODE`
- Semantics: point-in-time minimum progress, not transfer completion
- Scope: exact-manifest Works plus Topics/Subfields/Fields source-byte transfer only

## Canonical launch

- Git commit: `347bde998f6c76b7ccd264683288fdfd13b3fdd1`
- Downloader SHA-256: `7c16af6e38569889969f7528dbbd2cb1bccef87ab407fc5ae4cd23bd62f59e5f`
- systemd unit: `pivot-feedback-lean-h3-openalex-full-core-v1.service`
- Invocation ID: `3572b87997a44524b92edd1360d9acd8`
- Start: `2026-08-06T14:12:59+09:00`
- State at observation: `active/running`, state file `RUNNING`, run count `4`
- Restart policy: `on-failure`, 30-second delay

The first invocation was intentionally stopped at a file boundary to correct a
stale top-level state marker. It had checkpointed 1,079,675,758 bytes in 213
complete files with zero partial files. The canonical restart required no
uncheckpointed-tail recovery and no recovery-validation transfer.

## Progress at observation

- Manifest: 2,454 files, 724,973,106,144 source bytes
- Complete files: at least 222
- Materialized source bytes: at least 1,140,200,344
- Remaining source-byte upper bound: 723,832,905,800
- HTTP accounting: 223 HEAD requests and 225 range requests
- Canary pause count: 1

The latest capacity check before the most recent segment observed
3,022,607,728,640 available bytes. After reserving the remaining source bytes
and 1,087,459,659,216 working bytes, the margin was 1,211,315,095,957 bytes.

## Verification and boundary

- Whole-repository regression before the observability patch: 3,244 passed
- Transfer and release-boundary regression after the patch: 45 passed
- Ruff after the patch: pass
- Release artifact inspection before the patch: pass
- Source row values decoded: 0
- Current G1 label, model-selection, and holdout/private G4 rows read: 0
- Models fitted and seal-open events: 0

This receipt establishes only that the committed exact-manifest byte-transfer
process was active and had advanced beyond the verified canary while preserving
the capacity invariant. It is not transfer completion, row decoding,
historical completeness, H3 maturity, cohort eligibility, an association, a
null, a mechanism, a causal effect, or a scientific effect.
