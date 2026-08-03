from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_cutoff_primitives_source_replay_v2.py"
SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_cutoff_primitives_source_replay_full_20260723.summary.json"
)
SUMMARY_MD = SUMMARY.with_suffix(".md")
DOC = ROOT / "docs/research/pivot-feedback-cutoff-primitives-source-replay-v2.md"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_cutoff_source_replay_v2_test_module", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _tables() -> tuple[np.ndarray, pa.Table, pa.Table]:
    source = np.zeros(2, dtype=MODULE.SOURCE_RECORD_DTYPE)
    source["work"] = [4096, 8192]
    source["type"] = [1, 2]
    source["paratext"] = [0, 1]
    source["retracted"] = [2, 1]
    source["authors"] = [2, 7]
    source["topic"] = [0, 99]
    source["field"] = [0, 17]
    source["domain"] = [0, 3]
    output = pa.Table.from_arrays(
        [
            pa.array([4096, 8192], type=pa.uint64()),
            pa.array([10, 20], type=pa.int32()),
            pa.array([2000, 2001], type=pa.uint16()),
            pa.array([2, 3], type=pa.uint8()),
            pa.array([1, 2], type=pa.uint8()),
            pa.array([False, True], type=pa.bool_()),
            pa.array([None, True], type=pa.bool_()),
            pa.array([2, 7], type=pa.uint16()),
            pa.array([None, 99], type=pa.uint32()),
            pa.array([None, 17], type=pa.uint8()),
            pa.array([None, 3], type=pa.uint8()),
        ],
        schema=MODULE.WORK_SCHEMA,
    )
    canonical = pa.Table.from_arrays(
        [
            pa.array([4096, 8192], type=pa.uint64()),
            pa.array([10, 20], type=pa.int32()),
            pa.array([2000, 2001], type=pa.uint16()),
            pa.array([2, 3], type=pa.uint8()),
        ],
        schema=MODULE.CANONICAL_SCHEMA,
    )
    return source, output, canonical


def test_verifier_is_independent_and_record_abi_is_fixed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import build_pivot_feedback_cutoff_primitives" not in source
    assert "from build_pivot_feedback_cutoff_primitives" not in source
    assert MODULE.SOURCE_RECORD_DTYPE.itemsize == 19
    assert MODULE.PARTITION_COUNT == 4096


def test_exact_source_projection_and_canonical_time_pass() -> None:
    source, output, canonical = _tables()
    expected, rows = MODULE._compare_partition_records(
        source,
        output,
        canonical,
        partition=0,
        full_scope=True,
    )
    assert rows == 2
    assert np.array_equal(source, expected)


def test_metadata_row_permutation_is_detected_despite_identical_margins() -> None:
    source, output, canonical = _tables()
    permuted = source.copy()
    for field in (
        "type",
        "paratext",
        "retracted",
        "authors",
        "topic",
        "field",
        "domain",
    ):
        permuted[field] = source[field][::-1]
        assert sorted(permuted[field].tolist()) == sorted(source[field].tolist())
    with pytest.raises(
        MODULE.ReplayVerificationError,
        match=r"source replay mismatch field=work_type_code work_id=4096",
    ):
        MODULE._compare_partition_records(
            permuted,
            output,
            canonical,
            partition=0,
            full_scope=True,
        )


def test_nullable_value_mask_mismatch_is_detected() -> None:
    source, output, canonical = _tables()
    changed_retraction = source.copy()
    changed_retraction["retracted"][0] = 0
    with pytest.raises(
        MODULE.ReplayVerificationError,
        match=r"source replay mismatch field=is_retracted work_id=4096",
    ):
        MODULE._compare_partition_records(
            changed_retraction,
            output,
            canonical,
            partition=0,
            full_scope=True,
        )
    changed_topic = source.copy()
    changed_topic["topic"] = [99, 0]
    changed_topic["field"] = [17, 0]
    changed_topic["domain"] = [3, 0]
    with pytest.raises(
        MODULE.ReplayVerificationError,
        match=r"source replay mismatch field=primary_topic_id work_id=4096",
    ):
        MODULE._compare_partition_records(
            changed_topic,
            output,
            canonical,
            partition=0,
            full_scope=True,
        )


def test_canonical_time_mutation_is_detected() -> None:
    source, output, canonical = _tables()
    changed = canonical.set_column(
        canonical.schema.get_field_index("publication_year"),
        "publication_year",
        pa.array([2000, 2002], type=pa.uint16()),
    )
    with pytest.raises(
        MODULE.ReplayVerificationError,
        match=r"publication_year value mismatch in partition 0",
    ):
        MODULE._compare_partition_records(
            source,
            output,
            changed,
            partition=0,
            full_scope=True,
        )


def test_source_parser_rejects_topic_hierarchy_null_mask_disagreement() -> None:
    columns = [
        "id",
        "type",
        "is_paratext",
        "is_retracted",
        "authors_count",
        "primary_topic__id",
        "primary_topic__domain__id",
        "primary_topic__field__id",
    ]
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(["https://openalex.org/W1"]),
            pa.array(["article"]),
            pa.array([False]),
            pa.array([None], type=pa.bool_()),
            pa.array([1], type=pa.int64()),
            pa.array([None], type=pa.string()),
            pa.array(["https://openalex.org/domains/3"]),
            pa.array([None], type=pa.string()),
        ],
        names=columns,
    )
    with pytest.raises(
        MODULE.ReplayVerificationError,
        match="source topic/domain null-mask mismatch",
    ):
        MODULE._parse_source_batch(
            batch,
            projected_columns=columns,
            type_code_by_name={"article": 1},
        )


def test_content_checkpoint_self_digest_and_binding_are_recomputed(
    tmp_path: Path,
) -> None:
    relative = "openalex_works_20260330/part-0.parquet"
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
    checkpoint_dir = tmp_path / "checkpoints" / digest[:2]
    checkpoint_dir.mkdir(parents=True)
    source_stat = {
        "bytes": 100,
        "ctime_ns": 6,
        "device": 1,
        "inode": 2,
        "mode": 0o600,
        "mtime_ns": 5,
    }
    checkpoint = {
        "bytes_hashed": 100,
        "content_sha256": "a" * 64,
        "expected_parent_identity": {"bytes": 100, "mtime_ns": 5},
        "hash_algorithm": "sha256",
        "path_redacted": True,
        "plan_sha256": "b" * 64,
        "relative_path": relative,
        "schema_version": MODULE.CONTENT_CHECKPOINT_SCHEMA,
        "source_identity_after": source_stat,
        "source_identity_before": source_stat,
        "table_name": "openalex_works_20260330",
        "task_fingerprint_sha256": "c" * 64,
        "task_index": 7,
    }
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        MODULE._canonical_json_bytes(checkpoint)
    ).hexdigest()
    path = checkpoint_dir / f"{digest}.json"
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    raw_task = {
        "bytes": 100,
        "relative_path": relative,
        "source_stat": source_stat,
        "table_name": "openalex_works_20260330",
    }
    task = MODULE._validate_content_checkpoint(
        tmp_path,
        raw_task,
        task_index=7,
        plan_sha256="b" * 64,
        task_fingerprint="c" * 64,
    )
    assert task.content_sha256 == "a" * 64

    checkpoint["content_sha256"] = "d" * 64
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(
        MODULE.ReplayVerificationError,
        match="content checkpoint self-digest mismatch",
    ):
        MODULE._validate_content_checkpoint(
            tmp_path,
            raw_task,
            task_index=7,
            plan_sha256="b" * 64,
            task_fingerprint="c" * 64,
        )


def test_full_profile_closure_accepts_only_exact_frozen_margins() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "experiments/configs/pivot_feedback_cutoff_primitives_v0.yaml"
        ).read_text(encoding="utf-8")
    )
    source = config["source"]
    profile = config["input_profile"]
    code_by_name = {
        name: int(code)
        for code, name in profile["work_type_dictionary"].items()
    }
    metrics = {
        "files": source["files"],
        "source_bytes": source["source_bytes"],
        "raw_rows": source["raw_rows"],
        "quarantined_w_minus_1_rows": source["quarantined_w_minus_1_rows"],
        "retained_rows": source["retained_rows"],
        "null_counts": dict(profile["null_counts"]),
        "work_type_codes": {
            code_by_name[name]: count
            for name, count in profile["exact_work_type_counts"].items()
        },
        "authors_count_minimum": profile["authors_count_minimum"],
        "authors_count_maximum": profile["authors_count_maximum"],
    }
    MODULE._validate_full_profile(config, metrics)
    changed = dict(metrics)
    changed["retained_rows"] += 1
    with pytest.raises(
        MODULE.ReplayVerificationError,
        match="full source retained_rows mismatch",
    ):
        MODULE._validate_full_profile(config, changed)


def test_repository_summary_is_exact_and_claim_bounded() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == MODULE.FULL_STATUS
    assert summary["scope"] == {
        "full_scope": True,
        "production_source_replay_pass": True,
        "gpu_used": False,
        "source_files": 493,
        "source_bytes": 150_062_449_219,
        "source_content_rehashed_in_this_run": False,
    }
    assert summary["source_accounting"]["raw_rows"] == 492_293_207
    assert summary["source_accounting"]["quarantined_w_minus_1_rows"] == 1
    assert summary["source_accounting"]["retained_rows"] == 492_293_206
    replay = summary["row_level_verification"]
    assert replay["rows"] == 492_293_206
    assert replay["all_projected_values_and_null_masks_exact"] is True
    assert (
        replay["source_projection_record_sha256"]
        == replay["output_projection_record_sha256"]
        == "9a8f9b13250ab8b11aecd19b0d67b2a1dba9d4395e1d6e63da79f50f8989abf3"
    )
    assert summary["output_verification"]["output_tree_unchanged"] is True
    assert (
        summary["provenance"]["verifier_sha256"]
        == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    )
    assert summary["gate_status"]["g_minus_1_data_readiness"] == "PENDING"
    assert summary["gate_status"]["g0_measurement_freeze"] == "PENDING"
    assert summary["gate_status"]["scientific_effect"] == "NOT_ESTABLISHED"
    assert "not rehashed again" in summary["claim_boundary"]
    assert not Path(
        summary["provenance"]["operator_local_report_basename"]
    ).is_absolute()
    assert SUMMARY_MD.is_file()
    assert DOC.is_file()
