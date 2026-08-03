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
SCRIPT = ROOT / "scripts/build_pivot_feedback_s0_authorship_orcid_affiliation.py"
CONFIG = (
    ROOT
    / "experiments/configs/pivot_feedback_s0_authorship_orcid_affiliation_v1.yaml"
)
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v2.yaml"
CORE = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_s0_threshold_free_core_full_20260723.summary.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_s0_authorship_orcid_affiliation_test_module", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _config() -> dict:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def test_contract_is_parent_bound_cpu_only_and_fail_closed() -> None:
    config = _config()

    assert config["schema_version"] == MODULE.CONFIG_SCHEMA
    assert config["gate_parent"]["sha256"] == hashlib.sha256(
        GATE.read_bytes()
    ).hexdigest()
    assert config["s0_core_parent"]["sha256"] == hashlib.sha256(
        CORE.read_bytes()
    ).hexdigest()
    assert config["execution"]["gpu_used"] is False
    assert config["orcid_contract"]["natural_person_identity_claim_allowed"] is False
    assert (
        config["affiliation_contract"]["historical_affiliation_observability_established"]
        is False
    )
    assert config["output_contract"]["component_may_complete_s0"] is False
    assert config["output_contract"]["component_may_authorize_d0"] is False
    assert config["output_contract"]["component_may_authorize_model_fitting"] is False


def test_orcid_parser_separates_syntax_and_iso_7064_checksum() -> None:
    values = pa.array(
        [
            "https://orcid.org/0000-0002-1825-0097",
            "https://orcid.org/0000-0002-1694-233X",
            "https://orcid.org/0000-0002-1825-0098",
            "https://orcid.org/0000-0002-1694-233x",
            "0000-0002-1825-0097",
            None,
        ]
    )
    code, syntax, checksum = MODULE._parse_orcid(
        values, _config()["orcid_contract"]["accepted_form"]
    )

    assert syntax.tolist() == [True, True, True, False, False, False]
    assert checksum.tolist() == [True, True, False, False, False, False]
    assert code[0] != 0
    assert code[1] != 0
    assert code[0] != code[1]
    assert code[2:].tolist() == [0, 0, 0, 0]


def test_openalex_id_and_affiliation_projection_are_row_exact() -> None:
    ids = pa.array(
        [
            "https://openalex.org/A1",
            "https://openalex.org/A0",
            "https://openalex.org/A01",
            "https://openalex.org/W1",
            None,
        ]
    )
    numeric, valid = MODULE._parse_openalex_ids(
        ids,
        pattern=_config()["source"]["author_id_pattern"],
        prefix=MODULE.AUTHOR_PREFIX,
    )
    assert numeric.tolist() == [1, 0, 0, 0, 0]
    assert valid.tolist() == [True, True, False, False, False]

    raw = pa.array(
        [None, [], [""], ["alpha"], ["", "beta"]],
        type=pa.list_(pa.string()),
    )
    assert MODULE._nonempty_list_row_mask(raw).tolist() == [
        False,
        False,
        False,
        True,
        True,
    ]

    institution_type = pa.list_(
        pa.struct([pa.field("id", pa.string()), pa.field("name", pa.string())])
    )
    institutions = pa.array(
        [
            None,
            [],
            [{"id": "https://openalex.org/I7", "name": "x"}],
            [
                {"id": "https://openalex.org/I08", "name": "bad"},
                {"id": None, "name": "missing"},
            ],
            [
                {"id": "https://openalex.org/I9", "name": "y"},
                {"id": "https://openalex.org/I9", "name": "y"},
            ],
        ],
        type=institution_type,
    )
    row_flags, parents, projected = MODULE._institution_projection(
        institutions, _config()["source"]["institution_id_pattern"]
    )
    assert row_flags.tolist() == [False, False, True, False, True]
    assert parents.tolist() == [2, 4, 4]
    assert projected.tolist() == [7, 9, 9]


def test_author_orcid_reducer_is_stable_and_reports_shared_codes(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "author-input"
    input_root.mkdir()
    paths = []
    records_by_bucket: dict[int, list[tuple[int, int]]] = {
        1: [(1, 101), (1, 101)],
        2: [(2, 202), (2, 303)],
        3: [(3, 101)],
    }
    for bucket in range(MODULE.PARTITION_COUNT):
        path = input_root / f"part-{bucket:03d}.bin"
        rows = records_by_bucket.get(bucket, [])
        array = np.array(rows, dtype=MODULE.AUTHOR_ORCID_DTYPE)
        array.tofile(path)
        paths.append(path)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    stable_paths, diagnostics = MODULE._reduce_author_orcid(paths, scratch)
    stable = np.concatenate(
        [
            np.fromfile(path, dtype=MODULE.STABLE_AUTHOR_DTYPE)
            for path in stable_paths
            if path.stat().st_size
        ]
    )

    assert stable["author_numeric_id"].tolist() == [1, 3]
    assert stable["orcid_code"].tolist() == [101, 101]
    assert diagnostics == {
        "distinct_author_orcid_pairs": 4,
        "author_tokens_with_valid_orcid": 3,
        "stable_single_valid_orcid_author_tokens": 2,
        "multiple_valid_orcid_author_tokens": 1,
        "distinct_valid_orcids_on_stable_tokens": 1,
        "shared_valid_orcids_across_stable_tokens": 1,
        "stable_author_tokens_on_shared_valid_orcids": 2,
        "maximum_stable_author_tokens_per_valid_orcid": 2,
    }


def test_institution_reducer_deduplicates_within_field_year(
    tmp_path: Path,
) -> None:
    key_2020 = 11 * MODULE.YEAR_MULTIPLIER + 2020
    key_2021 = 11 * MODULE.YEAR_MULTIPLIER + 2021
    rows_by_bucket = {
        key_2020 % MODULE.PARTITION_COUNT: [
            (key_2020, 10),
            (key_2020, 10),
            (key_2020, 11),
        ],
        key_2021 % MODULE.PARTITION_COUNT: [(key_2021, 10)],
    }
    paths = []
    for bucket in range(MODULE.PARTITION_COUNT):
        path = tmp_path / f"institution-{bucket:03d}.bin"
        np.array(
            rows_by_bucket.get(bucket, []), dtype=MODULE.INSTITUTION_DTYPE
        ).tofile(path)
        paths.append(path)
    cells: dict[int, dict[str, int]] = {}

    diagnostics = MODULE._reduce_institutions(paths, cells)

    assert diagnostics == {
        "distinct_institution_tokens": 2,
        "distinct_institution_field_year_pairs": 3,
    }
    assert cells[key_2020]["distinct_institutions"] == 2
    assert cells[key_2021]["distinct_institutions"] == 1


def test_field_year_table_enforces_monotonicity() -> None:
    key = 11 * MODULE.YEAR_MULTIPLIER + 2020
    metrics = {
        "raw_authorship_rows": 10,
        "grammar_valid_author_id_rows": 8,
        "nonnull_orcid_rows": 7,
        "syntax_valid_orcid_rows": 6,
        "checksum_valid_orcid_rows": 5,
        "author_work_affiliation_rows": 4,
        "author_work_institution_rows": 3,
        "stable_orcid_author_work_pairs": 2,
        "stable_orcid_author_tokens": 1,
        "distinct_institutions": 3,
    }
    table = MODULE._field_year_table({key: metrics})
    assert table.schema.metadata == {b"schema_version": MODULE.FIELD_YEAR_SCHEMA.encode()}
    assert table.to_pylist()[0]["checksum_valid_orcid_rows"] == 5

    broken = dict(metrics)
    broken["checksum_valid_orcid_rows"] = 8
    with pytest.raises(MODULE.ProjectionError, match="ORCID cell monotonicity"):
        MODULE._field_year_table({key: broken})


def test_selected_checkpoint_validation_rejects_wrong_table(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    content_root = tmp_path / "content"
    relative = "table/part.parquet"
    source_path = source_root / relative
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"fixture")
    current = MODULE._stat_identity(source_path)
    task_fingerprint = "a" * 64
    plan_sha = "b" * 64
    checkpoint = {
        "schema_version": MODULE.CHECKPOINT_SCHEMA,
        "task_fingerprint_sha256": task_fingerprint,
        "plan_sha256": plan_sha,
        "relative_path": relative,
        "table_name": "wrong-table",
        "task_index": 7,
        "source_identity_before": current,
        "source_identity_after": current,
        "expected_parent_identity": {
            "bytes": current["bytes"],
            "mtime_ns": current["mtime_ns"],
        },
        "bytes_hashed": current["bytes"],
        "content_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        _canonical_bytes(checkpoint)
    ).hexdigest()
    digest = hashlib.sha256(relative.encode()).hexdigest()
    checkpoint_path = content_root / "checkpoints" / digest[:2] / f"{digest}.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(MODULE.ProjectionError, match="checkpoint table mismatch"):
        MODULE._validate_source_tasks_and_checkpoints(
            source_root.resolve(),
            content_root.resolve(),
            [{"relative_path": relative, "source_stat": current}],
            {
                "task_fingerprint_sha256": task_fingerprint,
                "plan_sha256": plan_sha,
            },
            table_name="expected-table",
        )


def test_emitted_schema_excludes_outcome_and_selection_metrics() -> None:
    config = _config()
    emitted = set(MODULE.FIELD_YEAR_METRICS)

    assert emitted.isdisjoint(config["forbidden_metrics"])
    assert "predictive_score" not in emitted
    assert "selected_best_threshold" not in emitted
    assert "entry_effect_size" not in emitted


def test_linux_memavailable_parser_counts_reclaimable_memory(
    tmp_path: Path,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       1048576 kB\n"
        "MemFree:          10240 kB\n"
        "MemAvailable:    786432 kB\n",
        encoding="utf-8",
    )

    assert MODULE._available_memory_bytes(meminfo) == 786432 * 1024

    meminfo.write_text("MemFree: 10240 kB\n", encoding="utf-8")
    with pytest.raises(MODULE.ProjectionError, match="MemAvailable is missing"):
        MODULE._available_memory_bytes(meminfo)
