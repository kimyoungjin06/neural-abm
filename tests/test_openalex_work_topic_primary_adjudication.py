from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/adjudicate_openalex_work_topic_primary_scores.py"
VERIFIER_PATH = REPO_ROOT / "scripts/verify_openalex_work_topic_primary_adjudication.py"
CONFIG_PATH = (
    REPO_ROOT
    / "experiments/configs/openalex_g1_work_topic_relation_primary_adjudication_v1.yaml"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_openalex_work_topic_primary_adjudication_module", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adjudication = _load_module()


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_openalex_work_topic_primary_adjudication_verifier_module",
        VERIFIER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


def _base_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _key(work: int, topic: int) -> int:
    return (work << adjudication.TOPIC_BITS) | topic


def _locator(file_index: int, file_row: int) -> int:
    return (file_index << adjudication.LOCATOR_ROW_BITS) | file_row


def _events(rows: list[tuple[int, int, int, int]]) -> np.ndarray:
    result = np.zeros(len(rows), dtype=adjudication.EVENT_DTYPE)
    for index, (key, kind, score_bits, locator) in enumerate(rows):
        result[index]["key"] = np.uint64(key)
        result[index]["kind"] = np.uint8(kind)
        result[index]["score_bits"] = np.uint64(score_bits)
        result[index]["locator"] = np.uint32(locator)
    result.sort(order=["key", "kind", "score_bits", "locator"])
    return result


def _synthetic_conflict(
    base: Path, *, primary_score: float = 0.8
) -> tuple[dict[str, Any], Path, Path, dict[str, Any], dict[str, int]]:
    source = base / "source"
    parent = base / "parent"
    source.mkdir(parents=True)
    parent.mkdir(parents=True)
    main_path = source / "main/part-0.parquet"
    main_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["https://openalex.org/W1"], type=pa.string()),
                "primary_topic__id": pa.array(
                    ["https://openalex.org/T10001"], type=pa.string()
                ),
                "primary_topic__score": pa.array([primary_score], type=pa.float64()),
            }
        ),
        main_path,
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
    )
    key = _key(1, 10001)
    bits = {
        "lower": adjudication._score_bits(0.7),
        "primary": adjudication._score_bits(primary_score),
    }
    event_array = _events(
        [
            (key, int(adjudication.MAIN_KIND), 0, _locator(0, 0)),
            (key, int(adjudication.TOPIC_KIND), adjudication._score_bits(0.7), 0),
            (key, int(adjudication.TOPIC_KIND), adjudication._score_bits(0.8), 1),
        ]
    )
    event_array.tofile(parent / "work_topic_relation.events.sorted.bin")
    config = _base_config()
    config["blocked_parent"]["expected_conflict_keys"] = [
        {"work_numeric_id": 1, "topic_numeric_id": 10001}
    ]
    config["blocked_parent"]["required_full_counts"]["score_conflict_edge_groups"] = 1
    plan = {
        "tasks": [
            {
                "table_key": "main",
                "file_index": 0,
                "relative_path": "main/part-0.parquet",
                "footer_rows": 1,
                "source_rich_identity": adjudication.rich_identity(
                    main_path, "synthetic main"
                ),
            },
            {
                "table_key": "topics",
                "file_index": 0,
                "relative_path": "topics/part-0.parquet",
            },
        ]
    }
    return config, source, parent, plan, bits


def test_frozen_production_config_is_valid_and_hash_bound() -> None:
    config, observed, size = adjudication.read_config(CONFIG_PATH)

    assert observed == adjudication.PRODUCTION_CONFIG_SHA256
    assert size == CONFIG_PATH.stat().st_size
    assert config["execution"]["gpu_used"] is False
    assert config["blocked_parent"]["receipt_artifact_count"] == 1430
    assert verifier.sha256_file(SCRIPT_PATH) == verifier.PRODUCER_SHA256


def test_primary_exact_score_selects_only_matching_child_and_preserves_rows(
    tmp_path: Path,
) -> None:
    config, source, parent, plan, bits = _synthetic_conflict(tmp_path)

    selected, sidecar = adjudication.adjudicate_conflicts(
        config=config,
        source_root=source,
        parent_dir=parent,
        plan=plan,
        sort_manifest={"event_rows": 3},
    )

    key = _key(1, 10001)
    assert selected == {key: bits["primary"]}
    assert sidecar["unresolved_group_count"] == 0
    conflict = sidecar["conflicts"][0]
    assert conflict["canonical_score"] == 0.8
    assert [row["score"] for row in conflict["raw_topic_rows"]] == [0.7, 0.8]
    assert [row["selected"] for row in conflict["raw_topic_rows"]] == [False, True]
    assert conflict["main_primary_readback"]["file_row"] == 0


def test_primary_score_absent_from_children_remains_blocked(tmp_path: Path) -> None:
    config, source, parent, plan, _ = _synthetic_conflict(tmp_path, primary_score=0.9)

    with pytest.raises(
        adjudication.AdjudicationError,
        match="main primary score is absent from child variants",
    ):
        adjudication.adjudicate_conflicts(
            config=config,
            source_root=source,
            parent_dir=parent,
            plan=plan,
            sort_manifest={"event_rows": 3},
        )


def test_materialized_relation_collapses_same_score_and_adjudicates_conflict(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    output = tmp_path / "output"
    parent.mkdir()
    output.mkdir()
    key_conflict = _key(1, 10001)
    key_same = _key(2, 10002)
    key_single = _key(3, 10003)
    high = adjudication._score_bits(0.8)
    low = adjudication._score_bits(0.7)
    same = adjudication._score_bits(0.5)
    single = adjudication._score_bits(0.2)
    event_array = _events(
        [
            (key_conflict, int(adjudication.MAIN_KIND), 0, 0),
            (key_conflict, int(adjudication.TOPIC_KIND), low, 0),
            (key_conflict, int(adjudication.TOPIC_KIND), high, 1),
            (key_same, int(adjudication.TOPIC_KIND), same, 2),
            (key_same, int(adjudication.TOPIC_KIND), same, 3),
            (key_single, int(adjudication.TOPIC_KIND), single, 4),
        ]
    )
    event_array.tofile(parent / "work_topic_relation.events.sorted.bin")
    config = _base_config()
    config["execution"]["aggregation_chunk_rows"] = 4
    config["blocked_parent"]["required_full_counts"].update(
        {
            "event_rows": 6,
            "topic_raw_rows": 5,
            "unique_work_topic_edges": 3,
            "duplicate_edge_groups": 2,
            "duplicate_row_excess": 2,
            "score_conflict_edge_groups": 1,
        }
    )

    artifact = adjudication.materialize_relation(
        config=config,
        parent_dir=parent,
        output_dir=output,
        sort_manifest={"event_rows": 6},
        adjudicated={key_conflict: high},
    )

    records = np.fromfile(
        output / artifact["relative_path"], dtype=adjudication.DEDUP_DTYPE
    )
    assert artifact["rows"] == 3
    assert list(records["key"]) == [key_conflict, key_same, key_single]
    assert list(records["score_bits"]) == [high, same, single]

    observed = verifier.verify_payload_equivalence(
        parent_events_path=parent / "work_topic_relation.events.sorted.bin",
        relation_path=output / artifact["relative_path"],
        expected_counts={
            "event_rows": 6,
            "topic_raw_rows": 5,
            "unique_work_topic_edges": 3,
            "duplicate_edge_groups": 2,
            "duplicate_row_excess": 2,
            "score_conflict_edge_groups": 1,
        },
        selected_scores={key_conflict: high},
        chunk_rows=4,
    )
    assert observed["score_conflict_edge_groups"] == 1


def test_independent_payload_verifier_rejects_changed_canonical_score(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    key = _key(1, 10001)
    low = adjudication._score_bits(0.7)
    high = adjudication._score_bits(0.8)
    _events(
        [
            (key, int(adjudication.TOPIC_KIND), low, 0),
            (key, int(adjudication.TOPIC_KIND), high, 1),
        ]
    ).tofile(parent / "events.bin")
    changed = np.zeros(1, dtype=adjudication.DEDUP_DTYPE)
    changed[0]["key"] = key
    changed[0]["score_bits"] = low
    changed.tofile(parent / "relation.bin")

    with pytest.raises(
        verifier.VerificationError,
        match="relation score differs from adjudicated parent view",
    ):
        verifier.verify_payload_equivalence(
            parent_events_path=parent / "events.bin",
            relation_path=parent / "relation.bin",
            expected_counts={
                "event_rows": 2,
                "topic_raw_rows": 2,
                "unique_work_topic_edges": 1,
                "duplicate_edge_groups": 1,
                "duplicate_row_excess": 1,
                "score_conflict_edge_groups": 1,
            },
            selected_scores={key: high},
            chunk_rows=4,
        )


def test_unexpected_score_conflict_does_not_publish_relation(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    output = tmp_path / "output"
    parent.mkdir()
    output.mkdir()
    key = _key(1, 10001)
    event_array = _events(
        [
            (key, int(adjudication.TOPIC_KIND), adjudication._score_bits(0.7), 0),
            (key, int(adjudication.TOPIC_KIND), adjudication._score_bits(0.8), 1),
        ]
    )
    event_array.tofile(parent / "work_topic_relation.events.sorted.bin")
    config = _base_config()
    config["execution"]["aggregation_chunk_rows"] = 4
    config["blocked_parent"]["required_full_counts"].update(
        {
            "event_rows": 2,
            "topic_raw_rows": 2,
            "unique_work_topic_edges": 1,
            "duplicate_edge_groups": 1,
            "duplicate_row_excess": 1,
            "score_conflict_edge_groups": 1,
        }
    )

    with pytest.raises(
        adjudication.AdjudicationError, match="unexpected unresolved score conflict"
    ):
        adjudication.materialize_relation(
            config=config,
            parent_dir=parent,
            output_dir=output,
            sort_manifest={"event_rows": 2},
            adjudicated={},
        )

    assert not (output / "work_topic_edges.primary_adjudicated.sorted.bin").exists()
    assert not list(output.glob(".*.tmp-*"))


def test_policy_validation_rejects_unconditional_score_selection() -> None:
    config = copy.deepcopy(_base_config())
    config["adjudication_policy"]["forbid_unconditional_first_max_mean_or_min"] = False

    with pytest.raises(adjudication.AdjudicationError, match="unsafe score policy"):
        adjudication._validate_config(config)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_primary_score_is_rejected(value: float) -> None:
    with pytest.raises(adjudication.AdjudicationError, match="not finite"):
        adjudication._score_bits(value)


def test_negative_zero_uses_canonical_positive_zero_bits() -> None:
    assert adjudication._score_bits(-0.0) == adjudication._score_bits(0.0) == 0
