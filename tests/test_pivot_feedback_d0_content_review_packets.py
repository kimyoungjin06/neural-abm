from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pivot_feedback_d0_content_review_packets.py"
CONFIG = ROOT / "experiments/configs/pivot_feedback_d0_content_review_v1.yaml"
SOURCE_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724"
)
SOURCE_TEMPLATE = SOURCE_ROOT / "content_review_template.csv"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_content_review_packets_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_builds_two_blinded_packets_without_completing_review(tmp_path: Path) -> None:
    output = tmp_path / "packets"
    source_before = _sha256(SOURCE_TEMPLATE)

    result = MODULE.build_packets(
        config_path=CONFIG,
        source_root=SOURCE_ROOT,
        output=output,
    )

    assert result["status"] == MODULE.STATUS
    assert result["reviewer_packet_rows_each"] == 384
    assert result["packet_orders_differ"] is True
    assert result["human_review_complete"] is False
    assert result["d0_complete"] is False
    assert _sha256(SOURCE_TEMPLATE) == source_before
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())

    packet_1 = _read_csv(output / "reviewer_1_packet.csv")
    packet_2 = _read_csv(output / "reviewer_2_packet.csv")
    assert len(packet_1) == len(packet_2) == 384
    assert list(packet_1[0]) == MODULE.PACKET_COLUMNS
    assert [row["title"] for row in packet_1] != [row["title"] for row in packet_2]
    assert all(not row["reviewer_label"] for row in packet_1 + packet_2)
    assert all(not row["reviewer_notes"] for row in packet_1 + packet_2)
    assert "openalex_id" not in packet_1[0]
    assert "episode_index" not in packet_1[0]
    assert "window_code" not in packet_1[0]
    assert "primary_topic_score" not in packet_1[0]
    assert "source_name" not in packet_1[0]

    key = _read_csv(output / "adjudication_key.csv")
    assert len(key) == 384
    assert all(row["reviewer_1_item_id"] for row in key)
    assert all(row["reviewer_2_item_id"] for row in key)
    assert all(
        row["reviewer_1_item_id"] != row["reviewer_2_item_id"] for row in key
    )

    receipt = json.loads(
        (output / "content_review_packets.outputs.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == MODULE.STATUS
    assert receipt["artifact_count"] == 5
    assert receipt["reviewer_labels_filled"] == 0
    assert receipt["human_review_complete"] is False
    assert receipt["automatic_title_classification_performed"] is False
    assert receipt["outcomes_read"] is False
    assert receipt["d0_complete"] is False
    assert receipt["g4_open"] is False


def test_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "packets"
    output.mkdir()

    with pytest.raises(MODULE.ReviewPacketError, match="already exists"):
        MODULE.build_packets(
            config_path=CONFIG,
            source_root=SOURCE_ROOT,
            output=output,
        )


def test_rejects_automatic_review_completion_contract(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["forbidden_inputs_and_outputs"]["automatic_review_completion"] = True
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(MODULE.ReviewPacketError, match="forbidden contract enabled"):
        MODULE.build_packets(
            config_path=config_path,
            source_root=SOURCE_ROOT,
            output=tmp_path / "packets",
        )


def test_cleanup_guard_accepts_only_matching_sibling(tmp_path: Path) -> None:
    output = (tmp_path / "packets").resolve()
    wrong = (tmp_path / ".other.tmp-123").resolve()
    wrong.mkdir()
    with pytest.raises(MODULE.ReviewPacketError, match="prefix"):
        MODULE._cleanup_temporary_directory(wrong, output=output)
    valid = (tmp_path / ".packets.tmp-123").resolve()
    valid.mkdir()
    MODULE._cleanup_temporary_directory(valid, output=output)
    assert not valid.exists()
    assert wrong.exists()


def test_config_preserves_human_and_outcome_boundaries() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    review = config["review_contract"]
    assert review["reviewers_must_be_distinct_humans"] is True
    assert review["artificial_agent_may_substitute_for_human_reviewer"] is False
    assert review["independent_completion_before_comparison"] is True
    assert config["forbidden_inputs_and_outputs"]["entry_rows_read"] is False
    assert config["forbidden_inputs_and_outputs"]["persistence_rows_read"] is False
    assert config["forbidden_inputs_and_outputs"]["g4_rows_read"] is False
    output = config["output_contract"]
    assert output["human_review_complete_after_packet_generation"] is False
    assert output["d0_complete_after_packet_generation"] is False
    assert output["discovery_outcome_opening_authorized"] is False
    assert output["model_fitting_authorized"] is False
    assert output["g4_open"] is False
