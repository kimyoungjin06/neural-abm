from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_pivot_feedback_d0_api_review_v2.py"
CONFIG = ROOT / "experiments/configs/pivot_feedback_d0_api_assisted_review_v2.yaml"
SOURCE_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724"
)
SOURCE_TEMPLATE = SOURCE_ROOT / "content_review_template.csv"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_api_review_v2_test_module",
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


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_prepares_dual_model_requests_and_blinded_calibration(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prepared"
    source_before = _sha256(SOURCE_TEMPLATE)

    result = MODULE.prepare(
        config_path=CONFIG,
        source_root=SOURCE_ROOT,
        output=output,
    )

    assert result["status"] == MODULE.STATUS
    assert result["request_rows_per_judge"] == 384
    assert result["total_request_rows"] == 768
    assert result["base_calibration_rows"] == 96
    assert result["base_calibration_strata"] == 12
    assert result["api_submitted"] is False
    assert result["human_calibration_complete"] is False
    assert result["d0_complete"] is False
    assert result["outcomes_read"] is False
    assert _sha256(SOURCE_TEMPLATE) == source_before
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())

    judge_a = _jsonl(output / "judge_a_requests.jsonl")
    judge_b = _jsonl(output / "judge_b_requests.jsonl")
    assert len(judge_a) == len(judge_b) == 384
    assert {item["body"]["model"] for item in judge_a} == {
        "gpt-5-mini-2025-08-07"
    }
    assert {item["body"]["model"] for item in judge_b} == {
        "gpt-4.1-mini-2025-04-14"
    }
    assert all(item["url"] == "/v1/responses" for item in judge_a + judge_b)
    assert all(item["body"]["store"] is False for item in judge_a + judge_b)
    assert all("temperature" not in item["body"] for item in judge_a + judge_b)

    forbidden = {
        "openalex_id",
        "episode_index",
        "window_code",
        "entry_outcome",
        "persistence_outcome",
    }
    for request in judge_a + judge_b:
        visible = json.loads(request["body"]["input"][0]["content"][0]["text"])
        assert set(visible) == set(MODULE.VISIBLE_SOURCE_MAP)
        assert not forbidden.intersection(visible)

    reviewer_1 = _csv_rows(output / "human_calibration_reviewer_1.csv")
    reviewer_2 = _csv_rows(output / "human_calibration_reviewer_2.csv")
    assert len(reviewer_1) == len(reviewer_2) == 96
    assert list(reviewer_1[0]) == MODULE.PACKET_COLUMNS
    assert [row["title"] for row in reviewer_1] != [
        row["title"] for row in reviewer_2
    ]
    assert all(not row["reviewer_label"] for row in reviewer_1 + reviewer_2)
    assert all(not row["reviewer_notes"] for row in reviewer_1 + reviewer_2)
    assert "episode_index" not in reviewer_1[0]
    assert "openalex_id" not in reviewer_1[0]


def test_preparation_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    output.mkdir()

    with pytest.raises(MODULE.ApiReviewPreparationError, match="output already exists"):
        MODULE.prepare(
            config_path=CONFIG,
            source_root=SOURCE_ROOT,
            output=output,
        )


def test_packet_ids_are_keyed_by_source_id_when_titles_repeat() -> None:
    rows = [
        {
            "openalex_id": "W1",
            "observed_primary_topic_name": "topic",
            "title": "Repeated title",
            "language": "en",
            "observed_work_type": "article",
        },
        {
            "openalex_id": "W2",
            "observed_primary_topic_name": "topic",
            "title": "Repeated title",
            "language": "en",
            "observed_work_type": "article",
        },
    ]

    packet_rows, item_ids = MODULE._packet_rows(
        rows,
        reviewer=1,
        seed="test-seed",
    )

    assert len(packet_rows) == 2
    assert set(item_ids) == {"W1", "W2"}
    assert len(set(item_ids.values())) == 2


def test_temporary_cleanup_only_accepts_private_output_sibling(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "prepared").resolve()
    private_temporary = tmp_path / ".prepared.tmp-safe"
    private_temporary.mkdir()
    (private_temporary / "partial.txt").write_text("partial", encoding="utf-8")

    MODULE._cleanup_temporary_directory(
        private_temporary.resolve(),
        output=output,
    )

    assert not private_temporary.exists()
    MODULE._cleanup_temporary_directory(None, output=output)


@pytest.mark.parametrize(
    "unsafe_kind",
    ["relative", "output", "wrong_parent", "wrong_prefix"],
)
def test_temporary_cleanup_rejects_unsafe_targets(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    output = (tmp_path / "prepared").resolve()
    if unsafe_kind == "relative":
        unsafe = Path(".")
    elif unsafe_kind == "output":
        unsafe = output
    elif unsafe_kind == "wrong_parent":
        unsafe = (tmp_path.parent / ".prepared.tmp-unsafe").resolve()
    else:
        unsafe = (tmp_path / "unrelated").resolve()

    with pytest.raises(MODULE.ApiReviewPreparationError):
        MODULE._cleanup_temporary_directory(unsafe, output=output)
