from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_pivot_feedback_d0_content_indexing.py"
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_content_indexing_audit_v1.yaml"
)

spec = importlib.util.spec_from_file_location(
    "test_pivot_feedback_d0_content_indexing_module", SCRIPT_PATH
)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _synthetic_contract(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = tmp_path / "parent"
    parent.mkdir()
    episodes = config["parent_reselection"]["expected_episodes"]
    selected_rows = []
    source_rows = []
    work_id = 1000
    for episode in episodes:
        for window_code, year_offset in ((0, -7), (1, -5), (2, -2)):
            work_id += 1
            year = int(episode["entry_year"]) + year_offset
            selected_rows.append(
                {
                    **episode,
                    "work_numeric_id": work_id,
                    "publication_year": year,
                    "work_type_code": 1,
                    "declared_authors_count": 2,
                    "is_retracted_null": False,
                    "window_code": window_code,
                }
            )
            source_rows.append(
                {
                    "id": f"https://openalex.org/W{work_id}",
                    "title": f"Synthetic topic {episode['topic_id']} work {work_id}",
                    "display_name": f"Synthetic topic {episode['topic_id']} work",
                    "publication_year": float(year),
                    "language": "en",
                    "type": "article",
                    "authors_count": 2,
                    "is_paratext": False,
                    "is_retracted": False,
                    "primary_topic__id": (
                        f"https://openalex.org/T{episode['topic_id']}"
                    ),
                    "primary_topic__display_name": (
                        f"Synthetic topic {episode['topic_id']}"
                    ),
                    "primary_topic__score": 0.8,
                    "primary_topic__field__id": (
                        f"https://openalex.org/fields/{episode['field_id']}"
                    ),
                    "primary_topic__subfield__id": (
                        f"https://openalex.org/subfields/{episode['field_id']}01"
                    ),
                    "primary_location__source__id": "https://openalex.org/S1",
                    "primary_location__source__display_name": "Synthetic Journal",
                    "created_date": "2020-01-01",
                    "updated_date": "2026-01-01",
                }
            )
    selected_path = parent / "selected_episode_predecision_works.parquet"
    pq.write_table(pa.Table.from_pylist(selected_rows), selected_path)
    report = {
        "status": config["parent_reselection"]["required_status"],
        "scope": {"outcomes_read": False},
        "abi": {
            "sha256": config["parent_reselection"]["abi_sha256"],
            "included_work_type_codes": [1],
            "included_work_type_names": ["article"],
        },
        "episode_reselection": {
            "selected_episodes": [
                {
                    "field_id": row["field_id"],
                    "topic_id": row["topic_id"],
                    "entry_year": row["entry_year"],
                    "baseline_link_rate": 0.8,
                    "recent_link_rate": 0.82,
                }
                for row in episodes
            ]
        },
    }
    report_path = parent / "measurement_abi_reselection.json"
    _write_json(report_path, report)
    receipt = {
        "status": config["parent_reselection"]["required_status"],
        "outcomes_read": False,
        "d0_complete": False,
        "g0_authorized": False,
        "g4_open": False,
    }
    receipt_path = parent / "measurement_abi_reselection.outputs.json"
    _write_json(receipt_path, receipt)
    parent_config = config["parent_reselection"]
    parent_config.update(
        {
            "receipt_sha256": _sha256(receipt_path),
            "report_sha256": _sha256(report_path),
            "selected_work_sha256": _sha256(selected_path),
            "selected_work_rows": len(selected_rows),
        }
    )
    config["sampling"]["per_episode_window"] = 1
    source_root = tmp_path / "synthetic_openalex_root"
    source_table = source_root / "works"
    source_table.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(source_rows), source_table / "part-0.parquet")
    source = config["current_snapshot_source"]
    source.update(
        {
            "root_id": source_root.name,
            "main_table_directory": source_table.name,
            "expected_main_file_count": 1,
            "expected_main_footer_rows": len(source_rows),
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, parent, source_root


def test_synthetic_audit_is_outcome_blind_and_human_review_gated(
    tmp_path: Path,
) -> None:
    config, parent, source = _synthetic_contract(tmp_path)
    output = tmp_path / "output"

    report = audit.run_audit(
        config_path=config,
        reselection_root=parent,
        openalex_root=source,
        output=output,
    )

    assert report["status"] == (
        "PASS_D0_CONTENT_INDEXING_EXTRACTION_HUMAN_REVIEW_REQUIRED"
    )
    assert report["sampling"]["sample_rows_found"] == 12
    assert report["blocking_issues"] == []
    assert report["human_review"]["complete"] is False
    assert report["authorization"] == {
        "d0_complete": False,
        "g0_authorized": False,
        "leaveout_network_bridge_authorized": False,
        "discovery_outcome_opening_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }
    receipt = json.loads(
        (output / "content_indexing_audit.outputs.json").read_text(encoding="utf-8")
    )
    assert receipt["artifact_count"] == 5
    assert receipt["sample_rows"] == 12
    assert receipt["outcomes_read"] is False
    assert receipt["human_review_complete"] is False
    assert (output.stat().st_mode & 0o777) == 0o700
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in output.iterdir())


def test_cleanup_guard_rejects_cwd_and_accepts_only_matching_sibling(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "audit-output").resolve()
    with pytest.raises(audit.AuditError, match="absolute"):
        audit._cleanup_temporary_directory(Path(), output=output)
    wrong = (tmp_path / ".other.tmp-123").resolve()
    wrong.mkdir()
    with pytest.raises(audit.AuditError, match="prefix"):
        audit._cleanup_temporary_directory(wrong, output=output)
    valid = (tmp_path / ".audit-output.tmp-123").resolve()
    valid.mkdir()
    audit._cleanup_temporary_directory(valid, output=output)
    assert not valid.exists()
    assert wrong.exists()


def test_config_keeps_outcomes_and_downstream_authorization_closed() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["human_review"]["required"] is True
    assert config["forbidden_inputs_and_outputs"]["entry_rows_read"] is False
    assert config["forbidden_inputs_and_outputs"]["persistence_rows_read"] is False
    assert config["forbidden_inputs_and_outputs"]["g4_rows_read"] is False
    assert config["output_contract"]["d0_complete_after_automated_pass"] is False
    assert (
        config["output_contract"][
            "leaveout_network_bridge_authorized_after_automated_pass"
        ]
        is False
    )
