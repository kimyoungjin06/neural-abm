#!/usr/bin/env python3
"""Independently verify the exact globally certified episode reselection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_certified_episode_reselection_v1.yaml"
)
DEFAULT_ROOT = Path(
    "/ssd/pivot-feedback-d0-certified-episode-reselection-v1-20260730"
)
DEFAULT_VERIFICATION = Path(
    "/ssd/pivot-feedback-d0-certified-episode-reselection-v1-20260730.verification-v1.json"
)
STATUS = "PASS_INDEPENDENT_EXACT_FOUR_CERTIFIED_EPISODE_RESELECTION"


class CertifiedEpisodeReselectionVerificationError(RuntimeError):
    """Fail-closed independent episode-reselection verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CertifiedEpisodeReselectionVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _compatible(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        int(first["subfield_id"]) != int(second["subfield_id"])
        and abs(int(first["entry_year"]) - int(second["entry_year"])) >= 9
    )


def _frontier_pair(rows: Sequence[Mapping[str, Any]], field_id: int) -> list[int]:
    field_rows = sorted(
        (row for row in rows if int(row["field_id"]) == field_id),
        key=lambda row: int(row["field_candidate_rank"]),
    )
    first: Mapping[str, Any] | None = None
    for row in field_rows:
        state = row["content_state"]
        if state == "CONTENT_FAIL":
            continue
        require(state == "CONTENT_PASS", f"blocked first: field {field_id}")
        first = row
        break
    require(first is not None, f"missing first: field {field_id}")
    for row in field_rows:
        if int(row["field_candidate_rank"]) <= int(first["field_candidate_rank"]):
            continue
        if not _compatible(first, row):
            continue
        state = row["content_state"]
        if state == "CONTENT_FAIL":
            continue
        require(state == "CONTENT_PASS", f"blocked second: field {field_id}")
        return [
            int(first["field_candidate_rank"]),
            int(row["field_candidate_rank"]),
        ]
    raise CertifiedEpisodeReselectionVerificationError(
        f"missing second: field {field_id}"
    )


def verify(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_certified_episode_reselection_config.v1",
        "config schema",
    )
    output = _mapping(config.get("output_contract"), "output")
    root = output_root.resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "root parent")
    require(root.name == output["root_basename"], "root basename")
    require(root.stat().st_mode & 0o077 == 0, "root mode")
    receipt_path = root / str(output["receipt_basename"])
    receipt = _json(receipt_path)
    require(
        receipt.get("status")
        == "PASS_EXACT_FOUR_GLOBAL_CERTIFICATE_BOUND_EPISODE_RESELECTION",
        "receipt status",
    )
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    artifacts = _mapping(receipt.get("artifacts"), "artifacts")
    for label, artifact_value in artifacts.items():
        artifact = _mapping(artifact_value, label)
        path = root / str(artifact["basename"])
        require(path.is_file() and not path.is_symlink(), f"artifact: {label}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {label}")
        require(path.stat().st_mode & 0o077 == 0, f"artifact mode: {label}")

    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    candidates_binding = _mapping(
        frozen.get("measurement_abi_candidates"), "candidates"
    )
    candidates_root = (
        Path("/ssd") / str(candidates_binding["root_basename"])
    ).resolve(strict=True)
    candidate_path = candidates_root / str(candidates_binding["candidate_basename"])
    require(_sha256(candidate_path) == candidates_binding["candidate_sha256"], "candidate SHA")
    candidate_table = pq.read_table(candidate_path)
    require(candidate_table.num_rows == 70277, "candidate rows")
    candidates = {
        (int(row["field_id"]), int(row["field_candidate_rank"])): row
        for row in candidate_table.to_pylist()
    }
    require(len(candidates) == 70277, "candidate uniqueness")

    certificate_binding = _mapping(
        frozen.get("selection_certificate"), "certificate"
    )
    certificate_root = (
        Path("/ssd") / str(certificate_binding["root_basename"])
    ).resolve(strict=True)
    frontier_path = certificate_root / str(certificate_binding["frontier_basename"])
    certificate_path = certificate_root / str(
        certificate_binding["certificate_basename"]
    )
    require(_sha256(frontier_path) == certificate_binding["frontier_sha256"], "frontier SHA")
    require(_sha256(certificate_path) == certificate_binding["certificate_sha256"], "certificate SHA")
    frontier_table = pq.read_table(frontier_path)
    frontier_rows = frontier_table.to_pylist()
    require(len(frontier_rows) == 51, "frontier rows")
    independent_pairs = {
        "computer_science": _frontier_pair(frontier_rows, 17),
        "medicine": _frontier_pair(frontier_rows, 27),
    }
    require(
        independent_pairs
        == {"computer_science": [1, 45], "medicine": [5, 6]},
        "independent pairs",
    )
    certificate = _json(certificate_path)
    require(
        certificate.get("computer_science_selected_candidate_ranks")
        == independent_pairs["computer_science"],
        "CS certificate",
    )
    require(
        certificate.get("medicine_selected_candidate_ranks")
        == independent_pairs["medicine"],
        "Medicine certificate",
    )
    require(certificate.get("global_selection_certificate_issued") is True, "global certificate")

    selection_path = root / str(output["selection_basename"])
    selection_table = pq.read_table(selection_path)
    require(selection_table.num_rows == 4, "selection rows")
    require(
        (selection_table.schema.metadata or {}).get(b"schema_version", b"").decode()
        == output["selection_schema_version"],
        "selection schema",
    )
    selection_rows = sorted(
        selection_table.to_pylist(), key=lambda row: row["certified_episode_index"]
    )
    expected_config_rows = config["selection_contract"]["exact_rows"]
    require(
        [(row["field_id"], row["field_candidate_rank"]) for row in selection_rows]
        == [(17, 1), (17, 45), (27, 5), (27, 6)],
        "selection identities",
    )
    added_fields = {
        "certified_episode_index",
        "field_episode_order",
        "content_state",
        "selection_certificate_version",
        "prior_measurement_abi_selected",
        "selection_change",
    }
    for row, expected_value in zip(selection_rows, expected_config_rows, strict=True):
        expected = _mapping(expected_value, "expected row")
        key = (int(expected["field_id"]), int(expected["field_candidate_rank"]))
        source = candidates[key]
        for source_key, source_value in source.items():
            if source_key in {"episode_selected", "episode_order"}:
                continue
            require(row[source_key] == source_value, f"source replay: {key} {source_key}")
        require(row["certified_episode_index"] == expected["certified_episode_index"], f"episode index: {key}")
        require(row["field_episode_order"] == expected["field_episode_order"], f"field order: {key}")
        require(row["content_state"] == "CONTENT_PASS", f"content state: {key}")
        require(row["selection_certificate_version"] == "v5", f"certificate version: {key}")
        require(row["prior_measurement_abi_selected"] == expected["prior_measurement_abi_selected"], f"prior selection: {key}")
        require(row["selection_change"] == expected["selection_change"], f"selection change: {key}")
        require(set(row) == (set(source) - {"episode_selected", "episode_order"}) | added_fields, f"row columns: {key}")
    for field_id in (17, 27):
        pair = [row for row in selection_rows if row["field_id"] == field_id]
        require(len(pair) == 2 and _compatible(pair[0], pair[1]), f"compatibility: {field_id}")
    require(sum(row["prior_measurement_abi_selected"] for row in selection_rows) == 3, "retained count")
    require(sum(not row["prior_measurement_abi_selected"] for row in selection_rows) == 1, "new count")
    medicine_rank5 = next(
        row
        for row in selection_rows
        if row["field_id"] == 27 and row["field_candidate_rank"] == 5
    )
    require(medicine_rank5["topic_id"] == 10151, "Medicine rank5 topic")
    require(medicine_rank5["entry_year"] == 1993, "Medicine rank5 year")
    require(medicine_rank5["subfield_id"] == 2721, "Medicine rank5 subfield")
    require(medicine_rank5["selection_change"] == "REPLACED_PRIOR_FIELD_FIRST", "Medicine replacement")

    decision = _json(root / str(output["decision_basename"]))
    require(decision.get("status") == receipt["status"], "decision status")
    require(decision.get("selected_field_candidate_ranks") == independent_pairs, "decision pairs")
    require(decision.get("source_work_rescan_executed") is False, "decision work rescan")
    require(decision.get("author_opportunity_panel_rebuild_executed") is False, "decision panel rebuild")
    require(receipt.get("selected_field_candidate_ranks") == independent_pairs, "receipt pairs")
    require(receipt.get("selected_episode_rows") == 4, "receipt rows")
    require(receipt.get("retained_episode_rows") == 3, "receipt retained")
    require(receipt.get("newly_selected_episode_rows") == 1, "receipt newly selected")
    require(receipt.get("deselected_prior_episode_rows") == 1, "receipt deselected")
    require(receipt.get("global_selection_certificate_bound") is True, "receipt certificate")
    require(receipt.get("deterministic_reselection_executed") is True, "receipt execution")
    for key in (
        "source_work_rescan_executed",
        "author_opportunity_panel_rebuild_executed",
        "production_pivot_classification_executed",
        "outcomes_read",
        "g4_open",
    ):
        require(receipt.get(key) is False, f"closed boundary: {key}")
    return {
        "schema_version": "pivot_feedback_d0_certified_episode_reselection_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(config_path),
        "receipt_sha256": _sha256(receipt_path),
        "selection_sha256": _sha256(selection_path),
        "decision_sha256": _sha256(root / str(output["decision_basename"])),
        "source_candidate_rows_replayed": len(candidates),
        "frontier_rows_replayed": len(frontier_rows),
        "selected_episode_rows": 4,
        "retained_episode_rows": 3,
        "newly_selected_episode_rows": 1,
        "deselected_prior_episode_rows": 1,
        "selected_field_candidate_ranks": independent_pairs,
        "deterministic_reselection_executed": True,
        "source_work_rescan_executed": False,
        "author_opportunity_panel_rebuild_executed": False,
        "production_pivot_classification_executed": False,
        "outcomes_read": False,
        "g4_open": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--verification-output", type=Path, default=DEFAULT_VERIFICATION)
    args = parser.parse_args(argv)
    try:
        result = verify(args.config, args.output_root)
        output = args.verification_output.absolute()
        require(output.parent == Path("/ssd"), "verification parent")
        require(not output.exists(), "verification exists")
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.chmod(0o600)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        CertifiedEpisodeReselectionVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_CERTIFIED_EPISODE_RESELECTION_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
