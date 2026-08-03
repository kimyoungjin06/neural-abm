#!/usr/bin/env python3
"""Independently verify blinded D0 human title-review packets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_content_review_v1.yaml"
)
DEFAULT_SOURCE_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-review-packets-v1-20260729"
)
PACKET_STATUS = (
    "PASS_D0_BLINDED_TWO_HUMAN_REVIEWER_PACKETS_PREPARED_REVIEW_NOT_RUN"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0_BLINDED_TWO_HUMAN_REVIEWER_PACKETS_REVIEW_NOT_RUN"
)
PACKET_COLUMNS = [
    "review_item_id",
    "target_topic_name",
    "title",
    "language",
    "work_type",
    "reviewer_label",
    "reviewer_notes",
]
OUTPUT_FILES = {
    "REVIEW_INSTRUCTIONS.md",
    "reviewer_1_packet.csv",
    "reviewer_2_packet.csv",
    "adjudication_key.csv",
    "review_packet_manifest.json",
    "content_review_packets.outputs.json",
}


class ReviewPacketVerificationError(RuntimeError):
    """Fail-closed review-packet verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewPacketVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing {dotted_path}",
        )
        current = current[part]
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewPacketVerificationError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ReviewPacketVerificationError(f"invalid config: {error}") from error
    return _mapping(value, "config")


def _safe_repo_file(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file() and not path.is_symlink(), f"invalid CSV: {path.name}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"missing CSV header: {path.name}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _opaque_item_id(seed: str, reviewer: int, openalex_id: str) -> str:
    payload = f"{seed}\0reviewer-{reviewer}\0{openalex_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _order_key(seed: str, reviewer: int, openalex_id: str) -> tuple[bytes, str]:
    payload = f"{seed}\0reviewer-{reviewer}\0{openalex_id}".encode()
    return hashlib.sha256(payload).digest(), openalex_id


def _artifact_tree_sha256(artifacts: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for artifact in sorted(artifacts, key=lambda item: str(item["relative_path"])):
        digest.update(str(artifact["relative_path"]).encode())
        digest.update(b"\0")
        digest.update(str(artifact["sha256"]).encode())
        digest.update(b"\0")
        digest.update(str(artifact["bytes"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_gate_parent(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    binding = _mapping(config.get("gate_parent"), "gate parent")
    path = _safe_repo_file(repo_root, binding.get("relative_path"), "gate parent path")
    digest = _sha256(path)
    require(digest == binding.get("sha256"), "gate parent SHA mismatch")
    parent = _load_yaml(path)
    require(
        parent.get("schema_version") == binding.get("required_schema_version"),
        "gate parent schema mismatch",
    )
    require(
        parent.get("status") == binding.get("required_status"),
        "gate parent status mismatch",
    )
    for dotted_path, expected in _mapping(
        binding.get("required_fields", {}),
        "gate parent required fields",
    ).items():
        observed = _nested(parent, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"gate parent field mismatch: {dotted_path}",
        )
    return {
        "relative_path": binding["relative_path"],
        "sha256": digest,
        "status": parent["status"],
    }


def _validate_source(
    config: Mapping[str, Any],
    *,
    source_root: Path,
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    source = _mapping(config.get("source_audit"), "source audit")
    root = source_root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "source root invalid")
    require(root.name == source.get("output_root_basename"), "source basename differs")
    receipt_path = root / str(source["receipt_basename"])
    template_path = root / str(source["template_basename"])
    require(_sha256(receipt_path) == source.get("receipt_sha256"), "source receipt SHA mismatch")
    require(
        _sha256(template_path) == source.get("template_sha256"),
        "source template SHA mismatch",
    )
    receipt = _load_json(receipt_path)
    require(
        receipt.get("schema_version") == source.get("required_receipt_schema_version"),
        "source receipt schema mismatch",
    )
    require(
        receipt.get("status") == source.get("required_receipt_status"),
        "source receipt status mismatch",
    )
    columns, rows = _read_csv(template_path)
    require(len(rows) == source.get("expected_rows"), "source row count differs")
    for row in rows:
        for column in source.get("required_blank_columns", []):
            require(not str(row[column]).strip(), f"source review not blank: {column}")
    require(
        len({row["openalex_id"] for row in rows}) == len(rows),
        "source IDs not unique",
    )
    return columns, rows, {
        "root_basename": root.name,
        "receipt_sha256": _sha256(receipt_path),
        "template_sha256": _sha256(template_path),
        "template_rows": len(rows),
    }


def _expected_packet(
    source_rows: Sequence[Mapping[str, str]],
    *,
    reviewer: int,
    item_seed: str,
    order_seed: str,
) -> list[dict[str, str]]:
    ordered = sorted(
        source_rows,
        key=lambda row: _order_key(order_seed, reviewer, row["openalex_id"]),
    )
    return [
        {
            "review_item_id": _opaque_item_id(
                item_seed,
                reviewer,
                row["openalex_id"],
            ),
            "target_topic_name": row["observed_primary_topic_name"],
            "title": row["title"],
            "language": row["language"],
            "work_type": row["observed_work_type"],
            "reviewer_label": "",
            "reviewer_notes": "",
        }
        for row in ordered
    ]


def validate(
    *,
    config_path: Path,
    source_root: Path,
    output_root: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    config = _load_yaml(config_path)
    require(
        config.get("schema_version") == "pivot_feedback_d0_content_review_config.v1",
        "config schema mismatch",
    )
    require(
        config.get("status") == "FROZEN_PACKET_GENERATION_CONTRACT_HUMAN_REVIEW_NOT_RUN",
        "config status mismatch",
    )
    parent = _validate_gate_parent(config, repo_root=repo_root)
    source_columns, source_rows, source = _validate_source(
        config,
        source_root=source_root,
    )

    require(output_root.is_dir() and not output_root.is_symlink(), "output root invalid")
    require(output_root.stat().st_mode & 0o777 == 0o700, "output mode differs")
    observed_files = {path.name for path in output_root.iterdir()}
    require(observed_files == OUTPUT_FILES, "output file set differs")
    for path in output_root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid output: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"file mode differs: {path.name}")

    receipt_path = output_root / "content_review_packets.outputs.json"
    manifest_path = output_root / "review_packet_manifest.json"
    receipt = _load_json(receipt_path)
    manifest = _load_json(manifest_path)
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(
        receipt.get("schema_version") == output_contract.get("receipt_schema_version"),
        "receipt schema mismatch",
    )
    require(receipt.get("status") == PACKET_STATUS, "receipt status mismatch")
    require(
        manifest.get("schema_version") == output_contract.get("manifest_schema_version"),
        "manifest schema mismatch",
    )
    require(manifest.get("status") == PACKET_STATUS, "manifest status mismatch")
    require(
        manifest["producer"]["config_sha256"] == _sha256(config_path),
        "manifest config SHA mismatch",
    )
    require(
        manifest["source"]["template_sha256"] == source["template_sha256"],
        "manifest source SHA mismatch",
    )

    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, list), "receipt artifacts missing")
    require(receipt.get("artifact_count") == len(artifacts) == 5, "artifact count differs")
    for artifact in artifacts:
        artifact = _mapping(artifact, "artifact")
        path = output_root / str(artifact["relative_path"])
        require(path.is_file(), f"artifact missing: {path.name}")
        require(path.stat().st_size == artifact.get("bytes"), f"artifact size differs: {path.name}")
        require(_sha256(path) == artifact.get("sha256"), f"artifact SHA differs: {path.name}")
    require(
        receipt.get("artifact_tree_sha256") == _artifact_tree_sha256(artifacts),
        "artifact tree SHA mismatch",
    )

    blinding = _mapping(config.get("blinding"), "blinding")
    item_seed = str(blinding["item_id_seed"])
    order_seed = str(blinding["reviewer_order_seed"])
    packets: dict[int, list[dict[str, str]]] = {}
    for reviewer in (1, 2):
        columns, rows = _read_csv(
            output_root / f"reviewer_{reviewer}_packet.csv"
        )
        require(columns == PACKET_COLUMNS, f"reviewer {reviewer} columns differ")
        require(len(rows) == len(source_rows) == 384, f"reviewer {reviewer} rows differ")
        expected = _expected_packet(
            source_rows,
            reviewer=reviewer,
            item_seed=item_seed,
            order_seed=order_seed,
        )
        require(rows == expected, f"reviewer {reviewer} packet content differs")
        require(
            all(not row["reviewer_label"] and not row["reviewer_notes"] for row in rows),
            f"reviewer {reviewer} packet contains decisions",
        )
        packets[reviewer] = rows
    require(
        [row["title"] for row in packets[1]]
        != [row["title"] for row in packets[2]],
        "packet orders are identical",
    )

    key_columns, key_rows = _read_csv(output_root / "adjudication_key.csv")
    expected_key_columns = [
        "source_row_number",
        "reviewer_1_item_id",
        "reviewer_2_item_id",
        *[
            column
            for column in source_columns
            if column
            not in {
                "reviewer_1_label",
                "reviewer_2_label",
                "adjudicated_label",
                "review_notes",
            }
        ],
    ]
    require(key_columns == expected_key_columns, "adjudication columns differ")
    require(len(key_rows) == len(source_rows), "adjudication rows differ")
    by_openalex = {row["openalex_id"]: row for row in key_rows}
    require(len(by_openalex) == len(source_rows), "adjudication IDs not unique")
    for index, row in enumerate(source_rows, start=1):
        observed = by_openalex[row["openalex_id"]]
        require(observed["source_row_number"] == str(index), "source row number differs")
        require(
            observed["reviewer_1_item_id"]
            == _opaque_item_id(item_seed, 1, row["openalex_id"]),
            "reviewer 1 adjudication ID differs",
        )
        require(
            observed["reviewer_2_item_id"]
            == _opaque_item_id(item_seed, 2, row["openalex_id"]),
            "reviewer 2 adjudication ID differs",
        )

    for key in (
        "human_review_complete",
        "automatic_title_classification_performed",
        "outcomes_read",
        "d0_complete",
        "leaveout_network_bridge_authorized",
        "discovery_outcome_opening_authorized",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
        "gpu_used",
    ):
        require(receipt.get(key) is False, f"unsafe receipt flag: {key}")
    require(receipt.get("scientific_effect") == "NOT_ESTABLISHED", "effect overstated")
    return {
        "schema_version": "pivot_feedback_d0_content_review_packet_verification.v1",
        "status": VERIFY_STATUS,
        "verifier_relative_path": Path(__file__).resolve().relative_to(repo_root).as_posix(),
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(config_path),
        "gate_parent": parent,
        "source": source,
        "output_root_basename": output_root.name,
        "output_receipt_sha256": _sha256(receipt_path),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "reviewer_packet_rows_each": 384,
        "packet_orders_differ": True,
        "packet_content_exact_match": True,
        "adjudication_key_exact_match": True,
        "reviewer_labels_filled": 0,
        "human_review_complete": False,
        "d0_complete": False,
        "outcomes_read": False,
        "g0_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
        "producer_module_imported": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(
            config_path=args.config,
            source_root=args.source_root,
            output_root=args.output_root,
        )
        if args.report is not None:
            report = args.report.resolve()
            require(report.is_absolute(), "report path must be absolute")
            require(report.parent.is_dir(), "report parent missing")
            require(not report.exists(), "report already exists")
            with report.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
                handle.write("\n")
            report.chmod(0o600)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (
        ReviewPacketVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0_CONTENT_REVIEW_PACKET_VERIFICATION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
