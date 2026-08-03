#!/usr/bin/env python3
"""Build two separately ordered, outcome-blind human title-review packets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_content_review_v1.yaml"
)
DEFAULT_SOURCE_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724"
)
STATUS = "PASS_D0_BLINDED_TWO_HUMAN_REVIEWER_PACKETS_PREPARED_REVIEW_NOT_RUN"
PACKET_COLUMNS = [
    "review_item_id",
    "target_topic_name",
    "title",
    "language",
    "work_type",
    "reviewer_label",
    "reviewer_notes",
]
SOURCE_REVIEW_COLUMNS = [
    "reviewer_1_label",
    "reviewer_2_label",
    "adjudicated_label",
    "review_notes",
]
ADJUDICATION_PREFIX_COLUMNS = [
    "source_row_number",
    "reviewer_1_item_id",
    "reviewer_2_item_id",
]


class ReviewPacketError(RuntimeError):
    """Fail-closed content-review packet generation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewPacketError(message)


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ReviewPacketError(f"invalid config: {error}") from error
    return _mapping(value, "config")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewPacketError(f"invalid JSON {path.name}: {error}") from error
    return _mapping(value, path.name)


def _safe_repo_file(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


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


def _read_source_template(
    config: Mapping[str, Any],
    *,
    source_root: Path,
) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
    source = _mapping(config.get("source_audit"), "source audit")
    root = source_root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "source audit root invalid")
    require(root.name == source.get("output_root_basename"), "source root differs")
    receipt_path = root / str(source["receipt_basename"])
    template_path = root / str(source["template_basename"])
    require(receipt_path.is_file() and not receipt_path.is_symlink(), "receipt invalid")
    require(template_path.is_file() and not template_path.is_symlink(), "template invalid")
    require(_sha256(receipt_path) == source.get("receipt_sha256"), "receipt SHA mismatch")
    require(_sha256(template_path) == source.get("template_sha256"), "template SHA mismatch")
    receipt = _load_json(receipt_path)
    require(
        receipt.get("schema_version") == source.get("required_receipt_schema_version"),
        "receipt schema mismatch",
    )
    require(
        receipt.get("status") == source.get("required_receipt_status"),
        "receipt status mismatch",
    )
    require(receipt.get("sample_rows") == source.get("expected_rows"), "sample rows differ")
    require(receipt.get("human_review_complete") is False, "source review already complete")
    for key in (
        "d0_complete",
        "g0_authorized",
        "leaveout_network_bridge_authorized",
        "model_fitting_authorized",
        "g4_open",
        "outcomes_read",
        "gpu_used",
    ):
        require(receipt.get(key) is False, f"unsafe source receipt flag: {key}")

    with template_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "source template lacks header")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    require(len(rows) == source.get("expected_rows"), "template rows differ")
    required_source_columns = set(
        _mapping(config.get("blinding"), "blinding")["visible_source_columns"]
    )
    required_source_columns.update(
        {
            "episode_index",
            "field_id",
            "topic_id",
            "entry_year",
            "window_code",
            "openalex_id",
            *source.get("required_blank_columns", []),
        }
    )
    require(
        required_source_columns.issubset(fieldnames),
        "source template columns missing",
    )
    blank_columns = list(source.get("required_blank_columns", []))
    require(blank_columns == SOURCE_REVIEW_COLUMNS, "source blank-column contract differs")
    for row in rows:
        for column in blank_columns:
            require(not str(row[column]).strip(), f"source review column not blank: {column}")
        require(str(row["openalex_id"]).strip(), "source OpenAlex ID missing")
        require(str(row["title"]).strip(), "source title missing")
        require(
            str(row["observed_primary_topic_name"]).strip(),
            "source topic name missing",
        )
    require(
        len({row["openalex_id"] for row in rows}) == len(rows),
        "source OpenAlex IDs not unique",
    )
    return rows, fieldnames, {
        "root_basename": root.name,
        "receipt_sha256": _sha256(receipt_path),
        "template_sha256": _sha256(template_path),
        "template_rows": len(rows),
    }


def _opaque_item_id(seed: str, reviewer: int, openalex_id: str) -> str:
    payload = f"{seed}\0reviewer-{reviewer}\0{openalex_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _order_key(seed: str, reviewer: int, openalex_id: str) -> tuple[bytes, str]:
    payload = f"{seed}\0reviewer-{reviewer}\0{openalex_id}".encode()
    return hashlib.sha256(payload).digest(), openalex_id


def _build_packet_rows(
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
    packet = []
    for row in ordered:
        packet.append(
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
        )
    require(
        len({row["review_item_id"] for row in packet}) == len(packet),
        f"reviewer {reviewer} item IDs not unique",
    )
    return packet


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def _write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
    path.chmod(0o600)


def _artifact_record(path: Path, *, root: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


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


def _relative_or_name(path: Path, *, root: Path) -> str:
    path = path.resolve()
    root = root.resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name


def _cleanup_temporary_directory(path: Path, *, output: Path) -> None:
    path = path.resolve()
    output = output.resolve()
    require(path.is_absolute(), "temporary path must be absolute")
    require(path.parent == output.parent, "temporary path parent differs")
    require(
        path.name.startswith(f".{output.name}.tmp-"),
        "temporary path prefix differs",
    )
    if path.exists():
        shutil.rmtree(path)


def _review_instructions(labels: Mapping[str, Any]) -> str:
    lines = [
        "# Blinded D0 title review",
        "",
        "Complete every row independently. Do not compare with the other reviewer",
        "before both completed files are frozen and hashed.",
        "",
        "Allowed labels:",
        "",
    ]
    for label, definition in labels.items():
        lines.append(f"- `{label}`: `{definition}`")
    lines.extend(
        [
            "",
            "Notes are required for AMBIGUOUS, OFF_TOPIC, and UNREVIEWABLE.",
            "Do not add, delete, or reorder columns. Row order may be preserved.",
            "This packet contains no entry, persistence, citation, or model outcome.",
            "",
        ]
    )
    return "\n".join(lines)


def build_packets(
    *,
    config_path: Path,
    source_root: Path,
    output: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    output = output.resolve()
    require(output.is_absolute(), "output must be absolute")
    require(output.parent.is_dir(), "output parent missing")
    require(not output.exists(), "output already exists")
    config = _load_yaml(config_path)
    require(
        config.get("schema_version") == "pivot_feedback_d0_content_review_config.v1",
        "config schema mismatch",
    )
    require(
        config.get("status") == "FROZEN_PACKET_GENERATION_CONTRACT_HUMAN_REVIEW_NOT_RUN",
        "config status mismatch",
    )
    gate_parent = _validate_gate_parent(config, repo_root=repo_root)
    source_rows, source_fieldnames, source = _read_source_template(
        config,
        source_root=source_root,
    )

    review_contract = _mapping(config.get("review_contract"), "review contract")
    require(review_contract.get("reviewer_count") == 2, "reviewer count differs")
    require(
        review_contract.get("reviewers_must_be_distinct_humans") is True,
        "distinct humans not required",
    )
    require(
        review_contract.get("artificial_agent_may_substitute_for_human_reviewer")
        is False,
        "artificial reviewer substitution allowed",
    )
    labels = _mapping(review_contract.get("label_definitions"), "label definitions")
    require(list(labels) == review_contract.get("labels"), "label definitions differ")

    blinding = _mapping(config.get("blinding"), "blinding")
    require(
        blinding.get("reviewer_packet_columns") == PACKET_COLUMNS,
        "packet columns differ",
    )
    require(
        set(blinding.get("hidden_from_reviewers", [])).isdisjoint(PACKET_COLUMNS),
        "hidden source field exposed in packet",
    )
    forbidden = _mapping(
        config.get("forbidden_inputs_and_outputs"),
        "forbidden inputs and outputs",
    )
    for key, value in forbidden.items():
        require(value is False, f"forbidden contract enabled: {key}")
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output_contract.get("packet_status") == STATUS, "packet status differs")
    require(
        output_contract.get("reviewer_packet_rows_each") == len(source_rows),
        "packet row contract differs",
    )
    require(
        output_contract.get("reviewer_packet_orders_must_differ") is True,
        "separate packet order not required",
    )
    require(
        output_contract.get("reviewer_packet_content_sets_must_match") is True,
        "matching packet content not required",
    )
    for key in (
        "human_review_complete_after_packet_generation",
        "d0_complete_after_packet_generation",
        "leaveout_network_bridge_authorized_after_packet_generation",
        "discovery_outcome_opening_authorized",
        "model_fitting_authorized",
        "g4_open",
        "gpu_used",
    ):
        require(output_contract.get(key) is False, f"unsafe output contract: {key}")
    item_seed = str(blinding["item_id_seed"])
    order_seed = str(blinding["reviewer_order_seed"])
    packets = {
        reviewer: _build_packet_rows(
            source_rows,
            reviewer=reviewer,
            item_seed=item_seed,
            order_seed=order_seed,
        )
        for reviewer in (1, 2)
    }
    require(
        [row["title"] for row in packets[1]]
        != [row["title"] for row in packets[2]],
        "reviewer packet orders are identical",
    )
    require(
        sorted(
            (
                row["target_topic_name"],
                row["title"],
                row["language"],
                row["work_type"],
            )
            for row in packets[1]
        )
        == sorted(
            (
                row["target_topic_name"],
                row["title"],
                row["language"],
                row["work_type"],
            )
            for row in packets[2]
        ),
        "reviewer packet content sets differ",
    )

    reviewer_ids: dict[int, dict[str, str]] = {}
    for reviewer in (1, 2):
        reviewer_ids[reviewer] = {
            source_row["openalex_id"]: packet_row["review_item_id"]
            for source_row, packet_row in zip(
                sorted(
                    source_rows,
                    key=lambda row: _order_key(
                        order_seed,
                        reviewer,
                        row["openalex_id"],
                    ),
                ),
                packets[reviewer],
                strict=True,
            )
        }
    adjudication_rows = []
    for index, row in enumerate(source_rows, start=1):
        adjudication_rows.append(
            {
                "source_row_number": index,
                "reviewer_1_item_id": reviewer_ids[1][row["openalex_id"]],
                "reviewer_2_item_id": reviewer_ids[2][row["openalex_id"]],
                **{
                    key: value
                    for key, value in row.items()
                    if key not in SOURCE_REVIEW_COLUMNS
                },
            }
        )
    adjudication_columns = [
        *ADJUDICATION_PREFIX_COLUMNS,
        *[column for column in source_fieldnames if column not in SOURCE_REVIEW_COLUMNS],
    ]

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=output.parent,
        )
    ).resolve()
    try:
        temporary.chmod(0o700)
        instructions_path = temporary / "REVIEW_INSTRUCTIONS.md"
        reviewer_1_path = temporary / "reviewer_1_packet.csv"
        reviewer_2_path = temporary / "reviewer_2_packet.csv"
        adjudication_path = temporary / "adjudication_key.csv"
        manifest_path = temporary / "review_packet_manifest.json"
        _write_text(instructions_path, _review_instructions(labels))
        _write_csv(reviewer_1_path, packets[1], PACKET_COLUMNS)
        _write_csv(reviewer_2_path, packets[2], PACKET_COLUMNS)
        _write_csv(adjudication_path, adjudication_rows, adjudication_columns)

        artifact_paths_and_rows = [
            (instructions_path, None),
            (reviewer_1_path, len(packets[1])),
            (reviewer_2_path, len(packets[2])),
            (adjudication_path, len(adjudication_rows)),
        ]
        preliminary_artifacts = [
            _artifact_record(path, root=temporary, rows=rows)
            for path, rows in artifact_paths_and_rows
        ]
        manifest = {
            "schema_version": config["output_contract"]["manifest_schema_version"],
            "status": STATUS,
            "generated_at_utc": _utc_now(),
            "producer": {
                "relative_path": _relative_or_name(
                    Path(__file__),
                    root=repo_root,
                ),
                "sha256": _sha256(Path(__file__).resolve()),
                "config_relative_path": _relative_or_name(
                    config_path,
                    root=repo_root,
                ),
                "config_sha256": _sha256(config_path),
            },
            "gate_parent": gate_parent,
            "source": source,
            "review_contract": {
                "reviewer_count": 2,
                "reviewers_must_be_distinct_humans": True,
                "artificial_agent_may_substitute_for_human_reviewer": False,
                "labels": list(labels),
            },
            "packet_accounting": {
                "source_rows": len(source_rows),
                "reviewer_1_rows": len(packets[1]),
                "reviewer_2_rows": len(packets[2]),
                "adjudication_key_rows": len(adjudication_rows),
                "packet_orders_differ": True,
                "packet_content_sets_match": True,
                "reviewer_labels_filled": 0,
                "human_review_complete": False,
            },
            "artifacts_before_manifest": preliminary_artifacts,
            "scope": {
                "outcomes_read": False,
                "automatic_title_classification_performed": False,
                "human_review_complete": False,
                "d0_complete": False,
                "leaveout_network_bridge_authorized": False,
                "discovery_outcome_opening_authorized": False,
                "g0_authorized": False,
                "model_fitting_authorized": False,
                "g4_open": False,
                "gpu_used": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
        }
        _write_json(manifest_path, manifest)
        artifacts = [
            *preliminary_artifacts,
            _artifact_record(manifest_path, root=temporary),
        ]
        receipt = {
            "schema_version": config["output_contract"]["receipt_schema_version"],
            "status": STATUS,
            "artifact_count": len(artifacts),
            "artifact_tree_sha256": _artifact_tree_sha256(artifacts),
            "artifacts": artifacts,
            "source_template_sha256": source["template_sha256"],
            "reviewer_packet_rows_each": len(source_rows),
            "packet_orders_differ": True,
            "packet_content_sets_match": True,
            "reviewer_labels_filled": 0,
            "human_review_complete": False,
            "automatic_title_classification_performed": False,
            "outcomes_read": False,
            "d0_complete": False,
            "leaveout_network_bridge_authorized": False,
            "discovery_outcome_opening_authorized": False,
            "g0_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "gpu_used": False,
            "scientific_effect": "NOT_ESTABLISHED",
        }
        receipt_path = temporary / "content_review_packets.outputs.json"
        _write_json(receipt_path, receipt)
        os.replace(temporary, output)
        output.chmod(0o700)
    except BaseException:
        _cleanup_temporary_directory(temporary, output=output)
        raise
    return {
        "status": STATUS,
        "output": str(output),
        "receipt_sha256": _sha256(output / "content_review_packets.outputs.json"),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "reviewer_packet_rows_each": len(source_rows),
        "packet_orders_differ": True,
        "human_review_complete": False,
        "d0_complete": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_packets(
            config_path=args.config,
            source_root=args.source_root,
            output=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ReviewPacketError, OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0_CONTENT_REVIEW_PACKET_GENERATION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
