#!/usr/bin/env python3
"""Prepare outcome-blind dual-model API review requests and human calibration."""

from __future__ import annotations

import argparse
import csv
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
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_api_assisted_review_v2.yaml"
)
DEFAULT_SOURCE_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-api-review-v2-prepared-20260729"
)
STATUS = "PASS_D0_API_ASSISTED_REVIEW_BATCH_REQUESTS_PREPARED_NOT_SUBMITTED"
PACKET_COLUMNS = [
    "review_item_id",
    "target_topic_name",
    "title",
    "language",
    "work_type",
    "reviewer_label",
    "reviewer_notes",
]
VISIBLE_SOURCE_MAP = {
    "target_topic_name": "observed_primary_topic_name",
    "title": "title",
    "language": "language",
    "work_type": "observed_work_type",
}
SOURCE_REVIEW_COLUMNS = [
    "reviewer_1_label",
    "reviewer_2_label",
    "adjudicated_label",
    "review_notes",
]


class ApiReviewPreparationError(RuntimeError):
    """Fail-closed API-review preparation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ApiReviewPreparationError(message)


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


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ApiReviewPreparationError(f"invalid YAML {path.name}: {error}") from error
    return _mapping(value, path.name)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ApiReviewPreparationError(f"invalid JSON {path.name}: {error}") from error
    return _mapping(value, path.name)


def _safe_repo_file(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _validate_bound_yaml(
    binding: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> dict[str, Any]:
    path = _safe_repo_file(repo_root, binding.get("relative_path"), label)
    require(_sha256(path) == binding.get("sha256"), f"{label} SHA mismatch")
    value = _load_yaml(path)
    if "required_schema_version" in binding:
        require(
            value.get("schema_version") == binding.get("required_schema_version"),
            f"{label} schema mismatch",
        )
    if "required_status" in binding:
        require(
            value.get("status") == binding.get("required_status"),
            f"{label} status mismatch",
        )
    for dotted_path, expected in _mapping(
        binding.get("required_fields", {}),
        f"{label} required fields",
    ).items():
        observed = _nested(value, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"{label} field mismatch: {dotted_path}",
        )
    return value


def _validate_config(config: Mapping[str, Any], repo_root: Path) -> None:
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_api_assisted_review_config.v2",
        "config schema mismatch",
    )
    require(
        config.get("status")
        == "FROZEN_API_ASSISTED_REVIEW_CONTRACT_BATCH_REQUESTS_NOT_SUBMITTED",
        "config status mismatch",
    )
    _validate_bound_yaml(
        _mapping(config.get("gate_parent"), "gate parent"),
        repo_root=repo_root,
        label="gate parent",
    )
    superseded = _mapping(
        config.get("superseded_execution_path"),
        "superseded execution path",
    )
    legacy_path = _safe_repo_file(
        repo_root,
        superseded.get("contract_relative_path"),
        "legacy contract",
    )
    require(
        _sha256(legacy_path) == superseded.get("contract_sha256"),
        "legacy contract SHA mismatch",
    )
    require(
        superseded.get("disposition") == "PRESERVED_NOT_RUN_NOT_RELABELED",
        "legacy contract disposition unsafe",
    )
    require(
        superseded.get("prepared_packets_may_be_counted_as_completed_review") is False,
        "legacy packet completion overstated",
    )


def _read_source(
    config: Mapping[str, Any],
    *,
    source_root: Path,
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    source = _mapping(config.get("source_audit"), "source audit")
    root = source_root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "source root invalid")
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
    for key in (
        "human_review_complete",
        "d0_complete",
        "leaveout_network_bridge_authorized",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
        "outcomes_read",
        "gpu_used",
    ):
        require(receipt.get(key) is False, f"unsafe source receipt flag: {key}")
    with template_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "source template lacks header")
        columns = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    require(len(rows) == source.get("expected_rows"), "source rows differ")
    required = {
        "episode_index",
        "window_code",
        "openalex_id",
        *VISIBLE_SOURCE_MAP.values(),
        *SOURCE_REVIEW_COLUMNS,
    }
    require(required.issubset(columns), "source columns missing")
    for row in rows:
        require(str(row["openalex_id"]).strip(), "source OpenAlex ID missing")
        require(str(row["title"]).strip(), "source title missing")
        for column in SOURCE_REVIEW_COLUMNS:
            require(not str(row[column]).strip(), f"source review populated: {column}")
    require(
        len({row["openalex_id"] for row in rows}) == len(rows),
        "source OpenAlex IDs not unique",
    )
    strata: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["episode_index"], row["window_code"])
        strata[key] = strata.get(key, 0) + 1
    require(
        len(strata) == source.get("expected_episode_window_strata"),
        "source strata differ",
    )
    require(
        set(strata.values()) == {source.get("expected_rows_per_stratum")},
        "source rows per stratum differ",
    )
    return columns, rows, {
        "root_basename": root.name,
        "receipt_sha256": _sha256(receipt_path),
        "template_sha256": _sha256(template_path),
        "template_rows": len(rows),
        "strata": len(strata),
    }


def _opaque_id(seed: str, kind: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{kind}\0{value}".encode()).hexdigest()


def _order_key(seed: str, kind: str, value: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{seed}\0{kind}\0{value}".encode()).digest()
    return digest, value


def _response_schema(
    labels: Sequence[str],
    confidence_levels: Sequence[str],
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "topic_title_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": list(labels)},
                "rationale": {"type": "string"},
                "confidence": {
                    "type": "string",
                    "enum": list(confidence_levels),
                },
            },
            "required": ["label", "rationale", "confidence"],
            "additionalProperties": False,
        },
    }


def _request(
    *,
    api_item_id: str,
    judge: Mapping[str, Any],
    row: Mapping[str, str],
    api: Mapping[str, Any],
) -> dict[str, Any]:
    visible = {
        output: row[source]
        for output, source in VISIBLE_SOURCE_MAP.items()
    }
    body = {
        "model": judge["model"],
        "store": False,
        "max_output_tokens": api["max_output_tokens"],
        "instructions": api["prompt_text"],
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            visible,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                ],
            }
        ],
        "text": {
            "format": _response_schema(
                api["allowed_labels"],
                api["confidence_levels"],
            )
        },
    }
    return {
        "custom_id": f"{judge['judge_id']}-{api_item_id}",
        "method": "POST",
        "url": api["endpoint"],
        "body": body,
    }


def _base_sample(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: str,
    rows_per_stratum: int,
) -> list[Mapping[str, str]]:
    strata: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        strata.setdefault((row["episode_index"], row["window_code"]), []).append(row)
    selected: list[Mapping[str, str]] = []
    for key in sorted(strata, key=lambda item: (int(item[0]), int(item[1]))):
        ordered = sorted(
            strata[key],
            key=lambda row: _order_key(seed, "base", row["openalex_id"]),
        )
        require(len(ordered) >= rows_per_stratum, "insufficient stratum rows")
        selected.extend(ordered[:rows_per_stratum])
    return selected


def _packet_rows(
    selected: Sequence[Mapping[str, str]],
    *,
    reviewer: int,
    seed: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    ordered = sorted(
        selected,
        key=lambda row: _order_key(
            seed,
            f"reviewer-{reviewer}",
            row["openalex_id"],
        ),
    )
    packet_rows: list[dict[str, str]] = []
    item_id_by_source_id: dict[str, str] = {}
    for row in ordered:
        source_id = row["openalex_id"]
        review_item_id = _opaque_id(
            seed,
            f"reviewer-{reviewer}",
            source_id,
        )
        packet_rows.append(
            {
                "review_item_id": review_item_id,
                "target_topic_name": row["observed_primary_topic_name"],
                "title": row["title"],
                "language": row["language"],
                "work_type": row["observed_work_type"],
                "reviewer_label": "",
                "reviewer_notes": "",
            }
        )
        item_id_by_source_id[source_id] = review_item_id
    return packet_rows, item_id_by_source_id


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    path.chmod(0o600)


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _cleanup_temporary_directory(
    temporary: Path | None,
    *,
    output: Path,
) -> None:
    """Remove only the private sibling directory created for this output."""
    if temporary is None:
        return
    require(temporary.is_absolute(), "temporary path must be absolute")
    require(output.is_absolute(), "output path must be absolute")
    require(temporary != output, "temporary path equals output")
    require(temporary.parent == output.parent, "temporary path is not an output sibling")
    require(
        temporary.name.startswith(f".{output.name}.tmp-"),
        "temporary path prefix differs",
    )
    if temporary.exists():
        require(
            temporary.is_dir() and not temporary.is_symlink(),
            "temporary path is not a private directory",
        )
        shutil.rmtree(temporary)


def prepare(
    *,
    config_path: Path,
    source_root: Path,
    output: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    require(config_path.is_relative_to(repo_root), "config outside repository")
    config = _load_yaml(config_path)
    _validate_config(config, repo_root)
    _, source_rows, source_evidence = _read_source(
        config,
        source_root=source_root,
    )
    api = _mapping(config.get("api_review"), "api review")
    calibration = _mapping(config.get("human_calibration"), "human calibration")
    require(api.get("provider") == "OpenAI", "provider differs")
    require(api.get("transport") == "BATCH_API", "transport differs")
    require(api.get("endpoint") == "/v1/responses", "endpoint differs")
    require(api.get("store") is False, "response storage enabled")
    require(api.get("tools_enabled") is False, "API tools enabled")
    require(api.get("web_search_enabled") is False, "web search enabled")
    require(
        api.get("sampling_parameter_policy")
        == "PINNED_MODEL_PROVIDER_DEFAULT_NO_TEMPERATURE_FIELD",
        "sampling policy differs",
    )
    require(
        api.get("request_visible_fields") == list(VISIBLE_SOURCE_MAP),
        "visible field contract differs",
    )
    judges = list(api.get("model_judges", []))
    require(len(judges) == 2, "two API judges required")
    require(len({judge["judge_id"] for judge in judges}) == 2, "judge IDs not unique")
    require(len({judge["model"] for judge in judges}) == 2, "model snapshots not unique")
    require(
        all(
            isinstance(judge["model"], str)
            and judge["model"].count("-") >= 3
            and any(char.isdigit() for char in judge["model"])
            for judge in judges
        ),
        "model snapshots not pinned",
    )
    request_seed = str(api["request_seed"])
    source_by_id = {row["openalex_id"]: row for row in source_rows}
    api_item_by_source_id = {
        source_id: _opaque_id(request_seed, "api-item", source_id)
        for source_id in source_by_id
    }
    ordered_source_ids = sorted(
        source_by_id,
        key=lambda source_id: api_item_by_source_id[source_id],
    )

    output = output.resolve(strict=False)
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir(), "output parent missing")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    temporary.chmod(0o700)
    try:
        request_artifacts: list[dict[str, Any]] = []
        for judge in judges:
            requests = [
                _request(
                    api_item_id=api_item_by_source_id[source_id],
                    judge=judge,
                    row=source_by_id[source_id],
                    api=api,
                )
                for source_id in ordered_source_ids
            ]
            request_path = temporary / f"{judge['judge_id']}_requests.jsonl"
            _write_jsonl(request_path, requests)
            request_artifacts.append(_artifact(request_path, rows=len(requests)))

        request_key_rows = []
        for source_row_number, source_id in enumerate(ordered_source_ids, start=1):
            row = source_by_id[source_id]
            request_key_rows.append(
                {
                    "api_item_id": api_item_by_source_id[source_id],
                    "source_row_number": source_row_number,
                    "openalex_id": source_id,
                    "episode_index": row["episode_index"],
                    "window_code": row["window_code"],
                    "target_topic_name": row["observed_primary_topic_name"],
                    "title": row["title"],
                    "language": row["language"],
                    "work_type": row["observed_work_type"],
                }
            )
        request_key_path = temporary / "api_request_key.csv"
        _write_csv(
            request_key_path,
            request_key_rows,
            [
                "api_item_id",
                "source_row_number",
                "openalex_id",
                "episode_index",
                "window_code",
                "target_topic_name",
                "title",
                "language",
                "work_type",
            ],
        )

        base_rows = _base_sample(
            source_rows,
            seed=str(calibration["base_sample_seed"]),
            rows_per_stratum=int(
                calibration["base_rows_per_episode_window_stratum"]
            ),
        )
        require(
            len(base_rows) == calibration.get("base_sample_rows"),
            "base sample rows differ",
        )
        packet_paths = []
        packet_rows_by_reviewer: dict[int, list[dict[str, str]]] = {}
        packet_id_maps: dict[int, dict[str, str]] = {}
        for reviewer in (1, 2):
            packet_rows, item_id_by_source_id = _packet_rows(
                base_rows,
                reviewer=reviewer,
                seed=str(calibration["base_sample_seed"]),
            )
            packet_rows_by_reviewer[reviewer] = packet_rows
            packet_id_maps[reviewer] = item_id_by_source_id
            packet_path = temporary / f"human_calibration_reviewer_{reviewer}.csv"
            _write_csv(packet_path, packet_rows, PACKET_COLUMNS)
            packet_paths.append(packet_path)
        require(
            [row["title"] for row in packet_rows_by_reviewer[1]]
            != [row["title"] for row in packet_rows_by_reviewer[2]],
            "reviewer packet orders identical",
        )

        calibration_key_rows = []
        for row in sorted(
            base_rows,
            key=lambda item: (
                int(item["episode_index"]),
                int(item["window_code"]),
                item["openalex_id"],
            ),
        ):
            calibration_key_rows.append(
                {
                    "api_item_id": api_item_by_source_id[row["openalex_id"]],
                    "reviewer_1_item_id": packet_id_maps[1][row["openalex_id"]],
                    "reviewer_2_item_id": packet_id_maps[2][row["openalex_id"]],
                    "openalex_id": row["openalex_id"],
                    "episode_index": row["episode_index"],
                    "window_code": row["window_code"],
                    "target_topic_name": row["observed_primary_topic_name"],
                    "title": row["title"],
                    "language": row["language"],
                    "work_type": row["observed_work_type"],
                }
            )
        calibration_key_path = temporary / "human_calibration_key.csv"
        _write_csv(
            calibration_key_path,
            calibration_key_rows,
            [
                "api_item_id",
                "reviewer_1_item_id",
                "reviewer_2_item_id",
                "openalex_id",
                "episode_index",
                "window_code",
                "target_topic_name",
                "title",
                "language",
                "work_type",
            ],
        )

        instructions_path = temporary / "API_RUN_INSTRUCTIONS.md"
        instructions_path.write_text(
            "# D0 API review v2 prepared inputs\n\n"
            "Status: `NOT_SUBMITTED`\n\n"
            "The two `judge_*_requests.jsonl` files are exact OpenAI Batch API "
            "inputs for `/v1/responses`. Upload each with purpose `batch`, then "
            "create a 24-hour batch. Preserve the upload file ID, batch ID, "
            "request ID, downloaded output bytes, and error-file bytes.\n\n"
            "Do not give `api_request_key.csv` or "
            "`human_calibration_key.csv` to reviewers. Give each human only "
            "their numbered packet. Reviewers must remain blind to API output. "
            "API execution and human calibration are both still not run.\n",
            encoding="utf-8",
        )
        instructions_path.chmod(0o600)

        primary_artifacts = [
            _artifact(instructions_path),
            *request_artifacts,
            _artifact(request_key_path, rows=len(request_key_rows)),
            *[
                _artifact(path, rows=len(base_rows))
                for path in packet_paths
            ],
            _artifact(calibration_key_path, rows=len(calibration_key_rows)),
        ]
        manifest_path = temporary / "preparation_manifest.json"
        manifest = {
            "schema_version": "pivot_feedback_d0_api_review_preparation_manifest.v2",
            "status": STATUS,
            "config_relative_path": config_path.relative_to(repo_root).as_posix(),
            "config_sha256": _sha256(config_path),
            "source": source_evidence,
            "api_endpoint": api["endpoint"],
            "completion_window": api["completion_window"],
            "judge_models": {
                judge["judge_id"]: judge["model"] for judge in judges
            },
            "request_rows_per_judge": len(source_rows),
            "total_request_rows": len(source_rows) * len(judges),
            "base_calibration_rows": len(base_rows),
            "base_calibration_strata": len(
                {(row["episode_index"], row["window_code"]) for row in base_rows}
            ),
            "artifacts": primary_artifacts,
            "api_submitted": False,
            "api_responses_received": False,
            "human_calibration_complete": False,
            "outcomes_read": False,
        }
        _write_json(manifest_path, manifest)
        artifacts = [*primary_artifacts, _artifact(manifest_path)]
        artifact_tree = hashlib.sha256()
        for item in sorted(artifacts, key=lambda value: value["relative_path"]):
            artifact_tree.update(str(item["relative_path"]).encode())
            artifact_tree.update(b"\0")
            artifact_tree.update(str(item["sha256"]).encode())
            artifact_tree.update(b"\0")
            artifact_tree.update(str(item["bytes"]).encode())
            artifact_tree.update(b"\n")
        receipt = {
            "schema_version": "pivot_feedback_d0_api_review_preparation_outputs.v2",
            "status": STATUS,
            "artifact_count": len(artifacts),
            "artifact_tree_sha256": artifact_tree.hexdigest(),
            "artifacts": artifacts,
            "source_template_sha256": source_evidence["template_sha256"],
            "judge_models": manifest["judge_models"],
            "request_rows_per_judge": len(source_rows),
            "total_request_rows": len(source_rows) * len(judges),
            "base_calibration_rows": len(base_rows),
            "base_calibration_strata": manifest["base_calibration_strata"],
            "request_visible_fields": list(VISIBLE_SOURCE_MAP),
            "forbidden_request_field_count": len(api["forbidden_request_fields"]),
            "request_bodies_contain_forbidden_fields": False,
            "api_submitted": False,
            "api_responses_received": False,
            "api_review_complete": False,
            "human_calibration_complete": False,
            "legacy_human_review_complete": False,
            "candidate_universe_screen_complete": False,
            "episode_3_quarantined": True,
            "remaining_episodes_promoted": False,
            "old_author_panel_current": False,
            "d0_complete": False,
            "discovery_outcome_opening_authorized": False,
            "g0_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "outcomes_read": False,
            "scientific_effect": "NOT_ESTABLISHED",
        }
        receipt_path = temporary / "api_review_preparation.outputs.json"
        _write_json(receipt_path, receipt)
        require(
            all(path.stat().st_mode & 0o777 == 0o600 for path in temporary.iterdir()),
            "output file mode differs",
        )
        os.replace(temporary, output)
        temporary = None
        return receipt
    finally:
        _cleanup_temporary_directory(temporary, output=output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = prepare(
            config_path=args.config,
            source_root=args.source_root,
            output=args.output,
        )
    except (ApiReviewPreparationError, OSError) as error:
        print(f"FAIL: {error}")
        return 1
    print(result["status"])
    print(f"output={args.output}")
    print(f"total_requests={result['total_request_rows']}")
    print(f"base_calibration_rows={result['base_calibration_rows']}")
    print("api_submitted=false")
    print("outcomes_read=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
