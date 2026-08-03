#!/usr/bin/env python3
"""Prepare outcome-blind OpenAI and Claude batch review inputs."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_UP
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
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v3.yaml"
)
DEFAULT_SOURCE_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-prepared-20260729"
)
STATUS = "PASS_D0_CROSS_PROVIDER_BATCH_REQUESTS_PREPARED_NOT_SUBMITTED"
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
PACKET_COLUMNS = [
    "review_item_id",
    "target_topic_name",
    "title",
    "language",
    "work_type",
    "reviewer_label",
    "reviewer_notes",
]


class CrossProviderPreparationError(RuntimeError):
    """Fail-closed cross-provider preparation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossProviderPreparationError(message)


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
        raise CrossProviderPreparationError(
            f"invalid YAML {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CrossProviderPreparationError(
            f"invalid JSON {path.name}: {error}"
        ) from error
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
) -> None:
    path = _safe_repo_file(repo_root, binding.get("relative_path"), label)
    require(_sha256(path) == binding.get("sha256"), f"{label} SHA mismatch")
    value = _load_yaml(path)
    require(
        value.get("schema_version") == binding.get("required_schema_version"),
        f"{label} schema mismatch",
    )
    require(
        value.get("status") == binding.get("required_status"),
        f"{label} status mismatch",
    )
    for dotted_path, expected in _mapping(
        binding.get("required_fields", {}),
        f"{label} fields",
    ).items():
        observed = _nested(value, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"{label} field mismatch: {dotted_path}",
        )


def _validate_config(config: Mapping[str, Any], repo_root: Path) -> None:
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_cross_provider_api_review_config.v3",
        "config schema mismatch",
    )
    require(
        config.get("status")
        == "FROZEN_CROSS_PROVIDER_REVIEW_CONTRACT_BATCH_REQUESTS_NOT_SUBMITTED",
        "config status mismatch",
    )
    _validate_bound_yaml(
        _mapping(config.get("gate_parent"), "gate parent"),
        repo_root=repo_root,
        label="gate parent",
    )
    predecessor = _mapping(
        config.get("superseded_openai_only_path"),
        "superseded OpenAI-only path",
    )
    for key in ("config", "summary"):
        path = _safe_repo_file(
            repo_root,
            predecessor.get(f"{key}_relative_path"),
            f"predecessor {key}",
        )
        require(
            _sha256(path) == predecessor.get(f"{key}_sha256"),
            f"predecessor {key} SHA mismatch",
        )
    require(
        predecessor.get("disposition")
        == "PRESERVED_PREPARED_NOT_SUBMITTED_SUPERSEDED_BY_CROSS_PROVIDER_V3",
        "predecessor disposition differs",
    )
    require(
        predecessor.get("v2_requests_may_be_submitted_after_v3_freeze") is False,
        "predecessor submission remains enabled",
    )


def _read_source(
    config: Mapping[str, Any],
    *,
    source_root: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    source = _mapping(config.get("source_audit"), "source audit")
    raw_root = source_root.absolute()
    require(not raw_root.is_symlink(), "source root symlinked")
    root = source_root.resolve(strict=True)
    require(root.is_dir(), "source root invalid")
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
        require(receipt.get(key) is False, f"unsafe source flag: {key}")
    with template_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "source template lacks header")
        columns = set(reader.fieldnames)
        rows = [dict(row) for row in reader]
    required = {
        "episode_index",
        "window_code",
        "openalex_id",
        *VISIBLE_SOURCE_MAP.values(),
        *SOURCE_REVIEW_COLUMNS,
    }
    require(required.issubset(columns), "source columns missing")
    require(len(rows) == source.get("expected_rows") == 384, "source rows differ")
    require(
        len({row["openalex_id"] for row in rows}) == len(rows),
        "source IDs not unique",
    )
    strata: dict[tuple[str, str], int] = {}
    for row in rows:
        require(row["openalex_id"].strip(), "source ID missing")
        require(row["title"].strip(), "source title missing")
        for column in SOURCE_REVIEW_COLUMNS:
            require(not row[column].strip(), f"source review populated: {column}")
        key = (row["episode_index"], row["window_code"])
        strata[key] = strata.get(key, 0) + 1
    require(len(strata) == 12, "source strata differ")
    require(set(strata.values()) == {32}, "source rows per stratum differ")
    return rows, {
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


def _json_schema(
    labels: Sequence[str],
    confidence_levels: Sequence[str],
) -> dict[str, Any]:
    return {
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
    }


def _visible(row: Mapping[str, str]) -> dict[str, str]:
    return {
        output: row[source]
        for output, source in VISIBLE_SOURCE_MAP.items()
    }


def _openai_request(
    *,
    item_id: str,
    judge: Mapping[str, Any],
    row: Mapping[str, str],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "custom_id": f"openai-{item_id}",
        "method": "POST",
        "url": judge["request_endpoint"],
        "body": {
            "model": judge["model"],
            "store": False,
            "max_output_tokens": contract["max_output_tokens"],
            "instructions": contract["prompt_text"],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                _visible(row),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "topic_title_review",
                    "strict": True,
                    "schema": _json_schema(
                        contract["allowed_labels"],
                        contract["confidence_levels"],
                    ),
                }
            },
        },
    }


def _claude_request(
    *,
    item_id: str,
    judge: Mapping[str, Any],
    row: Mapping[str, str],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "custom_id": f"claude-{item_id}",
        "params": {
            "model": judge["model"],
            "max_tokens": contract["max_output_tokens"],
            "system": contract["prompt_text"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                _visible(row),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    ],
                }
            ],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": _json_schema(
                        contract["allowed_labels"],
                        contract["confidence_levels"],
                    ),
                }
            },
        },
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
    packet: list[dict[str, str]] = []
    ids: dict[str, str] = {}
    for row in ordered:
        source_id = row["openalex_id"]
        item_id = _opaque_id(seed, f"reviewer-{reviewer}", source_id)
        ids[source_id] = item_id
        packet.append(
            {
                "review_item_id": item_id,
                "target_topic_name": row["observed_primary_topic_name"],
                "title": row["title"],
                "language": row["language"],
                "work_type": row["observed_work_type"],
                "reviewer_label": "",
                "reviewer_notes": "",
            }
        )
    return packet, ids


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
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


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _artifact_tree(artifacts: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(artifacts, key=lambda value: str(value["relative_path"])):
        digest.update(str(item["relative_path"]).encode())
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode())
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _estimated_maximum_cost(
    *,
    openai_path: Path,
    claude_path: Path,
    request_count: int,
    max_output_tokens: int,
    cost: Mapping[str, Any],
) -> dict[str, str]:
    overhead = int(cost["input_token_safety_overhead_per_request"]) * request_count
    openai_input = openai_path.stat().st_size + overhead
    claude_input = claude_path.stat().st_size + overhead
    openai_rates = _mapping(
        cost["openai_batch_price_per_million"],
        "OpenAI rates",
    )
    claude_rates = _mapping(
        cost["anthropic_haiku_4_5_batch_price_per_million"],
        "Claude rates",
    )
    output_tokens = request_count * max_output_tokens
    million = Decimal(1_000_000)
    openai_cost = (
        Decimal(openai_input) * Decimal(str(openai_rates["input"]))
        + Decimal(output_tokens) * Decimal(str(openai_rates["output"]))
    ) / million
    claude_cost = (
        Decimal(claude_input) * Decimal(str(claude_rates["input"]))
        + Decimal(output_tokens) * Decimal(str(claude_rates["output"]))
    ) / million
    total = openai_cost + claude_cost
    quant = Decimal("0.000001")
    return {
        "openai_input_token_upper_bound": str(openai_input),
        "claude_input_token_upper_bound": str(claude_input),
        "output_token_upper_bound_per_provider": str(output_tokens),
        "openai_maximum_usd": str(openai_cost.quantize(quant, rounding=ROUND_UP)),
        "claude_maximum_usd": str(claude_cost.quantize(quant, rounding=ROUND_UP)),
        "total_maximum_usd": str(total.quantize(quant, rounding=ROUND_UP)),
        "authorized_maximum_usd": str(
            Decimal(str(cost["maximum_authorized_total_usd"])).quantize(
                Decimal("0.00")
            )
        ),
    }


def _cleanup_temporary_directory(
    temporary: Path | None,
    *,
    output: Path,
) -> None:
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
    source_rows, source_evidence = _read_source(config, source_root=source_root)
    contract = _mapping(config.get("shared_review_contract"), "shared contract")
    calibration = _mapping(config.get("human_calibration"), "human calibration")
    cost = _mapping(config.get("operator_cost_control"), "cost control")
    judges = {
        str(item["judge_id"]): _mapping(item, "provider judge")
        for item in list(config.get("provider_judges", []))
    }
    require(set(judges) == {"openai", "claude"}, "provider judges differ")
    require(judges["openai"].get("provider") == "OpenAI", "OpenAI judge differs")
    require(judges["claude"].get("provider") == "Anthropic", "Claude judge differs")
    require(
        judges["openai"].get("model") == "gpt-5-mini-2025-08-07",
        "OpenAI snapshot differs",
    )
    require(
        judges["claude"].get("model") == "claude-haiku-4-5-20251001",
        "Claude snapshot differs",
    )
    require(
        contract.get("request_visible_fields") == list(VISIBLE_SOURCE_MAP),
        "visible fields differ",
    )

    request_seed = str(contract["request_seed"])
    source_by_id = {row["openalex_id"]: row for row in source_rows}
    item_ids = {
        source_id: _opaque_id(request_seed, "api-item", source_id)
        for source_id in source_by_id
    }
    ordered_ids = sorted(source_by_id, key=lambda value: item_ids[value])

    output = output.resolve(strict=False)
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir(), "output parent missing")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    temporary.chmod(0o700)
    try:
        openai_requests = [
            _openai_request(
                item_id=item_ids[source_id],
                judge=judges["openai"],
                row=source_by_id[source_id],
                contract=contract,
            )
            for source_id in ordered_ids
        ]
        openai_path = temporary / "openai_requests.jsonl"
        _write_jsonl(openai_path, openai_requests)

        claude_requests = [
            _claude_request(
                item_id=item_ids[source_id],
                judge=judges["claude"],
                row=source_by_id[source_id],
                contract=contract,
            )
            for source_id in ordered_ids
        ]
        claude_path = temporary / "claude_batch_request.json"
        _write_json(claude_path, {"requests": claude_requests})

        request_key_rows = []
        for row_number, source_id in enumerate(ordered_ids, start=1):
            row = source_by_id[source_id]
            request_key_rows.append(
                {
                    "api_item_id": item_ids[source_id],
                    "source_row_number": row_number,
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
        require(len(base_rows) == calibration.get("base_sample_rows"), "base sample differs")
        packet_paths: list[Path] = []
        packet_ids: dict[int, dict[str, str]] = {}
        packet_titles: dict[int, list[str]] = {}
        for reviewer in (1, 2):
            packet, ids = _packet_rows(
                base_rows,
                reviewer=reviewer,
                seed=str(calibration["base_sample_seed"]),
            )
            packet_ids[reviewer] = ids
            packet_titles[reviewer] = [row["title"] for row in packet]
            packet_path = temporary / f"human_calibration_reviewer_{reviewer}.csv"
            _write_csv(packet_path, packet, PACKET_COLUMNS)
            packet_paths.append(packet_path)
        require(packet_titles[1] != packet_titles[2], "packet orders identical")

        calibration_key_rows = []
        for row in sorted(
            base_rows,
            key=lambda item: (
                int(item["episode_index"]),
                int(item["window_code"]),
                item["openalex_id"],
            ),
        ):
            source_id = row["openalex_id"]
            calibration_key_rows.append(
                {
                    "api_item_id": item_ids[source_id],
                    "reviewer_1_item_id": packet_ids[1][source_id],
                    "reviewer_2_item_id": packet_ids[2][source_id],
                    "openalex_id": source_id,
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

        estimate = _estimated_maximum_cost(
            openai_path=openai_path,
            claude_path=claude_path,
            request_count=len(source_rows),
            max_output_tokens=int(contract["max_output_tokens"]),
            cost=cost,
        )
        require(
            Decimal(estimate["total_maximum_usd"])
            <= Decimal(estimate["authorized_maximum_usd"]),
            "estimated maximum cost exceeds authorization",
        )

        instructions_path = temporary / "API_RUN_INSTRUCTIONS.md"
        instructions_path.write_text(
            "# D0 cross-provider API review v3 prepared inputs\n\n"
            "Status: `NOT_SUBMITTED`\n\n"
            "`openai_requests.jsonl` is an OpenAI Batch API input for "
            "`/v1/responses`. Upload it with purpose `batch`, then create a "
            "24-hour batch. `claude_batch_request.json` is the exact JSON body "
            "for Anthropic `POST /v1/messages/batches`.\n\n"
            f"The frozen conservative total-cost ceiling is "
            f"`${estimate['authorized_maximum_usd']}`; the prepared-byte and "
            f"max-output estimate is `${estimate['total_maximum_usd']}`. "
            "Submission must fail closed if the estimate or artifacts differ.\n\n"
            "Do not provide either private key CSV to human reviewers. API "
            "execution and human calibration are both still not run.\n",
            encoding="utf-8",
        )
        instructions_path.chmod(0o600)

        primary = [
            _artifact(instructions_path),
            _artifact(openai_path, rows=len(openai_requests)),
            _artifact(claude_path, rows=len(claude_requests)),
            _artifact(request_key_path, rows=len(request_key_rows)),
            *[_artifact(path, rows=len(base_rows)) for path in packet_paths],
            _artifact(calibration_key_path, rows=len(calibration_key_rows)),
        ]
        manifest_path = temporary / "preparation_manifest.json"
        manifest = {
            "schema_version": "pivot_feedback_d0_cross_provider_manifest.v3",
            "status": STATUS,
            "config_relative_path": config_path.relative_to(repo_root).as_posix(),
            "config_sha256": _sha256(config_path),
            "source": source_evidence,
            "judges": {
                judge_id: {
                    "provider": judge["provider"],
                    "model": judge["model"],
                    "batch_endpoint": judge["batch_endpoint"],
                    "request_endpoint": judge["request_endpoint"],
                }
                for judge_id, judge in judges.items()
            },
            "request_rows_per_judge": len(source_rows),
            "total_request_rows": len(source_rows) * 2,
            "base_calibration_rows": len(base_rows),
            "base_calibration_strata": 12,
            "cost_control": estimate,
            "artifacts": primary,
            "api_submitted": False,
            "api_responses_received": False,
            "human_calibration_complete": False,
            "outcomes_read": False,
        }
        _write_json(manifest_path, manifest)
        artifacts = [*primary, _artifact(manifest_path)]
        receipt = {
            "schema_version": "pivot_feedback_d0_cross_provider_outputs.v3",
            "status": STATUS,
            "artifact_count": len(artifacts),
            "artifact_tree_sha256": _artifact_tree(artifacts),
            "artifacts": artifacts,
            "source_template_sha256": source_evidence["template_sha256"],
            "judge_models": {
                judge_id: judge["model"] for judge_id, judge in judges.items()
            },
            "request_rows_per_judge": len(source_rows),
            "total_request_rows": len(source_rows) * 2,
            "base_calibration_rows": len(base_rows),
            "base_calibration_strata": 12,
            "request_visible_fields": list(VISIBLE_SOURCE_MAP),
            "request_bodies_contain_forbidden_fields": False,
            "cost_control": estimate,
            "api_submitted": False,
            "api_responses_received": False,
            "api_review_complete": False,
            "human_calibration_complete": False,
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
        receipt_path = temporary / "cross_provider_preparation.outputs.json"
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = prepare(
            config_path=args.config,
            source_root=args.source_root,
            output=args.output,
        )
    except (
        CrossProviderPreparationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(f"FAIL: {error}")
        return 1
    print(result["status"])
    print(f"output={args.output}")
    print(f"total_requests={result['total_request_rows']}")
    print(
        "estimated_maximum_usd="
        f"{result['cost_control']['total_maximum_usd']}"
    )
    print("api_submitted=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
