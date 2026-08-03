#!/usr/bin/env python3
"""Independently verify prepared OpenAI and Claude D0 review inputs."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_UP
import hashlib
import json
from pathlib import Path, PurePosixPath
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
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-prepared-20260729"
)
PREPARATION_STATUS = "PASS_D0_CROSS_PROVIDER_BATCH_REQUESTS_PREPARED_NOT_SUBMITTED"
VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0_CROSS_PROVIDER_INPUT_VERIFICATION_NOT_SUBMITTED"
)
VISIBLE_SOURCE_MAP = {
    "target_topic_name": "observed_primary_topic_name",
    "title": "title",
    "language": "language",
    "work_type": "observed_work_type",
}
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
    "API_RUN_INSTRUCTIONS.md",
    "openai_requests.jsonl",
    "claude_batch_request.json",
    "api_request_key.csv",
    "human_calibration_reviewer_1.csv",
    "human_calibration_reviewer_2.csv",
    "human_calibration_key.csv",
    "preparation_manifest.json",
    "cross_provider_preparation.outputs.json",
}


class CrossProviderVerificationError(RuntimeError):
    """Fail-closed cross-provider verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossProviderVerificationError(message)


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
        raise CrossProviderVerificationError(
            f"invalid YAML {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CrossProviderVerificationError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file() and not path.is_symlink(), f"invalid CSV: {path.name}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"missing CSV header: {path.name}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSONL: {path.name}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            rows.append(_mapping(json.loads(line), f"{path.name}:{line_number}"))
        except json.JSONDecodeError as error:
            raise CrossProviderVerificationError(
                f"invalid JSONL {path.name}:{line_number}: {error}"
            ) from error
    return rows


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
) -> dict[str, str]:
    path = _safe_repo_file(
        repo_root,
        binding.get("relative_path"),
        "gate parent",
    )
    digest = _sha256(path)
    require(digest == binding.get("sha256"), "gate parent SHA mismatch")
    document = _load_yaml(path)
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        "gate parent schema mismatch",
    )
    require(
        document.get("status") == binding.get("required_status"),
        "gate parent status mismatch",
    )
    for dotted_path, expected in _mapping(
        binding.get("required_fields", {}),
        "gate parent fields",
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"gate parent field mismatch: {dotted_path}",
        )
    return {"relative_path": binding["relative_path"], "sha256": digest}


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


def _expected_openai(
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


def _expected_claude(
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
        selected.extend(ordered[:rows_per_stratum])
    return selected


def _expected_packet(
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


def _expected_cost(
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
    openai = (
        Decimal(openai_input) * Decimal(str(openai_rates["input"]))
        + Decimal(output_tokens) * Decimal(str(openai_rates["output"]))
    ) / million
    claude = (
        Decimal(claude_input) * Decimal(str(claude_rates["input"]))
        + Decimal(output_tokens) * Decimal(str(claude_rates["output"]))
    ) / million
    quant = Decimal("0.000001")
    return {
        "openai_input_token_upper_bound": str(openai_input),
        "claude_input_token_upper_bound": str(claude_input),
        "output_token_upper_bound_per_provider": str(output_tokens),
        "openai_maximum_usd": str(openai.quantize(quant, rounding=ROUND_UP)),
        "claude_maximum_usd": str(claude.quantize(quant, rounding=ROUND_UP)),
        "total_maximum_usd": str((openai + claude).quantize(quant, rounding=ROUND_UP)),
        "authorized_maximum_usd": str(
            Decimal(str(cost["maximum_authorized_total_usd"])).quantize(
                Decimal("0.00")
            )
        ),
    }


def validate(
    *,
    config_path: Path,
    source_root: Path,
    output_root: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    require(config_path.is_relative_to(repo_root), "config outside repository")
    config = _load_yaml(config_path)
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
    parent = _validate_bound_yaml(
        _mapping(config.get("gate_parent"), "gate parent"),
        repo_root=repo_root,
    )
    predecessor = _mapping(
        config.get("superseded_openai_only_path"),
        "predecessor",
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
        predecessor.get("v2_requests_may_be_submitted_after_v3_freeze") is False,
        "v2 remains submit-authorized",
    )

    source = _mapping(config.get("source_audit"), "source")
    raw_source = source_root.absolute()
    require(not raw_source.is_symlink(), "source root symlinked")
    source_root = source_root.resolve(strict=True)
    require(source_root.name == source.get("output_root_basename"), "source differs")
    source_receipt_path = source_root / str(source["receipt_basename"])
    template_path = source_root / str(source["template_basename"])
    require(_sha256(source_receipt_path) == source.get("receipt_sha256"), "source receipt SHA differs")
    require(_sha256(template_path) == source.get("template_sha256"), "source template SHA differs")
    source_receipt = _load_json(source_receipt_path)
    require(
        source_receipt.get("schema_version")
        == source.get("required_receipt_schema_version"),
        "source receipt schema differs",
    )
    require(
        source_receipt.get("status") == source.get("required_receipt_status"),
        "source receipt status differs",
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
        require(source_receipt.get(key) is False, f"unsafe source flag: {key}")
    _, source_rows = _read_csv(template_path)
    require(len(source_rows) == 384, "source rows differ")
    require(
        len({row["openalex_id"] for row in source_rows}) == 384,
        "source IDs not unique",
    )
    for row in source_rows:
        for column in (
            "reviewer_1_label",
            "reviewer_2_label",
            "adjudicated_label",
            "review_notes",
        ):
            require(not row[column].strip(), f"source review populated: {column}")

    raw_output = output_root.absolute()
    require(not raw_output.is_symlink(), "output root symlinked")
    output_root = output_root.resolve(strict=True)
    require(
        output_root.name == DEFAULT_OUTPUT_ROOT.name,
        "output basename differs",
    )
    require(output_root.is_dir(), "output root invalid")
    require(output_root.stat().st_mode & 0o777 == 0o700, "output mode differs")
    require({path.name for path in output_root.iterdir()} == OUTPUT_FILES, "file set differs")
    for path in output_root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid output: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"file mode differs: {path.name}")

    contract = _mapping(config.get("shared_review_contract"), "contract")
    judges = {
        str(item["judge_id"]): _mapping(item, "judge")
        for item in list(config.get("provider_judges", []))
    }
    require(set(judges) == {"openai", "claude"}, "judges differ")
    require(
        judges["openai"].get("model") == "gpt-5-mini-2025-08-07",
        "OpenAI model differs",
    )
    require(
        judges["claude"].get("model") == "claude-haiku-4-5-20251001",
        "Claude model differs",
    )
    seed = str(contract["request_seed"])
    source_by_id = {row["openalex_id"]: row for row in source_rows}
    item_ids = {
        source_id: _opaque_id(seed, "api-item", source_id)
        for source_id in source_by_id
    }
    ordered_ids = sorted(source_by_id, key=lambda value: item_ids[value])

    openai_path = output_root / "openai_requests.jsonl"
    observed_openai = _read_jsonl(openai_path)
    expected_openai = [
        _expected_openai(
            item_id=item_ids[source_id],
            judge=judges["openai"],
            row=source_by_id[source_id],
            contract=contract,
        )
        for source_id in ordered_ids
    ]
    require(observed_openai == expected_openai, "OpenAI requests differ")

    claude_path = output_root / "claude_batch_request.json"
    observed_claude = _load_json(claude_path).get("requests")
    require(isinstance(observed_claude, list), "Claude requests missing")
    expected_claude = [
        _expected_claude(
            item_id=item_ids[source_id],
            judge=judges["claude"],
            row=source_by_id[source_id],
            contract=contract,
        )
        for source_id in ordered_ids
    ]
    require(observed_claude == expected_claude, "Claude requests differ")
    forbidden = set(contract["forbidden_request_fields"])
    for request in observed_openai:
        body = _mapping(request["body"], "OpenAI body")
        require("tools" not in body and "temperature" not in body, "unsafe OpenAI parameters")
        visible = json.loads(body["input"][0]["content"][0]["text"])
        require(set(visible) == set(VISIBLE_SOURCE_MAP), "OpenAI visible fields differ")
        require(not forbidden.intersection(visible), "forbidden OpenAI field")
    for request in observed_claude:
        params = _mapping(request["params"], "Claude params")
        require("tools" not in params and "temperature" not in params, "unsafe Claude parameters")
        visible = json.loads(params["messages"][0]["content"][0]["text"])
        require(set(visible) == set(VISIBLE_SOURCE_MAP), "Claude visible fields differ")
        require(not forbidden.intersection(visible), "forbidden Claude field")

    key_columns, key_rows = _read_csv(output_root / "api_request_key.csv")
    require(
        key_columns
        == [
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
        "request key columns differ",
    )
    expected_key = []
    for row_number, source_id in enumerate(ordered_ids, start=1):
        row = source_by_id[source_id]
        expected_key.append(
            {
                "api_item_id": item_ids[source_id],
                "source_row_number": str(row_number),
                "openalex_id": source_id,
                "episode_index": row["episode_index"],
                "window_code": row["window_code"],
                "target_topic_name": row["observed_primary_topic_name"],
                "title": row["title"],
                "language": row["language"],
                "work_type": row["observed_work_type"],
            }
        )
    require(key_rows == expected_key, "request key differs")

    calibration = _mapping(config.get("human_calibration"), "calibration")
    calibration_seed = str(calibration["base_sample_seed"])
    base_rows = _base_sample(
        source_rows,
        seed=calibration_seed,
        rows_per_stratum=int(
            calibration["base_rows_per_episode_window_stratum"]
        ),
    )
    require(len(base_rows) == 96, "calibration rows differ")
    require(
        len({(row["episode_index"], row["window_code"]) for row in base_rows}) == 12,
        "calibration strata differ",
    )
    packet_ids: dict[int, dict[str, str]] = {}
    packet_titles: dict[int, list[str]] = {}
    for reviewer in (1, 2):
        columns, observed = _read_csv(
            output_root / f"human_calibration_reviewer_{reviewer}.csv"
        )
        expected, ids = _expected_packet(
            base_rows,
            reviewer=reviewer,
            seed=calibration_seed,
        )
        require(columns == PACKET_COLUMNS, f"reviewer {reviewer} columns differ")
        require(observed == expected, f"reviewer {reviewer} packet differs")
        packet_ids[reviewer] = ids
        packet_titles[reviewer] = [row["title"] for row in observed]
    require(packet_titles[1] != packet_titles[2], "packet orders identical")

    calibration_columns, calibration_rows = _read_csv(
        output_root / "human_calibration_key.csv"
    )
    require(
        calibration_columns
        == [
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
        "calibration key columns differ",
    )
    expected_calibration = []
    for row in sorted(
        base_rows,
        key=lambda item: (
            int(item["episode_index"]),
            int(item["window_code"]),
            item["openalex_id"],
        ),
    ):
        source_id = row["openalex_id"]
        expected_calibration.append(
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
    require(calibration_rows == expected_calibration, "calibration key differs")

    manifest_path = output_root / "preparation_manifest.json"
    receipt_path = output_root / "cross_provider_preparation.outputs.json"
    manifest = _load_json(manifest_path)
    receipt = _load_json(receipt_path)
    require(
        manifest.get("schema_version")
        == "pivot_feedback_d0_cross_provider_manifest.v3",
        "manifest schema differs",
    )
    require(manifest.get("status") == PREPARATION_STATUS, "manifest status differs")
    require(manifest.get("config_sha256") == _sha256(config_path), "config SHA differs")
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0_cross_provider_outputs.v3",
        "receipt schema differs",
    )
    require(receipt.get("status") == PREPARATION_STATUS, "receipt status differs")
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, list), "artifacts missing")
    require(receipt.get("artifact_count") == len(artifacts) == 8, "artifact count differs")
    require(
        {str(item["relative_path"]) for item in artifacts}
        == OUTPUT_FILES - {"cross_provider_preparation.outputs.json"},
        "artifact names differ",
    )
    for raw in artifacts:
        artifact = _mapping(raw, "artifact")
        path = output_root / str(artifact["relative_path"])
        require(path.stat().st_size == artifact.get("bytes"), f"size differs: {path.name}")
        require(_sha256(path) == artifact.get("sha256"), f"SHA differs: {path.name}")
    require(
        receipt.get("artifact_tree_sha256") == _artifact_tree(artifacts),
        "artifact tree differs",
    )

    expected_cost = _expected_cost(
        openai_path=openai_path,
        claude_path=claude_path,
        request_count=384,
        max_output_tokens=int(contract["max_output_tokens"]),
        cost=_mapping(config.get("operator_cost_control"), "cost control"),
    )
    require(receipt.get("cost_control") == expected_cost, "cost estimate differs")
    require(manifest.get("cost_control") == expected_cost, "manifest cost differs")
    require(
        Decimal(expected_cost["total_maximum_usd"])
        <= Decimal(expected_cost["authorized_maximum_usd"])
        == Decimal("1.00"),
        "cost authorization exceeded",
    )
    for key in (
        "api_submitted",
        "api_responses_received",
        "api_review_complete",
        "human_calibration_complete",
        "candidate_universe_screen_complete",
        "remaining_episodes_promoted",
        "old_author_panel_current",
        "d0_complete",
        "discovery_outcome_opening_authorized",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
        "outcomes_read",
    ):
        require(receipt.get(key) is False, f"unsafe receipt flag: {key}")
    require(receipt.get("episode_3_quarantined") is True, "episode 3 opened")
    require(receipt.get("scientific_effect") == "NOT_ESTABLISHED", "effect overstated")

    return {
        "schema_version": "pivot_feedback_d0_cross_provider_verification.v3",
        "status": VERIFY_STATUS,
        "verifier_relative_path": Path(__file__).resolve().relative_to(repo_root).as_posix(),
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(config_path),
        "gate_parent": parent,
        "source_template_sha256": _sha256(template_path),
        "output_root_basename": output_root.name,
        "output_receipt_sha256": _sha256(receipt_path),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "openai_request_rows": 384,
        "claude_request_rows": 384,
        "total_request_rows": 768,
        "cross_provider_request_content_exact_match": True,
        "request_visible_fields_exact_match": True,
        "request_bodies_contain_forbidden_fields": False,
        "calibration_rows": 96,
        "calibration_strata": 12,
        "calibration_packets_exact_match": True,
        "calibration_key_exact_match": True,
        "cost_control_exact_match": True,
        "estimated_maximum_total_usd": expected_cost["total_maximum_usd"],
        "authorized_maximum_total_usd": expected_cost["authorized_maximum_usd"],
        "api_submitted": False,
        "api_responses_received": False,
        "human_calibration_complete": False,
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
            require(report.parent.is_dir(), "report parent missing")
            require(not report.exists(), "report already exists")
            with report.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
                handle.write("\n")
            report.chmod(0o600)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (
        CrossProviderVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0_CROSS_PROVIDER_VERIFICATION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
