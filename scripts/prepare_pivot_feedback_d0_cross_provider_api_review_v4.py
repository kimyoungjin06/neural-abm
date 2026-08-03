#!/usr/bin/env python3
"""Repair cross-provider custom IDs without changing review content."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_UP
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v4.yaml"
)
DEFAULT_PARENT_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-prepared-20260729"
)
DEFAULT_PARENT_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-prepared-20260729"
    ".verification-v3.json"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
)
PARENT_VERIFIER = (
    REPO_ROOT / "scripts/verify_pivot_feedback_d0_cross_provider_api_review_v3.py"
)
PARENT_STATUS = (
    "PASS_INDEPENDENT_D0_CROSS_PROVIDER_INPUT_VERIFICATION_NOT_SUBMITTED"
)
STATUS = "PASS_D0_CROSS_PROVIDER_CUSTOM_ID_REPAIR_PREPARED_NOT_SUBMITTED"
UNCHANGED_FILES = [
    "api_request_key.csv",
    "human_calibration_reviewer_1.csv",
    "human_calibration_reviewer_2.csv",
    "human_calibration_key.csv",
]


class CrossProviderV4PreparationError(RuntimeError):
    """Fail-closed v4 preparation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossProviderV4PreparationError(message)


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
        raise CrossProviderV4PreparationError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise CrossProviderV4PreparationError(
            f"invalid YAML {path.name}: {error}"
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


def _validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_cross_provider_api_review_config.v4",
        "config schema differs",
    )
    require(
        config.get("status")
        == "FROZEN_CROSS_PROVIDER_RETRY_CONTRACT_BATCH_REQUESTS_NOT_SUBMITTED",
        "config status differs",
    )
    parent = _mapping(config.get("contract_parent"), "contract parent")
    parent_path = _safe_repo_file(
        repo_root,
        parent.get("relative_path"),
        "contract parent",
    )
    require(_sha256(parent_path) == parent.get("sha256"), "parent SHA differs")
    parent_config = _load_yaml(parent_path)
    require(
        parent_config.get("schema_version") == parent.get("required_schema_version"),
        "parent schema differs",
    )
    require(
        parent_config.get("status") == parent.get("required_status"),
        "parent status differs",
    )
    failed = _mapping(config.get("failed_attempt"), "failed attempt")
    require(failed.get("claude_http_status") == 400, "failure HTTP differs")
    require(
        failed.get("claude_error_reason") == "STRING_AT_MOST_64_CHARACTERS",
        "failure reason differs",
    )
    require(failed.get("claude_batch_created") is False, "Claude batch overstated")
    require(failed.get("openai_cancel_requested") is True, "OpenAI cancel missing")
    repair = _mapping(config.get("repair"), "repair")
    require(repair.get("provider_custom_id_characters") == 64, "ID length differs")
    require(
        repair.get("provider_custom_id")
        == "API_ITEM_ID_WITHOUT_PROVIDER_PREFIX",
        "ID policy differs",
    )
    for key in (
        "source_rows_changed",
        "visible_fields_changed",
        "prompt_changed",
        "response_schema_changed",
        "models_changed",
        "human_calibration_changed",
        "cost_control_changed",
    ):
        require(repair.get(key) is False, f"repair scope expanded: {key}")
    return parent_config


def _validate_parent_package(
    *,
    parent_root: Path,
    parent_report: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    parent_root = parent_root.resolve(strict=True)
    parent_report = parent_report.resolve(strict=True)
    result = subprocess.run(
        [
            sys.executable,
            str(PARENT_VERIFIER),
            "--output-root",
            str(parent_root),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, "parent live verification failed")
    live = _mapping(json.loads(result.stdout), "parent live verification")
    require(live.get("status") == PARENT_STATUS, "parent status differs")
    require(_load_json(parent_report) == live, "parent report differs")
    failed = _mapping(config.get("failed_attempt"), "failed attempt")
    receipt_path = parent_root / "cross_provider_preparation.outputs.json"
    require(
        parent_root.name == failed.get("prepared_root_basename"),
        "parent basename differs",
    )
    require(
        _sha256(receipt_path) == failed.get("preparation_receipt_sha256"),
        "parent receipt SHA differs",
    )
    require(
        _sha256(parent_report) == failed.get("verification_report_sha256"),
        "parent report SHA differs",
    )
    return _load_json(receipt_path)


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


def _cost_estimate(
    *,
    openai_path: Path,
    claude_path: Path,
    parent_config: Mapping[str, Any],
) -> dict[str, str]:
    cost = _mapping(parent_config["operator_cost_control"], "cost control")
    request_count = 384
    max_output_tokens = int(
        _mapping(
            parent_config["shared_review_contract"],
            "review contract",
        )["max_output_tokens"]
    )
    overhead = int(cost["input_token_safety_overhead_per_request"]) * request_count
    openai_input = openai_path.stat().st_size + overhead
    claude_input = claude_path.stat().st_size + overhead
    openai_rates = _mapping(cost["openai_batch_price_per_million"], "OpenAI rates")
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
    require(temporary.parent == output.parent, "temporary path not output sibling")
    require(
        temporary.name.startswith(f".{output.name}.tmp-"),
        "temporary prefix differs",
    )
    if temporary.exists():
        require(
            temporary.is_dir() and not temporary.is_symlink(),
            "temporary path invalid",
        )
        shutil.rmtree(temporary)


def prepare(
    *,
    config_path: Path,
    parent_root: Path,
    parent_report: Path,
    output: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    require(config_path.is_relative_to(repo_root), "config outside repository")
    config = _load_yaml(config_path)
    parent_config = _validate_config(config, repo_root=repo_root)
    parent_receipt = _validate_parent_package(
        parent_root=parent_root,
        parent_report=parent_report,
        config=config,
    )
    parent_root = parent_root.resolve(strict=True)
    output = output.resolve(strict=False)
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir(), "output parent missing")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    temporary.chmod(0o700)
    try:
        openai_rows = [
            _mapping(json.loads(line), "OpenAI request")
            for line in (parent_root / "openai_requests.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for row in openai_rows:
            custom_id = str(row["custom_id"])
            require(custom_id.startswith("openai-"), "OpenAI parent ID differs")
            repaired = custom_id.removeprefix("openai-")
            require(len(repaired) == 64, "OpenAI repaired ID length differs")
            row["custom_id"] = repaired
        require(len({row["custom_id"] for row in openai_rows}) == 384, "OpenAI IDs not unique")
        openai_path = temporary / "openai_requests.jsonl"
        _write_jsonl(openai_path, openai_rows)

        claude_document = _load_json(parent_root / "claude_batch_request.json")
        parent_claude_rows = list(claude_document.get("requests", []))
        require(len(parent_claude_rows) == 384, "Claude parent rows differ")
        claude_rows: list[dict[str, Any]] = []
        for raw in parent_claude_rows:
            row = _mapping(raw, "Claude request")
            custom_id = str(row["custom_id"])
            require(custom_id.startswith("claude-"), "Claude parent ID differs")
            repaired = custom_id.removeprefix("claude-")
            require(len(repaired) == 64, "Claude repaired ID length differs")
            row["custom_id"] = repaired
            claude_rows.append(row)
        require(len({row["custom_id"] for row in claude_rows}) == 384, "Claude IDs not unique")
        claude_path = temporary / "claude_batch_request.json"
        _write_json(claude_path, {"requests": claude_rows})

        copied_paths: list[Path] = []
        for name in UNCHANGED_FILES:
            source = parent_root / name
            destination = temporary / name
            shutil.copyfile(source, destination)
            destination.chmod(0o600)
            require(_sha256(destination) == _sha256(source), f"copy differs: {name}")
            copied_paths.append(destination)

        estimate = _cost_estimate(
            openai_path=openai_path,
            claude_path=claude_path,
            parent_config=parent_config,
        )
        require(
            Decimal(estimate["total_maximum_usd"])
            <= Decimal(estimate["authorized_maximum_usd"]),
            "cost estimate exceeds authorization",
        )
        instructions_path = temporary / "API_RUN_INSTRUCTIONS.md"
        instructions_path.write_text(
            "# D0 cross-provider API review v4 repaired inputs\n\n"
            "Status: `NOT_SUBMITTED`\n\n"
            "This create-only successor changes only provider `custom_id` "
            "values from prefixed 71-character strings to the unique "
            "64-character API item digest. The OpenAI and Claude request "
            "content, models, prompts, response schemas, private keys, and "
            "human calibration packets are unchanged.\n\n"
            f"Conservative maximum cost: `${estimate['total_maximum_usd']}` "
            f"under the frozen `${estimate['authorized_maximum_usd']}` cap.\n",
            encoding="utf-8",
        )
        instructions_path.chmod(0o600)

        primary = [
            _artifact(instructions_path),
            _artifact(openai_path, rows=384),
            _artifact(claude_path, rows=384),
            *[_artifact(path, rows=384 if path.name == "api_request_key.csv" else 96) for path in copied_paths],
        ]
        manifest_path = temporary / "preparation_manifest.json"
        manifest = {
            "schema_version": "pivot_feedback_d0_cross_provider_manifest.v4",
            "status": STATUS,
            "config_relative_path": config_path.relative_to(repo_root).as_posix(),
            "config_sha256": _sha256(config_path),
            "parent": {
                "root_basename": parent_root.name,
                "receipt_sha256": _sha256(
                    parent_root / "cross_provider_preparation.outputs.json"
                ),
                "verification_report_sha256": _sha256(parent_report),
                "artifact_tree_sha256": parent_receipt["artifact_tree_sha256"],
            },
            "repair": _mapping(config["repair"], "repair"),
            "request_rows_per_judge": 384,
            "total_request_rows": 768,
            "base_calibration_rows": 96,
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
            "schema_version": "pivot_feedback_d0_cross_provider_outputs.v4",
            "status": STATUS,
            "artifact_count": len(artifacts),
            "artifact_tree_sha256": _artifact_tree(artifacts),
            "artifacts": artifacts,
            "parent_artifact_tree_sha256": parent_receipt["artifact_tree_sha256"],
            "custom_id_characters": 64,
            "custom_ids_unique_per_provider": True,
            "only_custom_ids_changed_from_parent_requests": True,
            "unchanged_private_artifacts_exact_match": True,
            "judge_models": parent_receipt["judge_models"],
            "request_rows_per_judge": 384,
            "total_request_rows": 768,
            "base_calibration_rows": 96,
            "base_calibration_strata": 12,
            "request_visible_fields": parent_receipt["request_visible_fields"],
            "request_bodies_contain_forbidden_fields": False,
            "cost_control": estimate,
            "api_submitted": False,
            "api_responses_received": False,
            "api_review_complete": False,
            "human_calibration_complete": False,
            "candidate_universe_screen_complete": False,
            "episode_3_quarantined": True,
            "d0_complete": False,
            "outcomes_read": False,
            "g0_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        }
        _write_json(
            temporary / "cross_provider_preparation.outputs.json",
            receipt,
        )
        require(
            all(path.stat().st_mode & 0o777 == 0o600 for path in temporary.iterdir()),
            "file mode differs",
        )
        os.replace(temporary, output)
        temporary = None
        return receipt
    finally:
        _cleanup_temporary_directory(temporary, output=output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--parent-report", type=Path, default=DEFAULT_PARENT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare(
            config_path=args.config,
            parent_root=args.parent_root,
            parent_report=args.parent_report,
            output=args.output,
        )
        print(result["status"])
        print(f"output={args.output}")
        print(f"estimated_maximum_usd={result['cost_control']['total_maximum_usd']}")
        print("api_submitted=false")
        return 0
    except (
        CrossProviderV4PreparationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
