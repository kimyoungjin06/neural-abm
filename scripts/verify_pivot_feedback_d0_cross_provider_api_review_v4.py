#!/usr/bin/env python3
"""Independently verify the v4 cross-provider custom-ID repair."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_UP
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
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
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
)
PARENT_VERIFIER = (
    REPO_ROOT / "scripts/verify_pivot_feedback_d0_cross_provider_api_review_v3.py"
)
PARENT_STATUS = (
    "PASS_INDEPENDENT_D0_CROSS_PROVIDER_INPUT_VERIFICATION_NOT_SUBMITTED"
)
PREPARATION_STATUS = (
    "PASS_D0_CROSS_PROVIDER_CUSTOM_ID_REPAIR_PREPARED_NOT_SUBMITTED"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0_CROSS_PROVIDER_CUSTOM_ID_REPAIR_NOT_SUBMITTED"
)
UNCHANGED_FILES = {
    "api_request_key.csv",
    "human_calibration_reviewer_1.csv",
    "human_calibration_reviewer_2.csv",
    "human_calibration_key.csv",
}
OUTPUT_FILES = {
    "API_RUN_INSTRUCTIONS.md",
    "openai_requests.jsonl",
    "claude_batch_request.json",
    *UNCHANGED_FILES,
    "preparation_manifest.json",
    "cross_provider_preparation.outputs.json",
}


class CrossProviderV4VerificationError(RuntimeError):
    """Fail-closed v4 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossProviderV4VerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


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
        raise CrossProviderV4VerificationError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise CrossProviderV4VerificationError(
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            rows.append(_mapping(json.loads(line), f"{path.name}:{line_number}"))
        except json.JSONDecodeError as error:
            raise CrossProviderV4VerificationError(
                f"invalid JSONL {path.name}:{line_number}: {error}"
            ) from error
    return rows


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
    parent_config: Mapping[str, Any],
) -> dict[str, str]:
    cost = _mapping(parent_config["operator_cost_control"], "cost")
    contract = _mapping(parent_config["shared_review_contract"], "contract")
    request_count = 384
    output_tokens = request_count * int(contract["max_output_tokens"])
    overhead = int(cost["input_token_safety_overhead_per_request"]) * request_count
    openai_input = openai_path.stat().st_size + overhead
    claude_input = claude_path.stat().st_size + overhead
    openai_rates = _mapping(cost["openai_batch_price_per_million"], "OpenAI rates")
    claude_rates = _mapping(
        cost["anthropic_haiku_4_5_batch_price_per_million"],
        "Claude rates",
    )
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
    parent_root: Path,
    parent_report: Path,
    output_root: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = _load_yaml(config_path)
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
    parent_binding = _mapping(config.get("contract_parent"), "parent binding")
    parent_config_path = _safe_repo_file(
        repo_root,
        parent_binding.get("relative_path"),
        "parent config",
    )
    require(_sha256(parent_config_path) == parent_binding.get("sha256"), "parent config SHA differs")
    parent_config = _load_yaml(parent_config_path)
    require(
        parent_config.get("schema_version")
        == parent_binding.get("required_schema_version"),
        "parent config schema differs",
    )

    parent_root = parent_root.resolve(strict=True)
    parent_report = parent_report.resolve(strict=True)
    live_result = subprocess.run(
        [
            sys.executable,
            str(PARENT_VERIFIER),
            "--output-root",
            str(parent_root),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    require(live_result.returncode == 0, "parent verification failed")
    parent_live = _mapping(json.loads(live_result.stdout), "parent verification")
    require(parent_live.get("status") == PARENT_STATUS, "parent status differs")
    require(_load_json(parent_report) == parent_live, "parent report differs")
    failed = _mapping(config.get("failed_attempt"), "failed attempt")
    require(
        _sha256(parent_root / "cross_provider_preparation.outputs.json")
        == failed.get("preparation_receipt_sha256"),
        "failed preparation receipt differs",
    )
    require(
        _sha256(parent_report) == failed.get("verification_report_sha256"),
        "failed verification report differs",
    )
    failure_receipt = (
        Path("/ssd")
        / str(failed["execution_root_basename"])
        / "submission_failure.outputs.json"
    )
    cancel_receipt = (
        Path("/ssd")
        / str(failed["execution_root_basename"])
        / "openai_batch_cancel.response.json"
    )
    require(_sha256(failure_receipt) == failed.get("failure_receipt_sha256"), "failure receipt differs")
    require(_sha256(cancel_receipt) == failed.get("openai_cancel_response_sha256"), "cancel receipt differs")
    require(failed.get("claude_batch_created") is False, "Claude failure overstated")
    require(failed.get("openai_cancel_requested") is True, "OpenAI cancel missing")

    raw_output = output_root.absolute()
    require(not raw_output.is_symlink(), "output root symlinked")
    output_root = output_root.resolve(strict=True)
    require(
        output_root.name == "pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729",
        "output basename differs",
    )
    require(output_root.stat().st_mode & 0o777 == 0o700, "output mode differs")
    require({path.name for path in output_root.iterdir()} == OUTPUT_FILES, "file set differs")
    for path in output_root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid file: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"file mode differs: {path.name}")

    parent_openai = _read_jsonl(parent_root / "openai_requests.jsonl")
    repaired_openai = _read_jsonl(output_root / "openai_requests.jsonl")
    require(len(parent_openai) == len(repaired_openai) == 384, "OpenAI rows differ")
    for parent, repaired in zip(parent_openai, repaired_openai, strict=True):
        parent_id = str(parent.pop("custom_id"))
        repaired_id = str(repaired.pop("custom_id"))
        require(parent_id == f"openai-{repaired_id}", "OpenAI ID repair differs")
        require(len(repaired_id) == 64, "OpenAI ID length differs")
        require(parent == repaired, "OpenAI non-ID content changed")
    require(
        len(
            {
                row["custom_id"]
                for row in _read_jsonl(output_root / "openai_requests.jsonl")
            }
        )
        == 384,
        "OpenAI IDs not unique",
    )

    parent_claude = _load_json(parent_root / "claude_batch_request.json")["requests"]
    repaired_claude = _load_json(output_root / "claude_batch_request.json")["requests"]
    require(len(parent_claude) == len(repaired_claude) == 384, "Claude rows differ")
    repaired_claude_ids = set()
    for parent_raw, repaired_raw in zip(parent_claude, repaired_claude, strict=True):
        parent = _mapping(parent_raw, "parent Claude request")
        repaired = _mapping(repaired_raw, "repaired Claude request")
        parent_id = str(parent.pop("custom_id"))
        repaired_id = str(repaired.pop("custom_id"))
        require(parent_id == f"claude-{repaired_id}", "Claude ID repair differs")
        require(len(repaired_id) == 64, "Claude ID length differs")
        require(parent == repaired, "Claude non-ID content changed")
        repaired_claude_ids.add(repaired_id)
    require(len(repaired_claude_ids) == 384, "Claude IDs not unique")
    for name in UNCHANGED_FILES:
        require(
            _sha256(parent_root / name) == _sha256(output_root / name),
            f"private artifact changed: {name}",
        )

    manifest = _load_json(output_root / "preparation_manifest.json")
    receipt_path = output_root / "cross_provider_preparation.outputs.json"
    receipt = _load_json(receipt_path)
    require(
        manifest.get("schema_version")
        == "pivot_feedback_d0_cross_provider_manifest.v4",
        "manifest schema differs",
    )
    require(manifest.get("status") == PREPARATION_STATUS, "manifest status differs")
    require(manifest.get("config_sha256") == _sha256(config_path), "config SHA differs")
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0_cross_provider_outputs.v4",
        "receipt schema differs",
    )
    require(receipt.get("status") == PREPARATION_STATUS, "receipt status differs")
    require(receipt.get("custom_id_characters") == 64, "receipt ID length differs")
    require(receipt.get("custom_ids_unique_per_provider") is True, "receipt IDs not unique")
    require(
        receipt.get("only_custom_ids_changed_from_parent_requests") is True,
        "receipt repair scope differs",
    )
    require(
        receipt.get("unchanged_private_artifacts_exact_match") is True,
        "private artifacts unverified",
    )
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, list), "artifacts missing")
    require(receipt.get("artifact_count") == len(artifacts) == 8, "artifact count differs")
    for raw in artifacts:
        artifact = _mapping(raw, "artifact")
        path = output_root / str(artifact["relative_path"])
        require(path.stat().st_size == artifact.get("bytes"), f"size differs: {path.name}")
        require(_sha256(path) == artifact.get("sha256"), f"SHA differs: {path.name}")
    require(receipt.get("artifact_tree_sha256") == _artifact_tree(artifacts), "artifact tree differs")
    expected_cost = _expected_cost(
        openai_path=output_root / "openai_requests.jsonl",
        claude_path=output_root / "claude_batch_request.json",
        parent_config=parent_config,
    )
    require(receipt.get("cost_control") == expected_cost, "cost differs")
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
        "d0_complete",
        "outcomes_read",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt.get(key) is False, f"unsafe receipt flag: {key}")
    return {
        "schema_version": "pivot_feedback_d0_cross_provider_verification.v4",
        "status": VERIFY_STATUS,
        "verifier_relative_path": Path(__file__).resolve().relative_to(repo_root).as_posix(),
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(config_path),
        "parent_verification_report_sha256": _sha256(parent_report),
        "failed_submission_receipt_sha256": _sha256(failure_receipt),
        "output_root_basename": output_root.name,
        "output_receipt_sha256": _sha256(receipt_path),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "openai_request_rows": 384,
        "claude_request_rows": 384,
        "custom_id_characters": 64,
        "custom_ids_unique_per_provider": True,
        "only_custom_ids_changed_from_parent_requests": True,
        "unchanged_private_artifacts_exact_match": True,
        "cross_provider_request_content_exact_match": True,
        "cost_control_exact_match": True,
        "estimated_maximum_total_usd": expected_cost["total_maximum_usd"],
        "authorized_maximum_total_usd": expected_cost["authorized_maximum_usd"],
        "api_submitted": False,
        "api_responses_received": False,
        "human_calibration_complete": False,
        "d0_complete": False,
        "outcomes_read": False,
        "producer_module_imported": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--parent-report", type=Path, default=DEFAULT_PARENT_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(
            config_path=args.config,
            parent_root=args.parent_root,
            parent_report=args.parent_report,
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
        return 0
    except (
        CrossProviderV4VerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0_CROSS_PROVIDER_V4_VERIFICATION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
