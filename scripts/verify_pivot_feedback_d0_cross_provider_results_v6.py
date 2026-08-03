#!/usr/bin/env python3
"""Independently verify v6 API results and blinded human-review packets."""

from __future__ import annotations

import argparse
from collections import Counter
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
    / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v6.yaml"
)
DEFAULT_V4_PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
)
DEFAULT_V4_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-execution-20260729"
)
DEFAULT_V5_PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729"
)
DEFAULT_V5_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729.verification-v5.json"
)
DEFAULT_V5_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "execution-20260729"
)
DEFAULT_TERMINAL_SNAPSHOT = (
    DEFAULT_V5_EXECUTION_ROOT
    / "status_snapshots/status-20260729T050931994847Z.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v6-results-20260729"
)
RESULT_STATUS = "PASS_VALID_CROSS_PROVIDER_RESULTS_HUMAN_REVIEW_PENDING"
VERIFY_STATUS = "PASS_INDEPENDENT_D0_CROSS_PROVIDER_RESULT_VERIFICATION"
LABELS = {"ON_TOPIC", "AMBIGUOUS", "OFF_TOPIC", "UNREVIEWABLE"}
CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}
PACKET_FIELDS = [
    "review_item_id",
    "target_topic_name",
    "title",
    "language",
    "work_type",
    "reviewer_label",
    "reviewer_notes",
]
OUTPUT_FILES = {
    "API_RESULT_SUMMARY.md",
    "api_auto_accept_candidates.csv",
    "api_cross_provider_decisions.csv",
    "api_result_summary.json",
    "cross_provider_result.outputs.json",
    "human_review_queue_key.csv",
    "human_review_reviewer_1.csv",
    "human_review_reviewer_2.csv",
    "result_manifest.json",
}


class CrossProviderResultVerificationError(RuntimeError):
    """Fail-closed result verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossProviderResultVerificationError(message)


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
        raise CrossProviderResultVerificationError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise CrossProviderResultVerificationError(
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
            raise CrossProviderResultVerificationError(
                f"invalid JSONL {path.name}:{line_number}: {error}"
            ) from error
    return rows


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, [dict(row) for row in reader]


def _decision(value: Any, label: str) -> dict[str, str]:
    result = _mapping(value, label)
    require(
        set(result) == {"confidence", "label", "rationale"},
        f"{label} keys differ",
    )
    require(result["label"] in LABELS, f"{label} label differs")
    require(result["confidence"] in CONFIDENCES, f"{label} confidence differs")
    require(
        isinstance(result["rationale"], str) and bool(result["rationale"].strip()),
        f"{label} rationale differs",
    )
    return {
        "label": str(result["label"]),
        "confidence": str(result["confidence"]),
        "rationale": str(result["rationale"]).strip(),
    }


def _parse_openai(path: Path) -> tuple[dict[str, dict[str, str]], tuple[int, int]]:
    decisions = {}
    input_tokens = 0
    output_tokens = 0
    for row in _read_jsonl(path):
        custom_id = row.get("custom_id")
        require(
            isinstance(custom_id, str)
            and len(custom_id) == 64
            and custom_id not in decisions,
            "OpenAI custom ID differs",
        )
        require(row.get("error") is None, "OpenAI provider error present")
        response = _mapping(row.get("response"), "OpenAI response")
        require(response.get("status_code") == 200, "OpenAI HTTP differs")
        body = _mapping(response.get("body"), "OpenAI body")
        require(
            body.get("status") == "completed"
            and not body.get("incomplete_details"),
            "OpenAI completion differs",
        )
        usage = _mapping(body.get("usage"), "OpenAI usage")
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        texts = [
            content.get("text")
            for item in body.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        ]
        require(len(texts) == 1 and isinstance(texts[0], str), "OpenAI text differs")
        decisions[custom_id] = _decision(
            json.loads(texts[0]),
            "OpenAI decision",
        )
    require(len(decisions) == 384, "OpenAI rows differ")
    return decisions, (input_tokens, output_tokens)


def _parse_claude(path: Path) -> tuple[dict[str, dict[str, str]], tuple[int, int]]:
    decisions = {}
    input_tokens = 0
    output_tokens = 0
    for row in _read_jsonl(path):
        custom_id = row.get("custom_id")
        require(
            isinstance(custom_id, str)
            and len(custom_id) == 64
            and custom_id not in decisions,
            "Claude custom ID differs",
        )
        result = _mapping(row.get("result"), "Claude result")
        require(result.get("type") == "succeeded", "Claude result differs")
        message = _mapping(result.get("message"), "Claude message")
        usage = _mapping(message.get("usage"), "Claude usage")
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        content = message.get("content")
        require(isinstance(content, list) and len(content) == 1, "Claude content differs")
        item = _mapping(content[0], "Claude text")
        require(
            item.get("type") == "text" and isinstance(item.get("text"), str),
            "Claude text differs",
        )
        decisions[custom_id] = _decision(
            json.loads(str(item["text"])),
            "Claude decision",
        )
    require(len(decisions) == 384, "Claude rows differ")
    return decisions, (input_tokens, output_tokens)


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


def _review_id(api_item_id: str, reviewer: int) -> str:
    return hashlib.sha256(
        f"pivot-feedback-d0-v6|reviewer-{reviewer}|{api_item_id}".encode()
    ).hexdigest()


def _money(
    tokens: tuple[int, int],
    rates: Mapping[str, Any],
) -> Decimal:
    return (
        Decimal(tokens[0]) * Decimal(str(rates["input"]))
        + Decimal(tokens[1]) * Decimal(str(rates["output"]))
    ) / Decimal(1_000_000)


def validate(
    *,
    config_path: Path,
    v4_prepared_root: Path,
    v4_execution_root: Path,
    v5_prepared_root: Path,
    v5_report: Path,
    v5_execution_root: Path,
    terminal_snapshot: Path,
    output_root: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = _load_yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_cross_provider_api_review_config.v6",
        "config schema differs",
    )
    require(
        config.get("status")
        == "FROZEN_VALID_RESULT_REDUCTION_CONTRACT_HUMAN_REVIEW_PENDING",
        "config status differs",
    )
    parent = _mapping(config.get("contract_parent"), "parent")
    parent_path = _safe_repo_file(repo_root, parent.get("relative_path"), "parent config")
    require(_sha256(parent_path) == parent.get("sha256"), "parent SHA differs")
    require(
        _load_yaml(parent_path).get("schema_version")
        == parent.get("required_schema_version"),
        "parent schema differs",
    )
    inputs = _mapping(config.get("inputs"), "inputs")
    v4_prepared_root = v4_prepared_root.resolve(strict=True)
    v4_execution_root = v4_execution_root.resolve(strict=True)
    v5_prepared_root = v5_prepared_root.resolve(strict=True)
    v5_report = v5_report.resolve(strict=True)
    v5_execution_root = v5_execution_root.resolve(strict=True)
    terminal_snapshot = terminal_snapshot.resolve(strict=True)
    require(
        v4_prepared_root.name == inputs["v4_prepared_root_basename"],
        "v4 prepared root differs",
    )
    require(
        v4_execution_root.name == inputs["v4_execution_root_basename"],
        "v4 execution root differs",
    )
    require(
        v5_prepared_root.name == inputs["v5_prepared_root_basename"],
        "v5 prepared root differs",
    )
    require(
        v5_execution_root.name == inputs["v5_execution_root_basename"],
        "v5 execution root differs",
    )
    api_key_path = v4_prepared_root / "api_request_key.csv"
    calibration_key_path = v4_prepared_root / "human_calibration_key.csv"
    v4_submission = v4_execution_root / "cross_provider_submission.outputs.json"
    claude_path = v4_execution_root / "claude_results.jsonl"
    v5_preparation = v5_prepared_root / "openai_retry_preparation.outputs.json"
    v5_submission = v5_execution_root / "openai_retry_submission.outputs.json"
    openai_path = v5_execution_root / "openai_retry_output.jsonl"
    bindings = {
        api_key_path: inputs["api_request_key_sha256"],
        calibration_key_path: inputs["human_calibration_key_sha256"],
        v4_submission: inputs["v4_submission_receipt_sha256"],
        claude_path: inputs["claude_results_sha256"],
        v5_preparation: inputs["v5_preparation_receipt_sha256"],
        v5_report: inputs["v5_verification_report_sha256"],
        v5_submission: inputs["v5_submission_receipt_sha256"],
        openai_path: inputs["openai_retry_output_sha256"],
        terminal_snapshot: inputs["terminal_snapshot_sha256"],
    }
    for path, expected in bindings.items():
        require(_sha256(path) == expected, f"input SHA differs: {path.name}")
    terminal = _load_json(terminal_snapshot)
    require(
        terminal.get("terminal") is True
        and _mapping(terminal.get("openai"), "terminal OpenAI").get("status")
        == "completed",
        "terminal status differs",
    )
    require(
        _mapping(_load_json(v4_submission)["claude"], "Claude receipt").get(
            "batch_id"
        )
        == inputs["claude_batch_id"],
        "Claude batch ID differs",
    )
    require(
        _mapping(_load_json(v5_submission)["openai"], "OpenAI receipt").get(
            "batch_id"
        )
        == inputs["openai_batch_id"],
        "OpenAI batch ID differs",
    )

    openai, openai_tokens = _parse_openai(openai_path)
    claude, claude_tokens = _parse_claude(claude_path)
    require(set(openai) == set(claude), "provider ID universes differ")
    _, source_rows = _read_csv(api_key_path)
    source_by_id = {row["api_item_id"]: row for row in source_rows}
    require(
        len(source_rows) == len(source_by_id) == 384
        and set(source_by_id) == set(openai),
        "source key universe differs",
    )
    _, calibration_rows = _read_csv(calibration_key_path)
    calibration_ids = {row["api_item_id"] for row in calibration_rows}
    require(
        len(calibration_rows) == len(calibration_ids) == 96
        and calibration_ids <= set(openai),
        "calibration universe differs",
    )

    raw_output = output_root.absolute()
    require(not raw_output.is_symlink(), "output root symlinked")
    output_root = output_root.resolve(strict=True)
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == output_contract["root_basename"], "output root differs")
    require(output_root.stat().st_mode & 0o777 == 0o700, "output mode differs")
    require(
        {path.name for path in output_root.iterdir()} == OUTPUT_FILES,
        "output file set differs",
    )
    for path in output_root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid file: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"mode differs: {path.name}")

    _, decision_rows = _read_csv(output_root / "api_cross_provider_decisions.csv")
    decision_by_id = {row["api_item_id"]: row for row in decision_rows}
    require(
        len(decision_rows) == len(decision_by_id) == 384
        and set(decision_by_id) == set(openai),
        "decision table universe differs",
    )
    escalation_ids = set()
    auto_ids = set()
    label_pairs: Counter[tuple[str, str]] = Counter()
    confidence_pairs: Counter[tuple[str, str]] = Counter()
    openai_labels: Counter[str] = Counter()
    claude_labels: Counter[str] = Counter()
    openai_confidences: Counter[str] = Counter()
    claude_confidences: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for api_item_id in openai:
        oa = openai[api_item_id]
        cl = claude[api_item_id]
        row = decision_by_id[api_item_id]
        source = source_by_id[api_item_id]
        for key, value in source.items():
            require(row.get(key) == value, f"source field differs: {key}")
        require(row["openai_label"] == oa["label"], "OpenAI label table differs")
        require(
            row["openai_confidence"] == oa["confidence"],
            "OpenAI confidence table differs",
        )
        require(
            row["openai_rationale"] == oa["rationale"],
            "OpenAI rationale table differs",
        )
        require(row["claude_label"] == cl["label"], "Claude label table differs")
        require(
            row["claude_confidence"] == cl["confidence"],
            "Claude confidence table differs",
        )
        require(
            row["claude_rationale"] == cl["rationale"],
            "Claude rationale table differs",
        )
        reasons = []
        if oa["label"] != cl["label"]:
            reasons.append("LABEL_DISAGREEMENT")
        if oa["label"] != "ON_TOPIC":
            reasons.append("OPENAI_NOT_ON_TOPIC")
        if cl["label"] != "ON_TOPIC":
            reasons.append("CLAUDE_NOT_ON_TOPIC")
        if oa["confidence"] != "HIGH":
            reasons.append("OPENAI_NOT_HIGH_CONFIDENCE")
        if cl["confidence"] != "HIGH":
            reasons.append("CLAUDE_NOT_HIGH_CONFIDENCE")
        if reasons:
            escalation_ids.add(api_item_id)
        for reason in reasons:
            reason_counts[reason] += 1
        dual_high_on_topic = (
            oa["label"] == cl["label"] == "ON_TOPIC"
            and oa["confidence"] == cl["confidence"] == "HIGH"
        )
        if dual_high_on_topic:
            auto_ids.add(api_item_id)
        require(
            row["label_agreement"] == str(oa["label"] == cl["label"]).lower(),
            "agreement flag differs",
        )
        require(
            row["dual_high_confidence_on_topic"]
            == str(dual_high_on_topic).lower(),
            "auto-accept flag differs",
        )
        require(
            row["base_calibration"]
            == str(api_item_id in calibration_ids).lower(),
            "calibration flag differs",
        )
        require(
            row["api_escalated"] == str(bool(reasons)).lower(),
            "escalation flag differs",
        )
        require(row["escalation_reasons"] == "|".join(reasons), "reasons differ")
        openai_labels[oa["label"]] += 1
        claude_labels[cl["label"]] += 1
        openai_confidences[oa["confidence"]] += 1
        claude_confidences[cl["confidence"]] += 1
        label_pairs[(oa["label"], cl["label"])] += 1
        confidence_pairs[(oa["confidence"], cl["confidence"])] += 1

    _, auto_rows = _read_csv(output_root / "api_auto_accept_candidates.csv")
    require(
        len(auto_rows) == len(auto_ids) == 197
        and {row["api_item_id"] for row in auto_rows} == auto_ids,
        "auto-accept table differs",
    )
    queue_ids = calibration_ids | escalation_ids
    _, queue_rows = _read_csv(output_root / "human_review_queue_key.csv")
    queue_by_id = {row["api_item_id"]: row for row in queue_rows}
    require(
        len(queue_rows) == len(queue_by_id) == len(queue_ids) == 233
        and set(queue_by_id) == queue_ids,
        "human queue universe differs",
    )
    for api_item_id, row in queue_by_id.items():
        source = source_by_id[api_item_id]
        require(
            row["reviewer_1_item_id"] == _review_id(api_item_id, 1)
            and row["reviewer_2_item_id"] == _review_id(api_item_id, 2),
            "review IDs differ",
        )
        for key in ("openalex_id", "episode_index", "window_code", "source_row_number"):
            require(row[key] == source[key], f"queue source differs: {key}")
        require(
            row["base_calibration"] == str(api_item_id in calibration_ids).lower(),
            "queue calibration flag differs",
        )
        require(
            row["api_escalated"] == str(api_item_id in escalation_ids).lower(),
            "queue escalation flag differs",
        )
    for reviewer in (1, 2):
        fields, packet_rows = _read_csv(
            output_root / f"human_review_reviewer_{reviewer}.csv"
        )
        require(fields == PACKET_FIELDS, "blinded packet fields differ")
        expected_by_review_id = {
            _review_id(api_item_id, reviewer): source_by_id[api_item_id]
            for api_item_id in queue_ids
        }
        packet_by_id = {row["review_item_id"]: row for row in packet_rows}
        require(
            len(packet_rows) == len(packet_by_id) == 233
            and set(packet_by_id) == set(expected_by_review_id),
            "blinded packet universe differs",
        )
        for review_id, row in packet_by_id.items():
            source = expected_by_review_id[review_id]
            for key in ("target_topic_name", "title", "language", "work_type"):
                require(row[key] == source[key], f"packet source differs: {key}")
            require(
                row["reviewer_label"] == "" and row["reviewer_notes"] == "",
                "human packet already populated",
            )

    agreement = sum(
        count for (left, right), count in label_pairs.items() if left == right
    )
    reduction = {
        "label_agreement_rows": agreement,
        "label_disagreement_rows": 384 - agreement,
        "api_escalation_rows": len(escalation_ids),
        "base_calibration_rows": len(calibration_ids),
        "calibration_escalation_overlap_rows": len(calibration_ids & escalation_ids),
        "blinded_human_queue_rows": len(queue_ids),
        "dual_high_confidence_on_topic_rows": len(auto_ids),
    }
    require(
        reduction == _mapping(config["expected_reduction"], "expected reduction"),
        "reduction differs",
    )
    cost = _mapping(config["cost_control"], "cost")
    openai_cost = _money(
        openai_tokens,
        _mapping(cost["openai_batch_price_per_million"], "OpenAI rates"),
    )
    claude_cost = _money(
        claude_tokens,
        _mapping(
            cost["claude_haiku_4_5_batch_price_per_million"],
            "Claude rates",
        ),
    )
    v4_actual = Decimal(str(cost["v4_actual_total_cost_usd"]))
    quant = Decimal("0.000001")
    cost_summary = {
        "openai_retry_input_tokens": openai_tokens[0],
        "openai_retry_output_tokens": openai_tokens[1],
        "openai_retry_actual_batch_cost_usd": str(
            openai_cost.quantize(quant, rounding=ROUND_UP)
        ),
        "claude_input_tokens": claude_tokens[0],
        "claude_output_tokens": claude_tokens[1],
        "claude_actual_batch_cost_usd": str(
            claude_cost.quantize(quant, rounding=ROUND_UP)
        ),
        "v4_actual_total_cost_usd": str(v4_actual.quantize(quant)),
        "cumulative_actual_cost_usd": str(
            (v4_actual + openai_cost).quantize(quant, rounding=ROUND_UP)
        ),
        "cumulative_authorized_maximum_usd": str(
            Decimal(str(cost["cumulative_authorized_maximum_usd"])).quantize(
                Decimal("0.00")
            )
        ),
    }
    require(
        Decimal(cost_summary["cumulative_actual_cost_usd"])
        <= Decimal(cost_summary["cumulative_authorized_maximum_usd"]),
        "cost exceeds authorization",
    )
    summary = _load_json(output_root / "api_result_summary.json")
    require(summary.get("status") == RESULT_STATUS, "summary status differs")
    require(summary.get("reduction") == reduction, "summary reduction differs")
    require(summary.get("cost_control") == cost_summary, "summary cost differs")
    require(
        summary.get("openai_label_counts") == dict(sorted(openai_labels.items()))
        and summary.get("claude_label_counts") == dict(sorted(claude_labels.items())),
        "summary label counts differ",
    )
    require(
        summary.get("openai_confidence_counts")
        == dict(sorted(openai_confidences.items()))
        and summary.get("claude_confidence_counts")
        == dict(sorted(claude_confidences.items())),
        "summary confidence counts differ",
    )
    require(
        summary.get("label_pair_counts")
        == {
            f"{left}|{right}": count
            for (left, right), count in sorted(label_pairs.items())
        },
        "summary label pairs differ",
    )
    require(
        summary.get("confidence_pair_counts")
        == {
            f"{left}|{right}": count
            for (left, right), count in sorted(confidence_pairs.items())
        },
        "summary confidence pairs differ",
    )
    require(
        summary.get("escalation_reason_counts")
        == dict(sorted(reason_counts.items())),
        "summary reason counts differ",
    )

    manifest = _load_json(output_root / "result_manifest.json")
    receipt_path = output_root / "cross_provider_result.outputs.json"
    receipt = _load_json(receipt_path)
    require(
        manifest.get("schema_version")
        == "pivot_feedback_d0_cross_provider_result_manifest.v6"
        and manifest.get("status") == RESULT_STATUS,
        "manifest differs",
    )
    require(manifest.get("config_sha256") == _sha256(config_path), "manifest config differs")
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0_cross_provider_result_outputs.v6"
        and receipt.get("status") == RESULT_STATUS,
        "receipt differs",
    )
    require(receipt.get("reduction") == reduction, "receipt reduction differs")
    require(receipt.get("cost_control") == cost_summary, "receipt cost differs")
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, list), "artifacts missing")
    require(receipt.get("artifact_count") == len(artifacts) == 8, "artifact count differs")
    for raw in artifacts:
        artifact = _mapping(raw, "artifact")
        path = output_root / str(artifact["relative_path"])
        require(path.stat().st_size == artifact.get("bytes"), f"size differs: {path.name}")
        require(_sha256(path) == artifact.get("sha256"), f"SHA differs: {path.name}")
    require(
        receipt.get("artifact_tree_sha256") == _artifact_tree(artifacts),
        "artifact tree differs",
    )
    require(
        receipt.get("human_packets_blinded_from_api_outputs") is True,
        "packet blinding differs",
    )
    for key in (
        "human_calibration_complete",
        "human_adjudication_complete",
        "candidate_universe_screen_complete",
        "d0_complete",
        "outcomes_read",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt.get(key) is False, f"unsafe receipt flag: {key}")
    return {
        "schema_version": "pivot_feedback_d0_cross_provider_result_verification.v6",
        "status": VERIFY_STATUS,
        "verifier_relative_path": Path(__file__).resolve().relative_to(repo_root).as_posix(),
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(config_path),
        "output_root_basename": output_root.name,
        "output_receipt_sha256": _sha256(receipt_path),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "openai_valid_structured_rows": 384,
        "claude_valid_structured_rows": 384,
        "shared_expected_custom_ids": 384,
        "label_agreement_rows": agreement,
        "label_disagreement_rows": 384 - agreement,
        "dual_high_confidence_on_topic_rows": len(auto_ids),
        "blinded_human_queue_rows": len(queue_ids),
        "human_packets_blinded_from_api_outputs": True,
        "cost_control_exact_match": True,
        "cumulative_actual_cost_usd": cost_summary["cumulative_actual_cost_usd"],
        "authorized_maximum_total_usd": cost_summary[
            "cumulative_authorized_maximum_usd"
        ],
        "human_calibration_complete": False,
        "d0_complete": False,
        "outcomes_read": False,
        "producer_module_imported": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--v4-prepared-root",
        type=Path,
        default=DEFAULT_V4_PREPARED_ROOT,
    )
    parser.add_argument(
        "--v4-execution-root",
        type=Path,
        default=DEFAULT_V4_EXECUTION_ROOT,
    )
    parser.add_argument(
        "--v5-prepared-root",
        type=Path,
        default=DEFAULT_V5_PREPARED_ROOT,
    )
    parser.add_argument("--v5-report", type=Path, default=DEFAULT_V5_REPORT)
    parser.add_argument(
        "--v5-execution-root",
        type=Path,
        default=DEFAULT_V5_EXECUTION_ROOT,
    )
    parser.add_argument(
        "--terminal-snapshot",
        type=Path,
        default=DEFAULT_TERMINAL_SNAPSHOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(
            config_path=args.config,
            v4_prepared_root=args.v4_prepared_root,
            v4_execution_root=args.v4_execution_root,
            v5_prepared_root=args.v5_prepared_root,
            v5_report=args.v5_report,
            v5_execution_root=args.v5_execution_root,
            terminal_snapshot=args.terminal_snapshot,
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
        CrossProviderResultVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0_CROSS_PROVIDER_RESULT_V6_VERIFICATION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
