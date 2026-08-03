#!/usr/bin/env python3
"""Verify the additive D0-A-partial pivot-feedback gate ledger v4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot_feedback_gate_state.v4"
LEDGER_STATUS = (
    "VALID_G_MINUS_1A_AND_S0_PASS_D0A_STRUCTURAL_FREEZE_D0B_AUTHORIZED"
)
VERIFY_STATUS = (
    "PASS_G_MINUS_1A_S0_D0A_STRUCTURAL_FREEZE_D0B_ONLY_AUTHORIZED"
)
D0A_STATUS = (
    "PASS_INDEPENDENT_D0A_STRUCTURAL_FREEZE_D0_PARTIAL_D0B_ONLY_AUTHORIZED"
)
D0A_OUTPUT_STATUS = (
    "PASS_D0A_OUTCOME_BLIND_TWO_FIELD_FOUR_EPISODE_STRUCTURAL_FREEZE"
)
D0A_VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0A_OUTCOME_BLIND_STRUCTURAL_FREEZE_"
    "D0B_AUTHOR_PANEL_STILL_NOT_RUN"
)
EXPECTED_ORDER = [
    "G_MINUS_1A",
    "S0",
    "D0",
    "G0",
    "G0_5",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
]
FROZEN_EPISODES = [
    (17, 14430, 2003, 1710),
    (17, 10579, 2012, 1705),
    (27, 11361, 2007, 2740),
    (27, 12505, 1991, 2730),
]


class GateStateV4Error(RuntimeError):
    """Fail-closed gate-state v4 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV4Error(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be a string")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing {dotted_path}",
        )
        current = current[part]
    return current


def _read(path: Path, *, yaml_document: bool) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid document: {path}")
    try:
        value = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            if yaml_document
            else json.loads(path.read_text(encoding="utf-8"))
        )
    except (UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError) as error:
        raise GateStateV4Error(f"invalid document {path.name}: {error}") from error
    return _mapping(value, path.name)


def _validate_binding(
    raw: Mapping[str, Any],
    *,
    repo_root: Path,
    yaml_document: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item = dict(raw)
    path = _safe_path(repo_root, item.get("relative_path"), "bound evidence path")
    observed_sha = _sha256(path)
    require(observed_sha == item.get("sha256"), f"bound evidence SHA mismatch: {path.name}")
    document = _read(path, yaml_document=yaml_document)
    require(
        document.get("schema_version") == item.get("required_schema_version"),
        "bound schema mismatch",
    )
    require(document.get("status") == item.get("required_status"), "bound status mismatch")
    for dotted_path, expected in _mapping(
        item.get("required_fields", {}),
        "required fields",
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"bound field mismatch: {dotted_path}",
        )
    return document, {
        "relative_path": item["relative_path"],
        "sha256": observed_sha,
        "status": document["status"],
    }


def _summary_episodes(summary: Mapping[str, Any]) -> list[tuple[int, int, int, int]]:
    result = []
    raw_rows = summary.get("selected_pilot_episodes")
    require(isinstance(raw_rows, list), "selected episode summary is not a list")
    for raw in raw_rows:
        row = _mapping(raw, "selected episode")
        result.append(
            (
                int(row["field_id"]),
                int(row["topic_id"]),
                int(row["entry_year"]),
                int(row["subfield_id"]),
            )
        )
    return result


def _ledger_episodes(ledger: Mapping[str, Any]) -> list[tuple[int, int, int, int]]:
    frozen = _mapping(ledger.get("frozen_d0a_selection"), "frozen selection")
    raw_fields = frozen.get("fields")
    require(isinstance(raw_fields, list), "frozen fields are not a list")
    result = []
    for raw_field in raw_fields:
        field = _mapping(raw_field, "frozen field")
        field_id = int(field["field_id"])
        raw_episodes = field.get("episode_topic_years")
        require(isinstance(raw_episodes, list), "frozen episodes are not a list")
        for raw_episode in raw_episodes:
            episode = _mapping(raw_episode, "frozen episode")
            result.append(
                (
                    field_id,
                    int(episode["topic_id"]),
                    int(episode["entry_year"]),
                    int(episode["subfield_id"]),
                )
            )
    return result


def _validate_live_d0a(
    summary: Mapping[str, Any],
    *,
    repo_root: Path,
    output_root: Path | None,
    verification_report: Path | None,
) -> dict[str, Any]:
    output_binding = _mapping(summary.get("output_binding"), "output binding")
    root = (
        Path("/ssd") / str(output_binding["output_root_basename"])
        if output_root is None
        else output_root.resolve(strict=True)
    )
    require(root.is_dir() and not root.is_symlink(), "D0-A output root is invalid")
    require(root.name == output_binding.get("output_root_basename"), "D0-A output basename mismatch")
    receipt_path = root / "d0_structural_pilot_freeze.outputs.json"
    receipt_sha = _sha256(receipt_path)
    require(
        receipt_sha == output_binding.get("output_receipt_sha256"),
        "D0-A output receipt SHA mismatch",
    )
    receipt = _read(receipt_path, yaml_document=False)
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0_structural_pilot_freeze_outputs.v1",
        "D0-A output receipt schema mismatch",
    )
    require(receipt.get("status") == D0A_OUTPUT_STATUS, "D0-A output status mismatch")
    require(receipt.get("d0a_structural_freeze_complete") is True, "D0-A output incomplete")
    require(receipt.get("d0_complete") is False, "D0-A output improperly completes D0")
    require(
        receipt.get("d0b_outcome_blind_author_panel_authorized") is True,
        "D0-A output does not authorize bounded D0-B",
    )
    for key in ("g0_authorized", "model_fitting_authorized", "g4_open", "gpu_used"):
        require(receipt.get(key) is False, f"unsafe D0-A output flag: {key}")

    report_path = (
        Path("/ssd") / str(output_binding["verification_report_basename"])
        if verification_report is None
        else verification_report.resolve(strict=True)
    )
    require(report_path.is_file() and not report_path.is_symlink(), "D0-A verification report invalid")
    require(
        report_path.name == output_binding.get("verification_report_basename"),
        "D0-A verification basename mismatch",
    )
    verification_sha = _sha256(report_path)
    require(
        verification_sha == output_binding.get("verification_report_sha256"),
        "D0-A verification report SHA mismatch",
    )
    verification = _read(report_path, yaml_document=False)
    require(
        verification.get("schema_version")
        == "pivot_feedback_d0_structural_pilot_freeze_verification.v1",
        "D0-A verification schema mismatch",
    )
    require(
        verification.get("status") == D0A_VERIFY_STATUS,
        "D0-A verification status mismatch",
    )
    require(
        verification.get("output_receipt_sha256") == receipt_sha,
        "D0-A verification receipt binding mismatch",
    )
    require(verification.get("d0_complete") is False, "verification improperly completes D0")
    require(verification.get("outcomes_read") is False, "verification read outcomes")
    require(
        verification.get("d0b_outcome_blind_author_panel_authorized") is True,
        "verification does not authorize bounded D0-B",
    )
    for key in (
        "d0_discovery_outcome_opening_authorized",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
        "gpu_used",
    ):
        require(verification.get(key) is False, f"unsafe D0-A verification flag: {key}")
    require(
        verification.get("scientific_effect") == "NOT_ESTABLISHED",
        "D0-A verification overstates effect",
    )

    execution = _mapping(summary.get("execution"), "D0-A execution")
    for role in ("builder", "verifier"):
        path = _safe_path(
            repo_root,
            execution.get(f"{role}_relative_path"),
            f"D0-A {role}",
        )
        require(
            _sha256(path) == execution.get(f"{role}_sha256"),
            f"D0-A {role} SHA mismatch",
        )
    config_path = _safe_path(
        repo_root,
        execution.get("config_relative_path"),
        "D0-A config",
    )
    require(_sha256(config_path) == execution.get("config_sha256"), "D0-A config SHA mismatch")
    require(
        verification.get("builder_sha256") == execution.get("builder_sha256"),
        "verification builder binding mismatch",
    )
    require(
        verification.get("verifier_sha256") == execution.get("verifier_sha256"),
        "verification verifier binding mismatch",
    )
    require(
        [
            (
                int(row["field_id"]),
                int(row["topic_id"]),
                int(row["entry_year"]),
            )
            for row in verification["selected_episodes"]
        ]
        == [(field, topic, year) for field, topic, year, _ in FROZEN_EPISODES],
        "verification selected episodes differ",
    )
    return {
        "output_root_basename": root.name,
        "output_receipt_sha256": receipt_sha,
        "verification_report_basename": report_path.name,
        "verification_report_sha256": verification_sha,
        "verification_status": verification["status"],
        "builder_sha256": execution["builder_sha256"],
        "verifier_sha256": execution["verifier_sha256"],
    }


def validate(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    output_root: Path | None = None,
    verification_report: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    path = path.resolve(strict=True)
    ledger = _read(path, yaml_document=True)
    require(ledger.get("schema_version") == SCHEMA, "ledger schema mismatch")
    require(ledger.get("label") == "pivot_feedback_gate_state_v4", "ledger label mismatch")
    require(
        ledger.get("mode") == "ADDITIVE_D0A_STRUCTURAL_FREEZE_GATE_LEDGER",
        "ledger mode mismatch",
    )
    require(ledger.get("status") == LEDGER_STATUS, "ledger status mismatch")

    parent_raw = _mapping(ledger.get("additive_parent"), "additive parent")
    require(parent_raw.get("mutation_allowed") is False, "v3 parent mutation is allowed")
    parent, parent_result = _validate_binding(
        parent_raw,
        repo_root=repo_root,
        yaml_document=True,
    )
    parent_states = _mapping(parent.get("gate_state"), "parent gate state")
    require(list(parent_states) == EXPECTED_ORDER, "parent gate order mismatch")
    require(
        _mapping(parent_states["D0"], "parent D0").get("state")
        == "AUTHORIZED_NOT_RUN",
        "v3 parent no longer records authorized unrun D0",
    )

    evidence_raw = _mapping(ledger.get("execution_evidence"), "execution evidence")
    require(set(evidence_raw) == {"d0a_structural_freeze"}, "execution evidence set mismatch")
    summary, summary_result = _validate_binding(
        _mapping(evidence_raw["d0a_structural_freeze"], "D0-A evidence"),
        repo_root=repo_root,
        yaml_document=False,
    )
    require(summary.get("status") == D0A_STATUS, "D0-A summary status mismatch")
    require(_summary_episodes(summary) == FROZEN_EPISODES, "summary frozen episodes differ")
    live_result = _validate_live_d0a(
        summary,
        repo_root=repo_root,
        output_root=output_root,
        verification_report=verification_report,
    )

    states = _mapping(ledger.get("gate_state"), "gate state")
    require(list(states) == EXPECTED_ORDER, "ledger gate order mismatch")
    require(
        _mapping(states["G_MINUS_1A"], "G-1A state")
        == {"state": "PASS_BOUNDED", "complete": True, "authorizes": ["S0"]},
        "G-1A state differs",
    )
    require(
        _mapping(states["S0"], "S0 state")
        == {
            "state": "PASS_THRESHOLD_FREE_INVENTORY",
            "complete": True,
            "authorizes": ["D0"],
        },
        "S0 state differs",
    )
    require(
        _mapping(states["D0"], "D0 state")
        == {
            "state": "PARTIAL_D0A_STRUCTURAL_FREEZE_PASS",
            "complete": False,
            "requires": ["S0"],
            "entry_year_end": 2012,
            "completed_components": [
                "D0A_OUTCOME_BLIND_FIELD_AND_EPISODE_FREEZE"
            ],
            "authorizes": ["D0B_OUTCOME_BLIND_AUTHOR_PANEL"],
        },
        "D0 state differs",
    )
    require(_mapping(states["G0"], "G0 state").get("state") == "PENDING", "G0 not pending")
    require(
        _mapping(states["G4"], "G4 state")
        == {"state": "SEALED_NOT_OPENED", "complete": False},
        "G4 is not sealed",
    )

    authorization = _mapping(ledger.get("authorization"), "authorization")
    require(authorization.get("d0a_structural_freeze") == "COMPLETE", "D0-A auth differs")
    require(
        authorization.get("d0b_outcome_blind_author_panel")
        == "AUTHORIZED_ENTRY_YEAR_AT_MOST_2012",
        "D0-B authorization differs",
    )
    require(
        authorization.get("d0_discovery_outcome_opening") == "NOT_YET_AUTHORIZED",
        "D0 outcomes are open",
    )
    for key in (
        "g0_measurement_freeze",
        "final_panel_materialization",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    require(
        authorization.get("scientific_effect") == "NOT_ESTABLISHED",
        "scientific effect is overstated",
    )

    frozen = _mapping(ledger.get("frozen_d0a_selection"), "frozen selection")
    require(frozen.get("field_ids") == [17, 27], "frozen field IDs differ")
    require(_ledger_episodes(ledger) == FROZEN_EPISODES, "ledger frozen episodes differ")
    require(frozen.get("selection_by_effect_size") is False, "effect-size selection allowed")
    require(frozen.get("final_g0_selection") is False, "D0-A promoted to final G0 selection")

    next_step = _mapping(ledger.get("next_required_implementation"), "next step")
    require(next_step.get("gate") == "D0B", "next gate differs")
    require(next_step.get("entry_year_end") == 2012, "D0-B cutoff differs")
    require(next_step.get("allowed_fields") == [17, 27], "D0-B fields differ")
    require(
        [tuple(row) for row in next_step["allowed_episode_topic_years"]]
        == [(field, topic, year) for field, topic, year, _ in FROZEN_EPISODES],
        "D0-B episode boundary differs",
    )
    require(
        next_step.get("entry_or_persistence_outcomes_may_be_opened") is False,
        "D0-B may open outcomes",
    )
    require(
        next_step.get("final_field_choice_by_effect_size_allowed") is False,
        "effect-size field choice is allowed",
    )
    require(
        next_step.get("confirmatory_year_changes_allowed") is False,
        "confirmatory year changes are allowed",
    )
    require(next_step.get("gpu_required") is False, "GPU is improperly required")
    require("D0 remains incomplete" in str(ledger.get("claim_boundary")), "claim boundary omits D0 partial state")
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v4",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(path),
        "additive_parent": parent_result,
        "d0a_summary": summary_result,
        "live_d0a": live_result,
        "gate_order": EXPECTED_ORDER,
        "d0a_complete": True,
        "d0_complete": False,
        "d0b_outcome_blind_author_panel_authorized": True,
        "d0_discovery_outcome_opening_authorized": False,
        "g0_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v4.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--verification-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(
            args.ledger,
            output_root=args.output_root,
            verification_report=args.verification_report,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except (GateStateV4Error, OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V4",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
