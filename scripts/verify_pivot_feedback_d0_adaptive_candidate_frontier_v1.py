#!/usr/bin/env python3
"""Independently verify the adaptive candidate-frontier result v1."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_adaptive_candidate_frontier_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-adaptive-candidate-frontier-v1-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_adaptive_candidate_frontier_config.v1"
PROOF_SCHEMA = "pivot_feedback_d0_adaptive_frontier_equivalence_proof.v1"
FRONTIER_SCHEMA = "pivot_feedback_d0_initial_structural_frontier.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_adaptive_candidate_frontier_outputs.v1"
RESULT_STATUS = (
    "PASS_ADAPTIVE_FRONTIER_STATE_MACHINE_EXHAUSTIVE_SMALL_FIXTURES_"
    "AND_INITIAL_STRUCTURE"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_ADAPTIVE_FRONTIER_STATE_MACHINE_EXHAUSTIVE_SMALL_FIXTURES_"
    "AND_INITIAL_STRUCTURE"
)
STATES = [
    "CONTENT_PASS",
    "CONTENT_FAIL",
    "CONTENT_UNRESOLVED",
    "NOT_EVALUATED",
]


class AdaptiveFrontierVerificationError(RuntimeError):
    """Fail-closed adaptive-frontier verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdaptiveFrontierVerificationError(message)


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


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise AdaptiveFrontierVerificationError(
            f"invalid YAML {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdaptiveFrontierVerificationError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _safe_repo_path(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _ssd_root(value: Any) -> Path:
    require(
        isinstance(value, str)
        and value
        and "/" not in value
        and value not in {".", ".."},
        "unsafe SSD root basename",
    )
    path = (Path("/ssd") / value).resolve(strict=True)
    require(path.parent == Path("/ssd").resolve(), "SSD root escapes /ssd")
    require(path.is_dir() and not path.is_symlink(), "invalid SSD root")
    return path


def _compatible(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    return (
        int(first["subfield_id"]) != int(second["subfield_id"])
        and abs(int(first["entry_year"]) - int(second["entry_year"])) >= 9
    )


def eager_select(
    candidates: Sequence[Mapping[str, Any]],
    states: Sequence[str],
) -> tuple[int, int] | None:
    require(len(candidates) == len(states), "candidate/state length")
    require(all(state in STATES for state in states), "unknown state")
    first: int | None = None
    for index, state in enumerate(states):
        if state == "CONTENT_PASS":
            first = index
            break
    if first is None:
        return None
    for index in range(first + 1, len(candidates)):
        if states[index] == "CONTENT_PASS" and _compatible(
            candidates[first],
            candidates[index],
        ):
            return (
                int(candidates[first]["rank"]),
                int(candidates[index]["rank"]),
            )
    return None


def issue_lazy_certificate(
    candidates: Sequence[Mapping[str, Any]],
    states: Sequence[str],
) -> tuple[int, int] | None:
    require(len(candidates) == len(states), "candidate/state length")
    require(all(state in STATES for state in states), "unknown state")
    first: int | None = None
    for index, state in enumerate(states):
        if state == "CONTENT_FAIL":
            continue
        if state == "CONTENT_PASS":
            first = index
        break
    if first is None:
        return None
    for index in range(first + 1, len(candidates)):
        if not _compatible(candidates[first], candidates[index]):
            continue
        if states[index] == "CONTENT_FAIL":
            continue
        if states[index] != "CONTENT_PASS":
            return None
        return (
            int(candidates[first]["rank"]),
            int(candidates[index]["rank"]),
        )
    return None


def _observed_for_pair(
    candidates: Sequence[Mapping[str, Any]],
    truth: Sequence[str],
    pair: tuple[int, int],
) -> tuple[list[str], list[int], list[int]]:
    first = pair[0] - 1
    second = pair[1] - 1
    observed = ["NOT_EVALUATED"] * len(candidates)
    required: list[int] = []
    skipped: list[int] = []
    for index in range(first):
        require(truth[index] == "CONTENT_FAIL", "earlier truth")
        observed[index] = "CONTENT_FAIL"
        required.append(index)
    observed[first] = "CONTENT_PASS"
    required.append(first)
    for index in range(first + 1, second):
        if _compatible(candidates[first], candidates[index]):
            require(truth[index] == "CONTENT_FAIL", "compatible truth")
            observed[index] = "CONTENT_FAIL"
            required.append(index)
        else:
            skipped.append(index)
    observed[second] = "CONTENT_PASS"
    required.append(second)
    return observed, required, skipped


def _replay_proof() -> dict[str, Any]:
    total = 0
    pairs = 0
    unresolved_checks = 0
    structural_checks = 0
    by_length: list[dict[str, int]] = []
    for length in range(2, 7):
        length_total = 0
        length_pairs = 0
        for assignment in product(range(8), repeat=length):
            candidates: list[dict[str, int]] = []
            truth: list[str] = []
            for rank, value in enumerate(assignment, start=1):
                attributes = value // 2
                candidates.append(
                    {
                        "rank": rank,
                        "subfield_id": 1 + attributes // 2,
                        "entry_year": 9 * (attributes % 2),
                    }
                )
                truth.append("CONTENT_PASS" if value % 2 else "CONTENT_FAIL")
            total += 1
            length_total += 1
            eager = eager_select(candidates, truth)
            if eager is None:
                continue
            pairs += 1
            length_pairs += 1
            observed, required, skipped = _observed_for_pair(
                candidates,
                truth,
                eager,
            )
            require(
                issue_lazy_certificate(candidates, observed) == eager,
                "eager/lazy mismatch",
            )
            for index in required:
                mutated = list(observed)
                mutated[index] = "CONTENT_UNRESOLVED"
                require(
                    issue_lazy_certificate(candidates, mutated) is None,
                    "unresolved mutation accepted",
                )
                unresolved_checks += 1
            for index in skipped:
                mutated = list(observed)
                mutated[index] = "CONTENT_PASS"
                require(
                    issue_lazy_certificate(candidates, mutated) == eager,
                    "structural skip changed pair",
                )
                structural_checks += 1
        by_length.append(
            {
                "candidate_length": length,
                "fixture_cases": length_total,
                "eager_pair_cases": length_pairs,
            }
        )
    return {
        "total_fixture_cases": total,
        "eager_pair_cases": pairs,
        "required_state_unresolved_mutation_cases": unresolved_checks,
        "structurally_incompatible_not_evaluated_cases": structural_checks,
        "by_candidate_length": by_length,
        "lazy_pair_equals_eager_pair_for_every_eager_pair_case": True,
        "every_required_state_unresolved_mutation_blocks": True,
        "structurally_incompatible_not_evaluated_rows_cannot_change_pair": True,
    }


def _validate_config(
    config_path: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, str], Path]:
    config = _read_yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status")
        == "FROZEN_ADAPTIVE_FRONTIER_PROOF_AND_INITIAL_MATERIALIZATION_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = _safe_repo_path(
        repo_root,
        parent.get("relative_path"),
        "parent",
    )
    require(_sha256(parent_path) == parent["sha256"], "parent SHA")
    parent_document = _read_yaml(parent_path)
    require(
        parent_document.get("schema_version") == parent["required_schema_version"],
        "parent schema",
    )
    require(
        parent_document.get("status") == parent["required_status"],
        "parent status",
    )
    for dotted_path, expected in _mapping(
        parent.get("required_fields"),
        "parent fields",
    ).items():
        observed = _nested(parent_document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"parent field: {dotted_path}",
        )
    inputs = _mapping(config.get("frozen_inputs"), "inputs")
    scope = _mapping(inputs.get("scope_audit"), "scope")
    scope_root = _ssd_root(scope["root_basename"])
    scope_receipt_path = scope_root / str(scope["receipt_basename"])
    require(_sha256(scope_receipt_path) == scope["receipt_sha256"], "scope SHA")
    scope_receipt = _read_json(scope_receipt_path)
    require(scope_receipt.get("status") == scope["required_status"], "scope status")
    require(
        scope_receipt["closure"]["candidate_rows"] == scope["candidate_rows"],
        "scope candidates",
    )
    require(
        scope_receipt["closure"]["distinct_candidate_topics"]
        == scope["candidate_topics"],
        "scope topics",
    )
    require(
        scope_receipt.get("complete_candidate_work_manifest_materialized") is False,
        "scope manifest",
    )
    router = _mapping(inputs.get("topic_router"), "router")
    router_path = _safe_repo_path(
        repo_root,
        router.get("relative_path"),
        "router",
    )
    require(_sha256(router_path) == router["sha256"], "router SHA")
    router_document = _read_yaml(router_path)
    require(
        sorted(int(topic) for topic in router_document["topic_cards"])
        == router["card_topic_ids"],
        "router cards",
    )
    selection = _mapping(config.get("selection_contract"), "selection")
    require(selection.get("content_states") == STATES, "states")
    require(selection.get("content_unresolved_is_pass") is False, "unresolved pass")
    require(selection.get("not_evaluated_is_pass") is False, "unevaluated pass")
    for key, value in _mapping(
        config.get("execution_boundary"),
        "execution boundary",
    ).items():
        require(value is False, f"unsafe execution flag: {key}")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "adaptive_frontier_contract",
        "exhaustive_small_fixture_proof",
        "initial_structural_frontier_materialization",
    ):
        require(authorization.get(key) == "AUTHORIZED", f"safe work closed: {key}")
    for key in (
        "raw_title_scan",
        "candidate_content_dispositions",
        "api_request_creation",
        "api_submission",
        "deterministic_reselection",
        "author_panel_rebuild",
        "pivot_classification",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(
            authorization.get(key) == "NOT_AUTHORIZED",
            f"unsafe authorization: {key}",
        )
    require(
        authorization.get("selection_certificate")
        == "NOT_AUTHORIZED_UNTIL_REQUIRED_CONTENT_DISPOSITIONS_FROZEN",
        "certificate authorization",
    )
    return (
        config,
        {
            "parent_sha256": _sha256(parent_path),
            "scope_receipt_sha256": _sha256(scope_receipt_path),
            "router_sha256": _sha256(router_path),
        },
        router_path,
    )


def _load_candidates(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Path]:
    binding = _mapping(
        config["frozen_inputs"]["candidate_table"],
        "candidate input",
    )
    root = _ssd_root(binding["root_basename"])
    path = root / str(binding["basename"])
    require(path.is_file() and not path.is_symlink(), "candidate path")
    require(path.stat().st_mode & 0o777 == 0o600, "candidate mode")
    require(_sha256(path) == binding["sha256"], "candidate SHA")
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == binding["rows"], "candidate rows")
    metadata = parquet.schema_arrow.metadata or {}
    require(
        metadata.get(b"schema_version", b"").decode() == binding["schema_version"],
        "candidate schema",
    )
    rows = pq.read_table(path).to_pylist()
    require(
        sorted({int(row["field_id"]) for row in rows})
        == binding["selected_field_ids"],
        "candidate fields",
    )
    return rows, path


def _replay_frontier(
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[pa.Table, dict[str, Any]]:
    cards = set(config["frozen_inputs"]["topic_router"]["card_topic_ids"])
    expected = _mapping(
        config.get("initial_structural_frontier"),
        "initial frontier",
    )
    output: list[dict[str, Any]] = []
    field_closure: dict[str, dict[str, Any]] = {}
    for field_id in (17, 27):
        field_rows = sorted(
            (row for row in candidates if int(row["field_id"]) == field_id),
            key=lambda row: int(row["field_candidate_rank"]),
        )
        require(
            [int(row["field_candidate_rank"]) for row in field_rows]
            == list(range(1, len(field_rows) + 1)),
            f"field rank sequence: {field_id}",
        )
        first = field_rows[0]
        second = next(row for row in field_rows[1:] if _compatible(first, row))
        frontier = field_rows[: int(second["field_candidate_rank"])]
        for row in frontier:
            rank = int(row["field_candidate_rank"])
            if rank == 1:
                role = "FIRST_CANDIDATE_REQUIRES_CONTENT"
                reason = "HIGHEST_RANKED_CANDIDATE"
                state = "CONTENT_UNRESOLVED"
            elif rank == int(second["field_candidate_rank"]):
                role = "SECOND_CANDIDATE_REQUIRES_CONTENT"
                reason = "FIRST_STRUCTURALLY_COMPATIBLE_WITH_ASSUMED_FIRST"
                state = "CONTENT_UNRESOLVED"
            else:
                require(not _compatible(first, row), "earlier compatible")
                role = "STRUCTURALLY_INCOMPATIBLE_WITH_ASSUMED_FIRST"
                reason = "SUBFIELD_OR_MINIMUM_YEAR_SEPARATION_FAILED"
                state = "NOT_EVALUATED"
            output.append(
                {
                    "field_id": field_id,
                    "field_candidate_rank": rank,
                    "topic_id": int(row["topic_id"]),
                    "topic_name": str(row["topic_name"]),
                    "subfield_id": int(row["subfield_id"]),
                    "subfield_name": str(row["subfield_name"]),
                    "entry_year": int(row["entry_year"]),
                    "frontier_role": role,
                    "structural_reason": reason,
                    "content_state": state,
                    "topic_card_available": int(row["topic_id"]) in cards,
                    "assumed_first_candidate_rank": 1,
                    "assumption_is_content_disposition": False,
                }
            )
        selected = [row for row in output if row["field_id"] == field_id]
        field_closure[str(field_id)] = {
            "frontier_rows": len(selected),
            "structurally_incompatible_rows": sum(
                row["content_state"] == "NOT_EVALUATED" for row in selected
            ),
            "content_required_rows": sum(
                row["content_state"] == "CONTENT_UNRESOLVED" for row in selected
            ),
            "first_candidate_rank": 1,
            "first_topic_id": int(first["topic_id"]),
            "first_entry_year": int(first["entry_year"]),
            "first_subfield_id": int(first["subfield_id"]),
            "first_compatible_candidate_rank": int(second["field_candidate_rank"]),
            "first_compatible_topic_id": int(second["topic_id"]),
            "first_compatible_entry_year": int(second["entry_year"]),
            "first_compatible_subfield_id": int(second["subfield_id"]),
        }
        require(
            field_closure[str(field_id)] == expected["fields"][field_id],
            f"field closure: {field_id}",
        )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
            pa.field("subfield_name", pa.string(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("frontier_role", pa.string(), nullable=False),
            pa.field("structural_reason", pa.string(), nullable=False),
            pa.field("content_state", pa.string(), nullable=False),
            pa.field("topic_card_available", pa.bool_(), nullable=False),
            pa.field("assumed_first_candidate_rank", pa.uint32(), nullable=False),
            pa.field("assumption_is_content_disposition", pa.bool_(), nullable=False),
        ],
        metadata={b"schema_version": FRONTIER_SCHEMA.encode()},
    )
    table = pa.Table.from_pylist(output, schema=schema)
    required = [row for row in output if row["content_state"] == "CONTENT_UNRESOLVED"]
    closure = {
        "fields": field_closure,
        "total_frontier_rows": len(output),
        "total_structurally_incompatible_rows": sum(
            row["content_state"] == "NOT_EVALUATED" for row in output
        ),
        "total_content_required_rows": len(required),
        "distinct_frontier_topics": len({row["topic_id"] for row in output}),
        "content_required_topic_ids": sorted({row["topic_id"] for row in required}),
        "every_content_required_topic_has_frozen_card": all(
            row["topic_card_available"] for row in required
        ),
        "initial_selection_certificate_issued": False,
    }
    for key in (
        "total_frontier_rows",
        "total_structurally_incompatible_rows",
        "total_content_required_rows",
        "distinct_frontier_topics",
        "content_required_topic_ids",
        "every_content_required_topic_has_frozen_card",
        "initial_selection_certificate_issued",
    ):
        require(closure[key] == expected[key], f"frontier closure: {key}")
    return table, closure


def validate(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    require(output_root.is_dir() and not output_root.is_symlink(), "output root")
    require(output_root.stat().st_mode & 0o777 == 0o700, "output root mode")
    config, bindings, _ = _validate_config(config_path, repo_root=repo_root)
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")
    expected_names = {
        contract["frontier_basename"],
        contract["proof_basename"],
        contract["markdown_basename"],
        contract["receipt_basename"],
    }
    require(
        {path.name for path in output_root.iterdir()} == expected_names,
        "output file set",
    )
    for path in output_root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid output: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"output mode: {path.name}")

    receipt_path = output_root / str(contract["receipt_basename"])
    receipt = _read_json(receipt_path)
    require(receipt.get("schema_version") == OUTPUT_SCHEMA, "receipt schema")
    require(receipt.get("status") == RESULT_STATUS, "receipt status")
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    producer_path = _safe_repo_path(
        repo_root,
        receipt.get("producer_relative_path"),
        "producer",
    )
    require(receipt.get("producer_sha256") == _sha256(producer_path), "producer SHA")
    require(receipt.get("repository_bindings") == bindings, "bindings")
    candidates, candidate_path = _load_candidates(config)
    require(
        receipt.get("input_evidence")
        == {"candidate_sha256": _sha256(candidate_path)},
        "input evidence",
    )

    artifacts = _mapping(receipt.get("artifacts"), "artifacts")
    require(set(artifacts) == {"frontier", "proof", "markdown"}, "artifact set")
    for raw in artifacts.values():
        artifact = _mapping(raw, "artifact")
        path = output_root / str(artifact["basename"])
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {path.name}")
        require(path.stat().st_size == artifact["bytes"], f"artifact size: {path.name}")
    require(
        receipt.get("artifact_tree_sha256")
        == hashlib.sha256(_canonical_json_bytes(artifacts)).hexdigest(),
        "artifact tree SHA",
    )

    proof = _replay_proof()
    expected_proof = _mapping(
        config.get("exhaustive_small_fixture_proof"),
        "expected proof",
    )
    for key in (
        "total_fixture_cases",
        "eager_pair_cases",
        "required_state_unresolved_mutation_cases",
        "structurally_incompatible_not_evaluated_cases",
    ):
        require(proof[key] == expected_proof[key], f"proof config: {key}")
    require(receipt.get("proof") == proof, "receipt proof")
    proof_path = output_root / str(contract["proof_basename"])
    proof_document = _read_json(proof_path)
    require(proof_document.get("schema_version") == PROOF_SCHEMA, "proof schema")
    require(proof_document.get("status") == RESULT_STATUS, "proof status")
    require(proof_document.get("proof") == proof, "proof replay")

    expected_frontier, frontier_closure = _replay_frontier(candidates, config)
    frontier_path = output_root / str(contract["frontier_basename"])
    parquet = pq.ParquetFile(frontier_path)
    metadata = parquet.schema_arrow.metadata or {}
    require(
        metadata.get(b"schema_version", b"").decode() == FRONTIER_SCHEMA,
        "frontier schema",
    )
    require(pq.read_table(frontier_path).equals(expected_frontier), "frontier replay")
    require(receipt.get("frontier") == frontier_closure, "receipt frontier")
    require(proof_document.get("frontier") == frontier_closure, "proof frontier")

    for document, label in ((receipt, "receipt"), (proof_document, "proof")):
        for key in (
            "raw_openalex_main_table_read",
            "titles_read",
            "work_ids_materialized",
            "network_accessed",
            "credentials_accessed",
            "content_dispositions_created",
            "selection_certificate_issued",
            "outcomes_read",
        ):
            require(document.get(key) is False, f"{label} unsafe flag: {key}")
        require(document.get("api_requests_created") == 0, f"{label} request creation")
        require(
            document.get("api_requests_submitted") == 0,
            f"{label} request submission",
        )
        require(document.get("api_cost_usd") == "0.000000", f"{label} API cost")
    require(
        receipt.get("deterministic_reselection_executed") is False,
        "reselection executed",
    )
    require(receipt.get("model_fitting_authorized") is False, "fitting opened")
    require(receipt.get("g4_open") is False, "G4 opened")

    return {
        "schema_version": "pivot_feedback_d0_adaptive_candidate_frontier_verification.v1",
        "status": VERIFY_STATUS,
        "config_sha256": _sha256(config_path),
        "producer_sha256": _sha256(producer_path),
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "receipt_sha256": _sha256(receipt_path),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "proof": proof,
        "frontier": frontier_closure,
        "producer_module_imported": False,
        "raw_openalex_main_table_read": False,
        "titles_read": False,
        "work_ids_materialized": False,
        "network_accessed": False,
        "credentials_accessed": False,
        "api_requests_submitted": 0,
        "api_cost_usd": "0.000000",
        "content_dispositions_created": False,
        "selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
        "outcomes_read": False,
        "model_fitting_authorized": False,
        "g4_open": False,
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    require(not path.exists(), "verification report already exists")
    require(path.parent.is_dir(), "verification report parent missing")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(result))
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args.config, args.output_root)
        if args.report is not None:
            _write_report(args.report.absolute(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        AdaptiveFrontierVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_ADAPTIVE_CANDIDATE_FRONTIER_V1_VERIFICATION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
