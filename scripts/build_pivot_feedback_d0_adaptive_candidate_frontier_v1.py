#!/usr/bin/env python3
"""Build the adaptive candidate-frontier proof and initial structure v1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import shutil
import tempfile
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
STATUS = (
    "PASS_ADAPTIVE_FRONTIER_STATE_MACHINE_EXHAUSTIVE_SMALL_FIXTURES_"
    "AND_INITIAL_STRUCTURE"
)
STATES = [
    "CONTENT_PASS",
    "CONTENT_FAIL",
    "CONTENT_UNRESOLVED",
    "NOT_EVALUATED",
]


class AdaptiveFrontierBuildError(RuntimeError):
    """Fail-closed adaptive-frontier build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdaptiveFrontierBuildError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


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
        raise AdaptiveFrontierBuildError(
            f"invalid YAML {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdaptiveFrontierBuildError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


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


def compatible(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    minimum_year_separation: int = 9,
) -> bool:
    return (
        int(first["subfield_id"]) != int(second["subfield_id"])
        and abs(int(first["entry_year"]) - int(second["entry_year"]))
        >= minimum_year_separation
    )


def eager_select(
    candidates: Sequence[Mapping[str, Any]],
    states: Sequence[str],
) -> tuple[int, int] | None:
    require(len(candidates) == len(states), "candidate/state length mismatch")
    require(all(state in STATES for state in states), "unknown content state")
    first_index = next(
        (index for index, state in enumerate(states) if state == "CONTENT_PASS"),
        None,
    )
    if first_index is None:
        return None
    second_index = next(
        (
            index
            for index in range(first_index + 1, len(candidates))
            if states[index] == "CONTENT_PASS"
            and compatible(candidates[first_index], candidates[index])
        ),
        None,
    )
    if second_index is None:
        return None
    return (
        int(candidates[first_index]["rank"]),
        int(candidates[second_index]["rank"]),
    )


def issue_lazy_certificate(
    candidates: Sequence[Mapping[str, Any]],
    states: Sequence[str],
) -> tuple[int, int] | None:
    require(len(candidates) == len(states), "candidate/state length mismatch")
    require(all(state in STATES for state in states), "unknown content state")
    first_index: int | None = None
    for index, state in enumerate(states):
        if state == "CONTENT_FAIL":
            continue
        if state == "CONTENT_PASS":
            first_index = index
        break
    if first_index is None:
        return None
    first = candidates[first_index]
    for index in range(first_index + 1, len(candidates)):
        if not compatible(first, candidates[index]):
            continue
        state = states[index]
        if state == "CONTENT_FAIL":
            continue
        if state != "CONTENT_PASS":
            return None
        return (int(first["rank"]), int(candidates[index]["rank"]))
    return None


def _minimal_observed_states(
    candidates: Sequence[Mapping[str, Any]],
    truth: Sequence[str],
    pair: tuple[int, int],
) -> tuple[list[str], list[int], list[int]]:
    first_index = pair[0] - 1
    second_index = pair[1] - 1
    observed = ["NOT_EVALUATED"] * len(candidates)
    required: list[int] = []
    skipped: list[int] = []
    for index in range(first_index):
        require(truth[index] == "CONTENT_FAIL", "earlier truth is not fail")
        observed[index] = "CONTENT_FAIL"
        required.append(index)
    observed[first_index] = "CONTENT_PASS"
    required.append(first_index)
    for index in range(first_index + 1, second_index):
        if compatible(candidates[first_index], candidates[index]):
            require(truth[index] == "CONTENT_FAIL", "compatible truth is not fail")
            observed[index] = "CONTENT_FAIL"
            required.append(index)
        else:
            skipped.append(index)
    observed[second_index] = "CONTENT_PASS"
    required.append(second_index)
    return observed, required, skipped


def exhaustive_fixture_proof() -> dict[str, Any]:
    total = 0
    eager_pair_cases = 0
    unresolved_mutations = 0
    structural_skip_checks = 0
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
            eager_pair_cases += 1
            length_pairs += 1
            observed, required, skipped = _minimal_observed_states(
                candidates,
                truth,
                eager,
            )
            require(
                issue_lazy_certificate(candidates, observed) == eager,
                "lazy/eager pair mismatch",
            )
            for index in required:
                mutated = list(observed)
                mutated[index] = "CONTENT_UNRESOLVED"
                require(
                    issue_lazy_certificate(candidates, mutated) is None,
                    "unresolved required state did not block",
                )
                unresolved_mutations += 1
            for index in skipped:
                mutated = list(observed)
                mutated[index] = "CONTENT_PASS"
                require(
                    issue_lazy_certificate(candidates, mutated) == eager,
                    "structurally incompatible skip changed pair",
                )
                structural_skip_checks += 1
        by_length.append(
            {
                "candidate_length": length,
                "fixture_cases": length_total,
                "eager_pair_cases": length_pairs,
            }
        )
    return {
        "total_fixture_cases": total,
        "eager_pair_cases": eager_pair_cases,
        "required_state_unresolved_mutation_cases": unresolved_mutations,
        "structurally_incompatible_not_evaluated_cases": structural_skip_checks,
        "by_candidate_length": by_length,
        "lazy_pair_equals_eager_pair_for_every_eager_pair_case": True,
        "every_required_state_unresolved_mutation_blocks": True,
        "structurally_incompatible_not_evaluated_rows_cannot_change_pair": True,
    }


def _validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config = _read_yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status")
        == "FROZEN_ADAPTIVE_FRONTIER_PROOF_AND_INITIAL_MATERIALIZATION_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = (REPO_ROOT / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(REPO_ROOT), "parent escapes repository")
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
    inputs = _mapping(config.get("frozen_inputs"), "inputs")
    scope = _mapping(inputs.get("scope_audit"), "scope")
    scope_root = _ssd_root(scope["root_basename"])
    scope_receipt_path = scope_root / str(scope["receipt_basename"])
    require(_sha256(scope_receipt_path) == scope["receipt_sha256"], "scope receipt")
    scope_receipt = _read_json(scope_receipt_path)
    require(scope_receipt.get("status") == scope["required_status"], "scope status")
    require(
        scope_receipt["closure"]["candidate_rows"] == scope["candidate_rows"],
        "scope candidate rows",
    )
    require(
        scope_receipt["closure"]["distinct_candidate_topics"]
        == scope["candidate_topics"],
        "scope candidate topics",
    )
    require(
        scope_receipt.get("complete_candidate_work_manifest_materialized")
        is scope["complete_candidate_work_manifest_materialized"]
        is False,
        "scope manifest state",
    )
    router = _mapping(inputs.get("topic_router"), "router")
    router_path = (REPO_ROOT / str(router["relative_path"])).resolve(strict=True)
    require(router_path.is_relative_to(REPO_ROOT), "router escapes repository")
    require(_sha256(router_path) == router["sha256"], "router SHA")
    router_document = _read_yaml(router_path)
    require(
        sorted(int(value) for value in router_document["topic_cards"])
        == router["card_topic_ids"],
        "router topic IDs",
    )
    selection = _mapping(config.get("selection_contract"), "selection")
    require(selection.get("content_states") == STATES, "content states")
    require(selection.get("content_unresolved_is_pass") is False, "unresolved pass")
    require(selection.get("not_evaluated_is_pass") is False, "unevaluated pass")
    boundary = _mapping(config.get("execution_boundary"), "boundary")
    for key, value in boundary.items():
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
    return config, {
        "parent_sha256": _sha256(parent_path),
        "scope_receipt_sha256": _sha256(scope_receipt_path),
        "router_sha256": _sha256(router_path),
    }


def _load_candidates(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    binding = _mapping(
        config["frozen_inputs"]["candidate_table"],
        "candidate input",
    )
    root = _ssd_root(binding["root_basename"])
    path = root / str(binding["basename"])
    require(path.is_file() and not path.is_symlink(), "candidate input")
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
    return rows, {"candidate_sha256": _sha256(path)}


def _initial_frontier(
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[pa.Table, dict[str, Any]]:
    cards = set(config["frozen_inputs"]["topic_router"]["card_topic_ids"])
    expected = _mapping(
        config.get("initial_structural_frontier"),
        "initial frontier",
    )
    rows: list[dict[str, Any]] = []
    by_field: dict[str, dict[str, Any]] = {}
    for field_id in (17, 27):
        field = sorted(
            (row for row in candidates if int(row["field_id"]) == field_id),
            key=lambda row: int(row["field_candidate_rank"]),
        )
        require(
            [int(row["field_candidate_rank"]) for row in field]
            == list(range(1, len(field) + 1)),
            f"candidate rank sequence: {field_id}",
        )
        first = field[0]
        second = next(row for row in field[1:] if compatible(first, row))
        frontier = field[: int(second["field_candidate_rank"])]
        for row in frontier:
            rank = int(row["field_candidate_rank"])
            if rank == 1:
                role = "FIRST_CANDIDATE_REQUIRES_CONTENT"
                state = "CONTENT_UNRESOLVED"
                reason = "HIGHEST_RANKED_CANDIDATE"
            elif rank == int(second["field_candidate_rank"]):
                role = "SECOND_CANDIDATE_REQUIRES_CONTENT"
                state = "CONTENT_UNRESOLVED"
                reason = "FIRST_STRUCTURALLY_COMPATIBLE_WITH_ASSUMED_FIRST"
            else:
                require(not compatible(first, row), "earlier compatible row")
                role = "STRUCTURALLY_INCOMPATIBLE_WITH_ASSUMED_FIRST"
                state = "NOT_EVALUATED"
                reason = "SUBFIELD_OR_MINIMUM_YEAR_SEPARATION_FAILED"
            rows.append(
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
        field_content = [row for row in rows if row["field_id"] == field_id]
        by_field[str(field_id)] = {
            "frontier_rows": len(field_content),
            "structurally_incompatible_rows": sum(
                row["frontier_role"]
                == "STRUCTURALLY_INCOMPATIBLE_WITH_ASSUMED_FIRST"
                for row in field_content
            ),
            "content_required_rows": sum(
                row["content_state"] == "CONTENT_UNRESOLVED"
                for row in field_content
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
            by_field[str(field_id)] == expected["fields"][field_id],
            f"field frontier differs: {field_id}",
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
    table = pa.Table.from_pylist(rows, schema=schema)
    content_required = [
        row for row in rows if row["content_state"] == "CONTENT_UNRESOLVED"
    ]
    closure = {
        "fields": by_field,
        "total_frontier_rows": len(rows),
        "total_structurally_incompatible_rows": sum(
            row["content_state"] == "NOT_EVALUATED" for row in rows
        ),
        "total_content_required_rows": len(content_required),
        "distinct_frontier_topics": len({row["topic_id"] for row in rows}),
        "content_required_topic_ids": sorted(
            {row["topic_id"] for row in content_required}
        ),
        "every_content_required_topic_has_frozen_card": all(
            row["topic_card_available"] for row in content_required
        ),
        "initial_selection_certificate_issued": False,
    }
    expected_closure = {
        key: expected[key]
        for key in (
            "total_frontier_rows",
            "total_structurally_incompatible_rows",
            "total_content_required_rows",
            "distinct_frontier_topics",
            "content_required_topic_ids",
            "every_content_required_topic_has_frozen_card",
            "initial_selection_certificate_issued",
        )
    }
    require(
        {key: closure[key] for key in expected_closure} == expected_closure,
        "frontier closure differs",
    )
    return table, closure


def _write_private(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o600)


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    path.chmod(0o600)
    return {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": _sha256(path),
    }


def _markdown(proof: Mapping[str, Any], frontier: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Adaptive candidate frontier v1",
            "",
            f"Status: `{STATUS}`",
            "",
            f"- exhaustive fixture cases: {proof['total_fixture_cases']:,}",
            f"- eager-pair cases matched: {proof['eager_pair_cases']:,}",
            (
                "- unresolved required-state mutations blocked: "
                f"{proof['required_state_unresolved_mutation_cases']:,}"
            ),
            (
                "- structural skip invariance checks: "
                f"{proof['structurally_incompatible_not_evaluated_cases']:,}"
            ),
            f"- initial structural frontier rows: {frontier['total_frontier_rows']:,}",
            f"- content-required rows: {frontier['total_content_required_rows']:,}",
            (
                "- structurally incompatible rows: "
                f"{frontier['total_structurally_incompatible_rows']:,}"
            ),
            "",
            "The initial branch assumes rank 1 passes only to expose structure.",
            "That assumption is not a content disposition. Four required rows remain",
            "CONTENT_UNRESOLVED, so no selection certificate is issued.",
            "",
        ]
    )


def build(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    require(config_path.is_relative_to(REPO_ROOT), "config escapes repository")
    require(not output_root.exists(), "output root already exists")
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config, repository_bindings = _validate_config(config_path)
    candidates, input_evidence = _load_candidates(config)
    proof = exhaustive_fixture_proof()
    expected_proof = config["exhaustive_small_fixture_proof"]
    for key in (
        "total_fixture_cases",
        "eager_pair_cases",
        "required_state_unresolved_mutation_cases",
        "structurally_incompatible_not_evaluated_cases",
    ):
        require(proof[key] == expected_proof[key], f"proof closure: {key}")
    frontier, frontier_closure = _initial_frontier(candidates, config)
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    temporary.chmod(0o700)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        artifacts["frontier"] = _write_parquet(
            frontier,
            temporary / str(contract["frontier_basename"]),
        )
        proof_document = {
            "schema_version": PROOF_SCHEMA,
            "status": STATUS,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": _sha256(config_path),
            "proof": proof,
            "frontier": frontier_closure,
            "selection_certificate_issued": False,
            "content_dispositions_created": False,
            "raw_openalex_main_table_read": False,
            "titles_read": False,
            "work_ids_materialized": False,
            "network_accessed": False,
            "credentials_accessed": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "api_cost_usd": "0.000000",
            "outcomes_read": False,
        }
        proof_path = temporary / str(contract["proof_basename"])
        _write_private(proof_path, _canonical_json_bytes(proof_document))
        artifacts["proof"] = {
            "basename": proof_path.name,
            "bytes": proof_path.stat().st_size,
            "sha256": _sha256(proof_path),
        }
        markdown_path = temporary / str(contract["markdown_basename"])
        _write_private(
            markdown_path,
            _markdown(proof, frontier_closure).encode("utf-8"),
        )
        artifacts["markdown"] = {
            "basename": markdown_path.name,
            "bytes": markdown_path.stat().st_size,
            "sha256": _sha256(markdown_path),
        }
        receipt = {
            "schema_version": OUTPUT_SCHEMA,
            "status": STATUS,
            "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": _sha256(config_path),
            "producer_relative_path": Path(__file__)
            .resolve()
            .relative_to(REPO_ROOT)
            .as_posix(),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "repository_bindings": repository_bindings,
            "input_evidence": input_evidence,
            "proof": proof,
            "frontier": frontier_closure,
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "raw_openalex_main_table_read": False,
            "titles_read": False,
            "work_ids_materialized": False,
            "network_accessed": False,
            "credentials_accessed": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "api_cost_usd": "0.000000",
            "content_dispositions_created": False,
            "selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        }
        receipt_path = temporary / str(contract["receipt_basename"])
        _write_private(receipt_path, _canonical_json_bytes(receipt))
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build(args.config, args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        AdaptiveFrontierBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_ADAPTIVE_CANDIDATE_FRONTIER_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
