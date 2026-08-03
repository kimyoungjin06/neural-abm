#!/usr/bin/env python3
"""Verify the exact-four entry-source and classifier contract without row reads."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_entry_source_classifier_contract_v1.yaml"
)
CONTRACT_SCHEMA = "pivot_feedback_d0_entry_source_classifier_contract.v1"
CONTRACT_STATUS = "FROZEN_EXACT_FOUR_ENTRY_SOURCE_CLASSIFIER_CONTRACT_FIXTURES_ONLY"
VERIFY_SCHEMA = "pivot_feedback_d0_entry_source_classifier_contract_verification.v1"
VERIFY_STATUS = (
    "PASS_EXACT_FOUR_ENTRY_SOURCE_CLASSIFIER_CONTRACT_RECEIPTS_AND_FIXTURES"
)
STATES = (
    "NO_FOCAL_ENTRY",
    "NEW_TOPIC_ENTRY",
    "RETURN",
    "CONTINUATION",
)
FEATURES = (
    "pre_entry_target_topic_work_count",
    "trailing_3y_target_topic_work_count",
    "focal_confirmed_eligible_target_topic_work_count",
)
WORK_COLUMNS = (
    "work_id_numeric",
    "publication_year",
    "temporal_disposition",
    "work_type_code",
    "is_paratext",
    "is_retracted",
    "declared_authors_count",
    "primary_topic_id",
    "primary_field_id",
)
EVENT_COLUMNS = (
    "author_numeric_id",
    "work_numeric_id",
    "publication_year",
    "temporal_disposition",
    "declared_authors_count",
    "primary_field_id",
)


class EntrySourceClassifierContractError(RuntimeError):
    """Fail-closed entry-source contract validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EntrySourceClassifierContractError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid YAML path: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise EntrySourceClassifierContractError(f"invalid YAML: {error}") from error
    return _mapping(value, "YAML")


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSON path: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EntrySourceClassifierContractError(f"invalid JSON: {error}") from error
    return _mapping(value, "JSON")


def _safe_repo_path(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _lookup_dotted(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing bound field: {dotted}",
        )
        current = current[part]
    return current


def _validate_repository_document(
    spec_value: Any,
    *,
    repo_root: Path,
    label: str,
) -> dict[str, Any]:
    spec = _mapping(spec_value, label)
    path = _safe_repo_path(repo_root, spec.get("relative_path"), label)
    require(_sha256(path) == spec.get("sha256"), f"{label} SHA-256")
    document = _read_yaml(path) if path.suffix in {".yaml", ".yml"} else _read_json(path)
    require(
        document.get("schema_version") == spec.get("required_schema_version"),
        f"{label} schema",
    )
    if "required_status" in spec:
        status_key = str(spec.get("required_status_key", "status"))
        require(
            document.get(status_key) == spec.get("required_status"),
            f"{label} status",
        )
    return {
        "relative_path": spec["relative_path"],
        "sha256": spec["sha256"],
        "schema_version": document["schema_version"],
    }


def _artifact_metadata(receipt: Mapping[str, Any], basename: str) -> dict[str, Any]:
    artifacts = receipt.get("artifacts")
    if isinstance(artifacts, Mapping):
        if basename in artifacts:
            return _mapping(artifacts[basename], f"artifact {basename}")
        for raw in artifacts.values():
            item = _mapping(raw, f"artifact {basename}")
            if item.get("basename") == basename:
                return item
    if isinstance(artifacts, list):
        for raw in artifacts:
            item = _mapping(raw, f"artifact {basename}")
            if item.get("relative_path") == basename:
                return item
    raise EntrySourceClassifierContractError(f"artifact not bound by receipt: {basename}")


def _validate_operator_receipt(
    spec_value: Any,
    *,
    operator_base: Path,
    label: str,
) -> dict[str, Any]:
    spec = _mapping(spec_value, label)
    root = operator_base / str(spec.get("root_basename"))
    require(root.is_dir() and not root.is_symlink(), f"{label} root")
    require((root.stat().st_mode & 0o777) == 0o700, f"{label} root mode")
    receipt_path = root / str(spec.get("receipt_basename"))
    require(receipt_path.is_file() and not receipt_path.is_symlink(), f"{label} receipt")
    require(_sha256(receipt_path) == spec.get("receipt_sha256"), f"{label} receipt SHA")
    receipt = _read_json(receipt_path)
    require(
        receipt.get("schema_version") == spec.get("receipt_schema_version"),
        f"{label} receipt schema",
    )
    require(receipt.get("status") == spec.get("receipt_status"), f"{label} receipt status")
    if "artifact_prefix" in spec:
        artifacts = receipt.get("artifacts")
        require(isinstance(artifacts, list), f"{label} artifact list")
        prefix = str(spec["artifact_prefix"])
        count = sum(
            str(_mapping(item, f"{label} artifact").get("relative_path", "")).startswith(
                prefix
            )
            for item in artifacts
        )
        require(count == spec.get("artifact_count"), f"{label} prefix artifact count")
    for basename_key, sha_key in (
        ("artifact_basename", "artifact_sha256"),
        ("certificate_basename", "certificate_sha256"),
    ):
        if basename_key in spec:
            item = _artifact_metadata(receipt, str(spec[basename_key]))
            require(item.get("sha256") == spec.get(sha_key), f"{label} artifact SHA")
            if "rows" in spec and "rows" in item:
                require(item.get("rows") == spec.get("rows"), f"{label} artifact rows")
    if label == "cutoff_work_metadata":
        require(receipt.get("work_metadata_rows") == spec.get("rows"), "cutoff rows")
    elif label == "predecision_author_measurement":
        require(receipt.get("measurement_rows") == spec.get("rows"), "measurement rows")
    elif label == "certified_episode_selection":
        require(receipt.get("selected_episode_rows") == spec.get("rows"), "episode rows")
    return {
        "root_basename": root.name,
        "receipt_sha256": spec["receipt_sha256"],
        "receipt_status": receipt["status"],
        "parquet_rows_opened": 0,
    }


def _classify(features: Mapping[str, int]) -> str:
    require(set(features) == set(FEATURES), "fixture feature set")
    values: dict[str, int] = {}
    for key in FEATURES:
        value = features[key]
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{key} must be nonnegative integer",
        )
        values[key] = value
    require(
        values["trailing_3y_target_topic_work_count"]
        <= values["pre_entry_target_topic_work_count"],
        "trailing count exceeds pre-entry count",
    )
    if values["focal_confirmed_eligible_target_topic_work_count"] == 0:
        return "NO_FOCAL_ENTRY"
    if values["pre_entry_target_topic_work_count"] == 0:
        return "NEW_TOPIC_ENTRY"
    if values["trailing_3y_target_topic_work_count"] == 0:
        return "RETURN"
    return "CONTINUATION"


def _work_key(row: Mapping[str, Any]) -> int:
    value = row.get("work_id_numeric", row.get("work_numeric_id"))
    require(isinstance(value, int) and value > 0, "invalid fixture work ID")
    return value


def _author_key(row: Mapping[str, Any]) -> int:
    value = row.get("author_numeric_id")
    require(isinstance(value, int) and value > 0, "invalid fixture author ID")
    return value


def _work_exclusion_reason(
    row: Mapping[str, Any],
    *,
    entry_year: int,
    field_id: int,
    topic_id: int,
    allowed_types: set[int],
) -> str | None:
    year = row.get("publication_year")
    require(isinstance(year, int), "fixture publication year")
    if year < entry_year - 3 or year > entry_year:
        return "outside_t_minus_3_through_t"
    if (
        row.get("primary_topic_id") != topic_id
        or row.get("primary_field_id") != field_id
    ):
        return "wrong_topic_or_field"
    if row.get("temporal_disposition") != 2:
        return "temporal_disposition"
    if row.get("work_type_code") not in allowed_types or row.get("is_paratext") is True:
        return "paratext_or_work_type"
    if row.get("is_retracted") is True:
        return "retracted"
    return None


def _replay_fixture(contract: Mapping[str, Any]) -> dict[str, Any]:
    fixture = _mapping(contract.get("synthetic_fixture"), "synthetic fixture")
    episode = _mapping(fixture.get("episode"), "fixture episode")
    require(episode.get("content_state") == "CONTENT_PASS", "fixture content state")
    entry_year = int(episode["entry_year"])
    field_id = int(episode["field_id"])
    topic_id = int(episode["topic_id"])
    allowed_types = set(
        _mapping(contract.get("normalized_work_eligibility"), "work eligibility")[
            "inherited_work_type_codes"
        ]
    )

    registered_raw = fixture.get("registered_authors")
    works_raw = fixture.get("work_rows")
    events_raw = fixture.get("author_event_rows")
    require(isinstance(registered_raw, list), "registered authors")
    require(isinstance(works_raw, list), "fixture works")
    require(isinstance(events_raw, list), "fixture events")
    registered: dict[int, int] = {}
    for raw in registered_raw:
        row = _mapping(raw, "registered author")
        author = _author_key(row)
        require(author not in registered, "duplicate registered author")
        prior = row.get("pre_entry_target_topic_work_count")
        require(isinstance(prior, int) and prior >= 0, "invalid pre-entry count")
        registered[author] = prior

    works: dict[int, dict[str, Any]] = {}
    reasons: Counter[str] = Counter()
    retained_null_retraction = 0
    for raw in works_raw:
        row = _mapping(raw, "fixture work")
        require(set(row) == set(WORK_COLUMNS), "fixture work columns")
        work = _work_key(row)
        require(work not in works, "duplicate fixture work")
        works[work] = row
        reason = _work_exclusion_reason(
            row,
            entry_year=entry_year,
            field_id=field_id,
            topic_id=topic_id,
            allowed_types=allowed_types,
        )
        if reason is not None:
            reasons[reason] += 1
        elif row.get("is_retracted") is None:
            retained_null_retraction += 1

    event_seen: set[tuple[int, int]] = set()
    distinct_events: list[dict[str, Any]] = []
    duplicate_excess = 0
    for raw in events_raw:
        row = _mapping(raw, "fixture author event")
        require(set(row) == set(EVENT_COLUMNS), "fixture event columns")
        work = _work_key(row)
        author = _author_key(row)
        require(work in works, "event work missing")
        pair = (work, author)
        if pair in event_seen:
            duplicate_excess += 1
            continue
        event_seen.add(pair)
        distinct_events.append(row)

    events_by_work: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in distinct_events:
        events_by_work[int(event["work_numeric_id"])].append(event)

    included_work_rows = 0
    trailing_pairs: set[tuple[int, int]] = set()
    focal_pairs: set[tuple[int, int]] = set()
    zero_token_rows = 0
    for work, row in works.items():
        if _work_exclusion_reason(
            row,
            entry_year=entry_year,
            field_id=field_id,
            topic_id=topic_id,
            allowed_types=allowed_types,
        ) is not None:
            continue
        events = events_by_work.get(work, [])
        if not events:
            zero_token_rows += 1
            continue
        included_work_rows += 1
        for event in events:
            for work_field, event_field in (
                ("publication_year", "publication_year"),
                ("temporal_disposition", "temporal_disposition"),
                ("declared_authors_count", "declared_authors_count"),
                ("primary_field_id", "primary_field_id"),
            ):
                require(
                    row[work_field] == event[event_field],
                    f"fixture work/event mismatch: {work_field}",
                )
            author = int(event["author_numeric_id"])
            if author not in registered:
                continue
            pair = (author, work)
            year = int(row["publication_year"])
            if entry_year - 3 <= year <= entry_year - 1:
                trailing_pairs.add(pair)
            elif year == entry_year:
                focal_pairs.add(pair)

    trailing_counts = Counter(author for author, _ in trailing_pairs)
    focal_counts = Counter(author for author, _ in focal_pairs)
    reconstructed: list[dict[str, Any]] = []
    for author in sorted(registered):
        features = {
            "pre_entry_target_topic_work_count": registered[author],
            "trailing_3y_target_topic_work_count": trailing_counts[author],
            "focal_confirmed_eligible_target_topic_work_count": focal_counts[author],
        }
        reconstructed.append(
            {
                "author_numeric_id": author,
                **features,
                "state": _classify(features),
            }
        )
    expected_rows = fixture.get("expected_rows")
    require(reconstructed == expected_rows, "fixture reconstructed rows")
    require({row["state"] for row in reconstructed} == set(STATES), "fixture state coverage")

    accounting = {
        "input_work_rows": len(works_raw),
        "input_author_event_rows": len(events_raw),
        "duplicate_author_work_event_excess_rows": duplicate_excess,
        "eligible_linked_target_work_rows_in_t_minus_3_through_t": included_work_rows,
        "trailing_distinct_author_work_rows": len(trailing_pairs),
        "focal_distinct_author_work_rows": len(focal_pairs),
        "excluded_outside_t_minus_3_through_t_rows": reasons[
            "outside_t_minus_3_through_t"
        ],
        "excluded_wrong_topic_or_field_rows": reasons["wrong_topic_or_field"],
        "excluded_paratext_or_work_type_rows": reasons["paratext_or_work_type"],
        "excluded_retracted_rows": reasons["retracted"],
        "excluded_temporal_disposition_rows": reasons["temporal_disposition"],
        "excluded_zero_valid_author_token_rows": zero_token_rows,
        "retained_is_retracted_null_rows": retained_null_retraction,
        "registered_author_rows": len(registered),
        "classified_author_rows": len(reconstructed),
    }
    require(accounting == fixture.get("expected_accounting"), "fixture accounting")
    return {
        "reconstructed_rows": reconstructed,
        "accounting": accounting,
        "row_mismatches": 0,
        "accounting_mismatches": 0,
    }


def _validate_semantics(contract: Mapping[str, Any]) -> dict[str, Any]:
    content = _mapping(
        contract.get("content_confirmation_resolution"),
        "content confirmation",
    )
    require(
        content.get("confirmation_unit")
        == "CERTIFIED_EPISODE_TOPIC_NOT_INDIVIDUAL_WORK",
        "content confirmation unit",
    )
    for key in (
        "individual_work_api_label_required",
        "individual_work_human_label_required",
        "new_api_request_required",
        "provider_consensus_is_ground_truth",
        "current_snapshot_primary_topic_is_semantic_ground_truth",
        "episode_content_pass_is_work_level_ground_truth",
        "historical_topic_observability_established",
    ):
        require(content.get(key) is False, f"content boundary {key}")
    require(content.get("retrospective_post_diagnostic_certificate") is True, "certificate boundary")

    work = _mapping(contract.get("normalized_work_eligibility"), "work eligibility")
    require(tuple(work.get("exact_projected_columns", ())) == WORK_COLUMNS, "work columns")
    require(work.get("inherited_work_type_codes") == [1, 2, 3, 4, 7, 16, 18, 21], "work types")
    require(work.get("allowed_temporal_dispositions") == [2, 3], "allowed dispositions")
    require(work.get("exact_four_expected_temporal_disposition") == 2, "exact disposition")
    require(work.get("team_cap_for_primary_event") == "NONE", "team cap")

    linkage = _mapping(contract.get("author_linkage"), "author linkage")
    require(tuple(linkage.get("exact_projected_columns", ())) == EVENT_COLUMNS, "event columns")
    require(linkage.get("registered_population_join") == "LEFT_JOIN_ALL_3256271_AUTHOR_EPISODE_ROWS", "population join")
    require(linkage.get("natural_person_identity_established") is False, "person identity")

    features = _mapping(contract.get("feature_construction"), "features")
    require(features.get("entry_year_is_reserved_response_offset_zero_not_attention_year") is True, "entry-year role")
    require(features.get("measurement_abi_latest_attention_year_does_not_truncate_focal_outcome_year") is True, "attention horizon leakage")
    require(features.get("post_entry_offsets_allowed") == [], "post-entry offsets")
    classification = _mapping(contract.get("classification"), "classification")
    require(classification.get("states") == list(STATES), "states")
    require(classification.get("minimum_history_threshold") == "NONE", "history threshold")
    require(classification.get("graph_team_cap") == "NONE", "graph cap")
    return {
        "content_confirmation_unit": content["confirmation_unit"],
        "individual_work_api_required": False,
        "new_api_request_required": False,
        "work_types": work["inherited_work_type_codes"],
        "exact_four_expected_temporal_disposition": 2,
        "team_cap": "NONE",
        "post_entry_offsets": [],
    }


def validate(
    path: Path = DEFAULT_CONTRACT,
    *,
    repo_root: Path = REPO_ROOT,
    operator_base: Path = Path("/ssd"),
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    operator_base = operator_base.resolve(strict=True)
    path = path.resolve(strict=True)
    contract = _read_yaml(path)
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema")
    require(
        contract.get("label")
        == "pivot_feedback_d0_entry_source_classifier_contract_v1",
        "contract label",
    )
    require(contract.get("status") == CONTRACT_STATUS, "contract status")

    parent = _mapping(contract.get("authorization_parent"), "authorization parent")
    parent_path = _safe_repo_path(repo_root, parent.get("relative_path"), "parent")
    require(_sha256(parent_path) == parent.get("sha256"), "parent SHA")
    parent_document = _read_yaml(parent_path)
    require(
        parent_document.get("schema_version") == parent.get("required_schema_version"),
        "parent schema",
    )
    require(parent_document.get("status") == parent.get("required_status"), "parent status")
    require(parent.get("mutation_allowed") is False, "parent mutation")
    for dotted, expected in _mapping(parent.get("required_fields"), "parent fields").items():
        require(_lookup_dotted(parent_document, str(dotted)) == expected, f"parent field {dotted}")

    repository = {
        label: _validate_repository_document(
            spec,
            repo_root=repo_root,
            label=label,
        )
        for label, spec in _mapping(
            contract.get("repository_bindings"),
            "repository bindings",
        ).items()
    }
    operator = {
        label: _validate_operator_receipt(
            spec,
            operator_base=operator_base,
            label=label,
        )
        for label, spec in _mapping(
            contract.get("operator_receipt_bindings"),
            "operator receipts",
        ).items()
    }
    semantics = _validate_semantics(contract)
    fixture = _replay_fixture(contract)

    verification = _mapping(contract.get("verification_contract"), "verification")
    for key in (
        "validator_must_not_import_pivot_v2_validator",
        "validator_must_not_import_future_source_builder",
        "repository_binding_hashes_required",
        "operator_receipt_hashes_and_statuses_required",
        "operator_parquet_rows_must_not_be_opened",
        "fixture_expected_rows_reconstructed_not_trusted",
        "fixture_expected_accounting_reconstructed_not_trusted",
        "all_four_states_required",
    ):
        require(verification.get(key) is True, f"verification flag {key}")

    authorization = _mapping(contract.get("authorization"), "authorization")
    require(
        authorization.get("contract_fixture_validation") == "AUTHORIZED",
        "fixture validation authorization",
    )
    for key, value in authorization.items():
        if key != "contract_fixture_validation":
            require(value == "NOT_AUTHORIZED", f"unsafe authorization: {key}")

    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "contract_sha256": _sha256(path),
        "repository_bindings": repository,
        "operator_receipt_bindings": operator,
        "operator_receipts_read": len(operator),
        "operator_parquet_files_opened": 0,
        "production_parquet_rows_read": 0,
        "semantics": semantics,
        "fixture_replay": fixture,
        "all_four_states_covered": True,
        "threshold_free": True,
        "primary_minimum_history_threshold": "NONE",
        "content_confirmation_is_topic_level_not_work_level": True,
        "new_api_request_required": False,
        "api_requests_created": 0,
        "outcomes_read": False,
        "production_source_build_authorized": False,
        "production_pivot_classification_authorized": False,
        "g0_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "claim_boundary": contract["claim_boundary"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--operator-base", type=Path, default=Path("/ssd"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args.contract, operator_base=args.operator_base)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        EntrySourceClassifierContractError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_ENTRY_SOURCE_CLASSIFIER_CONTRACT_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
