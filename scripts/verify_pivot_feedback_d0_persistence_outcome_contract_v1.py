#!/usr/bin/env python3
"""Independently validate the D0 persistence outcome contract and fixtures."""

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
    / "experiments/configs/pivot_feedback_d0_persistence_outcome_contract_v1.yaml"
)
CONTRACT_SCHEMA = "pivot_feedback_d0_persistence_outcome_contract.v1"
CONTRACT_STATUS = "FROZEN_D0_PERSISTENCE_OUTCOME_SOURCE_CONTRACT_FIXTURES_ONLY"
VERIFY_SCHEMA = "pivot_feedback_d0_persistence_outcome_contract_verification.v1"
VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0_PERSISTENCE_OUTCOME_CONTRACT_AND_FIXTURE_REPLAY_"
    "NO_PRODUCTION_ROW_READ"
)
PIVOT_STATES = (
    "NO_FOCAL_ENTRY",
    "NEW_TOPIC_ENTRY",
    "RETURN",
    "CONTINUATION",
)
OUTCOME_STATES = (
    "MATURED_NO_PERSISTENCE",
    "MATURED_PERSISTENCE",
    "RIGHT_CENSORED",
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
OUTPUT_COLUMNS = (
    "certified_episode_index",
    "field_id",
    "topic_id",
    "entry_year",
    "author_numeric_id",
    "observable_history_pivot_state",
    "persistence_followup_complete",
    "target_topic_work_count_t_plus_1",
    "target_topic_work_count_t_plus_2",
    "target_topic_work_count_t_plus_3",
    "target_topic_work_count_t_plus_1_through_t_plus_3",
    "three_year_target_topic_persistence_state",
    "three_year_target_topic_persistent",
)


class PersistenceOutcomeContractError(RuntimeError):
    """Raised when the contract fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PersistenceOutcomeContractError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid YAML path: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PersistenceOutcomeContractError(f"invalid YAML: {error}") from error
    return dict(_mapping(value, "YAML"))


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSON path: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PersistenceOutcomeContractError(f"invalid JSON: {error}") from error
    return dict(_mapping(value, "JSON"))


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
            _lookup_dotted(document, status_key) == spec.get("required_status"),
            f"{label} status",
        )
    return {
        "relative_path": spec["relative_path"],
        "sha256": spec["sha256"],
        "schema_version": document["schema_version"],
    }


def _artifact_metadata(receipt: Mapping[str, Any], basename: str) -> Mapping[str, Any]:
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, list), "receipt artifacts must be a list")
    for raw in artifacts:
        item = _mapping(raw, f"artifact {basename}")
        if item.get("relative_path") == basename:
            return item
    raise PersistenceOutcomeContractError(f"artifact not bound by receipt: {basename}")


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
        require(count == spec.get("artifact_count"), f"{label} artifact count")
    if "artifact_basename" in spec:
        item = _artifact_metadata(receipt, str(spec["artifact_basename"]))
        require(item.get("sha256") == spec.get("artifact_sha256"), f"{label} artifact SHA")
        require(item.get("rows") == spec.get("rows"), f"{label} artifact rows")
    if label == "cutoff_work_metadata":
        require(receipt.get("work_metadata_rows") == spec.get("rows"), "cutoff rows")
        require(
            receipt.get("work_metadata_tree_sha256")
            == spec.get("artifact_tree_sha256"),
            "cutoff partition tree",
        )
    if label == "production_classification":
        require(
            receipt.get("classification_rows") == spec.get("rows"),
            "classification rows",
        )
        require(
            receipt.get("classification_columns") == spec.get("columns"),
            "classification columns",
        )
        require(
            receipt.get("new_topic_entry_rows") == spec.get("primary_pivot_rows"),
            "classification primary rows",
        )
        require(
            receipt.get("artifact_tree_sha256") == spec.get("artifact_tree_sha256"),
            "classification artifact tree",
        )
    return {
        "root_basename": root.name,
        "receipt_sha256": spec["receipt_sha256"],
        "receipt_status": receipt["status"],
        "parquet_files_opened": 0,
    }


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
    allowed_dispositions: set[int],
) -> str | None:
    year = row.get("publication_year")
    if year is None:
        return "missing_publication_year"
    require(isinstance(year, int) and not isinstance(year, bool), "publication year")
    if year < entry_year + 1 or year > entry_year + 3:
        return "outside_t_plus_1_through_t_plus_3"
    if (
        row.get("primary_topic_id") != topic_id
        or row.get("primary_field_id") != field_id
    ):
        return "wrong_topic_or_field"
    if row.get("temporal_disposition") not in allowed_dispositions:
        return "temporal_disposition"
    if row.get("work_type_code") not in allowed_types or row.get("is_paratext") is True:
        return "paratext_or_work_type"
    if row.get("is_retracted") is True:
        return "retracted"
    return None


def _right_censored_rows(primary_authors: set[int]) -> list[dict[str, Any]]:
    return [
        {
            "author_numeric_id": author,
            "persistence_followup_complete": False,
            "target_topic_work_count_t_plus_1": None,
            "target_topic_work_count_t_plus_2": None,
            "target_topic_work_count_t_plus_3": None,
            "target_topic_work_count_t_plus_1_through_t_plus_3": None,
            "three_year_target_topic_persistence_state": "RIGHT_CENSORED",
            "three_year_target_topic_persistent": None,
        }
        for author in sorted(primary_authors)
    ]


def _replay_fixture(
    fixture_value: Any,
    *,
    allowed_types: set[int],
    allowed_dispositions: set[int],
) -> dict[str, Any]:
    fixture = _mapping(fixture_value, "synthetic fixture")
    episode = _mapping(fixture.get("episode"), "fixture episode")
    entry_year = episode.get("entry_year")
    followup_year = episode.get("followup_complete_through_year")
    field_id = episode.get("field_id")
    topic_id = episode.get("topic_id")
    for value, label in (
        (entry_year, "entry year"),
        (followup_year, "followup year"),
        (field_id, "field ID"),
        (topic_id, "topic ID"),
    ):
        require(isinstance(value, int) and not isinstance(value, bool), label)
    matured = entry_year + 3 <= followup_year

    classifications_raw = fixture.get("classification_rows")
    works_raw = fixture.get("work_rows")
    events_raw = fixture.get("author_event_rows")
    require(isinstance(classifications_raw, list), "fixture classifications")
    require(isinstance(works_raw, list), "fixture works")
    require(isinstance(events_raw, list), "fixture events")

    classifications: dict[int, str] = {}
    for raw in classifications_raw:
        row = _mapping(raw, "fixture classification")
        require(
            set(row) == {"author_numeric_id", "observable_history_pivot_state"},
            "fixture classification columns",
        )
        author = _author_key(row)
        state = row["observable_history_pivot_state"]
        require(state in PIVOT_STATES, "fixture pivot state")
        require(author not in classifications, "duplicate fixture classification")
        classifications[author] = str(state)
    primary_authors = {
        author for author, state in classifications.items() if state == "NEW_TOPIC_ENTRY"
    }
    require(primary_authors, "fixture primary pivots")

    zero_accounting = {
        "duplicate_author_work_event_excess_rows": 0,
        "eligible_linked_target_work_rows_in_t_plus_1_through_t_plus_3": 0,
        "registered_primary_distinct_author_work_rows": 0,
        "unregistered_or_nonprimary_distinct_author_work_rows": 0,
        "excluded_outside_t_plus_1_through_t_plus_3_rows": 0,
        "excluded_missing_publication_year_rows": 0,
        "excluded_wrong_topic_or_field_rows": 0,
        "excluded_paratext_or_work_type_rows": 0,
        "excluded_retracted_rows": 0,
        "excluded_temporal_disposition_rows": 0,
        "excluded_zero_valid_author_token_rows": 0,
        "retained_is_retracted_null_rows": 0,
    }
    if not matured:
        reconstructed = _right_censored_rows(primary_authors)
        accounting = {
            "input_classification_rows": len(classifications_raw),
            "primary_pivot_rows": len(primary_authors),
            "nonprimary_rows_excluded": len(classifications) - len(primary_authors),
            "input_work_rows": len(works_raw),
            "input_author_event_rows": len(events_raw),
            **zero_accounting,
            "matured_primary_rows": 0,
            "right_censored_primary_rows": len(primary_authors),
            "persistent_primary_rows": 0,
            "nonpersistent_primary_rows": 0,
        }
        require(reconstructed == fixture.get("expected_rows"), "fixture reconstructed rows")
        require(accounting == fixture.get("expected_accounting"), "fixture accounting")
        return {
            "fixture_id": fixture["fixture_id"],
            "matured": False,
            "reconstructed_rows": reconstructed,
            "accounting": accounting,
            "row_mismatches": 0,
            "accounting_mismatches": 0,
        }

    works: dict[int, Mapping[str, Any]] = {}
    reasons: Counter[str] = Counter()
    retained_null_retraction = 0
    eligibility: dict[int, bool] = {}
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
            allowed_dispositions=allowed_dispositions,
        )
        eligibility[work] = reason is None
        if reason is not None:
            reasons[reason] += 1
        elif row.get("is_retracted") is None:
            retained_null_retraction += 1

    event_seen: set[tuple[int, int]] = set()
    events_by_work: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    duplicate_excess = 0
    for raw in events_raw:
        row = _mapping(raw, "fixture event")
        require(set(row) == set(EVENT_COLUMNS), "fixture event columns")
        work = _work_key(row)
        author = _author_key(row)
        require(work in works, "fixture event work missing")
        pair = (work, author)
        if pair in event_seen:
            duplicate_excess += 1
            continue
        event_seen.add(pair)
        events_by_work[work].append(row)

    registered_pairs: set[tuple[int, int, int]] = set()
    other_pairs: set[tuple[int, int]] = set()
    eligible_linked_work_rows = 0
    zero_token_rows = 0
    for work, row in works.items():
        if not eligibility[work]:
            continue
        events = events_by_work.get(work, [])
        if not events:
            zero_token_rows += 1
            continue
        eligible_linked_work_rows += 1
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
            if author in primary_authors:
                offset = int(row["publication_year"]) - entry_year
                registered_pairs.add((author, work, offset))
            else:
                other_pairs.add((author, work))

    by_author_offset: Counter[tuple[int, int]] = Counter(
        (author, offset) for author, _, offset in registered_pairs
    )
    reconstructed: list[dict[str, Any]] = []
    persistent_rows = 0
    for author in sorted(primary_authors):
        counts = [by_author_offset[(author, offset)] for offset in (1, 2, 3)]
        total = sum(counts)
        persistent = total >= 1
        persistent_rows += int(persistent)
        reconstructed.append(
            {
                "author_numeric_id": author,
                "persistence_followup_complete": True,
                "target_topic_work_count_t_plus_1": counts[0],
                "target_topic_work_count_t_plus_2": counts[1],
                "target_topic_work_count_t_plus_3": counts[2],
                "target_topic_work_count_t_plus_1_through_t_plus_3": total,
                "three_year_target_topic_persistence_state": (
                    "MATURED_PERSISTENCE"
                    if persistent
                    else "MATURED_NO_PERSISTENCE"
                ),
                "three_year_target_topic_persistent": persistent,
            }
        )

    accounting = {
        "input_classification_rows": len(classifications_raw),
        "primary_pivot_rows": len(primary_authors),
        "nonprimary_rows_excluded": len(classifications) - len(primary_authors),
        "input_work_rows": len(works_raw),
        "input_author_event_rows": len(events_raw),
        "duplicate_author_work_event_excess_rows": duplicate_excess,
        "eligible_linked_target_work_rows_in_t_plus_1_through_t_plus_3": (
            eligible_linked_work_rows
        ),
        "registered_primary_distinct_author_work_rows": len(registered_pairs),
        "unregistered_or_nonprimary_distinct_author_work_rows": len(other_pairs),
        "excluded_outside_t_plus_1_through_t_plus_3_rows": reasons[
            "outside_t_plus_1_through_t_plus_3"
        ],
        "excluded_missing_publication_year_rows": reasons[
            "missing_publication_year"
        ],
        "excluded_wrong_topic_or_field_rows": reasons["wrong_topic_or_field"],
        "excluded_paratext_or_work_type_rows": reasons["paratext_or_work_type"],
        "excluded_retracted_rows": reasons["retracted"],
        "excluded_temporal_disposition_rows": reasons["temporal_disposition"],
        "excluded_zero_valid_author_token_rows": zero_token_rows,
        "retained_is_retracted_null_rows": retained_null_retraction,
        "matured_primary_rows": len(primary_authors),
        "right_censored_primary_rows": 0,
        "persistent_primary_rows": persistent_rows,
        "nonpersistent_primary_rows": len(primary_authors) - persistent_rows,
    }
    require(reconstructed == fixture.get("expected_rows"), "fixture reconstructed rows")
    require(accounting == fixture.get("expected_accounting"), "fixture accounting")
    return {
        "fixture_id": fixture["fixture_id"],
        "matured": True,
        "reconstructed_rows": reconstructed,
        "accounting": accounting,
        "row_mismatches": 0,
        "accounting_mismatches": 0,
    }


def _validate_semantics(contract: Mapping[str, Any]) -> dict[str, Any]:
    scope = _mapping(contract.get("exact_episode_scope"), "episode scope")
    require(scope.get("entry_year_maximum") == 2012, "discovery year boundary")
    require(scope.get("discovery_only") is True, "discovery scope")
    require(scope.get("confirmatory_entry_years_allowed") is False, "confirmatory scope")
    require(scope.get("field_selection_by_effect_size_allowed") is False, "field selection")

    population = _mapping(contract.get("outcome_population"), "outcome population")
    require(
        population.get("exact_denominator_rule")
        == "observable_history_pivot_state == NEW_TOPIC_ENTRY",
        "outcome denominator",
    )
    require(population.get("exact_denominator_rows") == 4_301, "denominator rows")
    require(
        population.get("nonprimary_rows_materialized_in_persistence_output") is False,
        "nonprimary materialization",
    )
    require(population.get("all_primary_pivot_rows_must_be_retained") is True, "retention")

    definition = _mapping(contract.get("persistence_definition"), "definition")
    require(definition.get("outcome_window_offsets") == [1, 2, 3], "outcome offsets")
    require(definition.get("qualifying_count_threshold") == 1, "persistence threshold")
    require(definition.get("aggregation") == "COUNT_DISTINCT_AUTHOR_WORK_PAIR", "aggregation")
    require(definition.get("team_cap") == "NONE", "team cap")
    require(
        definition.get("future_outcome_does_not_redefine_primary_pivot") is True,
        "entry/outcome separation",
    )
    require(
        definition.get("true_researcher_topic_persistence_claim_allowed") is False,
        "lifetime persistence boundary",
    )

    maturation = _mapping(contract.get("maturation_and_censoring"), "maturation")
    require(maturation.get("last_complete_calendar_year") == 2025, "complete year")
    require(maturation.get("exact_four_all_matured") is True, "exact-four maturity")
    require(maturation.get("exact_four_latest_required_outcome_year") == 2015, "latest year")
    require(
        maturation.get("partial_positive_work_may_override_right_censoring") is False,
        "right-censor precedence",
    )
    require(maturation.get("g0_final_freeze_status") == "PENDING", "G0 boundary")

    work = _mapping(contract.get("normalized_work_eligibility"), "work eligibility")
    require(tuple(work.get("exact_projected_columns", ())) == WORK_COLUMNS, "work columns")
    require(work.get("inherited_work_type_codes") == [1, 2, 3, 4, 7, 16, 18, 21], "work types")
    require(work.get("allowed_temporal_dispositions") == [2, 3], "dispositions")
    require(work.get("team_cap", "NONE") == "NONE", "work team cap")

    linkage = _mapping(contract.get("author_linkage"), "author linkage")
    require(tuple(linkage.get("exact_projected_columns", ())) == EVENT_COLUMNS, "event columns")
    require(linkage.get("natural_person_identity_established") is False, "person identity")

    source = _mapping(contract.get("production_source_contract"), "source contract")
    require(tuple(source.get("exact_output_columns", ())) == OUTPUT_COLUMNS, "output columns")
    require(source.get("required_output_rows") == 4_301, "output rows")
    require(source.get("permitted_work_year_offsets") == [1, 2, 3], "source offsets")
    require(source.get("output_private_numeric_ids_only") is True, "private IDs")
    require(
        source.get("output_may_include_impact_citation_peer_model_or_g4_columns")
        is False,
        "forbidden output columns",
    )
    return {
        "primary_pivot_state": "NEW_TOPIC_ENTRY",
        "primary_pivot_rows": 4_301,
        "outcome_window_offsets": [1, 2, 3],
        "last_complete_calendar_year": 2025,
        "all_exact_four_matured": True,
        "team_cap": "NONE",
        "current_snapshot_annotated_persistence_only": True,
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
        contract.get("label") == "pivot_feedback_d0_persistence_outcome_contract_v1",
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
        require(
            _lookup_dotted(parent_document, str(dotted)) == expected,
            f"parent field {dotted}",
        )

    repository = {
        label: _validate_repository_document(spec, repo_root=repo_root, label=label)
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

    fixtures_raw = contract.get("synthetic_fixtures")
    require(isinstance(fixtures_raw, list) and len(fixtures_raw) >= 2, "fixtures")
    work = _mapping(contract.get("normalized_work_eligibility"), "work eligibility")
    fixtures = [
        _replay_fixture(
            fixture,
            allowed_types=set(work["inherited_work_type_codes"]),
            allowed_dispositions=set(work["allowed_temporal_dispositions"]),
        )
        for fixture in fixtures_raw
    ]
    states = {
        row["three_year_target_topic_persistence_state"]
        for fixture in fixtures
        for row in fixture["reconstructed_rows"]
    }
    require(states == set(OUTCOME_STATES), "fixture outcome state coverage")

    verification = _mapping(contract.get("verification_contract"), "verification")
    for key in (
        "validator_must_not_import_source_builder",
        "repository_binding_hashes_required",
        "operator_receipt_hashes_and_statuses_required",
        "operator_parquet_rows_must_not_be_opened",
        "post_entry_production_rows_must_not_be_read",
        "fixture_expected_rows_reconstructed_not_trusted",
        "fixture_expected_accounting_reconstructed_not_trusted",
        "matured_positive_zero_and_right_censored_states_required",
        "partial_positive_cannot_override_right_censoring",
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
        "repository_bindings_validated": len(repository),
        "operator_receipt_bindings": operator,
        "operator_receipts_read": len(operator),
        "operator_parquet_files_opened": 0,
        "production_classification_rows_read": 0,
        "production_post_entry_rows_read": 0,
        "semantics": semantics,
        "fixture_replay": fixtures,
        "fixture_count": len(fixtures),
        "fixture_row_mismatches": 0,
        "fixture_accounting_mismatches": 0,
        "fixture_outcome_states": sorted(states),
        "partial_positive_right_censored_fixture_passed": True,
        "new_api_request_required": False,
        "api_requests_created": 0,
        "production_post_entry_source_build_authorized": False,
        "discovery_outcome_analysis_authorized": False,
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
        PersistenceOutcomeContractError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": (
                        "BLOCKED_PIVOT_FEEDBACK_D0_PERSISTENCE_OUTCOME_CONTRACT_V1"
                    ),
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
