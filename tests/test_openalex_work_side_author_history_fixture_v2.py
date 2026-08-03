from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT / "scripts/verify_openalex_work_side_author_history_fixture_v2.py"
)
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_work_side_author_history_v2.yaml"
)
PRODUCTION_ROOT_ID = "openalex_works_20260330_repairreplay_20260410_190630"

_SPEC = importlib.util.spec_from_file_location("author_history_fixture_v2", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(RUNNER)


def _parents(*, spine_rows: int) -> dict[str, Any]:
    return {
        "current_content_baseline": {
            "schema_version": "openalex_g1_work_side_author_history_synthetic_content_parent.v1",
            "status": "SYNTHETIC_CONTENT_BASELINE_PARENT_ONLY",
            "synthetic_content_tree_sha256": "1" * 64,
            "synthetic_content_result_fingerprint_sha256": "2" * 64,
        },
        "main_temporal_validity": {
            "schema_version": "openalex_g1_work_side_author_history_synthetic_temporal_parent.v1",
            "status": "SYNTHETIC_MAIN_TEMPORAL_PARENT_ONLY",
            "synthetic_content_tree_sha256": "1" * 64,
            "synthetic_content_result_fingerprint_sha256": "2" * 64,
            "synthetic_temporal_result_fingerprint_sha256": "3" * 64,
        },
        "canonical_work_time_spine": {
            "schema_version": "openalex_g1_work_side_author_history_synthetic_spine_parent.v1",
            "status": "SYNTHETIC_CANONICAL_SPINE_PARENT_ONLY",
            "synthetic_content_tree_sha256": "1" * 64,
            "synthetic_content_result_fingerprint_sha256": "2" * 64,
            "synthetic_temporal_result_fingerprint_sha256": "3" * 64,
            "synthetic_spine_result_fingerprint_sha256": "4" * 64,
            "synthetic_spine_data_tree_sha256": "5" * 64,
            "rows": spine_rows,
        },
    }


def _authorship(
    work: int,
    author: int | None,
    *,
    position: str | None = "middle",
    raw_name: str | None = None,
    orcid: str | None = None,
) -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/W{work}",
        "authorships__author__id": (
            None if author is None else f"https://openalex.org/A{author}"
        ),
        "authorships__author__orcid": orcid,
        "authorships__raw_author_name": raw_name,
        "authorships__author__display_name": raw_name,
        "authorships__author_position": position,
    }


def _fixture() -> dict[str, Any]:
    # W0 makes A0 history-usable. W1 has a duplicated A1 pair. W2 crosses the
    # numeric cap-10 boundary and therefore tests cap-specific recomputation.
    main = [
        {
            "id": "https://openalex.org/W0",
            "authors_count": 1,
            "primary_topic__field__id": "https://openalex.org/fields/1",
        },
        {
            "id": "https://openalex.org/W1",
            "authors_count": 2,
            "primary_topic__field__id": None,
        },
        {
            "id": "https://openalex.org/W2",
            "authors_count": 11,
            "primary_topic__field__id": "https://openalex.org/fields/26",
        },
    ]
    spine = [
        {
            "work_id_numeric": 0,
            "publication_day": 18_262,
            "publication_year": 2020,
            "temporal_disposition": 2,
        },
        {
            "work_id_numeric": 1,
            "publication_day": 18_628,
            "publication_year": 2021,
            "temporal_disposition": 2,
        },
        {
            "work_id_numeric": 2,
            "publication_day": 20_542,
            "publication_year": 2026,
            "temporal_disposition": 3,
        },
    ]
    authorships = [
        _authorship(0, 0, position="first", raw_name="A Zero"),
        _authorship(
            1,
            1,
            position="first",
            raw_name="Alice",
            orcid="https://orcid.org/0000-0002-1825-0097",
        ),
        _authorship(1, 1, position="last", raw_name="ALICE"),
        *[_authorship(2, author) for author in range(2, 13)],
    ]
    return {
        "schema_version": "openalex_g1_work_side_author_history_synthetic_fixture.v2",
        "surface_kind": "SYNTHETIC_FIXTURE_ONLY",
        "production_input_access_allowed": False,
        "root_id": "synthetic-author-history-v2-test",
        "snapshot_date": "2026-03-30",
        "synthetic_parents": _parents(spine_rows=len(spine)),
        "main_rows": main,
        "spine_rows": spine,
        "authorship_rows": authorships,
    }


def _write_fixture(path: Path, value: dict[str, Any]) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path.write_bytes(payload)
    return payload


def _run(path: Path, *extra: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(CONFIG_PATH),
            "--synthetic-fixture",
            str(path),
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )


def _assert_failed(result: subprocess.CompletedProcess[bytes], text: str) -> None:
    assert result.returncode == 2
    assert result.stdout == b""
    assert text.encode() in result.stderr


def test_temp_fixture_happy_path_is_canonical_closed_and_nonpromoting(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "fixture.json"
    payload = _write_fixture(fixture_path, _fixture())
    entries_before = {path.name for path in tmp_path.iterdir()}

    result = _run(fixture_path)

    assert result.returncode == 0, result.stderr.decode()
    assert result.stderr == b""
    assert result.stdout.endswith(b"\n")
    assert result.stdout.count(b"\n") == 1
    report = json.loads(result.stdout)
    assert result.stdout == RUNNER.canonical_json_bytes(report) + b"\n"
    assert {path.name for path in tmp_path.iterdir()} == entries_before
    assert report["status"] == (
        "VALID_UNATTESTED_GENERIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_V2_MECHANICS_ONLY"
    )
    assert report["evidence_class"] == "UNATTESTED_GENERIC_FIXTURE"
    assert report["production_status"] == "PRODUCTION_NOT_RUN"
    assert report["inputs"]["fixture_sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["scope"]["production_input_accessed"] is None
    assert report["scope"]["production_source_interface_used"] is False
    assert report["scope"]["parent_evidence_class"] == (
        "SELF_DECLARED_UNVERIFIED_SYNTHETIC_PARENT_STUBS"
    )
    assert report["scope"]["gpu_used"] is False
    assert report["scope"]["files_written"] == 0
    assert set(report["inputs"]) == {
        "config_sha256",
        "fixture_sha256",
        "runtime_fingerprint",
        "foundation_verification",
    }
    assert report["inputs"]["runtime_fingerprint"] == RUNNER._runtime_fingerprint()
    foundation_records = report["inputs"]["foundation_verification"]
    assert len(foundation_records) == 6
    assert foundation_records == sorted(
        foundation_records, key=lambda item: item["repository_relative_path"]
    )
    assert all(
        record["verified_against_config"] is True for record in foundation_records
    )
    assert all(
        not record["repository_relative_path"].startswith("/")
        for record in foundation_records
    )
    assert b"/home/" not in result.stdout
    assert report["pair_accounting"]["grammar_valid_author_slots"] == 14
    assert report["pair_accounting"]["distinct_author_work_pairs"] == 13
    assert report["pair_accounting"]["duplicate_pair_row_excess"] == 1
    assert report["history_accounting"]["base_view_tokens"] == 13
    assert report["history_accounting"]["positive_view_tokens"] == 12
    assert report["history_accounting"]["base_view_unique_pairs"] == 13
    assert report["history_accounting"]["positive_view_unique_pairs"] == 12
    sensitivity = report["team_and_hyper_sensitivity"]
    assert sensitivity["base_view_cap_pair_counts"]["10"] == 2
    assert sensitivity["positive_author_id_view_cap_pair_counts"]["10"] == 1
    assert sensitivity["base_view_cap_pair_counts"]["ALL"] == 13
    assert sensitivity["positive_author_id_view_cap_pair_counts"]["ALL"] == 12
    assert any(
        sensitivity["base_view_cap_pair_counts"][cap]
        > sensitivity["positive_author_id_view_cap_pair_counts"][cap]
        for cap in ("10", "50", "100", "ALL")
    )
    assert report["gate_status"]["g_minus_1_data_readiness"] == "PENDING"
    assert report["gate_status"]["g0_measurement_freeze"] == "PENDING"
    assert report["gate_status"]["pass_g1"] is False
    for sensitive in (
        b"Alice",
        b"A Zero",
        b"https://orcid.org/0000-0002-1825-0097",
        b"https://openalex.org/A1",
    ):
        assert sensitive not in result.stdout


def test_checked_in_fixture_matches_config_pinned_counts_and_all_digests() -> None:
    fixture_path = (
        REPO_ROOT / "tests/fixtures/openalex_work_side_author_history_v2_minimal.json"
    )
    result = _run(fixture_path)
    assert result.returncode == 0, result.stderr.decode()
    report = json.loads(result.stdout)
    assert report["status"] == (
        "PASS_SYNTHETIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_V2_PINNED_FIXTURE_CONTRACT_ONLY"
    )
    assert report["evidence_class"] == "PINNED_CHECKED_FIXTURE"
    assert report["scope"]["fixture_evidence_class"] == "PINNED_CHECKED_FIXTURE"
    assert report["scope"]["parent_evidence_class"] == (
        "CONFIG_PINNED_SYNTHETIC_PARENT_STUBS"
    )
    assert report["scope"]["production_input_accessed"] is False
    assert report["scope"]["production_source_interface_used"] is False
    assert report["scope"]["pinned_fixture_sha256_match"] is True
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = config["minimal_fixture"]
    assert report["inputs"]["fixture_sha256"] == frozen["sha256"]
    counts = frozen["expected_visible_counts"]
    for section in (
        "source_accounting",
        "work_and_slot_accounting",
        "author_id_accounting",
        "temporal_accounting",
    ):
        assert report[section] == counts[section]
    assert {
        key: report["pair_accounting"][key] for key in counts["pair_accounting"]
    } == counts["pair_accounting"]
    assert {
        key: report["history_accounting"][key] for key in counts["history_count_fields"]
    } == counts["history_count_fields"]
    sensitivity = report["team_and_hyper_sensitivity"]
    assert (
        sensitivity["base_view_cap_pair_counts"] == counts["base_view_cap_pair_counts"]
    )
    assert (
        sensitivity["positive_author_id_view_cap_pair_counts"]
        == counts["positive_author_id_view_cap_pair_counts"]
    )
    assert set(sensitivity["base_view_cap_threshold_token_grid"].values()) == {
        counts["every_base_and_positive_threshold_grid_cell"]
    }
    assert set(
        sensitivity["positive_author_id_view_cap_threshold_token_grid"].values()
    ) == {counts["every_base_and_positive_threshold_grid_cell"]}
    report_digests = {
        **{
            key: value
            for key, value in report["pair_accounting"].items()
            if key.endswith("sha256")
        },
        **{
            key: value
            for key, value in report["history_accounting"].items()
            if key.endswith("sha256")
        },
        **{key: value for key, value in sensitivity.items() if key.endswith("sha256")},
    }
    assert len(report_digests) == 8
    assert report_digests == frozen["expected_digest_values"]


def test_same_pinned_bytes_at_another_path_remain_unattested_generic(
    tmp_path: Path,
) -> None:
    source = (
        REPO_ROOT / "tests/fixtures/openalex_work_side_author_history_v2_minimal.json"
    )
    copied = tmp_path / "copied.json"
    copied.write_bytes(source.read_bytes())

    result = _run(copied)

    assert result.returncode == 0, result.stderr.decode()
    report = json.loads(result.stdout)
    assert report["status"] == (
        "VALID_UNATTESTED_GENERIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_V2_MECHANICS_ONLY"
    )
    assert report["evidence_class"] == "UNATTESTED_GENERIC_FIXTURE"
    assert report["scope"]["pinned_fixture_sha256_match"] is True
    assert report["scope"]["production_input_accessed"] is None


def test_exact_pinned_path_with_wrong_hash_fails_instead_of_downgrading() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    pinned_path = (
        REPO_ROOT / "tests/fixtures/openalex_work_side_author_history_v2_minimal.json"
    ).resolve(strict=True)

    with pytest.raises(RUNNER.ValidationError, match="cannot downgrade to generic"):
        RUNNER._classify_fixture(
            config,
            resolved_fixture_path=pinned_path,
            fixture_sha256="0" * 64,
        )


def test_logical_row_permutation_preserves_reduction_hashes_but_not_input_hash(
    tmp_path: Path,
) -> None:
    first = _fixture()
    second = copy.deepcopy(first)
    second["main_rows"].reverse()
    second["spine_rows"].reverse()
    second["authorship_rows"].reverse()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_fixture(first_path, first)
    _write_fixture(second_path, second)

    first_report = json.loads(_run(first_path).stdout)
    second_report = json.loads(_run(second_path).stdout)

    assert (
        first_report["inputs"]["fixture_sha256"]
        != second_report["inputs"]["fixture_sha256"]
    )
    assert first_report["history_accounting"] == second_report["history_accounting"]
    for key in (
        "primary_population_aggregate_sha256",
        "history_usable_population_aggregate_sha256",
    ):
        assert (
            first_report["pair_accounting"][key]
            == second_report["pair_accounting"][key]
        )
    assert (
        first_report["team_and_hyper_sensitivity"]
        == second_report["team_and_hyper_sensitivity"]
    )


def test_diagnostic_only_mutation_changes_pair_digest_not_history_or_grid(
    tmp_path: Path,
) -> None:
    first = _fixture()
    second = copy.deepcopy(first)
    second["authorship_rows"][1]["authorships__raw_author_name"] = "Alicia"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_fixture(first_path, first)
    _write_fixture(second_path, second)

    first_report = json.loads(_run(first_path).stdout)
    second_report = json.loads(_run(second_path).stdout)

    assert (
        first_report["pair_accounting"]["primary_population_aggregate_sha256"]
        != second_report["pair_accounting"]["primary_population_aggregate_sha256"]
    )
    assert (
        first_report["pair_accounting"]["history_usable_population_aggregate_sha256"]
        != second_report["pair_accounting"][
            "history_usable_population_aggregate_sha256"
        ]
    )
    assert first_report["history_accounting"] == second_report["history_accounting"]
    assert (
        first_report["team_and_hyper_sensitivity"]
        == second_report["team_and_hyper_sensitivity"]
    )


def _threshold_fixture(unique_works: int) -> dict[str, Any]:
    main: list[dict[str, Any]] = []
    spine: list[dict[str, Any]] = []
    authorships: list[dict[str, Any]] = []
    for offset in range(unique_works):
        work = 1_000 + offset
        main.append(
            {
                "id": f"https://openalex.org/W{work}",
                "authors_count": 1,
                "primary_topic__field__id": None,
            }
        )
        spine.append(
            {
                "work_id_numeric": work,
                "publication_day": 18_262,
                "publication_year": 2020,
                "temporal_disposition": 2,
            }
        )
        authorships.append(_authorship(work, 1))
    for offset in range(unique_works):
        work = 2_000 + offset
        main.append(
            {
                "id": f"https://openalex.org/W{work}",
                "authors_count": 11,
                "primary_topic__field__id": None,
            }
        )
        spine.append(
            {
                "work_id_numeric": work,
                "publication_day": 18_262,
                "publication_year": 2020,
                "temporal_disposition": 2,
            }
        )
        authorships.append(_authorship(work, 2))
        authorships.extend(_authorship(work, None) for _ in range(10))
    return {
        "schema_version": "openalex_g1_work_side_author_history_synthetic_fixture.v2",
        "surface_kind": "SYNTHETIC_FIXTURE_ONLY",
        "production_input_access_allowed": False,
        "root_id": f"synthetic-threshold-{unique_works}",
        "snapshot_date": "2026-03-30",
        "synthetic_parents": _parents(spine_rows=len(spine)),
        "main_rows": main,
        "spine_rows": spine,
        "authorship_rows": authorships,
    }


def test_strict_gt_50_threshold_and_cap_specific_recomputation(
    tmp_path: Path,
) -> None:
    fifty = tmp_path / "fifty.json"
    fifty_one = tmp_path / "fifty_one.json"
    _write_fixture(fifty, _threshold_fixture(50))
    _write_fixture(fifty_one, _threshold_fixture(51))
    at_boundary_result = _run(fifty)
    above_boundary_result = _run(fifty_one)
    assert at_boundary_result.returncode == 0, at_boundary_result.stderr.decode()
    assert above_boundary_result.returncode == 0, above_boundary_result.stderr.decode()
    at_boundary = json.loads(at_boundary_result.stdout)["team_and_hyper_sensitivity"]
    above_boundary = json.loads(above_boundary_result.stdout)[
        "team_and_hyper_sensitivity"
    ]

    assert at_boundary["base_view_cap_threshold_token_grid"]["cap_10_gt_50"] == 0
    assert at_boundary["base_view_cap_threshold_token_grid"]["cap_50_gt_50"] == 0
    assert above_boundary["base_view_cap_threshold_token_grid"]["cap_10_gt_50"] == 1
    assert above_boundary["base_view_cap_threshold_token_grid"]["cap_50_gt_50"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(extra=True),
            "fixture has a missing or extra key",
        ),
        (
            lambda value: value["main_rows"][0].update(authors_count=True),
            "must be a non-boolean integer",
        ),
        (
            lambda value: value["spine_rows"][0].update(temporal_disposition=3),
            "temporal disposition predicate mismatch",
        ),
        (
            lambda value: value["synthetic_parents"][
                "canonical_work_time_spine"
            ].update(synthetic_content_tree_sha256="9" * 64),
            "parent cross-binding mismatch",
        ),
        (
            lambda value: value["main_rows"][0].update(authors_count=2),
            "declared authors_count does not equal observed raw slots",
        ),
        (
            lambda value: value["authorship_rows"][0].update(
                authorships__author__id="https://openalex.org/A01"
            ),
            "malformed non-null author ID",
        ),
        (
            lambda value: value["spine_rows"].pop(),
            "synthetic spine parent row count mismatch",
        ),
    ],
)
def test_fixture_contract_mutations_fail_closed(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    fixture = _fixture()
    mutation(fixture)
    fixture_path = tmp_path / "bad.json"
    _write_fixture(fixture_path, fixture)

    _assert_failed(_run(fixture_path), message)


def test_duplicate_json_key_nan_float_and_surrogate_are_rejected() -> None:
    with pytest.raises(RUNNER.ValidationError, match="duplicate JSON key"):
        RUNNER._strict_json_loads(b'{"x":1,"x":2}', field="test")
    with pytest.raises(RUNNER.ValidationError, match="non-finite"):
        RUNNER._strict_json_loads(b'{"x":NaN}', field="test")
    with pytest.raises(RUNNER.ValidationError, match="floating-point"):
        RUNNER._strict_json_loads(b'{"x":1.5}', field="test")
    with pytest.raises(RUNNER.ValidationError, match="lone Unicode surrogate"):
        RUNNER._strict_json_loads(b'{"x":"\\ud800"}', field="test")


def test_deep_json_yaml_and_overlong_integer_are_controlled_validation_failures(
    tmp_path: Path,
) -> None:
    deep_json = b"[" * 2_000 + b"0" + b"]" * 2_000
    with pytest.raises(RUNNER.ValidationError):
        RUNNER._strict_json_loads(deep_json, field="deep JSON")
    with pytest.raises(RUNNER.ValidationError):
        RUNNER._strict_yaml_loads(("- " * 2_000 + "0\n").encode(), field="deep YAML")
    with pytest.raises(RUNNER.ValidationError, match="overlong integer lexeme"):
        RUNNER._strict_json_loads(
            b'{"value":' + b"9" * 5_000 + b"}",
            field="huge integer JSON",
        )

    fixture_path = tmp_path / "deep.json"
    fixture_path.write_bytes(deep_json)
    result = _run(fixture_path)
    _assert_failed(result, "exceeds maximum nesting depth")
    assert b"Traceback" not in result.stderr


@pytest.mark.parametrize("identifier_kind", ["work", "author"])
def test_overlong_identifier_suffix_is_controlled_exit2(
    tmp_path: Path,
    identifier_kind: str,
) -> None:
    fixture = _fixture()
    huge_suffix = "9" * 5_000
    if identifier_kind == "work":
        fixture["main_rows"][0]["id"] = f"https://openalex.org/W{huge_suffix}"
    else:
        fixture["authorship_rows"][0]["authorships__author__id"] = (
            f"https://openalex.org/A{huge_suffix}"
        )
    fixture_path = tmp_path / f"huge-{identifier_kind}.json"
    _write_fixture(fixture_path, fixture)

    result = _run(fixture_path)

    _assert_failed(result, "overlong decimal suffix")
    assert b"Traceback" not in result.stderr


@pytest.mark.parametrize(
    "value",
    [
        "https://orcid.org/0000-0002-1694-233X",
        "http://orcid.org/0000-0002-1694-233x",
        "https://www.orcid.org/0000-0002-1694-233x",
        "http://www.orcid.org/0000-0002-1694-233x",
        "0000-0002-1694-233x",
        "000000021694233x",
    ],
)
def test_exact_v0_orcid_lexical_forms_canonicalize(value: str) -> None:
    assert RUNNER._canonical_orcid(value) == ("https://orcid.org/0000-0002-1694-233X")


@pytest.mark.parametrize(
    "value",
    [
        "https://orcid.org/0000-0002-1694-233x",
        "www.orcid.org/0000-0002-1694-233X",
        " https://orcid.org/0000-0002-1694-233X",
        "0000-0002-1694-2330",
    ],
)
def test_undeclared_or_invalid_orcid_forms_do_not_canonicalize(value: str) -> None:
    assert RUNNER._canonical_orcid(value) is None


def test_exact_v0_name_normalization_and_unsigned_utf8_tag_sort() -> None:
    assert RUNNER._normalize_name("  ADA\t \u2003 LOVELACE  ") == "ada lovelace"
    assert RUNNER._sorted_tags({"a", '"', "\n", None}) == [
        {"kind": "null"},
        {"kind": "string", "value": "\n"},
        {"kind": "string", "value": '"'},
        {"kind": "string", "value": "a"},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        b"x: 1\nx: 2\n",
        b"x: &anchor 1\ny: *anchor\n",
        b"x: {a: 1}\ny: {<<: {a: 2}}\n",
        b"x: !unsafe value\n",
    ],
)
def test_yaml_duplicate_alias_merge_and_custom_tag_are_rejected(payload: bytes) -> None:
    with pytest.raises(RUNNER.ValidationError, match="strict safe YAML"):
        RUNNER._strict_yaml_loads(payload, field="test config")


def test_symlink_and_hardlink_fixtures_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    _write_fixture(target, _fixture())
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    _assert_failed(_run(symlink), "path component must not be a symlink")

    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)
    _assert_failed(_run(target), "must have link count one")


def test_production_root_path_and_payload_identifiers_are_rejected(
    tmp_path: Path,
) -> None:
    production_named = tmp_path / PRODUCTION_ROOT_ID
    production_named.mkdir()
    path = production_named / "fixture.json"
    _write_fixture(path, _fixture())
    _assert_failed(_run(path), "fixture path contains the production root ID")

    fixture = _fixture()
    fixture["root_id"] = f"synthetic-{PRODUCTION_ROOT_ID}"
    payload_path = tmp_path / "payload.json"
    _write_fixture(payload_path, fixture)
    _assert_failed(
        _run(payload_path), "fixture root_id contains the production root ID"
    )


def test_symlinked_parent_cannot_hide_production_named_resolved_path(
    tmp_path: Path,
) -> None:
    production_named = tmp_path / PRODUCTION_ROOT_ID
    production_named.mkdir()
    target = production_named / "fixture.json"
    _write_fixture(target, _fixture())
    alias = tmp_path / "synthetic-alias"
    alias.symlink_to(production_named, target_is_directory=True)

    _assert_failed(_run(alias / "fixture.json"), "path component must not be a symlink")


def test_oversize_fixture_is_rejected_before_json_parse(tmp_path: Path) -> None:
    path = tmp_path / "oversize.json"
    path.write_bytes(b"{" + b" " * RUNNER.MAX_FIXTURE_BYTES)

    result = _run(path)
    _assert_failed(result, "exceeds its hard byte cap")
    assert b"strict UTF-8 JSON" not in result.stderr


def test_config_copy_and_forbidden_cli_surfaces_are_rejected(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    _write_fixture(fixture_path, _fixture())
    config_copy = tmp_path / "config.yaml"
    config_copy.write_bytes(CONFIG_PATH.read_bytes())
    copied = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(config_copy),
            "--synthetic-fixture",
            str(fixture_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    _assert_failed(copied, "only the exact audited repository config path")

    for forbidden in ("--source-root", "--production-root", "--out-dir"):
        result = _run(fixture_path, forbidden, str(tmp_path / "out"))
        assert result.returncode == 2
        assert result.stdout == b""
        assert b"unrecognized arguments" in result.stderr


def test_foundation_file_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RUNNER._strict_yaml_loads(
        CONFIG_PATH.read_bytes(), field="test repository config"
    )
    original_reader = RUNNER._read_bounded_regular

    def drift_one_file(
        path: Path, *, field: str, maximum_bytes: int
    ) -> tuple[bytes, os.stat_result]:
        payload, identity = original_reader(
            path, field=field, maximum_bytes=maximum_bytes
        )
        if path.name == "openalex_g1_work_side_author_history_v1.yaml":
            payload += b"\n"
        return payload, identity

    monkeypatch.setattr(RUNNER, "_read_bounded_regular", drift_one_file)

    with pytest.raises(RUNNER.ValidationError, match="foundation byte count mismatch"):
        RUNNER._verify_foundation_files(config)


def test_row_caps_are_checked_before_row_processing() -> None:
    fixture = _fixture()
    fixture["main_rows"] = [{}] * (RUNNER.MAX_MAIN_ROWS + 1)
    with pytest.raises(RUNNER.ValidationError, match="main row cap exceeded"):
        RUNNER._validate_and_reduce(
            fixture,
            config_sha256="1" * 64,
            fixture_sha256="2" * 64,
            pinned_fixture=False,
            pinned_fixture_sha256_match=False,
            foundation_verification=[],
            runtime_fingerprint=RUNNER._runtime_fingerprint(),
        )
