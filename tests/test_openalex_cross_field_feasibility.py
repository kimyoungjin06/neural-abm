from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "scripts/summarize_openalex_cross_field_feasibility.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_openalex_cross_field_feasibility", TOOL_PATH
)
assert SPEC is not None and SPEC.loader is not None
cross_field = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cross_field)

CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_cross_field_feasibility_v0.yaml"
)
SUMMARY_PATH = (
    REPO_ROOT
    / "experiments/evidence/results/"
    "openalex_g1_cross_field_feasibility_20260718.summary.json"
)
MARKDOWN_PATH = SUMMARY_PATH.with_suffix(".md")
PARENT_COMPACT_PATH = (
    REPO_ROOT
    / "experiments/evidence/results/"
    "openalex_g1_work_source_membership_20260717.summary.json"
)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table(markdown: str, header: str) -> list[list[str]]:
    lines = markdown.splitlines()
    start = lines.index(header)
    assert re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", lines[start + 1])
    rows: list[list[str]] = []
    index = start + 2
    while index < len(lines) and lines[index].startswith("|"):
        rows.append([cell.strip() for cell in lines[index].strip("|").split("|")])
        index += 1
    return rows


def _integer(value: str) -> int:
    return int(value.replace(",", ""))


def _field_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    return [field for domain in summary["domains"] for field in domain["fields"]]


def test_config_freezes_aliases_hashes_windows_and_bounded_status() -> None:
    config = _yaml(CONFIG_PATH)
    parent = config["parent_work_source_profile"]
    taxonomy = config["taxonomy"]
    profile = config["profile_projection"]
    success = config["success_contract"]

    assert parent["compact_summary"] == {
        "relative_path": "experiments/evidence/results/openalex_g1_work_source_membership_20260717.summary.json",
        "sha256": "df626b897bcb53c1bbc3209dbfbec8f69b84361367e09e1a9ffdf39e293d7b83",
        "required_status": "PASS_WORK_SOURCE_MEMBERSHIP_AND_PROFILE_AUDIT_WITH_FROZEN_QUARANTINE",
        "result_fingerprint_sha256": "8b7d0273a04eb7a231a9d8b466151c562ac80eac6557ff7444508cb174862242",
    }
    assert parent["full_report"]["path_env"] == "OPENALEX_G1_WORK_SOURCE_FULL_REPORT"
    assert parent["full_report"]["sha256"] == (
        "1c385d42595fc34794116189ce940526e1095ccd2a5da9eccd3da902e8f152ce"
    )
    assert taxonomy["root_env"] == "OPENALEX_TAXONOMY_ROOT"
    assert taxonomy["canonical_sha256"] == (
        "8b9f4e2f492eb6939e1dbb6a8187f06d4ea8fdebb9c1bafce017cd39e54401b4"
    )
    assert taxonomy["expected_entity_counts"] == {
        "domains": 4,
        "fields": 26,
        "subfields": 252,
        "topics": 4516,
    }
    assert len(taxonomy["files"]) == 7
    assert [entry["expected_rows"] for entry in taxonomy["files"]] == [
        4,
        26,
        252,
        4,
        9,
        91,
        4412,
    ]
    assert all(not Path(entry["relative_path"]).is_absolute() for entry in taxonomy["files"])

    windows = {
        window["key"]: (
            window["start_year_inclusive"],
            window["end_year_inclusive"],
        )
        for window in profile["temporal_windows"]
    }
    assert windows == {
        "discovery_through_2012": (None, 2012),
        "g1_replication_2013_2016": (2013, 2016),
        "model_selection_2017_2019": (2017, 2019),
        "g4_temporal_2020_2022": (2020, 2022),
        "followup_diagnostic_2023_2025": (2023, 2025),
    }
    assert profile["select_final_fields"] is False
    assert success["success_status"] == "PASS_CROSS_FIELD_PROFILE_FEASIBILITY_ONLY"
    gates = success["required_gate_status"]
    assert gates["field_entity_validity"] == "UNVERIFIED"
    assert gates["candidate_eligibility"] == "NOT_ESTABLISHED"
    assert gates["final_field_selection"] == "PENDING_G0"
    assert gates["g_minus_1_data_readiness"] == "PENDING"
    assert gates["g0_measurement_freeze"] == "PENDING"
    assert gates["empirical_g1_signature"] == "NOT_RUN"
    assert gates["pass_g1"] is False
    assert config["pending_full_taxonomy_reference_audit"]["total_files"] == 1423

    serialized = json.dumps(config)
    assert "/home/" not in serialized
    assert "/hdd/" not in serialized
    assert "/ssd/" not in serialized


def test_summary_is_exact_no_rescan_feasibility_evidence() -> None:
    summary = _json(SUMMARY_PATH)
    assert summary["status"] == "PASS_CROSS_FIELD_PROFILE_FEASIBILITY_ONLY"
    assert summary["scope"] == {
        "final_field_selection_performed": False,
        "parent_full_report_reused": True,
        "parent_joint_sparse_cube_cells_reprojected": 122472,
        "taxonomy_entity_rows_read": 4798,
        "work_parquet_files_read": 0,
        "work_parquet_rows_scanned": 0,
    }
    assert summary["taxonomy"]["entity_counts"] == {
        "domains": 4,
        "fields": 26,
        "subfields": 252,
        "topics": 4516,
    }
    assert summary["taxonomy"]["distinct_display_name_counts"] == {
        "domains": 4,
        "fields": 26,
        "subfields": 244,
        "topics": 4516,
    }
    assert all(summary["taxonomy"]["internal_graph_checks"].values())

    profile = summary["profile"]
    assert profile["master_work_rows"] == 492_293_206
    assert profile["classified_primary_field_works"] == 354_145_248
    assert profile["null_primary_field_works"] == 138_147_958
    assert profile["classified_primary_field_works"] + profile[
        "null_primary_field_works"
    ] == profile["master_work_rows"]
    assert profile["global_temporal_diagnostics"] == {
        "future_after_snapshot_year": 4010,
        "observed_nonnull_year_max": 2050,
        "observed_nonnull_year_min": 1000,
        "pre_1900": 6_692_869,
        "publication_year_null": 27_565_532,
        "snapshot_year": 11_516_735,
    }
    assert profile["global_temporal_window_counts"] == {
        "discovery_through_2012": 200_148_775,
        "g1_replication_2013_2016": 58_763_594,
        "model_selection_2017_2019": 46_731_999,
        "g4_temporal_2020_2022": 50_823_699,
        "followup_diagnostic_2023_2025": 96_738_862,
    }
    assert all(profile["projection_checks"].values())


def test_all_26_fields_are_grouped_by_four_domains_without_selection() -> None:
    summary = _json(SUMMARY_PATH)
    domains = summary["domains"]
    fields = _field_rows(summary)
    assert [(domain["domain_id"], domain["domain_name"]) for domain in domains] == [
        (1, "Life Sciences"),
        (2, "Social Sciences"),
        (3, "Physical Sciences"),
        (4, "Health Sciences"),
    ]
    assert [domain["field_count"] for domain in domains] == [5, 6, 10, 5]
    assert [domain["works"] for domain in domains] == [
        50_073_228,
        106_588_124,
        136_069_643,
        61_414_253,
    ]
    assert len(fields) == 26
    assert {field["field_id"] for field in fields} == set(range(11, 37))
    assert sum(field["works"] for field in fields) == 354_145_248
    for field in fields:
        assert field["works"] == sum(field["authorship_status_counts"].values())
        assert field["subfield_count"] > 0
        assert field["topic_count"] > 0
        assert set(field["temporal_window_counts"]) == {
            "discovery_through_2012",
            "g1_replication_2013_2016",
            "model_selection_2017_2019",
            "g4_temporal_2020_2022",
            "followup_diagnostic_2023_2025",
        }

    by_id = {field["field_id"]: field for field in fields}
    assert by_id[17]["field_name"] == "Computer Science"
    assert by_id[17]["topic_count"] == 302
    assert by_id[17]["works"] == 22_363_553
    assert by_id[27]["field_name"] == "Medicine"
    assert by_id[27]["subfield_count"] == 42
    assert by_id[27]["topic_count"] == 692
    assert by_id[27]["works"] == 51_046_633

    assert summary["field_selection"] == {
        "final_domain_holdout": None,
        "selected_domain_ids": [],
        "selected_field_ids": [],
        "selection_used_scientific_outcomes": False,
        "status": "PENDING_G0",
    }


def test_summary_provenance_hashes_and_fingerprint_are_bound() -> None:
    summary = _json(SUMMARY_PATH)
    provenance = summary["provenance"]
    assert provenance["config_sha256"] == _sha256(CONFIG_PATH)
    assert provenance["tool_sha256"] == _sha256(TOOL_PATH)
    assert provenance["compact_summary_sha256"] == _sha256(PARENT_COMPACT_PATH)
    assert provenance["full_report_sha256"] == (
        "1c385d42595fc34794116189ce940526e1095ccd2a5da9eccd3da902e8f152ce"
    )
    assert provenance["parent_result_fingerprint_sha256"] == (
        "8b7d0273a04eb7a231a9d8b466151c562ac80eac6557ff7444508cb174862242"
    )
    assert provenance["taxonomy_canonical_sha256"] == (
        "8b9f4e2f492eb6939e1dbb6a8187f06d4ea8fdebb9c1bafce017cd39e54401b4"
    )
    assert len(provenance["taxonomy_file_commitments"]) == 7
    assert provenance["path_redacted"] is True

    parent_hashes = {
        "compact_summary_sha256": provenance["compact_summary_sha256"],
        "full_report_sha256": provenance["full_report_sha256"],
        "parent_result_fingerprint_sha256": provenance[
            "parent_result_fingerprint_sha256"
        ],
    }
    payload = {
        "status": summary["status"],
        "gate_status": summary["gate_status"],
        "scope": summary["scope"],
        "taxonomy": summary["taxonomy"],
        "profile": summary["profile"],
        "domains": summary["domains"],
        "field_selection": summary["field_selection"],
        "pending_full_taxonomy_reference_audit": summary[
            "pending_full_taxonomy_reference_audit"
        ],
        "claim_boundary": summary["claim_boundary"],
        "parent_hashes": parent_hashes,
    }
    assert hashlib.sha256(cross_field.canonical_json_bytes(payload)).hexdigest() == (
        provenance["result_fingerprint_sha256"]
    )


def test_markdown_quantitatively_matches_json_for_every_field_and_gate() -> None:
    summary = _json(SUMMARY_PATH)
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    field_rows = _table(
        markdown,
        "| Domain ID | Domain | Field ID | Field | Subfields | Topics | Works | Complete | Partial | None usable | Zero authorship | Through 2012 | 2013-16 | 2017-19 | 2020-22 | 2023-25 | Null year | After 2026 |",
    )
    assert len(field_rows) == 26
    markdown_by_id = {_integer(row[2]): row for row in field_rows}
    for field in _field_rows(summary):
        row = markdown_by_id[field["field_id"]]
        assert row[0:4] == [
            str(field["domain_id"]),
            field["domain_name"],
            str(field["field_id"]),
            field["field_name"],
        ]
        assert [_integer(value) for value in row[4:11]] == [
            field["subfield_count"],
            field["topic_count"],
            field["works"],
            field["authorship_status_counts"]["complete"],
            field["authorship_status_counts"]["partial"],
            field["authorship_status_counts"]["none_usable"],
            field["authorship_status_counts"]["zero_authorship_row"],
        ]
        assert [_integer(value) for value in row[11:16]] == [
            field["temporal_window_counts"]["discovery_through_2012"],
            field["temporal_window_counts"]["g1_replication_2013_2016"],
            field["temporal_window_counts"]["model_selection_2017_2019"],
            field["temporal_window_counts"]["g4_temporal_2020_2022"],
            field["temporal_window_counts"]["followup_diagnostic_2023_2025"],
        ]
        assert [_integer(value) for value in row[16:18]] == [
            field["temporal_diagnostics"]["publication_year_null"],
            field["temporal_diagnostics"]["future_after_snapshot_year"],
        ]

    gates = {
        row[0].strip("`"): row[1].strip("`")
        for row in _table(markdown, "| Gate | Status |")
    }
    assert gates == {
        key: "false" if value is False else str(value)
        for key, value in summary["gate_status"].items()
    }
    assert "1,423 total" in markdown
    assert "No final field or domain has been selected" in markdown
    assert "`pass_g1=false`" in markdown
    assert "/home/" not in markdown
    assert "/hdd/" not in markdown
    assert "/ssd/" not in markdown


def test_project_profile_independently_rejects_a_margin_mismatch() -> None:
    statuses = ["complete", "partial", "none_usable", "zero_authorship_row"]
    zero = {status: 0 for status in statuses}
    full_report = {
        "author_work_profile": {
            "master_work_rows_profiled": 1,
            "by_primary_field": [
                {"field_id": None, "statuses": zero},
                {
                    "field_id": 11,
                    "statuses": {**zero, "complete": 1},
                },
            ],
            "joint_sparse_cube": [
                {
                    "author_id_status": "complete",
                    "declared_authors_band": "1",
                    "primary_field_id": 11,
                    "publication_year": 2010,
                    "works": 1,
                }
            ],
        }
    }
    taxonomy = {
        "counts": {"domains": 1, "fields": 1, "subfields": 1, "topics": 1},
        "field_taxonomy": {
            11: {
                "field_id": 11,
                "field_name": "Field",
                "domain_id": 1,
                "domain_name": "Domain",
                "subfield_count": 1,
                "topic_count": 1,
            }
        },
    }
    config = {
        "parent_work_source_profile": {
            "full_report": {
                "required_joint_sparse_cube_cells": 1,
                "required_main_rows_profiled": 1,
            }
        },
        "profile_projection": {
            "expected_author_id_statuses": statuses,
            "expected_primary_field_buckets_including_null": 2,
            "snapshot_year": 2026,
            "temporal_windows": [
                {
                    "key": "discovery_through_2012",
                    "start_year_inclusive": None,
                    "end_year_inclusive": 2012,
                }
            ],
        },
    }
    projected = cross_field.project_profile(full_report, taxonomy, config)
    assert projected["classified_primary_field_works"] == 1

    broken = copy.deepcopy(full_report)
    broken["author_work_profile"]["by_primary_field"][1]["statuses"][
        "complete"
    ] = 2
    with pytest.raises(
        cross_field.CrossFieldFeasibilityError,
        match="joint cube projection",
    ):
        cross_field.project_profile(broken, taxonomy, config)


def test_small_taxonomy_loader_validates_a_complete_hierarchy(tmp_path: Path) -> None:
    domain = {
        "id": "https://openalex.org/domains/1",
        "display_name": "Domain",
        "fields": [
            {"id": "https://openalex.org/fields/11", "display_name": "Field"}
        ],
    }
    field = {
        "id": "https://openalex.org/fields/11",
        "display_name": "Field",
        "domain": domain | {"id": domain["id"], "display_name": domain["display_name"]},
        "subfields": [
            {"id": "https://openalex.org/subfields/1100", "display_name": "Subfield"}
        ],
    }
    field["domain"] = {"id": domain["id"], "display_name": domain["display_name"]}
    subfield = {
        "id": "https://openalex.org/subfields/1100",
        "display_name": "Subfield",
        "field": {"id": field["id"], "display_name": field["display_name"]},
        "domain": {"id": domain["id"], "display_name": domain["display_name"]},
        "topics": [
            {"id": "https://openalex.org/T10000", "display_name": "Topic"}
        ],
    }
    topic = {
        "id": "https://openalex.org/T10000",
        "display_name": "Topic",
        "subfield": {
            "id": subfield["id"],
            "display_name": subfield["display_name"],
        },
        "field": {"id": field["id"], "display_name": field["display_name"]},
        "domain": {"id": domain["id"], "display_name": domain["display_name"]},
    }
    inputs = [
        ("domains", "domains/part.gz", domain),
        ("fields", "fields/part.gz", field),
        ("subfields", "subfields/part.gz", subfield),
        ("topics", "topics/part.gz", topic),
    ]
    digest = hashlib.sha256()
    file_specs = []
    for entity, relative, row in inputs:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        digest.update(cross_field.canonical_json_bytes(row))
        digest.update(b"\n")
        file_specs.append(
            {
                "entity": entity,
                "relative_path": relative,
                "sha256": _sha256(path),
                "expected_rows": 1,
            }
        )
    contract = {
        "files": file_specs,
        "canonical_sha256": digest.hexdigest(),
        "expected_entity_counts": {
            "domains": 1,
            "fields": 1,
            "subfields": 1,
            "topics": 1,
        },
    }
    taxonomy = cross_field.load_and_validate_taxonomy(tmp_path, contract)
    assert taxonomy["internal_graph_checks"] == {
        "unique_entity_ids": True,
        "parent_ids_resolve": True,
        "parent_names_match": True,
        "parent_chains_match": True,
        "declared_child_sets_match": True,
        "declared_child_names_match": True,
        "declared_child_ids_unique": True,
    }
    assert taxonomy["field_taxonomy"][11]["topic_count"] == 1


def test_runtime_contract_and_declared_child_names_fail_closed() -> None:
    config = copy.deepcopy(_yaml(CONFIG_PATH))
    config["success_contract"]["success_status"] = "PASS_UNBOUND_STATUS"
    with pytest.raises(
        cross_field.CrossFieldFeasibilityError,
        match="success status contract mismatch",
    ):
        cross_field.validate_success_contract(config)

    child_map = {
        "https://openalex.org/fields/11": {
            "display_name": "Canonical field name"
        }
    }
    with pytest.raises(
        cross_field.CrossFieldFeasibilityError,
        match="child display_name mismatch",
    ):
        cross_field._declared_child_ids(
            [
                {
                    "id": "https://openalex.org/fields/11",
                    "display_name": "Drifted field name",
                }
            ],
            "domain.fields",
            child_map,
        )

    summary = copy.deepcopy(_json(SUMMARY_PATH))
    summary["taxonomy"]["snapshot_date"] = "2031-01-02"
    summary["pending_full_taxonomy_reference_audit"]["status"] = "COMPLETE"
    summary["pending_full_taxonomy_reference_audit"]["workload_class"] = "gpu"
    rendered = cross_field.render_markdown(summary)
    assert "Snapshot-year 2031" in rendered
    assert "Publication year after 2031" in rendered
    assert "Declared status: `COMPLETE`" in rendered
    assert "Workload class: `gpu`" in rendered
