from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_f_field_episode_assignment_v1.yaml"
BUILDER = ROOT / "scripts/build_pivot_feedback_f_field_episode_assignment_v1.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("f_assignment_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_binds_active_authority_and_exact_taxonomy() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert contract["schema_version"] == "pivot_feedback_f_field_episode_assignment.v1"
    assert contract["authorization_parent"]["sha256"] == _sha256(
        ROOT / contract["authorization_parent"]["relative_path"]
    )
    assert contract["bound_design_sources"]["scientific_sections"]["sha256"] == _sha256(
        ROOT
        / contract["bound_design_sources"]["scientific_sections"]["relative_path"]
    )
    taxonomy = contract["normalized_inputs"]["topic_taxonomy_projection"]
    assert taxonomy["rows"] == 4516
    assert taxonomy["sha256"] == (
        "566facdbc02bf4a37783fe64db4c1d104d00b73e250999bd824d560083827ba5"
    )


def test_contract_is_component_not_gate_and_keeps_f_closed() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    completion = contract["completion_accounting"]
    projection = contract["operator_projection"]

    assert completion["this_component_is_new_gate"] is False
    assert completion["this_component_alone_may_pass_f"] is False
    assert completion["final_panel_conformance_after_assignment"] == "PENDING"
    assert completion["confirmatory_entry_or_persistence_labels_read"] == 0
    assert projection["decision_year_information_used_as_feature"] is False
    assert projection["entry_or_persistence_label_computed"] is False
    assert projection["g4_outcome_read"] is False


def test_attention_screen_applies_registered_mad_rule() -> None:
    builder = _load_builder()
    taxonomy = {
        topic_id: {
            "topic_name": f"Topic {topic_id}",
            "field_id": 11,
            "field_name": "Field 11",
            "subfield_id": 1100 + topic_id,
            "subfield_name": f"Subfield {topic_id}",
        }
        for topic_id in range(1, 5)
    }
    recent = {1: 30, 2: 33, 3: 36, 4: 300}
    topic_year = {}
    for topic_id in taxonomy:
        annual = {year: 10 for year in (2007, 2008, 2009)}
        annual.update({2010: recent[topic_id], 2011: 0, 2012: 0})
        topic_year[(11, topic_id)] = annual
    contract = {
        "modified_z_multiplier": 0.6744897501960817,
        "modified_z_threshold_inclusive": 3.0,
        "reference_offsets": [-6, -5, -4],
        "recent_offsets": [-3, -2, -1],
    }

    candidates = builder._attention_candidates(topic_year, taxonomy, contract)

    assert any(
        row["field_id"] == 11
        and row["topic_id"] == 4
        and row["decision_year"] == 2013
        and row["modified_z"] >= 3.0
        for row in candidates
    )
    assert not any(
        row["topic_id"] == 1 and row["decision_year"] == 2013
        for row in candidates
    )


def test_lexicographic_assignment_is_domain_disjoint_and_g4_cold() -> None:
    builder = _load_builder()
    linked = np.empty(26, dtype=builder.d0.LINKED_WORK_DTYPE)
    for offset, field_id in enumerate(range(11, 37)):
        linked[offset]["work_numeric_id"] = offset + 1
        linked[offset]["publication_year"] = 2012
        linked[offset]["field_id"] = field_id
        linked[offset]["topic_id"] = 10000 + offset
        linked[offset]["work_type_code"] = 1
        linked[offset]["declared_authors_count"] = 1
        linked[offset]["is_retracted_null"] = 0
        linked[offset]["temporal_disposition"] = 2
    author = {}
    for name, _, _ in builder.PERIODS:
        values = np.arange(101, 127, dtype=np.uint64)
        author[name] = {
            "eligible_author_count": values,
            "matured_entry_opportunity_count": values * 3,
        }
    compact = {
        field_id: {
            "publication": (90 + field_id, 100 + field_id),
            "edge": (70 + field_id, 100 + field_id),
            "orcid": (40 + field_id, 100 + field_id),
        }
        for field_id in range(11, 37)
    }
    domain_fields = {
        "life_sciences": list(range(11, 17)),
        "social_sciences": list(range(17, 24)),
        "physical_sciences": list(range(24, 31)),
        "health_sciences": list(range(31, 37)),
    }
    field_meta = {}
    for domain_id, fields in domain_fields.items():
        for field_id in fields:
            field_meta[field_id] = {
                "field_name": f"Field {field_id}",
                "domain_id": domain_id,
                "domain_name": domain_id,
            }
    candidates = []
    for field_id in range(11, 37):
        for period, start, _ in builder.PERIODS:
            for family in range(3):
                candidates.append(
                    {
                        "field_id": field_id,
                        "subfield_id": field_id * 100 + family,
                        "temporal_partition_id": period,
                        "decision_year": start,
                    }
                )

    rows, slots, selected = builder._rank_and_assign(
        linked=linked,
        author=author,
        compact=compact,
        candidates=candidates,
        field_meta=field_meta,
        domain_codes={name: index for index, name in enumerate(domain_fields, 1)},
    )

    assert len(rows) == 26
    assert len(slots) == 4
    assert len(selected) == 4
    assert len({row["selected_domain_id"] for row in slots}) == 4
    assert [row["role"] for row in slots].count("g1_replication") == 3
    assert all(len(row["reserve_field_ids"]) == 2 for row in slots)
    assert all(row["reserve_activated"] is False for row in slots)
    slot_table = builder.pa.Table.from_pylist(slots, schema=builder.SLOT_SCHEMA)
    assert slot_table.schema.field("reserve_field_ids").type.list_size == 2
    g4 = next(row for row in slots if row["role"] == "g4_domain_author_cold")
    assert g4["sealed"] is True
    assert g4["author_cold"] is True
    assert Fraction(
        rows[-1]["minimum_structural_coverage_numerator"],
        rows[-1]["minimum_structural_coverage_denominator"],
    ) > 0

    first = slots[0]
    without_nominal_period = [
        row
        for row in candidates
        if row["temporal_partition_id"] != "model_selection"
        and not (
            row["field_id"] == first["nominal_primary_field_id"]
            and row["temporal_partition_id"]
            == first["temporal_partition_id"]
        )
    ]
    _, activated_slots, _ = builder._rank_and_assign(
        linked=linked,
        author=author,
        compact=compact,
        candidates=without_nominal_period,
        field_meta=field_meta,
        domain_codes={name: index for index, name in enumerate(domain_fields, 1)},
    )

    activated = activated_slots[0]
    assert activated["nominal_primary_field_id"] == first["nominal_primary_field_id"]
    assert activated["selected_field_id"] == first["reserve_field_ids"][0]
    assert activated["reserve_activated"] is True
    assert activated["reserve_activation_reason"].startswith(
        "NOMINAL_PRIMARY_FEWER_THAN_THREE"
    )
