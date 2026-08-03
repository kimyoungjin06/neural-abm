from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
BUILDER = (
    SCRIPTS
    / "build_pivot_feedback_d0_certified_raw_authorship_identity_affiliation_v1.py"
)
VERIFIER = (
    SCRIPTS
    / "verify_pivot_feedback_d0_certified_raw_authorship_identity_affiliation_v1.py"
)
CONFIG = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_certified_raw_authorship_identity_affiliation_v1.yaml"
)
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v39.yaml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load(BUILDER, "certified_raw_identity_affiliation_builder_test")
VERIFIER_MODULE = _load(
    VERIFIER,
    "certified_raw_identity_affiliation_verifier_test",
)


def test_config_binds_v39_and_keeps_outcomes_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == MODULE.CONFIG_SCHEMA
    assert config["repository_bindings"]["gate_state"]["sha256"] == hashlib.sha256(
        GATE.read_bytes()
    ).hexdigest()
    assert config["affiliation_contract"]["graph_team_cap_profiles"] == [10, 50, 100]
    assert config["affiliation_contract"]["raw_affiliation_string_matching"] == (
        "FORBIDDEN"
    )
    assert config["identity_contract"]["natural_person_identity_claim_allowed"] is False
    assert all(
        value is False
        for value in config["forbidden_inputs_and_outputs"].values()
    )


def test_orcid_checksum_rule_accepts_x_and_rejects_bad_checksum() -> None:
    assert MODULE._orcid_checksum_valid(
        "https://orcid.org/0000-0002-1825-0097"
    )
    assert MODULE._orcid_checksum_valid(
        "https://orcid.org/0000-0002-1694-233X"
    )
    assert not MODULE._orcid_checksum_valid(
        "https://orcid.org/0000-0002-1825-0098"
    )
    assert not MODULE._orcid_checksum_valid("0000-0002-1825-0097")
    assert not MODULE._orcid_checksum_valid(None)


def test_output_schemas_keep_coverage_and_same_affiliation_separate() -> None:
    assert MODULE.EDGE_SCHEMA.metadata == {
        b"schema_version": MODULE.EDGE_SCHEMA_VERSION.encode()
    }
    assert (
        "has_current_snapshot_work_time_institution_coverage"
        in MODULE.EDGE_SCHEMA.names
    )
    assert "has_same_affiliation_shared_prior_work" in MODULE.EDGE_SCHEMA.names
    for cap in MODULE.CAPS:
        assert (
            f"has_institution_comparable_prior_peer_cap_{cap}"
            in MODULE.AUTHOR_SCHEMA.names
        )
        assert (
            f"has_same_affiliation_bridge_cap_{cap}"
            in MODULE.AUTHOR_SCHEMA.names
        )
    source = MODULE._author_output_sql()
    assert "minimum_institution_comparable_team_size BETWEEN 1" in source
    assert "minimum_same_affiliation_team_size BETWEEN 1" in source


def test_verifier_is_pair_first_and_does_not_import_producer() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert (
        "import build_pivot_feedback_d0_certified_raw_authorship_identity_affiliation_v1"
        not in source
    )
    assert "PAIR_FIRST_AUTHOR_ORCID_GROUP_THEN_TOKEN_REDUCTION" in source
    assert "DUCKDB_PEER_WORK_FIRST_THEN_CANDIDATE_REVERSE_JOIN" in source
    assert "EXCEPT ALL" in source
    assert VERIFIER_MODULE.VERIFY_STATUS.startswith(
        "PASS_INDEPENDENT_EXACT_FOUR"
    )
