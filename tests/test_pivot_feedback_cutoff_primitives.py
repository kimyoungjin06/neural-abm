from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pivot_feedback_cutoff_primitives.py"
VERIFIER_SCRIPT = ROOT / "scripts/verify_pivot_feedback_cutoff_primitives.py"
CONFIG = ROOT / "experiments/configs/pivot_feedback_cutoff_primitives_v0.yaml"
DOC = ROOT / "docs/research/pivot-feedback-cutoff-primitives-v0.md"


def _load(name: str, path: Path):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load("pivot_feedback_cutoff_primitives_test_module", SCRIPT)
VERIFIER = _load("pivot_feedback_cutoff_primitives_verifier_test_module", VERIFIER_SCRIPT)


def test_config_is_hash_frozen_and_claim_bounded() -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == MODULE.CONFIG_SHA256
    config, digest = MODULE.load_config(CONFIG)
    assert digest == MODULE.CONFIG_SHA256
    assert config["source"]["raw_rows"] == 492_293_207
    assert config["source"]["retained_rows"] == 492_293_206
    assert "cited_by_count" in config["source"]["forbidden_columns"]
    assert "fwci" in config["source"]["forbidden_columns"]
    assert config["cutoff_predicate_abi"]["fixed_outcome_horizon"] == "UNRESOLVED_REQUIRES_G0_SUCCESSOR"
    assert config["cutoff_predicate_abi"]["candidate_author_eligibility"] == "UNRESOLVED_REQUIRES_G0_SUCCESSOR"
    assert config["scope"]["full_component_may_establish_g_minus_1"] is False
    assert config["execution"]["gpu_used"] is False
    assert DOC.is_file()


def test_full_input_profile_closes_and_codes_are_lossless() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    profile = config["input_profile"]
    assert sum(profile["exact_work_type_counts"].values()) == config["source"]["raw_rows"]
    dictionary = profile["work_type_dictionary"]
    assert sorted(dictionary) == list(range(1, 25))
    assert set(dictionary.values()) == set(profile["exact_work_type_counts"])
    assert profile["null_counts"]["primary_topic__id"] == profile["null_counts"]["primary_topic__field__id"]
    assert profile["null_counts"]["primary_topic__id"] == profile["null_counts"]["primary_topic__domain__id"]
    assert MODULE.RECORD_DTYPE.itemsize == 19


def test_cutoff_predicates_have_exact_t_minus_one_boundary() -> None:
    years = np.array([2011, 2012, 2012, 2013, 0], dtype="<u2")
    nulls = np.array([False, False, False, False, True])
    dispositions = np.array([2, 3, 1, 2, 0], dtype="u1")
    assert MODULE.precutoff_work_mask(years, nulls, dispositions, 2013).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    cutoff = MODULE.annual_cutoff_day(2013)
    available = np.array([cutoff - 1, cutoff, cutoff + 1], dtype="<i4")
    assert MODULE.citation_pre_cutoff_mask(available, 2013).tolist() == [True, True, False]


def test_nullable_identifier_parser_preserves_null_and_rejects_zero() -> None:
    values = pa.array(["https://openalex.org/T100", None, "https://openalex.org/T7"])
    parsed, nulls = MODULE._nullable_numeric_id(
        values,
        pattern=MODULE.TOPIC_PATTERN,
        prefix_length=len("https://openalex.org/T"),
        dtype="<u4",
        field="topic",
    )
    assert parsed.tolist() == [100, 0, 7]
    assert nulls.tolist() == [False, True, False]
    with pytest.raises(MODULE.CutoffPrimitivesError, match="reserved for null"):
        MODULE._nullable_numeric_id(
            pa.array(["https://openalex.org/T0"]),
            pattern=MODULE.TOPIC_PATTERN,
            prefix_length=len("https://openalex.org/T"),
            dtype="<u4",
            field="topic",
        )


def test_independent_tree_digest_detects_order_or_content_change() -> None:
    records = [
        {"relative_path": "a", "bytes": 1, "sha256": "a" * 64},
        {"relative_path": "b", "bytes": 2, "sha256": "b" * 64},
    ]
    assert VERIFIER._tree_digest(records) == MODULE.canonical._tree_digest(records)
    assert VERIFIER._tree_digest(records) != VERIFIER._tree_digest(list(reversed(records)))
    changed = [dict(records[0]), dict(records[1])]
    changed[1]["bytes"] = 3
    assert VERIFIER._tree_digest(records) != VERIFIER._tree_digest(changed)
