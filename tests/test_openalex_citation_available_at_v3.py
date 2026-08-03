from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/audit_openalex_citation_available_at_v3.py"
VERIFIER_SCRIPT = REPO_ROOT / "scripts/verify_openalex_citation_available_at_v3.py"
CONFIG = REPO_ROOT / "experiments/configs/openalex_g1_citation_available_at_v2.yaml"


def _load(name: str, path: Path):
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load("citation_available_at_v3_test_module", SCRIPT)
VERIFIER = _load("citation_available_at_v3_verifier_test_module", VERIFIER_SCRIPT)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packed(day: int, disposition: int) -> np.uint32:
    return np.uint32(((day + MODULE.DAY_BIAS) << 3) | (disposition + 1))


def test_production_config_is_exact_and_preserves_frozen_contracts() -> None:
    assert _sha256(CONFIG) == MODULE.CONFIG_SHA256
    config, digest = MODULE.load_config(CONFIG)
    assert digest == MODULE.CONFIG_SHA256
    assert config["semantics"]["edge_direction"] == "CITING_WORK_TO_CITED_WORK"
    assert config["semantics"]["openalex_first_observation_time_established"] is False
    assert config["semantics"]["maturation_policy"] == "PENDING_SEPARATE_HASHED_G0_POLICY"
    assert config["scope"]["full_component_may_establish_g_minus_1"] is False
    assert config["resource_envelope"]["gpu_used"] is False


def test_lookup_encoding_round_trips_all_dispositions(tmp_path: Path) -> None:
    path = tmp_path / "lookup.u32"
    lookup = MODULE._new_sparse_memmap(path, "<u4", 10)
    expected_days = np.array([-354_000, -1, 0, 20_000, 29_000, 1_000], dtype="<i4")
    expected_dispositions = np.arange(6, dtype="u1")
    values = np.array(
        [_packed(int(day), int(disposition)) for day, disposition in zip(expected_days, expected_dispositions, strict=True)],
        dtype="<u4",
    )
    lookup[np.arange(6)] = values
    days, dispositions = MODULE._decode_packed(lookup[np.arange(6)])
    assert np.array_equal(days, expected_days)
    assert np.array_equal(dispositions, expected_dispositions)


def test_closed_partition_pipeline_and_independent_classifier(
    tmp_path: Path, monkeypatch
) -> None:
    partition_count = 4
    monkeypatch.setattr(MODULE, "PARTITION_COUNT", partition_count)
    monkeypatch.setattr(MODULE.helper, "PARTITION_COUNT", partition_count)
    monkeypatch.setattr(VERIFIER, "PARTITION_COUNT", partition_count)
    output = tmp_path / "output"
    for relative in (
        "checkpoints/pair",
        "checkpoints/safe",
        "deduplicated_edge_events",
        "chronology_safe_edges_by_cited_work",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)

    lookup_path = tmp_path / "lookup.u32"
    lookup = MODULE._new_sparse_memmap(lookup_path, "<u4", 100)
    lookup[1] = _packed(100, 2)
    lookup[2] = _packed(90, 2)
    lookup[3] = _packed(110, 2)
    lookup[4] = _packed(0, 0)
    lookup[5] = _packed(80, 1)

    pairs = [
        (1, 2),
        (1, 2),
        (1, 1),
        (1, 3),
        (1, 99),
        (4, 2),
        (1, 5),
        (3, 2),
        (2, 3),
        (2, 2),
    ]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_spools = []
    for partition in range(partition_count):
        values = np.array(
            [pair for pair in pairs if pair[0] % partition_count == partition],
            dtype=MODULE.PAIR_DTYPE,
        )
        path = raw_root / f"pairs-{partition:04d}.bin"
        values.tofile(path)
        raw_spools.append(path)

    checkpoints, safe_spools, result = MODULE.process_pair_partitions(
        output=output,
        raw_spools=raw_spools,
        lookup=lookup,
        task_fingerprint="a" * 64,
    )
    assert len(checkpoints) == partition_count
    assert result["aggregate"]["raw_pair_rows"] == 10
    assert result["aggregate"]["unique_pair_rows"] == 9
    assert result["aggregate"]["duplicate_pair_groups"] == 1
    assert result["aggregate"]["duplicate_pair_row_excess"] == 1
    assert result["bucket_counts"] == {
        "self_loop": 2,
        "source_temporal_unusable": 1,
        "target_external_open_world": 1,
        "target_temporal_unusable": 1,
        "negative_lag_target_after_source": 2,
        "same_day": 0,
        "positive_lag": 2,
    }

    safe_checkpoints, safe_manifest = MODULE.process_safe_partitions(
        output=output,
        safe_spools=safe_spools,
        task_fingerprint="a" * 64,
    )
    assert len(safe_checkpoints) == partition_count
    assert safe_manifest["rows"] == 2

    independent_dedup = VERIFIER._verify_deduplicated_rows(
        output, result["deduplicated_manifest"]
    )
    independent_safe = VERIFIER._verify_safe_rows(output, safe_manifest)
    assert independent_dedup["totals"]["raw_pair_rows"] == 10
    assert independent_dedup["buckets"] == result["bucket_counts"]
    assert independent_dedup["sensitivity"] == dict(result["sensitivity"])
    assert independent_safe["rows"] == 2

    totals = independent_dedup["totals"]
    report = {
        "pair_accounting": {
            key: value
            for key, value in totals.items()
            if key not in VERIFIER.SOURCE_ROW_ACCOUNTING_KEYS
        },
        "source_accounting": {
            key: totals[key] for key in VERIFIER.SOURCE_ROW_ACCOUNTING_KEYS
        },
        "exclusive_pair_buckets": independent_dedup["buckets"],
        "chronology_sensitivity_unique_pairs": independent_dedup["sensitivity"],
        "cross_tabs": independent_dedup["cross_tabs"],
        "attribute_margins": independent_dedup["attribute_margins"],
    }
    VERIFIER._verify_report_accounting(
        report, independent_dedup, independent_safe, safe_manifest
    )
