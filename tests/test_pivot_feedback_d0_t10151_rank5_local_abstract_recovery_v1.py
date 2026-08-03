from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pivot_feedback_d0_t10151_rank5_local_abstract_recovery_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("local_abstract_recovery", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reconstruct_and_eligibility_are_deterministic() -> None:
    module = _load()
    value = json.dumps({"Hepatitis": [0], "C": [1], "study": [2]})
    reconstructed = module.reconstruct_abstract(value)
    assert reconstructed["abstract"] == "Hepatitis C study"
    assert reconstructed["position_count"] == 3
    assert reconstructed["unique_positions"] == 3
    assert reconstructed["collisions"] == 0
    assert module.eligibility_state(
        reconstructed,
        minimum_words=3,
        maximum_collisions=0,
    ) == ("ELIGIBLE_ABSTRACT_REVIEW", 3)
    assert module.eligibility_state(
        reconstructed,
        minimum_words=4,
        maximum_collisions=0,
    ) == ("FRAGMENT_UNUSABLE", 3)


def test_scan_locations_finds_each_id_once(tmp_path: Path) -> None:
    module = _load()
    first = tmp_path / "base-1.parquet"
    second = tmp_path / "delta-0.parquet"
    pq.write_table(
        pa.table({"id": ["https://openalex.org/W1", "https://openalex.org/W2"], "value": [None, "{}"]}),
        first,
        row_group_size=1,
    )
    pq.write_table(
        pa.table({"id": ["https://openalex.org/W3"], "value": [None]}),
        second,
        row_group_size=1,
    )
    locations = module.scan_locations(
        [first, second],
        ["https://openalex.org/W2", "https://openalex.org/W3"],
        workers=2,
    )
    assert locations["https://openalex.org/W2"][0] == first
    assert locations["https://openalex.org/W2"][1] == 1
    assert locations["https://openalex.org/W3"][0] == second
    assert locations["https://openalex.org/W3"][1] == 0
