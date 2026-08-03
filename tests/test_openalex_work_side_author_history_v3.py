from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_work_side_author_history_v3.py"
VERIFIER_PATH = (
    REPO_ROOT / "scripts/verify_openalex_work_side_author_history_v3.py"
)
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_work_side_author_history_v3.yaml"
)


def _load_module():
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("author_history_v3_test_module", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _load_verifier_module():
    spec = importlib.util.spec_from_file_location(
        "author_history_v3_verifier_test_module", VERIFIER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_partition(
    root: Path,
    partition: int,
    work: list[int],
    year: list[int | None],
    disposition: list[int],
) -> None:
    path = root / "spine" / f"part-{partition:04d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "work_id_numeric": pa.array(work, type=pa.uint64()),
            "publication_day": pa.array(
                [None if value is None else 0 for value in year], type=pa.int32()
            ),
            "publication_year": pa.array(year, type=pa.uint16()),
            "temporal_disposition": pa.array(disposition, type=pa.uint8()),
        }
    )
    pq.write_table(table, path, use_dictionary=False)


def test_config_is_exact_and_predecessors_are_preserved() -> None:
    assert _sha256(CONFIG_PATH) == MODULE.CONFIG_SHA256
    config, digest = MODULE.load_config(CONFIG_PATH)
    assert digest == MODULE.CONFIG_SHA256
    assert config["semantics"]["unit"] == "CURRENT_SNAPSHOT_OPENALEX_AUTHOR_ID_TOKEN_NOT_PERSON"
    assert config["execution"]["closure_strategy"] == "RECEIPT_BOUND_AUTHOR_WORK_EVENT_ARTIFACT"
    assert config["scope"]["full_component_may_establish_g_minus_1"] is False


def test_year_and_token_reductions_close_exactly() -> None:
    events = np.empty(5, dtype=MODULE.EVENT_DTYPE)
    events["author"] = [1, 1, 1, 2, 2]
    events["work"] = [10, 11, 12, 20, 21]
    events["year"] = [2000, 2000, 2002, 2001, 2003]
    events["disposition"] = [2, 2, 2, 2, 2]
    events["team"] = [2, 11, 51, 1, 101]
    events["field"] = [1, 1, 2, 0, 3]
    year, token = MODULE._year_and_token_arrays(events)

    assert year["author"].tolist() == [1, 1, 2, 2]
    assert year["year"].tolist() == [2000, 2002, 2001, 2003]
    assert year["all"].tolist() == [2, 1, 1, 1]
    assert year["cap10"].tolist() == [1, 0, 1, 0]
    assert year["cap50"].tolist() == [2, 0, 1, 0]
    assert year["cap100"].tolist() == [2, 1, 1, 0]

    assert token["author"].tolist() == [1, 2]
    assert token["first"].tolist() == [2000, 2001]
    assert token["last"].tolist() == [2002, 2003]
    assert token["active"].tolist() == [2, 2]
    assert token["span"].tolist() == [3, 3]
    assert token["gap"].tolist() == [1, 1]
    assert token["total_all"].tolist() == [3, 2]
    assert token["max_all"].tolist() == [2, 1]


def test_independent_semantic_digest_ignores_arrow_chunk_boundaries() -> None:
    one_chunk = pa.table(
        {
            "value": pa.array([1, 2, 3, 4], type=pa.uint64()),
        }
    )
    two_chunks = pa.table(
        {
            "value": pa.chunked_array(
                [
                    pa.array([1, 2], type=pa.uint64()),
                    pa.array([3, 4], type=pa.uint64()),
                ]
            ),
        }
    )

    assert VERIFIER.semantic_digest(one_chunk) == VERIFIER.semantic_digest(two_chunks)


def test_closed_core_pipeline_joins_deduplicates_and_reduces(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(MODULE, "PARTITION_COUNT", 2)
    monkeypatch.setattr(VERIFIER, "PARTITION_COUNT", 2)
    output = tmp_path / "output"
    canonical_root = tmp_path / "canonical"
    for relative in (
        "checkpoints/work",
        "checkpoints/author",
        "events",
        "author_year_history",
        "author_token_history",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)

    _write_canonical_partition(canonical_root, 0, [0, 2], [2000, None], [2, 0])
    _write_canonical_partition(canonical_root, 1, [1], [2026], [3])

    main_spools: list[Path] = []
    slot_spools: list[Path] = []
    main_values = [
        [(0, 3, 1), (2, 1, 0)],
        [(1, 3, 2)],
    ]
    slot_values = [
        [(0, 0), (0, 2), (0, 2), (2, 1)],
        [(1, 1), (1, int(MODULE.NULL_AUTHOR_SENTINEL)), (1, 3)],
    ]
    for partition in range(2):
        main_path = tmp_path / f"main-{partition}.bin"
        main = np.array(main_values[partition], dtype=MODULE.MAIN_DTYPE)
        main.tofile(main_path)
        main_spools.append(main_path)
        slot_path = tmp_path / f"slot-{partition}.bin"
        slots = np.array(slot_values[partition], dtype=MODULE.SLOT_DTYPE)
        slots.tofile(slot_path)
        slot_spools.append(slot_path)

    work_checkpoints, event_spools = MODULE.process_work_partitions(
        output=output,
        canonical_root=canonical_root,
        main_spools=main_spools,
        slot_spools=slot_spools,
        task_fingerprint="a" * 64,
        full_scope=True,
    )
    work = MODULE._sum_work_metrics(work_checkpoints)
    assert work["raw_authorship_slots_joined"] == 7
    assert work["grammar_valid_author_slots"] == 6
    assert work["null_author_slots"] == 1
    assert work["distinct_author_work_pairs"] == 5
    assert work["duplicate_author_work_pair_excess_rows"] == 1
    assert work["history_usable_distinct_author_work_pairs"] == 4

    author_checkpoints = MODULE.process_author_partitions(
        output=output,
        event_spools=event_spools,
        task_fingerprint="a" * 64,
    )
    author = MODULE._sum_author_metrics(author_checkpoints)
    assert author["event_rows"] == 5
    assert author["history_usable_event_rows"] == 4
    assert author["all_distinct_author_token_rows"] == 4
    assert author["author_year_rows"] == 4
    assert author["author_token_rows"] == 4
    assert author["a_zero_event_rows"] == 1
    assert author["a_zero_token_rows"] == 1
    assert author["pair_counts_by_cap"] == {
        "all": 4,
        "cap10": 4,
        "cap50": 4,
        "cap100": 4,
    }

    for partition in range(2):
        checkpoint = author_checkpoints[partition]
        for output_contract in checkpoint["outputs"].values():
            path = output / output_contract["relative_path"]
            assert path.is_file()
            assert _sha256(path) == output_contract["sha256"]

        independently_verified = VERIFIER.verify_author_partition(
            output, partition, "a" * 64
        )
        assert independently_verified == {
            key: checkpoint[key]
            for key in (
                "event_rows",
                "history_usable_event_rows",
                "all_distinct_author_token_rows",
                "author_year_rows",
                "author_token_rows",
                "a_zero_event_rows",
                "a_zero_token_rows",
                "pair_counts_by_cap",
                "sensitivity",
            )
        }


def test_full_execution_requires_both_acknowledgements() -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            "--source-root",
            "/source",
            "--content-receipt",
            "/content/receipt",
            "--temporal-receipt",
            "/temporal/receipt",
            "--canonical-root",
            "/canonical",
            "--output-dir",
            "/output",
            "--execute",
        ]
    )
    assert args.authorize_full is False
    assert args.acknowledge_exclusive_source_scan is False
    assert args.max_main_files is None
    assert args.max_authorship_files is None
