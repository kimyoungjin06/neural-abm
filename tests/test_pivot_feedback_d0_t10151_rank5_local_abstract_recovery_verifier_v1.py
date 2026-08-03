from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_t10151_rank5_local_abstract_recovery_v1.py"
SOURCE_ROOT = Path(
    "/hdd/raid_backup_20260624/data/OpenAlex/parquet_exports/"
    "openalex_works_20260330_repairreplay_20260410_190630"
)


def _load():
    spec = importlib.util.spec_from_file_location("local_abstract_recovery_verifier", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_abstract_recovery_is_independently_verified() -> None:
    result = _load().validate(source_root=SOURCE_ROOT)
    assert result["status"] == "PASS_INDEPENDENT_EXACT_16_LOCAL_ABSTRACT_RECOVERY_VERIFICATION"
    assert result["source_rows_verified"] == 16
    assert result["reconstructable_rows"] == 6
    assert result["eligible_review_rows"] == 5
    assert result["missing_rows"] == 10
    assert result["fragment_rows"] == 1
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
    assert result["semantic_labels_created"] == 0
    assert result["production_disposition_authorized"] is False
