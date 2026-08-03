from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pivot_feedback_d0_rank5_local_abstract_closeout_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("rank5_local_abstract_closeout", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_closeout_inputs_recompute_unresolved_tables() -> None:
    module = _load()
    config = module.load_config(module.DEFAULT_CONFIG)
    dispositions, frontier, candidate, _ = module.inputs(config)
    dispositions_out, frontier_out, counts = module.update_tables(
        dispositions, frontier, candidate, config
    )
    assert dispositions_out.num_rows == 8
    assert frontier_out.num_rows == 51
    assert counts["total_content_pass_rows"] == 3
    assert counts["total_content_fail_rows"] == 4
    assert counts["total_content_unresolved_rows"] == 1
    rank5 = next(
        row
        for row in dispositions_out.to_pylist()
        if row["field_id"] == 27 and row["field_candidate_rank"] == 5
    )
    assert rank5["content_state"] == "CONTENT_UNRESOLVED"
    assert rank5["human_review_used"] is False
    assert rank5["api_labels_used_as_ground_truth"] is False
