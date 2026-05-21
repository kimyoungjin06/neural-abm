from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from neural_abm.results import (
    write_binary_summary_artifact,
    write_domain_summary_artifact,
    write_run_metadata_artifacts,
)


class DummyConfig:
    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {"run": {"seed": 1}, "model": {"policy": {"rule": "dummy"}}}


def test_write_run_metadata_artifacts_records_capability_contract(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "input.yaml"
    config_path.write_text("run:\n  seed: 1\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    payload = write_run_metadata_artifacts(
        config_path=config_path,
        config=DummyConfig(),
        run_dir=run_dir,
        toy="toy2",
        metadata={"run_name": "demo", "policy_rule": "neural_policy"},
    )

    assert (run_dir / "config.yaml").read_text(encoding="utf-8") == (
        "run:\n  seed: 1\n"
    )
    assert yaml.safe_load((run_dir / "resolved_config.yaml").read_text()) == {
        "run": {"seed": 1},
        "model": {"policy": {"rule": "dummy"}},
    }
    assert json.loads((run_dir / "metadata.json").read_text()) == payload
    assert payload["nabm_status"] == "full"
    assert payload["reference_policies"] == [
        "rd_well_mixed",
        "fermi_imitation",
        "reputation_imitation",
    ]


def test_write_domain_summary_artifact_records_capability_contract(
    tmp_path: Path,
) -> None:
    payload = write_domain_summary_artifact(
        run_dir=tmp_path,
        toy="toy6",
        final_fragmentation_components=2,
        domain_metrics={"domain_metric": 1.5},
    )

    assert json.loads((tmp_path / "summary.json").read_text()) == payload
    assert payload["nabm_status"] == "compatible"
    assert payload["reference_policies"] == []
    assert payload["domain_metrics"] == {"domain_metric": 1.5}


def test_write_binary_summary_artifact_can_skip_unknown_test_toy_capability(
    tmp_path: Path,
) -> None:
    payload = write_binary_summary_artifact(
        run_dir=tmp_path,
        toy="fake",
        final_action_rate=0.5,
        final_mean_payoff=1.0,
        final_fragmentation_components=3,
        final_mean_policy_action_probability=0.6,
        final_mean_reputation=0.7,
        final_reputation_dispersion=0.1,
        domain_metrics={"domain_metric": 2.0},
        strict_capability=False,
    )

    assert "nabm_status" not in payload
    assert json.loads((tmp_path / "summary.json").read_text()) == payload
