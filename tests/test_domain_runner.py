from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from neural_abm.domain_runner import (
    DomainRunSettings,
    DomainToyRunner,
    make_domain_run_dir,
    write_domain_run_metadata,
)


@dataclass(frozen=True)
class FakeConfig:
    payload: dict[str, object]

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        assert mode == "json"
        return self.payload


@dataclass(frozen=True)
class FakeStep:
    epoch: int
    value: int
    fragmentation_components: int = 1


@dataclass
class FakeState:
    steps: int = 0


@dataclass
class FakeAdapter:
    epochs: int = 3
    fallback_enabled: bool = False

    def initialize(self) -> FakeState:
        return FakeState()

    def step_epochs(self, state: FakeState) -> range:
        return range(1, self.epochs + 1)

    def step(self, epoch: int, state: FakeState) -> FakeStep:
        state.steps += 1
        return FakeStep(epoch=epoch, value=epoch * 10)

    def fallback_step(self, state: FakeState) -> FakeStep | None:
        if not self.fallback_enabled:
            return None
        return FakeStep(epoch=0, value=99, fragmentation_components=7)

    def aggregate_row(
        self,
        epoch: int,
        state: FakeState,
        step: FakeStep,
    ) -> dict[str, object]:
        return {
            "epoch": epoch,
            "value": step.value,
            "fragmentation_components": step.fragmentation_components,
        }

    def micro_rows(
        self,
        epoch: int,
        state: FakeState,
        step: FakeStep,
    ) -> list[dict[str, object]]:
        return [{"epoch": epoch, "agent_id": 0, "value": step.value}]

    def final_epoch(self, state: FakeState, step: FakeStep) -> int:
        return step.epoch

    def domain_metrics(
        self,
        final_row: dict[str, object],
        state: FakeState,
        step: FakeStep,
    ) -> dict[str, object]:
        return {
            "domain_final_value": final_row["value"],
            "domain_steps": state.steps,
        }


def fake_settings(
    tmp_path: Path,
    *,
    logging_interval: int = 2,
    log_micro_state: bool = True,
    log_aggregate_metrics: bool = True,
) -> DomainRunSettings:
    config_path = tmp_path / "fake.yaml"
    config_path.write_text("run:\n  name: fake_run\n", encoding="utf-8")
    return DomainRunSettings(
        toy="fake",
        config=FakeConfig({"run": {"name": "fake_run"}}),
        config_path=config_path,
        output_dir=tmp_path / "runs",
        run_name="fake_run",
        seed=5,
        micro_state_fields=["epoch", "agent_id", "value"],
        aggregate_fields=["epoch", "value", "fragmentation_components"],
        metadata={"toy": "fake", "run_name": "fake_run", "seed": 5},
        logging_interval=logging_interval,
        log_micro_state=log_micro_state,
        log_aggregate_metrics=log_aggregate_metrics,
        no_step_error="fake produced no steps",
        strict_capability=False,
    )


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_domain_runner_writes_artifacts_and_respects_micro_interval(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    settings = fake_settings(tmp_path)

    result = DomainToyRunner(adapter, settings).run()

    assert result.toy == "fake"
    assert result.final_fragmentation_components == 1
    assert result.domain_metrics == {"domain_final_value": 30, "domain_steps": 3}
    assert json.loads((result.run_dir / "metadata.json").read_text()) == {
        "run_name": "fake_run",
        "seed": 5,
        "toy": "fake",
    }
    assert (result.run_dir / "config.yaml").read_text(encoding="utf-8") == (
        "run:\n  name: fake_run\n"
    )
    assert yaml.safe_load((result.run_dir / "resolved_config.yaml").read_text()) == {
        "run": {"name": "fake_run"}
    }
    assert csv_rows(result.run_dir / "aggregate_metrics.csv") == [
        {"epoch": "1", "value": "10", "fragmentation_components": "1"},
        {"epoch": "2", "value": "20", "fragmentation_components": "1"},
        {"epoch": "3", "value": "30", "fragmentation_components": "1"},
    ]
    assert csv_rows(result.run_dir / "micro_state.csv") == [
        {"epoch": "2", "agent_id": "0", "value": "20"}
    ]
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["toy"] == "fake"
    assert summary["domain_metrics"] == {
        "domain_final_value": 30,
        "domain_steps": 3,
    }


def test_domain_run_artifact_helpers_use_settings(tmp_path: Path) -> None:
    settings = fake_settings(tmp_path)

    run_dir = make_domain_run_dir(settings)
    write_domain_run_metadata(settings, run_dir)

    assert run_dir.exists()
    assert run_dir.name.endswith("_fake_run_seed05")
    assert json.loads((run_dir / "metadata.json").read_text()) == {
        "run_name": "fake_run",
        "seed": 5,
        "toy": "fake",
    }
    assert (run_dir / "config.yaml").read_text(encoding="utf-8") == (
        "run:\n  name: fake_run\n"
    )
    assert yaml.safe_load((run_dir / "resolved_config.yaml").read_text()) == {
        "run": {"name": "fake_run"}
    }


def test_domain_runner_respects_disabled_logging(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    settings = fake_settings(
        tmp_path,
        log_micro_state=False,
        log_aggregate_metrics=False,
    )

    result = DomainToyRunner(adapter, settings).run()

    assert csv_rows(result.run_dir / "aggregate_metrics.csv") == []
    assert csv_rows(result.run_dir / "micro_state.csv") == []
    assert json.loads((result.run_dir / "summary.json").read_text())["domain_metrics"][
        "domain_final_value"
    ] == 30


def test_domain_runner_writes_fallback_rows_even_when_logging_disabled(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter(
        epochs=0,
        fallback_enabled=True,
    )
    settings = fake_settings(
        tmp_path,
        log_micro_state=False,
        log_aggregate_metrics=False,
    )

    result = DomainToyRunner(adapter, settings).run()

    assert result.final_fragmentation_components == 7
    assert csv_rows(result.run_dir / "aggregate_metrics.csv") == [
        {"epoch": "0", "value": "99", "fragmentation_components": "7"}
    ]
    assert csv_rows(result.run_dir / "micro_state.csv") == [
        {"epoch": "0", "agent_id": "0", "value": "99"}
    ]


def test_domain_runner_raises_when_no_step_and_no_fallback(tmp_path: Path) -> None:
    adapter = FakeAdapter(epochs=0)
    settings = fake_settings(tmp_path)

    with pytest.raises(RuntimeError, match="fake produced no steps"):
        DomainToyRunner(adapter, settings).run()
