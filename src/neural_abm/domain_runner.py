"""Shared lifecycle runner for domain-specific Toy6-10 simulations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from neural_abm.logging import CsvLogWriter
from neural_abm.results import (
    DomainToyResult,
    write_domain_summary_artifact,
    write_run_metadata_artifacts,
)


StateT = TypeVar("StateT")
StepT = TypeVar("StepT")


class DomainToyAdapter(Protocol[StateT, StepT]):
    """Toy-specific callbacks required by :class:`DomainToyRunner`."""

    def initialize(self) -> StateT:
        ...

    def step_epochs(self, state: StateT) -> Iterable[int]:
        ...

    def step(self, epoch: int, state: StateT) -> StepT | None:
        ...

    def fallback_step(self, state: StateT) -> StepT | None:
        ...

    def aggregate_row(
        self,
        epoch: int,
        state: StateT,
        step: StepT,
    ) -> Mapping[str, object]:
        ...

    def micro_rows(
        self,
        epoch: int,
        state: StateT,
        step: StepT,
    ) -> Iterable[Mapping[str, object]]:
        ...

    def final_epoch(self, state: StateT, step: StepT) -> int:
        ...

    def domain_metrics(
        self,
        final_row: Mapping[str, object],
        state: StateT,
        step: StepT,
    ) -> Mapping[str, object]:
        ...


@dataclass(frozen=True)
class DomainRunSettings:
    """Run-level settings shared by Toy6-10-compatible domain runners."""

    toy: str
    config: Any
    config_path: Path
    output_dir: Path
    run_name: str
    seed: int
    micro_state_fields: Sequence[str]
    aggregate_fields: Sequence[str]
    metadata: Mapping[str, Any]
    logging_interval: int
    log_micro_state: bool
    log_aggregate_metrics: bool
    no_step_error: str
    strict_capability: bool = True


def make_timestamped_run_dir(*, output_dir: Path, run_name: str, seed: int) -> Path:
    """Create the timestamped run directory shape used by domain toy runners."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_dir / f"{timestamp}_{run_name}_seed{seed:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_domain_run_dir(settings: DomainRunSettings) -> Path:
    """Create a run directory from shared domain runner settings."""

    return make_timestamped_run_dir(
        output_dir=settings.output_dir,
        run_name=settings.run_name,
        seed=settings.seed,
    )


def write_domain_run_metadata(settings: DomainRunSettings, run_dir: Path) -> None:
    """Write the standard metadata artifacts for a domain toy run."""

    write_run_metadata_artifacts(
        config_path=settings.config_path,
        config=settings.config,
        run_dir=run_dir,
        toy=settings.toy,
        metadata=settings.metadata,
        strict_capability=settings.strict_capability,
    )


@dataclass
class DomainToyRunner(Generic[StateT, StepT]):
    """Generic lifecycle runner for Toy6-10-style domain simulations."""

    adapter: DomainToyAdapter[StateT, StepT]
    settings: DomainRunSettings

    def run(self) -> DomainToyResult:
        run_dir = make_domain_run_dir(self.settings)
        write_domain_run_metadata(self.settings, run_dir)
        state = self.adapter.initialize()
        final_step: StepT | None = None

        with (
            CsvLogWriter(
                run_dir / "micro_state.csv",
                self.settings.micro_state_fields,
            ) as micro,
            CsvLogWriter(
                run_dir / "aggregate_metrics.csv",
                self.settings.aggregate_fields,
            ) as aggregate,
        ):
            for epoch in self.adapter.step_epochs(state):
                step = self.adapter.step(epoch, state)
                if step is None:
                    break
                final_step = step
                row = self.adapter.aggregate_row(epoch, state, step)
                if self.settings.log_aggregate_metrics:
                    aggregate.write(dict(row))
                if (
                    self.settings.log_micro_state
                    and epoch % self.settings.logging_interval == 0
                ):
                    self._write_micro_rows(micro, epoch, state, step)

            if final_step is None:
                final_step = self.adapter.fallback_step(state)
                if final_step is None:
                    raise RuntimeError(self.settings.no_step_error)
                fallback_epoch = self.adapter.final_epoch(state, final_step)
                aggregate.write(
                    dict(self.adapter.aggregate_row(fallback_epoch, state, final_step))
                )
                self._write_micro_rows(micro, fallback_epoch, state, final_step)

        final_epoch = self.adapter.final_epoch(state, final_step)
        final_row = self.adapter.aggregate_row(final_epoch, state, final_step)
        domain_metrics = dict(
            self.adapter.domain_metrics(final_row, state, final_step)
        )
        write_domain_summary_artifact(
            run_dir=run_dir,
            toy=self.settings.toy,
            final_fragmentation_components=final_row["fragmentation_components"],
            domain_metrics=domain_metrics,
            strict_capability=self.settings.strict_capability,
        )
        return DomainToyResult(
            run_dir=run_dir,
            toy=self.settings.toy,
            final_fragmentation_components=int(
                final_row["fragmentation_components"]
            ),
            domain_metrics=domain_metrics,
        )

    def _write_micro_rows(
        self,
        writer: CsvLogWriter,
        epoch: int,
        state: StateT,
        step: StepT,
    ) -> None:
        for row in self.adapter.micro_rows(epoch, state, step):
            writer.write(dict(row))
