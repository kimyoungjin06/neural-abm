#!/usr/bin/env python3
"""Run the precommitted Pivot delayed-feedback learning prototype v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DEFAULT_CONFIG = (
    ROOT
    / "experiments"
    / "configs"
    / "pivot_delayed_feedback_learning_prototype_v1.yaml"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import research_pivot_learning_study as study2  # noqa: E402
import research_pivot_study as study1  # noqa: E402


FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
VISIBILITY_COMPONENT_ID = 9173
FINAL_METRICS = (
    "pivot_rate",
    "productive_pivot_rate",
    "failed_pivot_rate",
    "productive_share_of_pivots",
)


@dataclass(frozen=True)
class VisibilityRegime:
    """Domain-owned outcome availability rule."""

    regime_id: str
    success_delay_steps: int
    failure_delay_steps: int
    success_visibility_probability: float
    failure_visibility_probability: float

    def __post_init__(self) -> None:
        if not self.regime_id:
            raise ValueError("regime_id must be non-empty")
        for name in ("success_delay_steps", "failure_delay_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        for name in (
            "success_visibility_probability",
            "failure_visibility_probability",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> VisibilityRegime:
        return cls(
            regime_id=str(row["id"]),
            success_delay_steps=int(row["success_delay_steps"]),
            failure_delay_steps=int(row["failure_delay_steps"]),
            success_visibility_probability=float(
                row["success_visibility_probability"]
            ),
            failure_visibility_probability=float(
                row["failure_visibility_probability"]
            ),
        )

    def delay(self, event: study2.PivotEvent) -> int:
        if event.productive:
            return self.success_delay_steps
        return self.failure_delay_steps

    def visibility_probability(self, event: study2.PivotEvent) -> float:
        if event.productive:
            return self.success_visibility_probability
        return self.failure_visibility_probability


@dataclass
class DelayedPivotEnvironment:
    """Separate privately realized events from publicly available messages."""

    neighbors: list[list[int]]
    config: study2.LearningStudyConfig
    replicate: int
    visibility: VisibilityRegime
    step: int = 0
    last_events: list[study2.PivotEvent] = field(default_factory=list)
    new_events: list[study2.PivotEvent] = field(default_factory=list)
    private_events: list[study2.PivotEvent] = field(default_factory=list)
    public_events: list[study2.PivotEvent] = field(default_factory=list)
    never_public_events: list[study2.PivotEvent] = field(default_factory=list)
    release_queue: dict[int, list[study2.PivotEvent]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def _visibility_uniform(self, event: study2.PivotEvent) -> float:
        seed = np.random.SeedSequence(
            (
                self.config.base_seed,
                self.replicate,
                VISIBILITY_COMPONENT_ID,
                event.agent_id,
                event.step,
                int(event.productive),
            )
        )
        return float(np.random.default_rng(seed).random())

    def ingest_private_events(self) -> list[study2.PivotEvent]:
        """Move newly realized outcomes into the domain availability queue."""

        events = list(self.new_events)
        self.new_events = []
        for event in events:
            self.private_events.append(event)
            probability = self.visibility.visibility_probability(event)
            if self._visibility_uniform(event) >= probability:
                self.never_public_events.append(event)
                continue
            release_step = event.step + 1 + self.visibility.delay(event)
            self.release_queue[release_step].append(event)
        return events

    def advance(self) -> None:
        # Compatibility with the predecessor adapter if a caller did not
        # explicitly ingest after the prior unit step.
        self.ingest_private_events()
        self.step += 1
        self.last_events = list(self.release_queue.pop(self.step, []))
        self.public_events.extend(self.last_events)

    def local_noise(self, agent_id: int) -> float:
        rng = study1.component_rng(
            self.config.base_seed,
            self.replicate,
            "local_noise",
            self.step - 1,
            agent_id,
        )
        return float(rng.normal(0.0, self.config.local_noise_scale))

    @property
    def pending_public_events(self) -> list[study2.PivotEvent]:
        return [
            event
            for release_step in sorted(self.release_queue)
            for event in self.release_queue[release_step]
        ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prototype contract must be a mapping")
    validate_contract(payload)
    return payload


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != (
        "pivot_delayed_feedback_learning_prototype.v1"
    ):
        raise ValueError("unexpected prototype schema_version")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("prototype is not in the precommit design state")
    if contract.get("scenario", {}).get("name") != "hot_field_hype":
        raise ValueError("v1 must remain focused on hot_field_hype")
    if contract.get("learning_arms") != ["frozen", "imitative", "cautionary"]:
        raise ValueError("v1 learning arms changed")
    rows = contract.get("visibility_regimes")
    if not isinstance(rows, list):
        raise ValueError("visibility_regimes must be a list")
    regimes = [VisibilityRegime.from_mapping(row) for row in rows]
    if [regime.regime_id for regime in regimes] != [
        "immediate",
        "symmetric_lag_2",
        "failure_lag_2",
        "failure_lag_4",
        "selective_failure_lag_4",
    ]:
        raise ValueError("v1 visibility regimes changed")
    immediate = regimes[0]
    if asdict(immediate) != {
        "regime_id": "immediate",
        "success_delay_steps": 0,
        "failure_delay_steps": 0,
        "success_visibility_probability": 1.0,
        "failure_visibility_probability": 1.0,
    }:
        raise ValueError("immediate predecessor baseline changed")
    boundaries = contract.get("boundaries", {})
    if boundaries.get("generic_nabm_api_changed") is not False:
        raise ValueError("generic NABM API change is outside v1")
    if boundaries.get("empirical_rows_read") != 0:
        raise ValueError("empirical rows are outside v1")
    if contract.get("reporting", {}).get("decision_thresholds") != (
        "NONE_DESCRIPTIVE_ONLY"
    ):
        raise ValueError("v1 must not add a result threshold")


def _study_config(contract: Mapping[str, Any], profile: str) -> Any:
    try:
        profile_values = dict(contract["profiles"][profile])
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile}") from exc
    inherited = dict(contract["inherited_learning_parameters"])
    return study2.LearningStudyConfig(
        **profile_values,
        **inherited,
        arms=tuple(contract["learning_arms"]),
    )


def _scenario(contract: Mapping[str, Any]) -> Any:
    name = contract["scenario"]["name"]
    return next(row for row in study1.study_scenarios() if row.name == name)


def _regimes(contract: Mapping[str, Any]) -> list[VisibilityRegime]:
    return [
        VisibilityRegime.from_mapping(row)
        for row in contract["visibility_regimes"]
    ]


def run_delayed_arm(
    *,
    scenario: Any,
    config: study2.LearningStudyConfig,
    visibility: VisibilityRegime,
    replicate: int,
    learning_mode: str,
) -> dict[str, Any]:
    """Run one precommitted visibility regime and learning arm."""

    rng = study1.component_rng(config.base_seed, replicate, "population")
    study_config = study1.StudyConfig(
        agent_count=config.agent_count,
        steps=config.steps,
        replicates=config.replicates,
        base_seed=config.base_seed,
        neighbor_count=config.neighbor_count,
        within_stage_probability=config.within_stage_probability,
        local_alpha=config.local_alpha,
        local_noise_scale=config.local_noise_scale,
        social_alpha=config.social_alpha,
        peer_similarity_threshold=config.peer_similarity_threshold,
        pivot_threshold=config.pivot_threshold,
        productive_threshold=config.productive_threshold,
    )
    context = SimpleNamespace(scenario=scenario, replicate=replicate, rng=rng)
    researchers = study1.make_researchers(context, study_config)
    neighbors = study1.build_stage_assortative_network(
        researchers,
        context,
        study_config,
    )
    environment = DelayedPivotEnvironment(
        neighbors=neighbors,
        config=config,
        replicate=replicate,
        visibility=visibility,
    )
    agents = [
        study2.PivotPolicyAgent(
            agent_id=researcher.agent_id,
            base=researcher,
            environment=environment,
            learning_rate=(
                0.0 if learning_mode == "frozen" else config.learning_rate
            ),
            learning_mode=learning_mode,
            prior_anchor=config.prior_anchor,
        )
        for researcher in researchers
    ]
    unit = study2.NABMUnit(
        agents=agents,
        step=study2.NABMStep(
            social_block=study2.SocialBlock(alpha=config.social_alpha),
            channel=study2.SocialChannel(
                name="pivot_readiness",
                kind=study2.BOUNDED_SCALAR_CHANNEL,
                commit_mode="domain_pivot_threshold",
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            commit_adapter=study2.PivotCommitAdapter(
                agents=agents,
                environment=environment,
            ),
        ),
        peer_selector=study2._make_peer_selector(environment, config),
        social_value_builder=study2.scalar_message_values("pivot_readiness"),
    )

    realized_success_per_step: list[int] = []
    realized_failure_per_step: list[int] = []
    public_success_per_step: list[int] = []
    public_failure_per_step: list[int] = []
    public_backlog_per_step: list[int] = []
    cumulative_failed_rate: list[float] = []
    cumulative_productive_rate: list[float] = []
    mean_attention_weight: list[float] = []
    count = len(agents)
    for _step in range(config.steps):
        environment.advance()
        unit.run(collect_logs=False)
        realized = environment.ingest_private_events()
        released = environment.last_events
        realized_success_per_step.append(sum(event.productive for event in realized))
        realized_failure_per_step.append(
            sum(not event.productive for event in realized)
        )
        public_success_per_step.append(sum(event.productive for event in released))
        public_failure_per_step.append(sum(not event.productive for event in released))
        public_backlog_per_step.append(len(environment.pending_public_events))
        failed = sum(
            agent.base.pivoted and not agent.base.productive_pivot
            for agent in agents
        )
        productive = sum(agent.base.productive_pivot for agent in agents)
        cumulative_failed_rate.append(failed / count)
        cumulative_productive_rate.append(productive / count)
        mean_attention_weight.append(
            float(np.mean([agent.attention_weight() for agent in agents]))
        )

    pivots = sum(agent.base.pivoted for agent in agents)
    productive = sum(agent.base.productive_pivot for agent in agents)
    private_success = sum(event.productive for event in environment.private_events)
    public_success = sum(event.productive for event in environment.public_events)
    pending_success = sum(
        event.productive for event in environment.pending_public_events
    )
    never_success = sum(
        event.productive for event in environment.never_public_events
    )
    return {
        "pivot_rate": pivots / count,
        "productive_pivot_rate": productive / count,
        "failed_pivot_rate": (pivots - productive) / count,
        "productive_share_of_pivots": productive / pivots if pivots else 0.0,
        "private_success_count": private_success,
        "private_failure_count": len(environment.private_events) - private_success,
        "public_success_count": public_success,
        "public_failure_count": len(environment.public_events) - public_success,
        "horizon_censored_success_count": pending_success,
        "horizon_censored_failure_count": (
            len(environment.pending_public_events) - pending_success
        ),
        "never_public_success_count": never_success,
        "never_public_failure_count": (
            len(environment.never_public_events) - never_success
        ),
        "realized_success_per_step": realized_success_per_step,
        "realized_failure_per_step": realized_failure_per_step,
        "public_success_per_step": public_success_per_step,
        "public_failure_per_step": public_failure_per_step,
        "public_backlog_per_step": public_backlog_per_step,
        "cumulative_failed_rate": cumulative_failed_rate,
        "cumulative_productive_rate": cumulative_productive_rate,
        "mean_attention_weight": mean_attention_weight,
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), 8),
        "std": round(float(array.std(ddof=1)), 8) if array.size > 1 else 0.0,
        "min": round(float(array.min()), 8),
        "max": round(float(array.max()), 8),
        "values": [round(float(value), 8) for value in array],
    }


def _paired_record(
    *,
    comparison: str,
    regime: str,
    arm: str,
    reference_regime: str,
    reference_arm: str,
    metric: str,
    values: Sequence[float],
) -> dict[str, Any]:
    return {
        "comparison": comparison,
        "regime": regime,
        "arm": arm,
        "reference_regime": reference_regime,
        "reference_arm": reference_arm,
        "metric": metric,
        "sign_counts": {
            "negative": sum(value < 0.0 for value in values),
            "zero": sum(value == 0.0 for value in values),
            "positive": sum(value > 0.0 for value in values),
        },
        "paired_delta": _distribution(values),
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["regime"]), str(row["arm"]))].append(row)
    cells: list[dict[str, Any]] = []
    for regime in _regimes(contract):
        for arm in contract["learning_arms"]:
            cell_rows = grouped[(regime.regime_id, arm)]
            cells.append(
                {
                    "regime": regime.regime_id,
                    "arm": arm,
                    "replicates": len(cell_rows),
                    "metrics": {
                        metric: _distribution(
                            [float(row[metric]) for row in cell_rows]
                        )
                        for metric in FINAL_METRICS
                    },
                    "message_counts": {
                        metric: _distribution(
                            [float(row[metric]) for row in cell_rows]
                        )
                        for metric in (
                            "private_success_count",
                            "private_failure_count",
                            "public_success_count",
                            "public_failure_count",
                            "horizon_censored_success_count",
                            "horizon_censored_failure_count",
                            "never_public_success_count",
                            "never_public_failure_count",
                        )
                    },
                    "mean_trajectories": {
                        metric: [
                            round(float(value), 8)
                            for value in np.asarray(
                                [row[metric] for row in cell_rows],
                                dtype=np.float64,
                            ).mean(axis=0)
                        ]
                        for metric in (
                            "realized_success_per_step",
                            "realized_failure_per_step",
                            "public_success_per_step",
                            "public_failure_per_step",
                            "public_backlog_per_step",
                            "cumulative_failed_rate",
                            "cumulative_productive_rate",
                            "mean_attention_weight",
                        )
                    },
                }
            )

    comparisons: list[dict[str, Any]] = []
    for regime in _regimes(contract):
        if regime.regime_id == "immediate":
            continue
        for arm in contract["learning_arms"]:
            rows_by_rep = {
                int(row["replicate"]): row
                for row in grouped[(regime.regime_id, arm)]
            }
            refs_by_rep = {
                int(row["replicate"]): row
                for row in grouped[("immediate", arm)]
            }
            for metric in FINAL_METRICS:
                deltas = [
                    float(rows_by_rep[index][metric])
                    - float(refs_by_rep[index][metric])
                    for index in sorted(rows_by_rep)
                ]
                comparisons.append(
                    _paired_record(
                        comparison="delay_minus_immediate",
                        regime=regime.regime_id,
                        arm=arm,
                        reference_regime="immediate",
                        reference_arm=arm,
                        metric=metric,
                        values=deltas,
                    )
                )
    for regime in _regimes(contract):
        for arm in ("imitative", "cautionary"):
            rows_by_rep = {
                int(row["replicate"]): row
                for row in grouped[(regime.regime_id, arm)]
            }
            refs_by_rep = {
                int(row["replicate"]): row
                for row in grouped[(regime.regime_id, "frozen")]
            }
            for metric in FINAL_METRICS:
                deltas = [
                    float(rows_by_rep[index][metric])
                    - float(refs_by_rep[index][metric])
                    for index in sorted(rows_by_rep)
                ]
                comparisons.append(
                    _paired_record(
                        comparison="learning_minus_frozen",
                        regime=regime.regime_id,
                        arm=arm,
                        reference_regime=regime.regime_id,
                        reference_arm="frozen",
                        metric=metric,
                        values=deltas,
                    )
                )
    return {
        "schema_version": "pivot_delayed_feedback_phase_map.v1",
        "status": "DESCRIPTIVE_SYNTHETIC_RESULT",
        "profile": profile,
        "scenario": contract["scenario"]["name"],
        "cells": cells,
        "paired_comparisons": comparisons,
        "registered_result_space": contract["registered_result_space"],
        "boundaries": {
            **contract["boundaries"],
            "model_selected": "NO",
            "scientific_effect": "NOT_ESTABLISHED",
        },
    }


def run_phase_map(
    contract: Mapping[str, Any],
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = _study_config(contract, profile)
    scenario = _scenario(contract)
    rows: list[dict[str, Any]] = []
    for replicate in range(config.replicates):
        for visibility in _regimes(contract):
            for arm in contract["learning_arms"]:
                result = run_delayed_arm(
                    scenario=scenario,
                    config=config,
                    visibility=visibility,
                    replicate=replicate,
                    learning_mode=arm,
                )
                rows.append(
                    {
                        "replicate": replicate,
                        "regime": visibility.regime_id,
                        "arm": arm,
                        **result,
                    }
                )
    return rows, _summarize(rows, contract, profile)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_source_bindings(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    for binding in contract["source_bindings"]:
        relative = Path(binding["path"])
        actual = _sha256(ROOT / relative)
        expected = str(binding["sha256"])
        if actual != expected:
            raise RuntimeError(
                f"source binding mismatch for {relative}: {actual} != {expected}"
            )
        verified.append(
            {
                "path": relative.as_posix(),
                "sha256": actual,
                "role": str(binding["role"]),
            }
        )
    return verified


def verify_precommit(
    commit_sha: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if FULL_SHA_PATTERN.fullmatch(commit_sha) is None:
        raise RuntimeError("run requires a full 40-character lowercase Git SHA")
    head = _git("rev-parse", "HEAD")
    if head != commit_sha:
        raise RuntimeError(f"precommit SHA is not HEAD: {commit_sha} != {head}")
    paths = [str(path) for path in contract["precommit"]["precommit_paths"]]
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", *paths)
    if dirty:
        raise RuntimeError(f"precommit paths are dirty:\n{dirty}")
    tracked = set(_git("ls-tree", "-r", "--name-only", commit_sha, "--", *paths).split())
    missing = sorted(set(paths) - tracked)
    if missing:
        raise RuntimeError(f"precommit paths missing from commit: {missing}")
    return {
        "commit_sha": commit_sha,
        "precommit_paths": paths,
        "source_bindings": verify_source_bindings(contract),
    }


def _write_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    path.chmod(0o600)


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot delayed-feedback phase map v1",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        "Status: `DESCRIPTIVE_SYNTHETIC_RESULT`",
        "",
        "| Visibility regime | Learning arm | Pivot rate | Productive rate | "
        "Failed rate | Productive share |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        metrics = cell["metrics"]
        lines.append(
            "| {regime} | {arm} | {pivot:.4f} | {productive:.4f} | "
            "{failed:.4f} | {share:.4f} |".format(
                regime=cell["regime"],
                arm=cell["arm"],
                pivot=metrics["pivot_rate"]["mean"],
                productive=metrics["productive_pivot_rate"]["mean"],
                failed=metrics["failed_pivot_rate"]["mean"],
                share=metrics["productive_share_of_pivots"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "These values describe the frozen synthetic model only. No "
            "empirical fit, model selection, neural-necessity conclusion, or "
            "scientific effect is established.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    precommit: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    raw_path = output_dir / "replicates.jsonl"
    phase_path = output_dir / "phase_map.json"
    markdown_path = output_dir / "phase_map.md"
    receipt_path = output_dir / "receipt.json"
    _write_exclusive(
        raw_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    _write_exclusive(
        phase_path,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _write_exclusive(markdown_path, _markdown(summary))
    artifacts = [raw_path, phase_path, markdown_path]
    receipt = {
        "schema_version": "pivot_delayed_feedback_receipt.v1",
        "status": "COMPLETE",
        "precommit": precommit,
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": _sha256(contract_path),
        },
        "row_count": len(rows),
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
        "boundaries": summary["boundaries"],
    }
    _write_exclusive(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def _plan_payload(
    contract: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_path": str(config_path.resolve()),
        "contract_sha256": _sha256(config_path),
        "scenario": contract["scenario"]["name"],
        "regimes": [regime.regime_id for regime in _regimes(contract)],
        "learning_arms": contract["learning_arms"],
        "profiles": contract["profiles"],
        "source_bindings": verify_source_bindings(contract),
        "boundaries": contract["boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=("smoke", "development"),
        default="development",
    )
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    contract = load_contract(config_path)
    if args.command == "plan":
        print(json.dumps(_plan_payload(contract, config_path), indent=2, sort_keys=True))
        return
    if args.precommit_sha is None:
        parser.error("run requires --precommit-sha")
    if args.output_dir is None:
        parser.error("run requires --output-dir")
    precommit = verify_precommit(args.precommit_sha, contract)
    rows, summary = run_phase_map(contract, args.profile)
    summary = {
        **summary,
        "precommit_sha": args.precommit_sha,
        "contract_sha256": _sha256(config_path),
    }
    receipt = write_outputs(
        output_dir=args.output_dir.resolve(),
        rows=rows,
        summary=summary,
        precommit=precommit,
        contract_path=config_path,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output_dir": str(args.output_dir.resolve()),
                "row_count": receipt["row_count"],
                "receipt_sha256": _sha256(
                    args.output_dir.resolve() / "receipt.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
