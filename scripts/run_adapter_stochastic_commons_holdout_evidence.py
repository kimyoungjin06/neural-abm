#!/usr/bin/env python3
"""Run an adapter-only stochastic commons holdout manifest."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from neural_abm.spatial_binary import run_binary_policy_learning_step


@dataclass(frozen=True)
class CommonsAgent:
    agent_id: int


@dataclass
class StochasticCommonsHoldout:
    """Small endogenous resource commons backed by unit policy callbacks."""

    agent_count: int
    mode: str
    seed: int
    initial_resource_mean: float
    initial_resource_spread: float
    base_need: float
    need_spread: float
    regen_rate: float
    passive_regen_rate: float
    harvest_demand: float
    neighbor_depletion: float
    collapse_floor: float
    shock_epochs: set[int]
    shock_fraction: float
    shock_scale: float
    local_harvest_threshold: float
    global_harvest_threshold: float
    neighbor_radius: int = 2
    agents: list[CommonsAgent] = field(init=False)
    rng: np.random.Generator = field(init=False)
    resources: np.ndarray = field(init=False)
    needs: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    last_payoffs: np.ndarray = field(init=False)
    shock_memory: np.ndarray = field(init=False)
    resource_mean_history: list[float] = field(default_factory=list)
    harvest_rate_history: list[float] = field(default_factory=list)
    welfare_history: list[float] = field(default_factory=list)
    shock_epoch_history: list[int] = field(default_factory=list)
    unit_policy_steps: int = 0

    def __post_init__(self) -> None:
        self.agents = [CommonsAgent(agent_id) for agent_id in range(self.agent_count)]
        self.rng = np.random.default_rng(self.seed)
        resources = self.rng.normal(
            loc=self.initial_resource_mean,
            scale=self.initial_resource_spread,
            size=self.agent_count,
        )
        needs = self.rng.normal(
            loc=self.base_need,
            scale=self.need_spread,
            size=self.agent_count,
        )
        self.resources = np.clip(resources, 0.0, 1.0)
        self.needs = np.clip(needs, 0.0, 1.0)
        self.actions = np.zeros(self.agent_count, dtype=np.int64)
        self.last_payoffs = np.zeros(self.agent_count, dtype=np.float64)
        self.shock_memory = np.zeros(self.agent_count, dtype=np.float64)
        self.resource_mean_history.append(float(np.mean(self.resources)))
        self.harvest_rate_history.append(0.0)
        self.welfare_history.append(0.0)

    def peer_ids(self) -> list[list[int]]:
        peer_ids: list[list[int]] = []
        for agent_id in range(self.agent_count):
            peers: list[int] = []
            for offset in range(1, self.neighbor_radius + 1):
                peers.append((agent_id - offset) % self.agent_count)
                peers.append((agent_id + offset) % self.agent_count)
            peer_ids.append(peers)
        return peer_ids

    def neighbor_harvest_rate(self) -> np.ndarray:
        rates = np.zeros(self.agent_count, dtype=np.float64)
        for agent_id, peers in enumerate(self.peer_ids()):
            rates[agent_id] = float(np.mean(self.actions[peers])) if peers else 0.0
        return rates

    @property
    def global_resource_mean(self) -> float:
        return float(np.mean(self.resources))

    def observations(self) -> torch.Tensor:
        return torch.tensor(
            np.column_stack(
                [
                    self.resources,
                    self.neighbor_harvest_rate(),
                    np.full(
                        self.agent_count,
                        self.global_resource_mean,
                        dtype=np.float64,
                    ),
                    self.needs,
                    self.last_payoffs,
                    self.shock_memory,
                ]
            ),
            dtype=torch.float32,
        )

    def collect_policy_probs(
        self,
        agents: list[CommonsAgent],
        observations: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        if agents != self.agents:
            raise ValueError("commons holdout callback received unexpected agents")
        resource = observations[:, 0]
        neighbor_harvest = observations[:, 1]
        global_resource = observations[:, 2]
        need = observations[:, 3]
        last_payoff = observations[:, 4]
        shock_memory = observations[:, 5]
        if self.mode == "greedy_harvest_baseline":
            logits = 3.0 + 3.0 * resource + 1.0 * need + 0.25 * last_payoff
        elif self.mode == "global_pressure_negative_control":
            logits = 10.0 * (global_resource - self.global_harvest_threshold)
            logits = logits + 0.35 * (need - 0.5)
        elif self.mode == "adapter_local_resource_main":
            logits = 12.0 * (resource - self.local_harvest_threshold)
            logits = logits - 4.0 * neighbor_harvest - 3.0 * shock_memory
            logits = logits + 0.45 * (need - 0.5)
        else:
            raise ValueError(f"unknown stochastic commons mode: {self.mode}")
        harvest_probs = torch.sigmoid(logits / float(temperature))
        return torch.stack((1.0 - harvest_probs, harvest_probs), dim=1)

    def decision_action_probs(self, policy_probs: torch.Tensor) -> torch.Tensor:
        return policy_probs

    def sample_actions(self, action_probs: torch.Tensor) -> np.ndarray:
        return (action_probs[:, 1].detach().cpu().numpy() >= 0.5).astype(np.int64)

    def local_update(self, actions: np.ndarray) -> list[float]:
        self.actions = np.asarray(actions, dtype=np.int64)
        requested = self.harvest_demand * (0.75 + 0.5 * self.needs)
        harvested = np.minimum(self.resources, requested * self.actions)
        neighbor_pressure = self.neighbor_harvest_rate()
        depletion = harvested + self.neighbor_depletion * neighbor_pressure
        after_depletion = np.maximum(0.0, self.resources - depletion)
        active_regen = self.regen_rate * (1.0 - self.actions) * (1.0 - after_depletion)
        passive_regen = self.passive_regen_rate * (1.0 - after_depletion)
        self.resources = np.clip(after_depletion + active_regen + passive_regen, 0.0, 1.0)
        self.shock_memory *= 0.5
        epoch = self.unit_policy_steps + 1
        if epoch in self.shock_epochs:
            shock_count = max(1, int(round(self.agent_count * self.shock_fraction)))
            shocked = self.rng.choice(self.agent_count, size=shock_count, replace=False)
            self.resources[shocked] = np.maximum(
                0.0,
                self.resources[shocked] - self.shock_scale,
            )
            self.shock_memory[shocked] = 1.0
            self.shock_epoch_history.append(epoch)
        floor_penalty = np.maximum(0.0, self.collapse_floor - self.resources)
        payoff = harvested - 0.65 * floor_penalty
        self.last_payoffs = payoff
        self.resource_mean_history.append(float(np.mean(self.resources)))
        self.harvest_rate_history.append(float(np.mean(self.actions)))
        self.welfare_history.append(float(np.mean(payoff)))
        return (-payoff).tolist()

    def refresh_policy_cache(self, agents: list[CommonsAgent]) -> None:
        if len(agents) != self.agent_count:
            raise ValueError("commons refresh received unexpected agent count")

    def step(self, *, temperature: float) -> None:
        run_binary_policy_learning_step(
            agents=self.agents,
            observations=self.observations(),
            temperature=temperature,
            collect_policy_probs=self.collect_policy_probs,
            decision_action_probs=self.decision_action_probs,
            sample_actions=self.sample_actions,
            local_update=self.local_update,
            refresh_policy_cache=self.refresh_policy_cache,
            extras={"domain": "adapter_only_stochastic_commons_holdout"},
        )
        self.unit_policy_steps += 1

    def collapse_epochs(self) -> int:
        return sum(
            1 for value in self.resource_mean_history[1:] if value < self.collapse_floor
        )

    def recovery_epochs_after_first_shock(self, threshold: float) -> int | None:
        if not self.shock_epoch_history:
            return None
        first_shock = self.shock_epoch_history[0]
        for epoch, value in enumerate(self.resource_mean_history):
            if epoch > first_shock and value >= threshold:
                return epoch - first_shock
        return None


def _load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("adapter stochastic commons manifest must be a mapping")
    return data


def _simulate_row(
    *,
    label: str,
    case: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
) -> dict[str, object]:
    domain = StochasticCommonsHoldout(
        agent_count=int(case["agent_count"]),
        mode=str(variant["mode"]),
        seed=seed,
        initial_resource_mean=float(case["initial_resource_mean"]),
        initial_resource_spread=float(case["initial_resource_spread"]),
        base_need=float(case["base_need"]),
        need_spread=float(case["need_spread"]),
        regen_rate=float(case["regen_rate"]),
        passive_regen_rate=float(case["passive_regen_rate"]),
        harvest_demand=float(case["harvest_demand"]),
        neighbor_depletion=float(case["neighbor_depletion"]),
        collapse_floor=float(case["collapse_floor"]),
        shock_epochs={int(epoch) for epoch in case.get("shock_epochs", [])},
        shock_fraction=float(case.get("shock_fraction", 0.0)),
        shock_scale=float(case.get("shock_scale", 0.0)),
        local_harvest_threshold=float(case["local_harvest_threshold"]),
        global_harvest_threshold=float(case["global_harvest_threshold"]),
        neighbor_radius=int(case.get("neighbor_radius", 2)),
    )
    epochs = int(case["epochs"])
    for _ in range(epochs):
        domain.step(temperature=float(case.get("temperature", 1.0)))
    recovery_threshold = float(case.get("recovery_threshold", domain.collapse_floor))
    recovery_epochs = domain.recovery_epochs_after_first_shock(recovery_threshold)
    mean_welfare = float(np.mean(domain.welfare_history[1:]))
    mean_harvest_rate = float(np.mean(domain.harvest_rate_history[1:]))
    min_resource_mean = float(np.min(domain.resource_mean_history[1:]))
    final_resource_mean = float(domain.resource_mean_history[-1])
    return {
        "label": label,
        "case": case["name"],
        "variant": variant["name"],
        "group": variant["group"],
        "mode": variant["mode"],
        "seed": seed,
        "agent_count": domain.agent_count,
        "epochs": epochs,
        "final_resource_mean": final_resource_mean,
        "min_resource_mean": min_resource_mean,
        "collapse_floor": domain.collapse_floor,
        "collapse_epochs": domain.collapse_epochs(),
        "mean_welfare": mean_welfare,
        "mean_harvest_rate": mean_harvest_rate,
        "final_harvest_rate": float(domain.harvest_rate_history[-1]),
        "shock_count": len(domain.shock_epoch_history),
        "recovery_epochs_after_shock": (
            "" if recovery_epochs is None else recovery_epochs
        ),
        "recovered_after_shock": (
            "" if recovery_epochs is None else recovery_epochs <= epochs
        ),
        "unit_policy_lifecycle_used": domain.unit_policy_steps == epochs,
        "source_changes_required": False,
    }


def _case_summary(rows: list[dict[str, object]]) -> dict[str, Any]:
    variants: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant = str(row["variant"])
        info = variants.setdefault(
            variant,
            {
                "group": row["group"],
                "mode": row["mode"],
                "final_resources": [],
                "min_resources": [],
                "collapse_epochs": [],
                "welfare": [],
                "harvest_rates": [],
                "recovery_epochs": [],
            },
        )
        info["final_resources"].append(float(row["final_resource_mean"]))
        info["min_resources"].append(float(row["min_resource_mean"]))
        info["collapse_epochs"].append(int(row["collapse_epochs"]))
        info["welfare"].append(float(row["mean_welfare"]))
        info["harvest_rates"].append(float(row["mean_harvest_rate"]))
        recovery_value = row["recovery_epochs_after_shock"]
        if recovery_value != "":
            info["recovery_epochs"].append(int(recovery_value))
    for info in variants.values():
        final_resources = info["final_resources"]
        min_resources = info["min_resources"]
        collapse_epochs = info["collapse_epochs"]
        welfare = info["welfare"]
        harvest_rates = info["harvest_rates"]
        recovery_epochs = info["recovery_epochs"]
        info["mean_final_resource"] = float(np.mean(final_resources))
        info["min_resource_mean"] = float(np.min(min_resources))
        info["max_collapse_epochs"] = int(max(collapse_epochs))
        info["min_collapse_epochs"] = int(min(collapse_epochs))
        info["mean_welfare"] = float(np.mean(welfare))
        info["mean_harvest_rate"] = float(np.mean(harvest_rates))
        info["recovery_hits"] = len(recovery_epochs)
        info["max_recovery_epochs_after_shock"] = (
            int(max(recovery_epochs)) if recovery_epochs else None
        )
        del info["final_resources"]
        del info["min_resources"]
        del info["collapse_epochs"]
        del info["welfare"]
        del info["harvest_rates"]
        del info["recovery_epochs"]
    return variants


def _evaluate_success(
    *,
    rows: list[dict[str, object]],
    manifest: dict[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    criteria = manifest.get("success_criteria", {})
    for case in manifest["cases"]:
        case_name = case["name"]
        case_rows = [row for row in rows if row["case"] == case_name]
        for group, expected in criteria.get(case_name, {}).items():
            group_rows = [row for row in case_rows if row["group"] == group]
            if not group_rows:
                failures.append(f"{case_name}:{group}:missing_rows")
                continue
            collapse_epochs = [int(row["collapse_epochs"]) for row in group_rows]
            min_resources = [float(row["min_resource_mean"]) for row in group_rows]
            welfare = [float(row["mean_welfare"]) for row in group_rows]
            harvest_rates = [float(row["mean_harvest_rate"]) for row in group_rows]
            recovery_epochs = [
                int(row["recovery_epochs_after_shock"])
                for row in group_rows
                if row["recovery_epochs_after_shock"] != ""
            ]
            if "max_collapse_epochs" in expected:
                maximum = max(collapse_epochs)
                required = int(expected["max_collapse_epochs"])
                if maximum > required:
                    failures.append(
                        f"{case_name}:{group}:max_collapse_epochs={maximum}>"
                        f"{required}"
                    )
            if "min_collapse_epochs" in expected:
                minimum = min(collapse_epochs)
                required = int(expected["min_collapse_epochs"])
                if minimum < required:
                    failures.append(
                        f"{case_name}:{group}:min_collapse_epochs={minimum}<"
                        f"{required}"
                    )
            if "min_min_resource_mean" in expected:
                minimum = min(min_resources)
                required = float(expected["min_min_resource_mean"])
                if minimum < required:
                    failures.append(
                        f"{case_name}:{group}:min_min_resource_mean="
                        f"{minimum:.3f}<{required:.3f}"
                    )
            if "max_min_resource_mean" in expected:
                maximum = max(min_resources)
                required = float(expected["max_min_resource_mean"])
                if maximum > required:
                    failures.append(
                        f"{case_name}:{group}:max_min_resource_mean="
                        f"{maximum:.3f}>{required:.3f}"
                    )
            if "min_mean_welfare" in expected:
                minimum = min(welfare)
                required = float(expected["min_mean_welfare"])
                if minimum < required:
                    failures.append(
                        f"{case_name}:{group}:min_mean_welfare={minimum:.3f}<"
                        f"{required:.3f}"
                    )
            if "min_mean_harvest_rate" in expected:
                minimum = min(harvest_rates)
                required = float(expected["min_mean_harvest_rate"])
                if minimum < required:
                    failures.append(
                        f"{case_name}:{group}:min_mean_harvest_rate="
                        f"{minimum:.3f}<{required:.3f}"
                    )
            if "min_recovery_hits" in expected:
                required = int(expected["min_recovery_hits"])
                if len(recovery_epochs) < required:
                    failures.append(
                        f"{case_name}:{group}:recovery_hits="
                        f"{len(recovery_epochs)}<{required}"
                    )
            if "max_recovery_epochs_after_shock" in expected:
                required = int(expected["max_recovery_epochs_after_shock"])
                if not recovery_epochs:
                    failures.append(f"{case_name}:{group}:missing_recovery_epochs")
                elif max(recovery_epochs) > required:
                    failures.append(
                        f"{case_name}:{group}:max_recovery_epochs_after_shock="
                        f"{max(recovery_epochs)}>{required}"
                    )
    return not failures, failures


def run_adapter_stochastic_commons_holdout_evidence(
    manifest_path: Path,
) -> dict[str, Path]:
    manifest = _load_manifest(manifest_path)
    label = str(manifest["label"])
    results_dir = Path(manifest.get("results_dir", "experiments/results/nabm_effect_matrix"))
    summary_dir = Path(manifest.get("summary_dir", "experiments/evidence/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for case in manifest["cases"]:
        for variant in manifest["variants"]:
            for seed in case["seeds"]:
                rows.append(
                    _simulate_row(
                        label=label,
                        case=case,
                        variant=variant,
                        seed=int(seed),
                    )
                )
    runs_path = results_dir / f"{label}_runs.csv"
    with runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    cases = [
        {
            "case": case["name"],
            "collapse_floor": float(case["collapse_floor"]),
            "variants": _case_summary(
                [row for row in rows if row["case"] == case["name"]]
            ),
        }
        for case in manifest["cases"]
    ]
    passed, failures = _evaluate_success(rows=rows, manifest=manifest)
    summary = {
        "label": label,
        "status": "pass" if passed else "fail",
        "runs_path": str(runs_path),
        "cases": cases,
        "failures": failures,
        "claim_boundary": manifest.get("claim_boundary", ""),
    }
    summary_json_path = summary_dir / f"{label}.summary.json"
    summary_md_path = summary_dir / f"{label}.summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")
    findings_path = Path(
        manifest.get("findings_path", results_dir / f"{label}_findings.md")
    )
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    findings_path.write_text(_findings_markdown(summary, manifest), encoding="utf-8")
    return {
        "runs": runs_path,
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
        "findings": findings_path,
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Adapter Stochastic Commons Holdout Evidence: {summary['label']}",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"Runs: `{summary['runs_path']}`",
        "",
        "| Case | Variant | Group | Min resource | Max collapse epochs | "
        "Mean welfare | Mean harvest | Recovery hits | Max recovery |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        for variant, info in case["variants"].items():
            recovery = info["max_recovery_epochs_after_shock"]
            lines.append(
                f"| {case['case']} | `{variant}` | {info['group']} | "
                f"{info['min_resource_mean']:.3f} | "
                f"{info['max_collapse_epochs']} | {info['mean_welfare']:.3f} | "
                f"{info['mean_harvest_rate']:.3f} | {info['recovery_hits']} | "
                f"{'' if recovery is None else recovery} |"
            )
    if summary["failures"]:
        lines.extend(["", "Failures:", ""])
        lines.extend(f"- `{failure}`" for failure in summary["failures"])
    if summary["claim_boundary"]:
        lines.extend(["", "Claim boundary:", "", f"> {summary['claim_boundary']}"])
    return "\n".join(lines) + "\n"


def _findings_markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# Adapter-Only Stochastic Commons Holdout Findings",
        "",
        f"Manifest: `{manifest['label']}`",
        "",
        "## Purpose",
        "",
        "- Test an adapter-only binary ABM with endogenous state transitions.",
        "- Let actions deplete local resources, conservation regenerate them, and",
        "  stochastic shocks perturb local resource stock.",
        "- Keep the holdout outside `src/neural_abm` and use public binary policy",
        "  lifecycle callbacks only.",
        "",
        "## Result",
        "",
        f"Gate status: `{summary['status']}`.",
        "",
        "| Case | Variant | Group | Min resource | Max collapse epochs | "
        "Mean welfare | Mean harvest | Recovery hits | Max recovery |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        for variant, info in case["variants"].items():
            recovery = info["max_recovery_epochs_after_shock"]
            lines.append(
                f"| {case['case']} | `{variant}` | {info['group']} | "
                f"{info['min_resource_mean']:.3f} | "
                f"{info['max_collapse_epochs']} | {info['mean_welfare']:.3f} | "
                f"{info['mean_harvest_rate']:.3f} | {info['recovery_hits']} | "
                f"{'' if recovery is None else recovery} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This moves the adapter-only evidence beyond a fixed threshold or",
            "capacity target. The relevant state is endogenous: harvest decisions",
            "change future resource stock, shocks perturb the environment, and the",
            "main adapter uses local state rather than a direct target mask.",
            "",
            "The claim remains bounded. This is still a compact scripted commons",
            "holdout. It is not a full general-purpose ABM framework proof.",
            "",
            "## Artifacts",
            "",
            f"- Runs: `{summary['runs_path']}`",
            f"- Summary JSON: `experiments/evidence/results/{summary['label']}.summary.json`",
            f"- Summary Markdown: `experiments/evidence/results/{summary['label']}.summary.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/evidence/adapter_only_stochastic_commons_quick.yaml"),
    )
    args = parser.parse_args()
    outputs = run_adapter_stochastic_commons_holdout_evidence(args.manifest)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
