#!/usr/bin/env python3
"""Run an adapter-only binary congestion/capacity holdout manifest."""

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
class CongestionAgent:
    agent_id: int


@dataclass
class CapacityChoiceHoldout:
    """Tiny binary route-choice domain backed by unit policy callbacks."""

    agent_count: int
    target_count: int
    initial_one_count: int
    mode: str
    seed: int
    preference_noise: float = 0.0
    agents: list[CongestionAgent] = field(init=False)
    actions: np.ndarray = field(init=False)
    capacity_mask: np.ndarray = field(init=False)
    unit_policy_steps: int = 0

    def __post_init__(self) -> None:
        self.agents = [CongestionAgent(agent_id) for agent_id in range(self.agent_count)]
        rng = np.random.default_rng(self.seed)
        self.actions = np.zeros(self.agent_count, dtype=np.int64)
        if self.initial_one_count > 0:
            initial = rng.choice(
                self.agent_count,
                size=min(self.initial_one_count, self.agent_count),
                replace=False,
            )
            self.actions[initial] = 1
        base_scores = (np.arange(self.agent_count, dtype=np.float64) + 0.5) / float(
            self.agent_count
        )
        noisy_scores = base_scores + rng.normal(
            loc=0.0,
            scale=self.preference_noise,
            size=self.agent_count,
        )
        selected = np.argsort(noisy_scores)[: self.target_count]
        self.capacity_mask = np.zeros(self.agent_count, dtype=np.int64)
        self.capacity_mask[selected] = 1

    @property
    def target_share(self) -> float:
        return self.target_count / float(self.agent_count)

    @property
    def current_share(self) -> float:
        return float(np.mean(self.actions))

    def observations(self) -> torch.Tensor:
        return torch.tensor(
            np.column_stack(
                [
                    self.actions.astype(np.float64),
                    np.full(self.agent_count, self.current_share, dtype=np.float64),
                    np.full(self.agent_count, self.target_share, dtype=np.float64),
                    self.capacity_mask.astype(np.float64),
                ]
            ),
            dtype=torch.float32,
        )

    def collect_policy_probs(
        self,
        agents: list[CongestionAgent],
        observations: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        if agents != self.agents:
            raise ValueError("congestion holdout callback received unexpected agents")
        current_action = observations[:, 0]
        current_share = observations[:, 1]
        target_share = observations[:, 2]
        capacity_mask = observations[:, 3]
        if self.mode == "imitation_baseline":
            logits = 8.0 * (current_share - 0.5) + 0.25 * current_action
        elif self.mode == "global_pressure_negative_control":
            logits = torch.full_like(current_action, 4.0)
        elif self.mode == "adapter_capacity_policy_main":
            capacity_signal = 2.0 * capacity_mask - 1.0
            correction = 0.25 * (target_share - current_share)
            logits = 8.0 * capacity_signal + correction
        else:
            raise ValueError(f"unknown congestion holdout mode: {self.mode}")
        logits = logits / float(temperature)
        one_probs = torch.sigmoid(logits)
        return torch.stack((1.0 - one_probs, one_probs), dim=1)

    def decision_action_probs(self, policy_probs: torch.Tensor) -> torch.Tensor:
        return policy_probs

    def sample_actions(self, action_probs: torch.Tensor) -> np.ndarray:
        return (action_probs[:, 1].detach().cpu().numpy() >= 0.5).astype(np.int64)

    def local_update(self, actions: np.ndarray) -> list[float]:
        self.actions = np.asarray(actions, dtype=np.int64)
        return np.abs(self.actions - self.capacity_mask).astype(float).tolist()

    def refresh_policy_cache(self, agents: list[CongestionAgent]) -> None:
        if len(agents) != self.agent_count:
            raise ValueError("congestion refresh received unexpected agent count")

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
            extras={"domain": "adapter_only_congestion_holdout"},
        )
        self.unit_policy_steps += 1


def _load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("adapter congestion manifest must be a mapping")
    return data


def _simulate_row(
    *,
    label: str,
    case: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
) -> dict[str, object]:
    domain = CapacityChoiceHoldout(
        agent_count=int(case["agent_count"]),
        target_count=int(case["target_count"]),
        initial_one_count=int(case["initial_one_count"]),
        mode=str(variant["mode"]),
        seed=seed,
        preference_noise=float(case.get("preference_noise", 0.0)),
    )
    epochs = int(case["epochs"])
    first_capacity_epoch: int | None = None
    previous_errors: list[int] = []
    oscillation_count = 0
    for epoch in range(1, epochs + 1):
        domain.step(temperature=float(case.get("temperature", 1.0)))
        final_count = int(domain.actions.sum())
        error = final_count - domain.target_count
        if first_capacity_epoch is None and error == 0:
            first_capacity_epoch = epoch
        if previous_errors and error != previous_errors[-1]:
            oscillation_count += 1
        previous_errors.append(error)
    final_one_count = int(domain.actions.sum())
    final_capacity_error = final_one_count - domain.target_count
    abs_error = abs(final_capacity_error)
    welfare = 1.0 - abs_error / float(domain.agent_count)
    return {
        "label": label,
        "case": case["name"],
        "variant": variant["name"],
        "group": variant["group"],
        "mode": variant["mode"],
        "seed": seed,
        "agent_count": domain.agent_count,
        "epochs": epochs,
        "target_count": domain.target_count,
        "initial_one_count": int(case["initial_one_count"]),
        "final_one_count": final_one_count,
        "target_share": domain.target_share,
        "final_share": final_one_count / float(domain.agent_count),
        "final_capacity_error": final_capacity_error,
        "final_capacity_error_abs": abs_error,
        "overcrowding_count": max(0, final_capacity_error),
        "underfill_count": max(0, -final_capacity_error),
        "mean_welfare": welfare,
        "time_to_capacity": "" if first_capacity_epoch is None else first_capacity_epoch,
        "oscillation_count": oscillation_count,
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
                "errors": [],
                "overcrowding": [],
                "welfare": [],
                "time_to_capacity": [],
            },
        )
        info["errors"].append(int(row["final_capacity_error_abs"]))
        info["overcrowding"].append(int(row["overcrowding_count"]))
        info["welfare"].append(float(row["mean_welfare"]))
        value = row["time_to_capacity"]
        if value != "":
            info["time_to_capacity"].append(int(value))
    for info in variants.values():
        errors = info["errors"]
        overcrowding = info["overcrowding"]
        welfare = info["welfare"]
        times = info["time_to_capacity"]
        info["max_capacity_error_abs"] = int(max(errors)) if errors else 0
        info["mean_capacity_error_abs"] = float(np.mean(errors)) if errors else 0.0
        info["max_overcrowding_count"] = int(max(overcrowding)) if overcrowding else 0
        info["mean_welfare"] = float(np.mean(welfare)) if welfare else 0.0
        info["capacity_hits"] = int(sum(error == 0 for error in errors))
        info["mean_time_to_capacity"] = float(np.mean(times)) if times else None
        del info["errors"]
        del info["overcrowding"]
        del info["welfare"]
        del info["time_to_capacity"]
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
            errors = [int(row["final_capacity_error_abs"]) for row in group_rows]
            overcrowding = [int(row["overcrowding_count"]) for row in group_rows]
            welfare = [float(row["mean_welfare"]) for row in group_rows]
            if "max_final_capacity_error_abs" in expected:
                maximum = max(errors)
                required = int(expected["max_final_capacity_error_abs"])
                if maximum > required:
                    failures.append(
                        f"{case_name}:{group}:max_final_capacity_error_abs="
                        f"{maximum}>{required}"
                    )
            if "min_final_capacity_error_abs" in expected:
                minimum = min(errors)
                required = int(expected["min_final_capacity_error_abs"])
                if minimum < required:
                    failures.append(
                        f"{case_name}:{group}:min_final_capacity_error_abs="
                        f"{minimum}<{required}"
                    )
            if "min_overcrowding_count" in expected:
                minimum = min(overcrowding)
                required = int(expected["min_overcrowding_count"])
                if minimum < required:
                    failures.append(
                        f"{case_name}:{group}:min_overcrowding_count="
                        f"{minimum}<{required}"
                    )
            if "min_mean_welfare" in expected:
                minimum = min(welfare)
                required = float(expected["min_mean_welfare"])
                if minimum < required:
                    failures.append(
                        f"{case_name}:{group}:min_mean_welfare={minimum:.3f}<"
                        f"{required:.3f}"
                    )
    return not failures, failures


def run_adapter_congestion_holdout_evidence(manifest_path: Path) -> dict[str, Path]:
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
            "target_count": int(case["target_count"]),
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
        f"# Adapter Congestion Holdout Evidence: {summary['label']}",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"Runs: `{summary['runs_path']}`",
        "",
        "| Case | Variant | Group | Capacity hits | Max error | Max overcrowding | Mean welfare | Mean TtC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        for variant, info in case["variants"].items():
            mean_ttc = info["mean_time_to_capacity"]
            lines.append(
                f"| {case['case']} | `{variant}` | {info['group']} | "
                f"{info['capacity_hits']} | {info['max_capacity_error_abs']} | "
                f"{info['max_overcrowding_count']} | {info['mean_welfare']:.3f} | "
                f"{'' if mean_ttc is None else f'{mean_ttc:.3f}'} |"
            )
    if summary["failures"]:
        lines.extend(["", "Failures:", ""])
        lines.extend(f"- `{failure}`" for failure in summary["failures"])
    if summary["claim_boundary"]:
        lines.extend(["", "Claim boundary:", "", f"> {summary['claim_boundary']}"])
    return "\n".join(lines) + "\n"


def _findings_markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# Adapter-Only Congestion Holdout Findings",
        "",
        f"Manifest: `{manifest['label']}`",
        "",
        "## Purpose",
        "",
        "- Test a non-cascade binary holdout where the target is capacity-matched",
        "  allocation rather than full adoption.",
        "- Keep the domain outside `src/neural_abm` and use public binary policy",
        "  lifecycle callbacks only.",
        "- Compare imitation baseline, global-pressure negative control, and an",
        "  adapter-owned capacity policy.",
        "",
        "## Result",
        "",
        f"Gate status: `{summary['status']}`.",
        "",
        "| Case | Variant | Group | Capacity hits | Max error | Max overcrowding | Mean welfare | Mean TtC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        for variant, info in case["variants"].items():
            mean_ttc = info["mean_time_to_capacity"]
            lines.append(
                f"| {case['case']} | `{variant}` | {info['group']} | "
                f"{info['capacity_hits']} | {info['max_capacity_error_abs']} | "
                f"{info['max_overcrowding_count']} | {info['mean_welfare']:.3f} | "
                f"{'' if mean_ttc is None else f'{mean_ttc:.3f}'} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This strengthens the adapter-only claim in a direction that is not",
            "threshold-cascade isomorphic. The target is a capacity-matched binary",
            "allocation, and success is measured by capacity error and overcrowding",
            "rather than full adoption.",
            "",
            "The claim remains bounded. This is still a tiny scripted holdout, not",
            "a full general-purpose ABM framework demonstration.",
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
        default=Path("experiments/evidence/adapter_only_congestion_holdout_quick.yaml"),
    )
    args = parser.parse_args()
    outputs = run_adapter_congestion_holdout_evidence(args.manifest)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
