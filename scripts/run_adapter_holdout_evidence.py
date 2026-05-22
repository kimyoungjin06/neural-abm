#!/usr/bin/env python3
"""Run a small adapter-only NABM holdout evidence manifest.

The holdout intentionally lives outside ``src/neural_abm``. It exercises the
public binary policy lifecycle and readiness propagation units through
domain-supplied callbacks, then writes compact evidence artifacts.
"""

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

from neural_abm.readiness import BinaryReadinessPropagationUnit
from neural_abm.spatial_binary import run_binary_policy_learning_step


@dataclass(frozen=True)
class HoldoutAgent:
    agent_id: int


@dataclass
class ThresholdAdoptionHoldout:
    """Tiny irreversible binary adoption domain backed by unit callbacks."""

    agent_count: int
    threshold: float
    intrinsic_value: float
    exposure_weight: float
    readiness_weight: float
    mode: str
    seed: int
    initial_seed_count: int
    neighbor_radius: int = 2
    agents: list[HoldoutAgent] = field(init=False)
    actions: np.ndarray = field(init=False)
    peer_evidence_increment: np.ndarray = field(init=False)
    unit_policy_steps: int = 0
    readiness_steps: int = 0

    def __post_init__(self) -> None:
        self.agents = [HoldoutAgent(agent_id) for agent_id in range(self.agent_count)]
        self.actions = np.zeros(self.agent_count, dtype=np.int64)
        if self.initial_seed_count > 0:
            rng = np.random.default_rng(self.seed)
            seeded = rng.choice(
                self.agent_count,
                size=min(self.initial_seed_count, self.agent_count),
                replace=False,
            )
            self.actions[seeded] = 1
        self.peer_evidence_increment = np.zeros(self.agent_count, dtype=np.float64)

    def peer_ids(self) -> list[list[int]]:
        peer_ids: list[list[int]] = []
        for agent_id in range(self.agent_count):
            peers: list[int] = []
            for offset in range(1, self.neighbor_radius + 1):
                peers.append((agent_id - offset) % self.agent_count)
                peers.append((agent_id + offset) % self.agent_count)
            peer_ids.append(peers)
        return peer_ids

    def exposure(self) -> np.ndarray:
        peer_ids = self.peer_ids()
        exposure = np.zeros(self.agent_count, dtype=np.float64)
        for agent_id, peers in enumerate(peer_ids):
            exposure[agent_id] = float(np.mean(self.actions[peers])) if peers else 0.0
        return exposure

    def observations(self) -> torch.Tensor:
        return torch.tensor(
            np.column_stack(
                [
                    np.full(self.agent_count, self.intrinsic_value, dtype=np.float64),
                    self.exposure(),
                    np.full(self.agent_count, self.threshold, dtype=np.float64),
                    self.peer_evidence_increment,
                    self.actions.astype(np.float64),
                ]
            ),
            dtype=torch.float32,
        )

    def update_readiness(self) -> None:
        if self.mode != "adapter_threshold_readiness":
            self.peer_evidence_increment = np.zeros(self.agent_count, dtype=np.float64)
            return
        report = BinaryReadinessPropagationUnit(
            enabled=True,
            weight=self.readiness_weight,
            aggregation="max",
        ).propagate(
            peer_ids=self.peer_ids(),
            previous_readiness=self.actions.astype(np.float64),
            active=self.actions.astype(bool),
            direction_ok=np.ones(self.agent_count, dtype=bool),
        )
        self.readiness_steps += 1
        self.peer_evidence_increment = report.peer_evidence_increment

    def collect_policy_probs(
        self,
        agents: list[HoldoutAgent],
        observations: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        if agents != self.agents:
            raise ValueError("holdout agent callback received unexpected agents")
        intrinsic = observations[:, 0]
        exposure = observations[:, 1]
        threshold = observations[:, 2]
        peer_evidence = observations[:, 3]
        active_bonus = observations[:, 4] * 1.0
        if self.mode == "exposure_baseline":
            logits = intrinsic + self.exposure_weight * exposure - threshold
        elif self.mode == "thresholdless_global_pressure":
            logits = intrinsic + 0.30 - threshold
        elif self.mode == "adapter_threshold_readiness":
            logits = (
                intrinsic
                + self.exposure_weight * exposure
                + peer_evidence
                - threshold
            )
        else:
            raise ValueError(f"unknown adapter holdout mode: {self.mode}")
        logits = (logits + active_bonus) / float(temperature)
        adopt_probs = torch.sigmoid(logits)
        return torch.stack((1.0 - adopt_probs, adopt_probs), dim=1)

    def decision_action_probs(self, policy_probs: torch.Tensor) -> torch.Tensor:
        return policy_probs

    def sample_actions(self, action_probs: torch.Tensor) -> np.ndarray:
        selected = (action_probs[:, 1].detach().cpu().numpy() >= 0.5).astype(np.int64)
        return np.maximum(self.actions, selected)

    def local_update(self, actions: np.ndarray) -> list[float]:
        self.actions = np.asarray(actions, dtype=np.int64)
        return (1.0 - self.actions.astype(np.float64)).tolist()

    def refresh_policy_cache(self, agents: list[HoldoutAgent]) -> None:
        if len(agents) != self.agent_count:
            raise ValueError("holdout refresh received unexpected agent count")

    def step(self, *, temperature: float) -> None:
        self.update_readiness()
        run_binary_policy_learning_step(
            agents=self.agents,
            observations=self.observations(),
            temperature=temperature,
            collect_policy_probs=self.collect_policy_probs,
            decision_action_probs=self.decision_action_probs,
            sample_actions=self.sample_actions,
            local_update=self.local_update,
            refresh_policy_cache=self.refresh_policy_cache,
            extras={"domain": "adapter_only_threshold_holdout"},
        )
        self.unit_policy_steps += 1


def _load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("adapter holdout manifest must be a mapping")
    return data


def _simulate_row(
    *,
    label: str,
    case: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
) -> dict[str, object]:
    domain = ThresholdAdoptionHoldout(
        agent_count=int(case["agent_count"]),
        threshold=float(case["threshold"]),
        intrinsic_value=float(case["intrinsic_value"]),
        exposure_weight=float(case["exposure_weight"]),
        readiness_weight=float(case["readiness_weight"]),
        mode=str(variant["mode"]),
        seed=seed,
        initial_seed_count=int(case["initial_seed_count"]),
        neighbor_radius=int(case.get("neighbor_radius", 2)),
    )
    epochs = int(case["epochs"])
    time_to_full: int | None = (
        0 if int(domain.actions.sum()) == domain.agent_count else None
    )
    for epoch in range(1, epochs + 1):
        domain.step(temperature=float(case.get("temperature", 1.0)))
        if time_to_full is None and int(domain.actions.sum()) == domain.agent_count:
            time_to_full = epoch
    final_adoption_count = int(domain.actions.sum())
    return {
        "label": label,
        "case": case["name"],
        "variant": variant["name"],
        "group": variant["group"],
        "mode": variant["mode"],
        "seed": seed,
        "agent_count": domain.agent_count,
        "epochs": epochs,
        "initial_seed_count": int(case["initial_seed_count"]),
        "final_adoption_count": final_adoption_count,
        "final_adoption_rate": final_adoption_count / domain.agent_count,
        "time_to_full_adoption": "" if time_to_full is None else time_to_full,
        "reached_full_adoption": final_adoption_count == domain.agent_count,
        "preserved_safety": final_adoption_count == 0,
        "unit_policy_lifecycle_used": domain.unit_policy_steps == epochs,
        "readiness_unit_used": domain.readiness_steps > 0,
        "readiness_usage_matches_mode": (
            domain.readiness_steps == epochs
            if variant["mode"] == "adapter_threshold_readiness"
            else domain.readiness_steps == 0
        ),
        "source_changes_required": False,
    }


def _case_summary(rows: list[dict[str, object]], case: dict[str, Any]) -> dict[str, Any]:
    variants: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant = str(row["variant"])
        info = variants.setdefault(
            variant,
            {
                "group": row["group"],
                "mode": row["mode"],
                "final_adoption_counts": [],
                "time_to_full_adoption": [],
            },
        )
        info["final_adoption_counts"].append(int(row["final_adoption_count"]))
        value = row["time_to_full_adoption"]
        if value != "":
            info["time_to_full_adoption"].append(int(value))
    for info in variants.values():
        counts = info["final_adoption_counts"]
        times = info["time_to_full_adoption"]
        info["mean_final_adoption_count"] = float(np.mean(counts)) if counts else 0.0
        info["final_full_hits"] = int(sum(count == case["agent_count"] for count in counts))
        info["safety_hits"] = int(sum(count == 0 for count in counts))
        info["mean_time_to_full_adoption"] = (
            float(np.mean(times)) if times else None
        )
        del info["final_adoption_counts"]
        del info["time_to_full_adoption"]
    return {"case": case["name"], "variants": variants}


def _evaluate_success(
    *,
    rows: list[dict[str, object]],
    manifest: dict[str, Any],
) -> tuple[bool, list[str]]:
    criteria = manifest.get("success_criteria", {})
    failures: list[str] = []
    for case in manifest["cases"]:
        case_name = case["name"]
        case_rows = [row for row in rows if row["case"] == case_name]
        case_criteria = criteria.get(case_name, {})
        for group, expected in case_criteria.items():
            group_rows = [row for row in case_rows if row["group"] == group]
            if not group_rows:
                failures.append(f"{case_name}:{group}:missing_rows")
                continue
            counts = [int(row["final_adoption_count"]) for row in group_rows]
            if "min_final_adoption_count" in expected:
                minimum = min(counts)
                required = int(expected["min_final_adoption_count"])
                if minimum < required:
                    failures.append(
                        f"{case_name}:{group}:min_final_adoption_count={minimum}<"
                        f"{required}"
                    )
            if "max_final_adoption_count" in expected:
                maximum = max(counts)
                required = int(expected["max_final_adoption_count"])
                if maximum > required:
                    failures.append(
                        f"{case_name}:{group}:max_final_adoption_count={maximum}>"
                        f"{required}"
                    )
    return not failures, failures


def run_adapter_holdout_evidence(manifest_path: Path) -> dict[str, Path]:
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
    case_summaries = [
        _case_summary([row for row in rows if row["case"] == case["name"]], case)
        for case in manifest["cases"]
    ]
    passed, failures = _evaluate_success(rows=rows, manifest=manifest)
    summary = {
        "label": label,
        "status": "pass" if passed else "fail",
        "runs_path": str(runs_path),
        "cases": case_summaries,
        "failures": failures,
        "claim_boundary": manifest.get("claim_boundary", ""),
    }
    summary_json_path = summary_dir / f"{label}.summary.json"
    summary_md_path = summary_dir / f"{label}.summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")
    findings_path = Path(
        manifest.get(
            "findings_path",
            results_dir / f"{label}_findings.md",
        )
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
        f"# Adapter Holdout Evidence: {summary['label']}",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"Runs: `{summary['runs_path']}`",
        "",
        "| Case | Variant | Group | Final full hits | Safety hits | Mean final adoption | Mean TtF |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        for variant, info in case["variants"].items():
            mean_ttf = info["mean_time_to_full_adoption"]
            lines.append(
                f"| {case['case']} | `{variant}` | {info['group']} | "
                f"{info['final_full_hits']} | {info['safety_hits']} | "
                f"{info['mean_final_adoption_count']:.3f} | "
                f"{'' if mean_ttf is None else f'{mean_ttf:.3f}'} |"
            )
    if summary["failures"]:
        lines.extend(["", "Failures:", ""])
        lines.extend(f"- `{failure}`" for failure in summary["failures"])
    if summary["claim_boundary"]:
        lines.extend(["", "Claim boundary:", "", f"> {summary['claim_boundary']}"])
    return "\n".join(lines) + "\n"


def _findings_markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# Adapter-Only Threshold Holdout Findings",
        "",
        f"Manifest: `{manifest['label']}`",
        "",
        "## Purpose",
        "",
        "- Test whether a new binary threshold-like domain can use the NABM Unit v1",
        "  policy lifecycle and readiness propagation through adapter callbacks only.",
        "- Keep the holdout outside `src/neural_abm` so the generic unit does not",
        "  absorb domain semantics.",
        "- Include a baseline, a negative control, a main adapter path, and result",
        "  artifacts.",
        "",
        "## Result",
        "",
        f"Gate status: `{summary['status']}`.",
        "",
        "| Case | Variant | Group | Final full hits | Safety hits | Mean final adoption | Mean TtF |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        for variant, info in case["variants"].items():
            mean_ttf = info["mean_time_to_full_adoption"]
            lines.append(
                f"| {case['case']} | `{variant}` | {info['group']} | "
                f"{info['final_full_hits']} | {info['safety_hits']} | "
                f"{info['mean_final_adoption_count']:.3f} | "
                f"{'' if mean_ttf is None else f'{mean_ttf:.3f}'} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This strengthens the extensibility claim beyond a pure unit smoke test:",
            "a separate holdout manifest now runs a small domain with baseline,",
            "negative-control, and main variants while using only public unit APIs.",
            "",
            "The claim remains bounded. This is still a tiny binary holdout, not a",
            "full general-purpose ABM framework demonstration.",
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
        default=Path("experiments/evidence/adapter_only_threshold_holdout_quick.yaml"),
    )
    args = parser.parse_args()
    outputs = run_adapter_holdout_evidence(args.manifest)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
