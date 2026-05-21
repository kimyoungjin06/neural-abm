"""Shared runner result types."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from neural_abm.capabilities import (
    artifact_capability_metadata,
    optional_artifact_capability_metadata,
)


ResolvedConfigFormat = Literal["json", "yaml"]


@dataclass
class DomainToyResult:
    run_dir: Path
    toy: str
    final_fragmentation_components: int
    domain_metrics: dict[str, object]


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_artifact_metadata(
    toy: str,
    metadata: Mapping[str, Any],
    *,
    strict_capability: bool = True,
) -> dict[str, object]:
    capability = (
        artifact_capability_metadata(toy)
        if strict_capability
        else optional_artifact_capability_metadata(toy)
    )
    return {**dict(metadata), **capability}


def write_run_metadata_artifacts(
    *,
    config_path: Path,
    config: Any,
    run_dir: Path,
    toy: str,
    metadata: Mapping[str, Any],
    resolved_config_format: ResolvedConfigFormat = "yaml",
    strict_capability: bool = True,
) -> dict[str, object]:
    shutil.copy2(config_path, run_dir / "config.yaml")
    resolved_config = config.model_dump(mode="json")
    if resolved_config_format == "json":
        resolved_text = json.dumps(resolved_config, indent=2, sort_keys=True)
    elif resolved_config_format == "yaml":
        resolved_text = yaml.safe_dump(resolved_config, sort_keys=False)
    else:
        raise ValueError(f"Unsupported resolved config format: {resolved_config_format}")
    (run_dir / "resolved_config.yaml").write_text(
        resolved_text,
        encoding="utf-8",
    )
    payload = run_artifact_metadata(
        toy,
        metadata,
        strict_capability=strict_capability,
    )
    write_json_artifact(run_dir / "metadata.json", payload)
    return payload


def domain_summary_payload(
    *,
    run_dir: Path,
    toy: str,
    final_fragmentation_components: int | object,
    domain_metrics: Mapping[str, object],
    strict_capability: bool = True,
) -> dict[str, object]:
    return run_artifact_metadata(
        toy,
        {
            "run_dir": str(run_dir),
            "toy": toy,
            "final_fragmentation_components": final_fragmentation_components,
            "domain_metrics": dict(domain_metrics),
        },
        strict_capability=strict_capability,
    )


def write_domain_summary_artifact(
    *,
    run_dir: Path,
    toy: str,
    final_fragmentation_components: int | object,
    domain_metrics: Mapping[str, object],
    strict_capability: bool = True,
) -> dict[str, object]:
    payload = domain_summary_payload(
        run_dir=run_dir,
        toy=toy,
        final_fragmentation_components=final_fragmentation_components,
        domain_metrics=domain_metrics,
        strict_capability=strict_capability,
    )
    write_json_artifact(run_dir / "summary.json", payload)
    return payload


def binary_summary_payload(
    *,
    run_dir: Path,
    toy: str,
    final_action_rate: object,
    final_mean_payoff: object,
    final_fragmentation_components: object,
    final_mean_policy_action_probability: object,
    final_mean_reputation: object,
    final_reputation_dispersion: object,
    domain_metrics: Mapping[str, object],
    strict_capability: bool = True,
) -> dict[str, object]:
    return run_artifact_metadata(
        toy,
        {
            "run_dir": str(run_dir),
            "toy": toy,
            "final_action_rate": final_action_rate,
            "final_mean_payoff": final_mean_payoff,
            "final_fragmentation_components": final_fragmentation_components,
            "final_mean_policy_action_probability": (
                final_mean_policy_action_probability
            ),
            "final_mean_reputation": final_mean_reputation,
            "final_reputation_dispersion": final_reputation_dispersion,
            "domain_metrics": dict(domain_metrics),
        },
        strict_capability=strict_capability,
    )


def write_binary_summary_artifact(
    *,
    run_dir: Path,
    toy: str,
    final_action_rate: object,
    final_mean_payoff: object,
    final_fragmentation_components: object,
    final_mean_policy_action_probability: object,
    final_mean_reputation: object,
    final_reputation_dispersion: object,
    domain_metrics: Mapping[str, object],
    strict_capability: bool = True,
) -> dict[str, object]:
    payload = binary_summary_payload(
        run_dir=run_dir,
        toy=toy,
        final_action_rate=final_action_rate,
        final_mean_payoff=final_mean_payoff,
        final_fragmentation_components=final_fragmentation_components,
        final_mean_policy_action_probability=final_mean_policy_action_probability,
        final_mean_reputation=final_mean_reputation,
        final_reputation_dispersion=final_reputation_dispersion,
        domain_metrics=domain_metrics,
        strict_capability=strict_capability,
    )
    write_json_artifact(run_dir / "summary.json", payload)
    return payload
