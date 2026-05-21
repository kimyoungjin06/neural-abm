"""CSV logging helpers for experiment runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class CsvLogWriter:
    """Append dictionaries to a CSV file with a fixed field order."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=fieldnames)
        self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        normalized = {key: row.get(key, "") for key in self.fieldnames}
        for key, value in normalized.items():
            if isinstance(value, (list, dict, tuple)):
                normalized[key] = json.dumps(value, sort_keys=True)
        self._writer.writerow(normalized)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "CsvLogWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


MICRO_STATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "domain_shard_group",
    "coordination_mixer",
    "coordination_peer_rule",
    "model_init_mode",
    "social_channel",
    "commit_mode",
    "local_loss",
    "social_loss",
    "social_update_norm",
    "domain_global_accuracy",
    "domain_probe_accuracy",
    "domain_probe_entropy",
    "domain_confidence",
    "peer_ids",
    "edge_weights",
    "peer_count",
    "component_id",
    "message_norm",
    "latent_norm",
    "param_norm",
    "param_delta_norm",
    "domain_output_js_to_population_mean",
]


AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "coordination_mixer",
    "coordination_peer_rule",
    "model_init_mode",
    "domain_mean_global_accuracy",
    "domain_mean_probe_accuracy",
    "domain_mean_consensus",
    "domain_mean_output_js",
    "domain_polarization_clusters",
    "fragmentation_components",
    "mean_peer_count",
    "social_channel",
    "commit_mode",
    "mean_social_loss",
    "mean_social_update_norm",
    "max_social_update_norm",
    "active_social_agent_count",
    "edge_entropy",
]
