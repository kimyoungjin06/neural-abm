from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml

from neural_abm.config import (
    Toy1Config,
    Toy10Config,
    Toy2Config,
    Toy3Config,
    Toy4Config,
    Toy5Config,
    Toy6Config,
    Toy7Config,
    Toy8Config,
    Toy9Config,
    load_toy1_config,
    load_toy10_config,
    load_toy2_config,
    load_toy3_config,
    load_toy4_config,
    load_toy5_config,
    load_toy6_config,
    load_toy7_config,
    load_toy8_config,
    load_toy9_config,
)


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "configs"
REQUIRED_TOP_LEVEL_FIELDS = {"run", "simulation", "model", "domain", "logging"}
REQUIRED_MODEL_FIELDS_BY_TOY = {
    "toy1": {"agents", "coordination"},
    "toy2": {"policy", "agents", "coordination", "state"},
    "toy3": {"policy", "agents", "coordination"},
    "toy4": {"policy", "agents", "coordination", "state"},
    "toy5": {"policy", "agents", "coordination", "state"},
    "toy6": {"policy", "agents", "coordination"},
    "toy7": {"policy", "agents", "coordination"},
    "toy8": {"policy", "agents", "coordination"},
    "toy9": {"policy", "agents", "coordination"},
    "toy10": {"policy", "agents", "coordination"},
}


def toy_config_kind(path: Path) -> str | None:
    relative = path.relative_to(CONFIG_ROOT)
    candidates = (path.stem, *relative.parts[:-1])
    for candidate in candidates:
        if candidate.startswith("toy1_"):
            return "toy1"
        if candidate.startswith("toy2_"):
            return "toy2"
        if candidate.startswith("toy3_"):
            return "toy3"
        if candidate.startswith("toy4_"):
            return "toy4"
        if candidate.startswith("toy5_"):
            return "toy5"
        if candidate.startswith("toy6_"):
            return "toy6"
        if candidate.startswith("toy7_"):
            return "toy7"
        if candidate.startswith("toy8_"):
            return "toy8"
        if candidate.startswith("toy9_"):
            return "toy9"
        if candidate.startswith("toy10_"):
            return "toy10"
    return None


def iter_toy_configs() -> list[tuple[str, Path]]:
    configs = [
        (kind, path)
        for path in sorted(CONFIG_ROOT.rglob("*.yaml"))
        if (kind := toy_config_kind(path)) is not None
    ]
    assert configs, "Expected checked-in Toy1-10 YAML configs"
    return configs


LOADERS: dict[
    str,
    Callable[
        [Path],
        Toy1Config
        | Toy2Config
        | Toy3Config
        | Toy4Config
        | Toy5Config
        | Toy6Config
        | Toy7Config
        | Toy8Config
        | Toy9Config
        | Toy10Config,
    ],
] = {
    "toy1": load_toy1_config,
    "toy10": load_toy10_config,
    "toy2": load_toy2_config,
    "toy3": load_toy3_config,
    "toy4": load_toy4_config,
    "toy5": load_toy5_config,
    "toy6": load_toy6_config,
    "toy7": load_toy7_config,
    "toy8": load_toy8_config,
    "toy9": load_toy9_config,
}


def test_checked_in_toy_configs_use_model_domain_layout() -> None:
    failures: list[str] = []
    for kind, path in iter_toy_configs():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            failures.append(f"{path}: expected a YAML mapping")
            continue

        top_level = set(raw)
        if top_level != REQUIRED_TOP_LEVEL_FIELDS:
            failures.append(
                f"{path}: top-level fields {sorted(top_level)} != "
                f"{sorted(REQUIRED_TOP_LEVEL_FIELDS)}"
            )

        model = raw.get("model")
        if not isinstance(model, dict):
            failures.append(f"{path}: model must be a mapping")
        elif not REQUIRED_MODEL_FIELDS_BY_TOY[kind] <= set(model):
            missing = REQUIRED_MODEL_FIELDS_BY_TOY[kind].difference(model)
            failures.append(f"{path}: missing model field(s): {sorted(missing)}")

        domain = raw.get("domain")
        if not isinstance(domain, dict):
            failures.append(f"{path}: domain must be a mapping")
        elif domain.get("toy") != kind:
            failures.append(f"{path}: domain.toy={domain.get('toy')!r} != {kind!r}")

        try:
            LOADERS[kind](path)
        except Exception as exc:  # noqa: BLE001 - report all config failures together.
            failures.append(f"{path}: loader failed: {exc}")

    assert not failures, "\n".join(failures)
