from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml


_MODEL_KEYS = {"policy", "agents", "coordination", "state"}
_DOMAIN_KEYS = {"environment", "game", "graph", "data", "rewiring", "network"}
_ALIASES = {
    "social": ("model", "coordination"),
    "dynamics": ("model", "policy"),
}


class BinaryToyConfigDict(dict):
    """Test helper that mutates old section names while dumping the new layout."""

    def _section(self, key: str) -> Any:
        if key in _ALIASES:
            parent, field = _ALIASES[key]
            return dict.__getitem__(self, parent)[field]
        if key in _MODEL_KEYS:
            return dict.__getitem__(self, "model")[key]
        if key in _DOMAIN_KEYS:
            return dict.__getitem__(self, "domain")[key]
        return dict.__getitem__(self, key)

    def __getitem__(self, key: str) -> Any:
        if key in _ALIASES or key in _MODEL_KEYS or key in _DOMAIN_KEYS:
            return self._section(key)
        return dict.__getitem__(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in _ALIASES:
            parent, field = _ALIASES[key]
            dict.__getitem__(self, parent)[field] = value
            return
        if key in _MODEL_KEYS:
            dict.__getitem__(self, "model")[key] = value
            return
        if key in _DOMAIN_KEYS:
            dict.__getitem__(self, "domain")[key] = value
            return
        dict.__setitem__(self, key, value)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str) and key in _ALIASES:
            parent, field = _ALIASES[key]
            return field in dict.__getitem__(self, parent)
        if isinstance(key, str) and key in _MODEL_KEYS | _DOMAIN_KEYS:
            parent = "model" if key in _MODEL_KEYS else "domain"
            return key in dict.__getitem__(self, parent)
        return dict.__contains__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        return default

    def pop(self, key: str, default: Any = None) -> Any:
        if key in _ALIASES:
            parent, field = _ALIASES[key]
            return dict.__getitem__(self, parent).pop(field, default)
        if key in _MODEL_KEYS:
            return dict.__getitem__(self, "model").pop(key, default)
        if key in _DOMAIN_KEYS:
            return dict.__getitem__(self, "domain").pop(key, default)
        return dict.pop(self, key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in _ALIASES:
            parent, field = _ALIASES[key]
            return dict.__getitem__(self, parent).setdefault(field, default)
        if key in _MODEL_KEYS:
            return dict.__getitem__(self, "model").setdefault(key, default)
        if key in _DOMAIN_KEYS:
            return dict.__getitem__(self, "domain").setdefault(key, default)
        return dict.setdefault(self, key, default)

    def clone(self) -> "BinaryToyConfigDict":
        return BinaryToyConfigDict(deepcopy(dict(self)))


def _represent_binary_config(
    dumper: yaml.SafeDumper,
    data: BinaryToyConfigDict,
) -> yaml.nodes.MappingNode:
    return dumper.represent_dict(dict(data))


yaml.SafeDumper.add_representer(BinaryToyConfigDict, _represent_binary_config)


def binary_toy_config(raw: dict[str, Any], toy: str) -> BinaryToyConfigDict:
    raw = deepcopy(raw)
    domain: dict[str, Any] = {"toy": toy}
    for key in ("environment", "game", "graph"):
        if key in raw:
            domain[key] = raw.pop(key)
    model = {
        "policy": raw.pop("policy"),
        "agents": raw.pop("agents"),
        "coordination": raw.pop("coordination"),
        "state": raw.pop("state"),
    }
    return BinaryToyConfigDict(
        {
            "run": raw.pop("run"),
            "simulation": raw.pop("simulation"),
            "model": model,
            "domain": domain,
            "logging": raw.pop("logging"),
            **raw,
        }
    )


def toy1_config(raw: dict[str, Any]) -> BinaryToyConfigDict:
    raw = deepcopy(raw)
    domain = {
        "toy": "toy1",
        "data": raw.pop("data"),
        "graph": raw.pop("graph"),
    }
    model = {
        "agents": raw.pop("agents"),
        "coordination": raw.pop("social"),
    }
    return BinaryToyConfigDict(
        {
            "run": raw.pop("run"),
            "simulation": raw.pop("simulation"),
            "model": model,
            "domain": domain,
            "logging": raw.pop("logging"),
            **raw,
        }
    )


def toy3_config(raw: dict[str, Any]) -> BinaryToyConfigDict:
    raw = deepcopy(raw)
    domain = {
        "toy": "toy3",
        "environment": raw.pop("environment"),
        "graph": raw.pop("graph"),
        "rewiring": raw.pop("rewiring"),
    }
    model = {
        "policy": raw.pop("dynamics"),
        "agents": raw.pop("agents"),
        "coordination": raw.pop("social"),
    }
    return BinaryToyConfigDict(
        {
            "run": raw.pop("run"),
            "simulation": raw.pop("simulation"),
            "model": model,
            "domain": domain,
            "logging": raw.pop("logging"),
            **raw,
        }
    )


def toy6_config(raw: dict[str, Any]) -> BinaryToyConfigDict:
    raw = deepcopy(raw)
    domain = {
        "toy": "toy6",
        "environment": raw.pop("environment"),
        "game": raw.pop("game"),
        "graph": raw.pop("graph"),
    }
    model = {
        "policy": raw.pop("policy"),
        "agents": raw.pop("agents"),
        "coordination": raw.pop("coordination"),
    }
    return BinaryToyConfigDict(
        {
            "run": raw.pop("run"),
            "simulation": raw.pop("simulation"),
            "model": model,
            "domain": domain,
            "logging": raw.pop("logging"),
            **raw,
        }
    )


def toy7_config(raw: dict[str, Any]) -> BinaryToyConfigDict:
    raw = deepcopy(raw)
    domain = {
        "toy": "toy7",
        "environment": raw.pop("environment"),
        "graph": raw.pop("graph"),
    }
    model = {
        "policy": raw.pop("policy"),
        "agents": raw.pop("agents"),
        "coordination": raw.pop("coordination"),
    }
    return BinaryToyConfigDict(
        {
            "run": raw.pop("run"),
            "simulation": raw.pop("simulation"),
            "model": model,
            "domain": domain,
            "logging": raw.pop("logging"),
            **raw,
        }
    )


def toy8_config(raw: dict[str, Any]) -> BinaryToyConfigDict:
    raw = deepcopy(raw)
    domain = {
        "toy": "toy8",
        "environment": raw.pop("environment"),
        "graph": raw.pop("graph"),
    }
    model = {
        "policy": raw.pop("policy"),
        "agents": raw.pop("agents"),
        "coordination": raw.pop("coordination"),
    }
    return BinaryToyConfigDict(
        {
            "run": raw.pop("run"),
            "simulation": raw.pop("simulation"),
            "model": model,
            "domain": domain,
            "logging": raw.pop("logging"),
            **raw,
        }
    )


def toy9_config(raw: dict[str, Any]) -> BinaryToyConfigDict:
    raw = deepcopy(raw)
    domain = {
        "toy": "toy9",
        "environment": raw.pop("environment"),
        "graph": raw.pop("graph"),
    }
    model = {
        "policy": raw.pop("policy"),
        "agents": raw.pop("agents"),
        "coordination": raw.pop("coordination"),
    }
    return BinaryToyConfigDict(
        {
            "run": raw.pop("run"),
            "simulation": raw.pop("simulation"),
            "model": model,
            "domain": domain,
            "logging": raw.pop("logging"),
            **raw,
        }
    )


def toy10_config(raw: dict[str, Any]) -> BinaryToyConfigDict:
    raw = deepcopy(raw)
    domain = {
        "toy": "toy10",
        "environment": raw.pop("environment"),
        "network": raw.pop("network"),
    }
    model = {
        "policy": raw.pop("policy"),
        "agents": raw.pop("agents"),
        "coordination": raw.pop("coordination"),
    }
    return BinaryToyConfigDict(
        {
            "run": raw.pop("run"),
            "simulation": raw.pop("simulation"),
            "model": model,
            "domain": domain,
            "logging": raw.pop("logging"),
            **raw,
        }
    )
