from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_t10543_local_routing_v1.yaml"
)
OUTPUT = Path("/ssd/pivot-feedback-d0-t10543-local-routing-v1-20260729")
VERIFIER_SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_t10543_local_routing_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_SCRIPT, "t10543_local_routing_verifier_test")


def _document() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_current_t10543_local_routing_replays() -> None:
    result = VERIFIER.validate(config_path=CONFIG, output_root=OUTPUT)
    candidate = result["candidate"]

    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert candidate["recent_accept_rows"] == 8
    assert candidate["recent_reject_rows"] == 1
    assert candidate["recent_ambiguous_rows"] == 3
    assert candidate["recent_quarantine_rows"] == 52
    assert candidate["recent_opaque_title_fraction"] == 0.8125
    assert candidate["recent_opaque_dominant_source_share"] == 1.0
    assert candidate["dry_run_state"] == "WOULD_CONTENT_UNRESOLVED"
    assert result["api_requests_created"] == 0
    assert result["production_content_dispositions_created"] == 0


def test_api_or_rule_amendment_cannot_be_opened(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["new_api_request_creation"] = "AUTHORIZED"
    document["authorization"]["candidate_rule_amendment"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.T10543LocalRoutingVerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(config_path=mutated, output_root=OUTPUT)


def test_route_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    routes = copied / "t10543_local_title_routes.csv"
    routes.write_text(routes.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    routes.chmod(0o600)

    with pytest.raises(
        VERIFIER.T10543LocalRoutingVerificationError,
        match="artifact",
    ):
        VERIFIER.validate(config_path=CONFIG, output_root=copied)


def test_verifier_does_not_import_producer() -> None:
    tree = ast.parse(VERIFIER_SCRIPT.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "scripts.build_pivot_feedback_d0_t10543_local_routing_v1" not in imports
