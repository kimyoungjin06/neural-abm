from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
import shutil
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_api_only_topic_routing_v1.yaml"
)
OUTPUT = Path("/ssd/pivot-feedback-d0-api-only-topic-routing-v1-dry-run-20260729")
PRODUCER_SCRIPT = (
    ROOT / "scripts/build_pivot_feedback_d0_api_only_topic_routing_v1.py"
)
VERIFIER_SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_api_only_topic_routing_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRODUCER = _load(PRODUCER_SCRIPT, "pivot_feedback_topic_routing_producer_test")
VERIFIER = _load(VERIFIER_SCRIPT, "pivot_feedback_topic_routing_verifier_test")


def _document() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _route(title: str, topic_id: int) -> dict:
    document = _document()
    routing = document["routing_contract"]
    return PRODUCER.route_title(
        title,
        document["topic_cards"][topic_id],
        opaque_patterns=[
            re.compile(pattern)
            for pattern in routing["global_opaque_title_patterns"]
        ],
        minimum_tokens=routing["minimum_informative_token_count"],
    )


@pytest.mark.parametrize(
    ("title", "topic_id", "expected"),
    [
        ("Cognitive radio spectrum sensing", 10579, "NO_API_RULE_ACCEPT"),
        ("The Compton effect", 10579, "NO_API_RULE_REJECT"),
        (
            "Cognitive radio and the Compton effect",
            10579,
            "API_REQUIRED_AMBIGUOUS",
        ),
        ("Signal detection under noise", 10579, "API_REQUIRED_AMBIGUOUS"),
        ("PVN LOT 123 ABC", 10579, "QUARANTINE_UNRESOLVED"),
        ("Trypsin proteolysis", 12505, "NO_API_RULE_ACCEPT"),
        ("Amylase activity", 12505, "NO_API_RULE_REJECT"),
    ],
)
def test_fixture_routes(
    title: str,
    topic_id: int,
    expected: str,
) -> None:
    assert _route(title, topic_id)["local_route"] == expected


def test_current_dry_run_replays_independently() -> None:
    result = VERIFIER.validate(CONFIG, OUTPUT)

    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["sample_rows"] == 384
    assert result["route_counts"] == {
        "NO_API_RULE_ACCEPT": 229,
        "NO_API_RULE_REJECT": 24,
        "API_REQUIRED_AMBIGUOUS": 94,
        "QUARANTINE_UNRESOLVED": 37,
    }
    assert result["avoided_api_request_rows"] == 290
    assert result["counterfactual_api_request_rows"] == 94
    assert result["api_request_avoidance_fraction"] == "0.755208"
    assert result["network_accessed"] is False
    assert result["credentials_accessed"] is False
    assert result["api_requests_submitted"] == 0
    assert result["accuracy_established"] is False
    assert result["candidate_universe_screen_complete"] is False
    assert result["production_dispositions_authorized"] is False


def test_rule_function_has_no_score_source_or_api_inputs() -> None:
    arguments = PRODUCER.route_title.__annotations__

    assert set(arguments) == {
        "title",
        "card",
        "opaque_patterns",
        "minimum_tokens",
        "return",
    }


def test_paid_or_accuracy_flags_fail_closed(tmp_path: Path) -> None:
    document = _document()
    document["dry_run_contract"]["accuracy_claim_allowed"] = True
    document["authorization"]["paid_api_submission"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.TopicRoutingVerificationError,
        match="unsafe dry-run flag",
    ):
        VERIFIER.validate(mutated, OUTPUT)


def test_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    decisions = copied / "topic_routing_decisions.csv"
    decisions.write_text(
        decisions.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    decisions.chmod(0o600)

    with pytest.raises(
        VERIFIER.TopicRoutingVerificationError,
        match="artifact SHA",
    ):
        VERIFIER.validate(CONFIG, copied)


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

    assert "scripts.build_pivot_feedback_d0_api_only_topic_routing_v1" not in imports
