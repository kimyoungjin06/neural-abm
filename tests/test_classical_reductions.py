"""Golden tests for classical instances and explicitly labeled near variants.

These pin the claim-bearing deterministic outcomes in
``examples/classical_reductions.py``: exact DeGroot consensus, FJ-like anchored
disagreement, self-excluding HK cluster collapse, and the Granovetter knife-edge
cascade.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSICAL_REDUCTIONS_EXAMPLE = ROOT / "examples" / "classical_reductions.py"


def _run_reductions() -> dict:
    completed = subprocess.run(
        [sys.executable, str(CLASSICAL_REDUCTIONS_EXAMPLE)],
        check=True,
        capture_output=True,
        cwd=ROOT,
        text=True,
    )
    return json.loads(completed.stdout)


def test_classical_examples_run_torch_free_and_match_claimed_outcomes() -> None:
    payload = _run_reductions()

    assert payload["status"] == "ok"
    assert payload["torch_loaded"] is False
    assert payload["surface"] == "neural_abm.scenario_lite"

    degroot = payload["degroot"]
    assert degroot["consensus_reached"] is True
    assert degroot["variance_monotone_decreasing"] is True
    assert abs(degroot["final_mean"] - 0.5) < 1e-6
    assert degroot["final_range"] < 1e-3

    friedkin_johnsen = payload["friedkin_johnsen"]
    assert friedkin_johnsen["model"] == "friedkin_johnsen_like_pre_mix_anchor"
    assert friedkin_johnsen["converged"] is True
    assert friedkin_johnsen["disagreement_persists"] is True
    assert friedkin_johnsen["final_range"] == 0.651821

    clusters = {
        row["epsilon"]: row["cluster_count"] for row in payload["hegselmann_krause"]
    }
    assert clusters == {0.05: 8, 0.15: 3, 0.35: 1}
    assert {row["model"] for row in payload["hegselmann_krause"]} == {
        "hegselmann_krause_self_excluding_variant"
    }

    granovetter = payload["granovetter"]
    assert granovetter["baseline_adopters"] == 100
    assert granovetter["perturbed_adopters"] == 1
    assert granovetter["knife_edge_reproduced"] is True
    assert granovetter["comparison"]["delta"] == -99.0
    assert granovetter["switches"]["peer_similarity_threshold"] == 0.0


def test_classical_reductions_example_imports_only_api_lite() -> None:
    import ast

    tree = ast.parse(CLASSICAL_REDUCTIONS_EXAMPLE.read_text(encoding="utf-8"))
    neural_imports = sorted(
        {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("neural_abm")
        }
    )
    assert neural_imports == ["neural_abm.api_lite"]
