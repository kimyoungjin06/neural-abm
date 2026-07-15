from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_nabm_unit_schematic.py"
SVG = ROOT / "docs" / "figures" / "nabm_unit_recurrent_block.svg"
DOC = ROOT / "docs" / "classical-reductions.md"


def _load_generator():
    spec = importlib.util.spec_from_file_location("plot_nabm_unit_schematic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_in_order(text: str, labels: list[str]) -> None:
    positions = [text.index(label) for label in labels]
    assert positions == sorted(positions), positions


def test_schematic_encodes_full_unit_execution_order_and_ownership() -> None:
    svg = SVG.read_text(encoding="utf-8")
    _assert_in_order(
        svg,
        [
            "Local update ×N",
            "Validated messages ×N",
            "PeerSelector(messages)",
            "SocialValueBuilder",
            "NABMStep.run",
            "Audit + report",
        ],
    )
    assert "agent.local_update(...)" in svg
    assert "domain-supplied method" in svg
    assert "agent.local_update(...) · injected adapter" not in svg
    assert "injected CommitAdapter inside step" in svg
    assert "external caller after report" in svg
    assert "next timestep" in svg
    assert "kind · bounds" in svg
    assert "commit mode" in svg
    assert "kind · α · validation" not in svg
    assert "NABMStepResult" in svg
    assert "NABMUnitReport" in svg
    assert "StepReport output" not in svg


def test_schematic_encodes_lite_runtime_order_and_timestep_recurrence() -> None:
    svg = SVG.read_text(encoding="utf-8")
    _assert_in_order(
        svg,
        [
            "Scenario + agents",
            "Candidate neighbors",
            "Local adaptation",
            "Peer filter",
            "Bounded-scalar mix",
            "Domain transition",
            "Aggregate + micro audit",
        ],
    )
    assert "build before workflow" in svg
    assert "uses post-local xᵢ" in svg
    assert "repeat each timestep" in svg
    for transformer_label in ("Q / K / V", "Multi-Head", "Add &amp; Norm"):
        assert transformer_label not in svg


def test_caption_explains_lite_order_and_generic_execution_sites() -> None:
    doc = DOC.read_text(encoding="utf-8")
    caption = doc[doc.index("**Figure 1.") : doc.index("Four explicit controls")]
    assert "builds candidate neighbors *before*" in caption
    assert "post-local values" in caption
    assert "injected `CommitAdapter` inside the step" in caption
    assert "external caller after" in caption
    assert "recurrence returns to candidate-neighbor construction" in caption


def test_publication_typography_budget_is_at_least_seven_points() -> None:
    generator = _load_generator()
    final_minimum = (
        generator.MIN_TEXT_SIZE
        * generator.PUBLICATION_WIDTH_IN
        / generator.FIGURE_SIZE[0]
    )
    assert final_minimum >= 7.0


def test_tracked_schematic_assets_match_generator() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--docs-only"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
