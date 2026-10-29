"""Paper Sections 1, 9, and 10 frame the audited results without widening them."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SECTIONS = {
    "introduction": PAPER / "sections" / "01-introduction.md",
    "discussion": PAPER / "sections" / "09-discussion.md",
    "conclusion": PAPER / "sections" / "10-conclusion.md",
}
SOURCES = {
    "claim_matrix": PAPER / "claim-matrix.md",
    "section_08": PAPER / "sections" / "08-policy-ladder.md",
    "readme": ROOT / "README.md",
    "consolidation_v2": ROOT / "docs" / "research" / "pivot-policy-ladder-consolidation-v2.md",
}


def _text(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _central_claim(text: str) -> str:
    match = re.search(r"> NABM is a reusable framework (.+?) lifecycle\.", text)
    assert match, "central claim quote not found"
    return " ".join(match.group(0).replace(">", " ").split())


def test_each_section_is_a_draft_with_the_paper_status() -> None:
    for path in SECTIONS.values():
        assert "Status: draft prose candidate." in _text(path), path.name


def test_the_introduction_quotes_the_central_claim_verbatim() -> None:
    assert _central_claim(_text(SECTIONS["introduction"])) == _central_claim(_text(SOURCES["claim_matrix"]))


# Each phrase a section uses, and the phrase in an audited source that it restates.
QUOTED = [
    ("introduction", "1.9 to 6.7 times", "claim_matrix", "cost 1.9 to 6.7 times the rule ABM per agent-step"),
    ("introduction", "about a quarter of runs", "section_08", "about a quarter of runs"),
    ("discussion", "1.9 to 6.7 times", "claim_matrix", "cost 1.9 to 6.7 times the rule ABM per agent-step"),
    ("discussion", "about 60 times", "claim_matrix", "per-agent learners 57 to 65 times"),
    ("discussion", "under 10 GPU minutes", "claim_matrix", "Elicitation took under 10 GPU minutes"),
    ("discussion", "in 7 of 9 world and learner settings", "readme", "in 7 of 9 world-and-learner settings"),
    ("discussion", "about 110 GPU-seconds", "claim_matrix", "about 110 GPU-seconds per run"),
    ("discussion", "about 5 CPU-seconds", "claim_matrix", "about 5 CPU-seconds for an NABM learner run"),
    ("discussion", "beat the adaptive learner in all 12 study groups", "section_08",
     "The best table beats the adaptive learner in all 12 groups."),
    ("discussion", "donors ran in other worlds of the same generator", "section_08",
     "whose donors ran in other worlds beats the learner in all 12 groups"),
    ("discussion", "about a quarter of runs", "section_08", "about a quarter of runs"),
    ("discussion", "1.35 to 1.48 times", "section_08", "1.35 to 1.48 times as much as F1"),
    ("discussion", "0.984 to 1.001 times", "section_08", "0.984 to 1.001 times as much as their sources"),
    ("discussion", "(5 of 5 against 0 of 5)", "claim_matrix", "output-average baseline reaches `0/5` in all spread cases"),
    ("conclusion", "under 10 GPU minutes", "claim_matrix", "Elicitation took under 10 GPU minutes"),
    ("conclusion", "38 to 125 GPU hours", "consolidation_v2", "38 to 125 GPU hours"),
    ("conclusion", "about a quarter of runs", "section_08", "about a quarter of runs"),
]


@pytest.mark.parametrize(("section", "phrase", "source", "evidence"), QUOTED)
def test_numbers_restate_audited_sources(section: str, phrase: str, source: str, evidence: str) -> None:
    assert phrase in _text(SECTIONS[section])
    assert evidence in _text(SOURCES[source])


def test_the_stated_scale_matches_the_policy_ladder_configs() -> None:
    configs = ROOT / "experiments" / "configs"
    pivot = yaml.safe_load((configs / "pivot_llm_prior_pilot_v4.yaml").read_text(encoding="utf-8"))
    spread = yaml.safe_load((configs / "info_spread_policy_ladder_study_v2.yaml").read_text(encoding="utf-8"))
    sizes = (_find(pivot, "agents_per_replicate"), spread["profiles"]["development"]["agents"])
    assert sizes == (256, 1000)
    for section in ("introduction", "discussion"):
        assert "256 to 1,000 agents per run" in _text(SECTIONS[section])


def _find(document: object, key: str) -> object:
    if isinstance(document, dict):
        if key in document:
            return document[key]
        for value in document.values():
            found = _find(value, key)
            if found is not None:
                return found
    return None


def test_the_released_tag_is_the_package_version() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert f"released in `v{version}`" in _text(SECTIONS["introduction"])


def test_the_sections_keep_their_boundaries() -> None:
    introduction, discussion, conclusion = (_text(path) for path in SECTIONS.values())
    for required in (
        "No performance claim is central.",
        "This does not imply the absence of learning.",
        "Neural necessity is not established.",
        "one local 4B model family",
        "The worlds are synthetic",
    ):
        assert required in introduction, required
    for required in (
        "This is an engineering architecture claim.",
        "not dominance over hand-coded rules",
        "This does not imply the absence of learning.",
        "Neural necessity is not established.",
        "one generator, one veracity relation",
        "not the reach of false items",
        "It is not a Transformer-grade universal primitive.",
    ):
        assert required in discussion, required
    for required in ("None of these answers is the claim.", "remain open"):
        assert required in conclusion, required


def test_the_outline_links_each_draft() -> None:
    outline = (PAPER / "outline.md").read_text(encoding="utf-8")
    for path in SECTIONS.values():
        assert f"(sections/{path.name})" in outline, path.name
