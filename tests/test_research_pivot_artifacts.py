from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "docs" / "case-studies" / "researcher-pivot"
STUDY1 = CASE_ROOT / "data" / "study_results.json"
STUDY2 = CASE_ROOT / "data" / "learning_study_results.json"


def _comparison(
    payload: dict[str, object],
    *,
    scenario: str,
    arm: str | None = None,
    outcome: str | None = None,
) -> dict[str, object]:
    rows = payload["comparisons"]
    assert isinstance(rows, list)
    matches = [
        row
        for row in rows
        if row["scenario"] == scenario
        and (arm is None or row.get("arm") == arm)
        and (outcome is None or row.get("outcome_field") == outcome)
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_source_snapshot_matches_current_tree(payload: dict[str, object]) -> None:
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    snapshot = provenance["source_snapshot"]
    assert isinstance(snapshot, dict)
    paths = snapshot["paths"]
    assert isinstance(paths, list)
    digest = hashlib.sha256()
    for relative in paths:
        assert isinstance(relative, str)
        path = ROOT / relative
        assert path.is_file(), relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    assert snapshot["file_count"] == len(paths)
    assert snapshot["sha256"] == digest.hexdigest()


def test_study1_claim_ledger_matches_tracked_artifact() -> None:
    payload = json.loads(STUDY1.read_text(encoding="utf-8"))
    grants = _comparison(payload, scenario="interdisciplinary_seed_grants")
    hype = _comparison(payload, scenario="hot_field_hype")
    placebo = payload["sensitivity"]["grant_scale_sweep"][0]["seed_grants"]
    direct = payload["direct_contrasts"][0]

    assert grants["mean_delta"] == pytest.approx(0.19125)
    assert grants["mean_effect_ci95"] == pytest.approx([0.1808956564, 0.2016043436])
    assert hype["mean_delta"] == pytest.approx(0.1869166667)
    assert placebo["mean_delta"] == 0.0
    assert placebo["mean_effect_ci95"] == [0.0, 0.0]
    assert all(value == 0.0 for value in placebo["paired_deltas"])
    assert direct["mean_delta"] == pytest.approx(-0.004333)
    assert direct["mean_effect_ci95"] == pytest.approx([-0.012397, 0.00373])
    grant_outcomes = payload["scenarios"]["interdisciplinary_seed_grants"][
        "outcome_values"
    ]
    hype_outcomes = payload["scenarios"]["hot_field_hype"]["outcome_values"]
    recomputed_direct = [
        hype_value - grant_value
        for grant_value, hype_value in zip(
            grant_outcomes,
            hype_outcomes,
            strict=True,
        )
    ]
    assert direct["paired_deltas"] == pytest.approx(recomputed_direct, abs=1e-8)
    assert payload["provenance"]["rng_contract"].startswith("keyed SeedSequence")
    _assert_source_snapshot_matches_current_tree(payload)


def test_study2_claim_ledger_keeps_raw_outcomes_and_parameter_audit() -> None:
    payload = json.loads(STUDY2.read_text(encoding="utf-8"))
    baseline_imitation = _comparison(
        payload,
        scenario="baseline",
        arm="imitative - frozen",
        outcome="failed_pivot_rate",
    )
    hype_imitation = _comparison(
        payload,
        scenario="hot_field_hype",
        arm="imitative - frozen",
        outcome="failed_pivot_rate",
    )
    hype_failure_only = _comparison(
        payload,
        scenario="hot_field_hype",
        arm="cautionary - frozen",
        outcome="failed_pivot_rate",
    )

    assert baseline_imitation["mean"] == pytest.approx(0.2806)
    assert baseline_imitation["mean_ci95"] == pytest.approx([0.235, 0.3217])
    assert hype_imitation["mean"] == pytest.approx(-0.0086)
    assert hype_failure_only["mean"] == pytest.approx(-0.1436)
    assert hype_failure_only["mean_ci95"] == pytest.approx([-0.1592, -0.1294])
    assert payload["scenario_definitions"]["hot_field_hype"]["parameters"]
    assert (
        "attention_signal"
        in payload["scenario_definitions"]["hot_field_hype"]["parameters"]
    )
    _assert_source_snapshot_matches_current_tree(payload)

    for arms in payload["scenarios"].values():
        for summary in arms.values():
            assert len(summary["replicate_runs"]) == payload["config"]["replicates"]
            assert set(summary["mean_weight_trajectories"]) == set(payload["features"])
            assert len(summary["mean_bias_trajectory"]) == payload["config"]["steps"]

    for comparison in payload["comparisons"]:
        scenario = payload["scenarios"][comparison["scenario"]]
        arm_name = comparison["arm"].removesuffix(" - frozen")
        outcome = comparison["outcome_field"]
        recomputed = [
            adaptive[outcome] - frozen[outcome]
            for frozen, adaptive in zip(
                scenario["frozen"]["replicate_runs"],
                scenario[arm_name]["replicate_runs"],
                strict=True,
            )
        ]
        assert comparison["paired_deltas"] == pytest.approx(recomputed, abs=1e-8)
        assert comparison["mean"] == pytest.approx(
            sum(recomputed) / len(recomputed),
            abs=5e-5,
        )


def test_claim_prose_drops_the_reviewed_overclaims() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            CASE_ROOT / "README.md",
            ROOT / "paper" / "sections" / "07-research-application.md",
            ROOT / "paper" / "claim-matrix.md",
            ROOT / "docs" / "classical-reductions.md",
        )
    )

    for stale in (
        "baseline world is left exactly unchanged",
        "could not have been *asked*",
        "the 95% CI is the percentile",
        "moves nowhere else",
        "textbook information-cascade mechanism",
        "acquired hype immunity",
    ):
        assert stale not in text
    assert "not an equivalence claim" in text
    assert "not an identified single route" in text


def test_research_pivot_figures_match_the_tracked_evidence() -> None:
    subprocess.run(
        [sys.executable, "scripts/plot_research_pivot_study.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
