from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "evidence" / "results"
RESULT = RESULTS / "pivot_peer_outcome_contrast_v1_20261030.summary.json"
SUMMARY = RESULTS / "pivot_peer_outcome_contrast_v1_20261030.summary.md"
SCENARIOS = ("baseline", "interdisciplinary_seed_grants", "hot_field_hype")


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _text() -> str:
    return " ".join(SUMMARY.read_text(encoding="utf-8").split())


def _r(value: float, places: str = "0.001") -> str:
    text = str(Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP))
    return text.replace("-", "−") if text.startswith("-") else f"+{text}"


def _row(row: dict, places: str = "0.001") -> str:
    return f"{_r(row['mean'], places)} [{_r(row['low'], places)}, {_r(row['high'], places)}]"


def test_reproduced_under_its_freeze_with_every_check_passing() -> None:
    result = _result()
    assert result["precommit_commit"] == "586f33bff0b50b4d7363555bb8d718d008be369a"
    assert all(result["execution"]["byte_identical"].values()) and len(result["execution"]["byte_identical"]) == 3
    checks = result["checks"]
    assert checks["study2_parity"] == {"delay_0_runs_checked": 270, "delay_0_runs_equal": 270, "frozen_delay_3_runs_equal": 90, "passes": True}
    assert not any(checks["n11b_parity"].values()) and checks["n11b_receipt_identity"] and checks["n11b_output_drift"] == []
    assert result["data"]["unconverged_fits"] == 0
    assert all(row["unconverged_fits"] == 0 for row in result["simulation"].values())
    assert result["data_access"] == {"partition": "DISCOVERY_ONLY", "confirmatory_outcome_rows_read": 0, "g4_outcome_rows_read": 0}


def test_the_estimator_flags_the_frozen_arm_so_the_data_did_is_not_read() -> None:
    result = _result()
    assert result["status"] == "E4_NOT_INTERPRETABLE_THE_ESTIMATOR_FLAGS_STUDY2_FROZEN_ARM"
    expectations = result["decision"]["expectations"]
    assert {key: row["held"] for key, row in expectations.items()} == {"ES2": False, "ES3": False, "EE1": True, "EE2": False}
    assert expectations["ES3"]["covers_zero"] == {"baseline": None, "interdisciplinary_seed_grants": False, "hot_field_hype": True}
    assert expectations["ES3"]["rows"]["interdisciplinary_seed_grants"]["high"] < 0
    assert expectations["ES2"]["rows"]["baseline"]["mean"] is None
    assert result["simulation"]["simulation/baseline/delay_0"]["state_events"]["frozen"]["F"] == 9.0
    for scenario in SCENARIOS:
        imitative = result["simulation"][f"simulation/{scenario}/delay_3"]["by_arm"]["imitative"]
        assert imitative["state/success_following"]["low"] > 0 and imitative["state/failure_avoidance"]["high"] < 0
        assert result["simulation"][f"simulation/{scenario}/delay_3"]["state_events"]["cautionary"]["F"] == 0.0


def test_the_summary_quotes_the_record() -> None:
    result = _result()
    text = _text()
    expectations = result["decision"]["expectations"]
    assert f"Grants: {_r(expectations['ES2']['rows']['interdisciplinary_seed_grants']['mean'], '0.01')}" in text
    assert f"Grants: {_row(expectations['ES3']['rows']['interdisciplinary_seed_grants'])}" in text
    assert f"Hype: {_row(expectations['ES3']['rows']['hot_field_hype'])}" in text
    assert f"Held: {_row(expectations['EE1']['row'])} log-odds" in text
    assert f"Failed: {_row(expectations['EE2']['row'])}" in text
    labels = {"baseline": "baseline", "interdisciplinary_seed_grants": "grants", "hot_field_hype": "hype"}
    for key, scenario, arm in (("baseline", "baseline", "imitative"), ("grants", "interdisciplinary_seed_grants", "frozen"),
                               ("grants", "interdisciplinary_seed_grants", "imitative"), ("hype", "hot_field_hype", "frozen"),
                               ("hype", "hot_field_hype", "imitative")):
        rows = result["simulation"][f"simulation/{scenario}/delay_3"]["by_arm"][arm]
        line = f"| {labels[scenario]} | {arm} | " + " | ".join(
            _row(rows[f"state/{q}"], "0.01") for q in ("visibility_did", "success_following", "failure_avoidance")) + " |"
        assert line in text, line
    events = result["data"]["state_events"]
    assert "| Events | " + " | ".join(str(int(events[s])) for s in ("S", "F", "P_S", "P_F", "U")) + " |" in text
    estimates = result["data"]["estimates"]
    for label, key in (("Exposure, unrecruited entries", "exposure/exposed"),
                       ("Exposure, recruited entries counted as events", "exposure_recruited_as_events/exposed"),
                       ("Outcome gap, visible (S − F)", "state/outcome_gap_visible"),
                       ("Outcome gap, before visible (P_S − P_F)", "state/outcome_gap_before_visible"),
                       ("Visibility DiD", "state/visibility_did"), ("Success-following", "state/success_following"),
                       ("Failure-avoidance", "state/failure_avoidance"), ("Entry-order gradient of the label", "entry_order/entry_time")):
        assert f"| {label} | {_row(estimates[key])} |" in text, label
    counts = result["data"]["counts"]
    assert f"{counts['discovery_entrants']:,} entrants, {counts['discovery_precise_entrants']:,} of them precise" in text
    assert f"{counts['dyads']:,} dyads" in text and f"{counts['exposed_authors']:,} exposed authors" in text
    assert f"{counts['exposed_rows_with_a_visible_entry']:,} rows with a visible coauthor entry" in text
    post = result["post_hoc"]
    for key in ("label_rate_of_precise_entrants", "recruited_share_of_precise_entrants", "incumbent_entry_coauthor_share_of_precise_entrants"):
        assert f"{Decimal(str(post[key] * 100)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%" in text, key
    assert f"at least {Decimal(str(post['imitative_outcome_gap_visible_min_mean'])).quantize(Decimal('0.01'))} log-odds" in text
    assert f"at least {Decimal(str(post['imitative_visibility_did_at_delay_3_min_mean'])).quantize(Decimal('0.01'))}" in text
    assert f"upper bounds are {Decimal(str(post['data_outcome_gap_visible_high'])).quantize(Decimal('0.01'))} and {Decimal(str(post['data_visibility_did_high'])).quantize(Decimal('0.01'))}" in text
    half = (estimates["state/visibility_did"]["high"] - estimates["state/visibility_did"]["low"]) / 2
    assert f"interval half-width, {Decimal(str(half)).quantize(Decimal('0.01'))}" in text
    walls = result["execution"]["wall_seconds"]
    assert f"{round(walls['development']):,} s for the development run" in text
    assert f"{round(walls['reproduction']):,} s for the reproduction run" in text
