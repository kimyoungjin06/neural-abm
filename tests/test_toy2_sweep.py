from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_toy2_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_toy2_sweep", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_toy2_sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_toy2_sweep
SPEC.loader.exec_module(run_toy2_sweep)

REGIME_PRESETS = run_toy2_sweep.REGIME_PRESETS
CoordinationSweepPoint = run_toy2_sweep.CoordinationSweepPoint
Toy2SweepSettings = run_toy2_sweep.Toy2SweepSettings
build_toy2_rows_from_args = run_toy2_sweep.build_toy2_rows_from_args
build_grouped_summary = run_toy2_sweep.build_grouped_summary
effective_action_selection_modes = run_toy2_sweep.effective_action_selection_modes
effective_action_temperatures = run_toy2_sweep.effective_action_temperatures
effective_calibration_strengths = run_toy2_sweep.effective_calibration_strengths
effective_decision_calibrations = run_toy2_sweep.effective_decision_calibrations
effective_interaction_modes = run_toy2_sweep.effective_interaction_modes
effective_local_update_rules = run_toy2_sweep.effective_local_update_rules
effective_neural_peer_modes = run_toy2_sweep.effective_neural_peer_modes
effective_policy_prior_specs = run_toy2_sweep.effective_policy_prior_specs
effective_policy_temperatures = run_toy2_sweep.effective_policy_temperatures
iter_toy2_case_points = run_toy2_sweep.iter_toy2_case_points
iter_toy2_sweep_blocks = run_toy2_sweep.iter_toy2_sweep_blocks
parse_args = run_toy2_sweep.parse_args
parse_policy_prior_specs = run_toy2_sweep.parse_policy_prior_specs
resolve_policy_prior_probability = run_toy2_sweep.resolve_policy_prior_probability
result_row = run_toy2_sweep.result_row
non_overwriting_path = run_toy2_sweep.non_overwriting_path
write_case_config = run_toy2_sweep.write_case_config
write_rd_config = run_toy2_sweep.write_rd_config
write_summary_csv = run_toy2_sweep.write_summary_csv


def load_base_config(tmp_path: Path) -> dict:
    base_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "configs"
        / "toy2_spatial_pd_baseline.yaml"
    )
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    raw["run"]["output_dir"] = str(tmp_path / "runs")
    return raw


def stag_hunt_regime():
    return next(regime for regime in REGIME_PRESETS if regime.name == "stag_hunt")


def sample_toy2_row(tmp_path: Path, label: str = "toy2_main") -> dict:
    return result_row(
        label=label,
        regime=stag_hunt_regime(),
        update_rule="neural_policy",
        mixer="none",
        peer_rule="none",
        seed=1,
        initial_action_probability=0.6,
        policy_prior_mode="default",
        policy_prior_action_probability=None,
        local_update_rule="sampled_policy_gradient",
        neural_peer_mode="spatial",
        interaction_mode="spatial",
        decision_mode="sampled",
        action_temperature=1.0,
        decision_calibration_mode="none",
        decision_calibration_strength=4.0,
        decision_threshold=None,
        revision_rate=0.1,
        alpha=0.0,
        selection_strength=1.0,
        policy_temperature=1.0,
        payoff_transform="linear",
        exploration_epsilon=0.0,
        learning_enabled=True,
        reputation_decay=0.9,
        reputation_temperature=1.0,
        reputation_noise=0.0,
        reputation_observation_mode="none",
        mobility_enabled=False,
        run_dir=tmp_path / "run",
        final_action_rate=0.6,
        final_mean_payoff=2.0,
        final_mean_policy_action_probability=0.6,
        final_mean_reputation=0.55,
        final_reputation_dispersion=0.1,
        final_mobility_rate=0.0,
        final_mean_mobility_gain=0.0,
        domain_action_components=1,
        domain_largest_action_cluster_fraction=0.5,
        final_fragmentation_components=1,
    )


def test_toy2_sweep_parse_args_preserves_legacy_common_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_toy2_sweep.py"])

    args = parse_args()

    assert args.base_config == Path("experiments/configs/toy2_spatial_pd_baseline.yaml")
    assert args.label == "toy2_regime_sweep_seeds01_05"
    assert args.seeds == [1, 2, 3, 4, 5]
    assert args.epochs == 50
    assert args.mixers == ["none", "output_average"]
    assert args.peer_rules is None
    assert args.alpha == pytest.approx(0.25)
    assert args.alphas is None
    assert args.coordination_thresholds == [0.0]
    assert args.config_dir == Path("experiments/configs/generated")
    assert args.results_dir == Path("experiments/results")


def test_toy2_sweep_exports_common_non_overwriting_helper(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    path.write_text("existing", encoding="utf-8")

    assert non_overwriting_path(path) == tmp_path / "summary_01.csv"


def test_toy2_sweep_main_writes_outputs_with_non_overwrite_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "run": {"output_dir": str(tmp_path / "runs")},
                "simulation": {"epochs": 1},
                "domain": {
                    "environment": {"initial_action_probability": 0.6},
                },
                "model": {"policy": {"revision_rate": 0.1}},
            }
        ),
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for name in (
        "toy2_main_summary.csv",
        "toy2_main_grouped_summary.csv",
        "toy2_main_grouped_summary.md",
    ):
        (results_dir / name).write_text("existing", encoding="utf-8")

    calls: list[dict[str, object]] = []

    def fake_rows(**kwargs: object) -> list[dict]:
        calls.append(dict(kwargs))
        return [sample_toy2_row(tmp_path, label=str(kwargs["label"]))]

    monkeypatch.setattr(run_toy2_sweep, "run_toy2_sweep_rows", fake_rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_toy2_sweep.py",
            "--base-config",
            str(base_path),
            "--label",
            "toy2_main",
            "--results-dir",
            str(results_dir),
            "--config-dir",
            str(tmp_path / "configs"),
            "--regimes",
            "stag_hunt",
            "--seeds",
            "1",
            "--epochs",
            "1",
        ],
    )

    run_toy2_sweep.main()

    summary_path = results_dir / "toy2_main_summary_01.csv"
    grouped_path = results_dir / "toy2_main_grouped_summary_01.csv"
    markdown_path = results_dir / "toy2_main_grouped_summary_01.md"
    assert summary_path.exists()
    assert grouped_path.exists()
    assert markdown_path.exists()
    assert calls[0]["label"] == "toy2_main"
    assert calls[0]["config_dir"] == tmp_path / "configs"


def test_toy2_sweep_policy_priors_generate_expected_yaml(tmp_path: Path) -> None:
    base = load_base_config(tmp_path)
    regime = stag_hunt_regime()
    specs = parse_policy_prior_specs(["default", "match_p0", "0.5"])
    expected = {
        "default": None,
        "match_p0": 0.6,
        "0.5": 0.5,
    }

    for spec in specs:
        prior_probability = resolve_policy_prior_probability(
            spec,
            initial_action_probability=0.6,
        )
        path = write_case_config(
            base=base,
            label="policy_prior_test",
            regime=regime,
            update_rule="neural_policy",
            mixer="none",
            seed=1,
            epochs=1,
            initial_action_probability=0.6,
            policy_prior_spec=spec,
            policy_prior_action_probability=prior_probability,
            local_update_rule="counterfactual_advantage",
            neural_peer_mode="well_mixed",
            interaction_mode="well_mixed_resampled",
            action_selection_mode="argmax",
            action_temperature=0.5,
            decision_calibration_mode="payoff_threshold",
            calibration_strength=2.0,
            alpha=0.0,
            revision_rate=0.1,
            selection_strength=1.0,
            policy_temperature=1.0,
            payoff_transform="linear",
            exploration_epsilon=0.0,
            learning_enabled=False,
            config_dir=tmp_path / "configs",
        )
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert raw["run"]["name"].startswith("policy_prior_test_stag_hunt")
        assert raw["model"]["policy"]["domain"]["local_update_rule"] == (
            "counterfactual_advantage"
        )
        assert raw["model"]["policy"]["domain"]["neural_peer_mode"] == "well_mixed"
        assert raw["model"]["policy"]["domain"]["interaction_mode"] == (
            "well_mixed_resampled"
        )
        assert raw["model"]["policy"]["decision"]["mode"] == "argmax"
        assert raw["model"]["policy"]["decision"]["action_temperature"] == pytest.approx(1.0)
        assert raw["model"]["policy"]["decision"]["exploration_epsilon"] == pytest.approx(0.0)
        assert raw["model"]["policy"]["decision"]["calibration"]["mode"] == "none"
        assert raw["model"]["policy"]["decision"]["calibration"]["strength"] == (
            pytest.approx(4.0)
        )
        if expected[spec.mode] is None:
            assert "policy_prior_action_probability" not in raw["model"]["agents"]
        else:
            assert raw["model"]["agents"]["policy_prior_action_probability"] == (
                pytest.approx(expected[spec.mode])
            )


def test_toy2_sweep_payoff_threshold_calibration_generates_yaml(
    tmp_path: Path,
) -> None:
    base = load_base_config(tmp_path)
    regime = stag_hunt_regime()
    spec = parse_policy_prior_specs(["match_p0"])[0]
    path = write_case_config(
        base=base,
        label="payoff_threshold_test",
        regime=regime,
        update_rule="neural_policy",
        mixer="none",
        seed=1,
        epochs=1,
        initial_action_probability=0.7,
        policy_prior_spec=spec,
        policy_prior_action_probability=0.7,
        local_update_rule="counterfactual_advantage",
        neural_peer_mode="spatial",
        interaction_mode="spatial",
        action_selection_mode="sampled",
        action_temperature=0.5,
        decision_calibration_mode="payoff_threshold",
        calibration_strength=4.0,
        alpha=0.0,
        revision_rate=0.25,
        selection_strength=1.0,
        policy_temperature=1.0,
        payoff_transform="linear",
        exploration_epsilon=0.0,
        learning_enabled=True,
        config_dir=tmp_path / "configs",
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert raw["model"]["policy"]["decision"]["mode"] == "sampled"
    assert raw["model"]["policy"]["decision"]["action_temperature"] == pytest.approx(0.5)
    assert raw["model"]["policy"]["decision"]["calibration"]["mode"] == "payoff_threshold"
    assert raw["model"]["policy"]["decision"]["calibration"]["strength"] == pytest.approx(4.0)


def test_toy2_sweep_rd_reference_generates_expected_yaml(tmp_path: Path) -> None:
    base = load_base_config(tmp_path)
    regime = stag_hunt_regime()
    path = write_rd_config(
        base=base,
        label="rd_reference_test",
        regime=regime,
        seed=2,
        epochs=3,
        initial_action_probability=0.4,
        revision_rate=0.25,
        selection_strength=2.0,
        payoff_transform="tanh",
        exploration_epsilon=0.5,
        learning_enabled=False,
        config_dir=tmp_path / "configs",
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert raw["run"]["name"] == (
        "rd_reference_test_stag_hunt_rd_well_mixed_p0p4_r0p25_reference"
    )
    assert raw["run"]["seed"] == 2
    assert raw["simulation"]["epochs"] == 3
    assert raw["model"]["policy"]["rule"] == "rd_well_mixed"
    assert raw["model"]["policy"]["learning_enabled"] is False
    assert raw["model"]["policy"]["revision_rate"] == pytest.approx(0.25)
    assert raw["model"]["policy"]["selection_strength"] == pytest.approx(2.0)
    assert raw["model"]["policy"]["decision"]["exploration_epsilon"] == pytest.approx(0.0)
    assert raw["model"]["policy"]["domain"]["payoff_transform"] == "tanh"
    assert "policy_prior_action_probability" not in raw["model"]["agents"]
    assert raw["model"]["coordination"]["mixer"] == "none"
    assert raw["model"]["coordination"]["peer_rule"] == "none"
    assert raw["model"]["coordination"]["alpha"] == pytest.approx(0.0)
    assert raw["model"]["state"]["reputation"]["enabled"] is True
    assert raw["model"]["state"]["mobility"]["enabled"] is False


def test_toy2_sweep_output_similarity_coordination_generates_yaml(
    tmp_path: Path,
) -> None:
    base = load_base_config(tmp_path)
    regime = stag_hunt_regime()
    spec = parse_policy_prior_specs(["default"])[0]
    path = write_case_config(
        base=base,
        label="coordination_test",
        regime=regime,
        update_rule="fermi_imitation",
        mixer="output_average",
        peer_rule="output_similarity",
        coordination_threshold=0.3,
        seed=1,
        epochs=1,
        initial_action_probability=0.5,
        policy_prior_spec=spec,
        policy_prior_action_probability=None,
        local_update_rule="sampled_policy_gradient",
        neural_peer_mode="spatial",
        interaction_mode="spatial",
        action_selection_mode="sampled",
        action_temperature=1.0,
        decision_calibration_mode="none",
        calibration_strength=4.0,
        alpha=0.25,
        revision_rate=1.0,
        selection_strength=1.0,
        policy_temperature=1.0,
        payoff_transform="linear",
        exploration_epsilon=0.0,
        learning_enabled=True,
        config_dir=tmp_path / "configs",
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert "output_similarity_coord_th0p3" in raw["run"]["name"]
    assert "output_similarity_coord_th0p3" in path.name
    assert raw["model"]["coordination"]["mixer"] == "output_average"
    assert raw["model"]["coordination"]["peer_rule"] == "output_similarity"
    assert raw["model"]["coordination"]["alpha"] == pytest.approx(0.25)
    assert raw["model"]["coordination"]["threshold"] == pytest.approx(0.3)


def test_toy2_sweep_case_point_generation_preserves_conditional_grid() -> None:
    regime = stag_hunt_regime()
    settings = Toy2SweepSettings(
        regimes=[regime],
        coordination_points=[
            CoordinationSweepPoint(
                mixer="none",
                peer_rule="none",
                alpha=0.0,
                threshold=0.0,
            ),
            CoordinationSweepPoint(
                mixer="output_average",
                peer_rule="none",
                alpha=0.25,
                threshold=0.0,
            ),
            CoordinationSweepPoint(
                mixer="output_average",
                peer_rule="none",
                alpha=0.5,
                threshold=0.0,
            ),
        ],
        initial_action_probabilities=[0.6],
        revision_rates=[0.1],
        selection_strengths=[1.0],
        policy_temperatures=[0.5],
        action_temperatures=[1.0, 0.5],
        learning_enabled_values=[True],
        policy_prior_specs=parse_policy_prior_specs(["default", "match_p0"]),
        local_update_rules=["sampled_policy_gradient"],
        neural_peer_modes=["spatial"],
        interaction_modes=["spatial", "well_mixed_resampled"],
        action_selection_modes=["sampled", "argmax"],
        decision_calibrations=["none", "payoff_threshold"],
        calibration_strengths=[2.0, 4.0],
    )
    block = next(iter_toy2_sweep_blocks(settings))

    points = list(
        iter_toy2_case_points(
            block,
            settings=settings,
            update_rules=["neural_policy", "fermi_imitation"],
            seeds=[1],
        )
    )
    neural_points = [
        point for point in points if point.update_rule == "neural_policy"
    ]
    fermi_points = [
        point for point in points if point.update_rule == "fermi_imitation"
    ]
    argmax_points = [
        point for point in neural_points if point.action_selection_mode == "argmax"
    ]

    assert len(points) == 90
    assert len(neural_points) == 84
    assert len(fermi_points) == 6
    assert points[0].policy_prior_spec.mode == "default"
    assert points[0].policy_prior_action_probability is None
    assert points[0].mixer == "none"
    assert points[0].alpha == pytest.approx(0.0)
    assert points[0].decision_calibration_mode == "none"
    assert {
        point.alpha for point in neural_points if point.mixer == "output_average"
    } == {0.25, 0.5}
    assert {
        point.alpha for point in neural_points if point.mixer == "none"
    } == {0.0}
    assert {point.action_temperature for point in argmax_points} == {1.0}
    assert {point.decision_calibration_mode for point in argmax_points} == {"none"}
    assert {point.policy_prior_spec.mode for point in fermi_points} == {"default"}
    assert {point.policy_temperature for point in fermi_points} == {1.0}
    assert {point.action_temperature for point in fermi_points} == {1.0}
    assert {point.local_update_rule for point in fermi_points} == {
        "sampled_policy_gradient"
    }


def test_toy2_sweep_case_point_generation_supports_output_similarity() -> None:
    regime = stag_hunt_regime()
    settings = Toy2SweepSettings(
        regimes=[regime],
        coordination_points=[
            CoordinationSweepPoint(
                mixer="output_average",
                peer_rule="output_similarity",
                alpha=0.25,
                threshold=0.3,
            ),
            CoordinationSweepPoint(
                mixer="output_average",
                peer_rule="output_similarity",
                alpha=0.25,
                threshold=0.6,
            ),
        ],
        initial_action_probabilities=[0.6],
        revision_rates=[0.1],
        selection_strengths=[1.0],
        policy_temperatures=[1.0],
        action_temperatures=[1.0],
        learning_enabled_values=[True],
        policy_prior_specs=parse_policy_prior_specs(["default"]),
        local_update_rules=["sampled_policy_gradient"],
        neural_peer_modes=["spatial"],
        interaction_modes=["spatial"],
        action_selection_modes=["sampled"],
        decision_calibrations=["none"],
        calibration_strengths=[4.0],
    )
    block = next(iter_toy2_sweep_blocks(settings))

    points = list(
        iter_toy2_case_points(
            block,
            settings=settings,
            update_rules=["neural_policy"],
            seeds=[1],
        )
    )

    assert len(points) == 2
    assert {point.peer_rule for point in points} == {"output_similarity"}
    assert {point.coordination_threshold for point in points} == {0.3, 0.6}


def test_toy2_sweep_rows_uses_point_adapter_and_preserves_rd_insertion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regime = stag_hunt_regime()
    settings = Toy2SweepSettings(
        regimes=[regime],
        coordination_points=[
            CoordinationSweepPoint(
                mixer="none",
                peer_rule="none",
                alpha=0.0,
                threshold=0.0,
            ),
        ],
        initial_action_probabilities=[0.6],
        revision_rates=[0.1],
        selection_strengths=[1.0],
        policy_temperatures=[1.0],
        action_temperatures=[1.0],
        learning_enabled_values=[True],
        policy_prior_specs=parse_policy_prior_specs(["default"]),
        local_update_rules=["sampled_policy_gradient"],
        neural_peer_modes=["spatial"],
        interaction_modes=["spatial"],
        action_selection_modes=["sampled"],
        decision_calibrations=["none"],
        calibration_strengths=[4.0],
    )
    calls: list[tuple[str, int]] = []

    def fake_case_point(**kwargs):
        point = kwargs["point"]
        calls.append(("case", point.seed))
        return (
            {
                "kind": "case",
                "seed": point.seed,
                "policy_rule": point.update_rule,
            },
            f"case seed={point.seed}",
        )

    def fake_rd_point(**kwargs):
        point = kwargs["point"]
        calls.append(("rd", point.seed))
        return (
            {
                "kind": "rd",
                "seed": point.seed,
                "policy_rule": "rd_well_mixed",
            },
            f"rd seed={point.seed}",
        )

    monkeypatch.setattr(run_toy2_sweep, "run_toy2_case_point", fake_case_point)
    monkeypatch.setattr(run_toy2_sweep, "run_toy2_rd_point", fake_rd_point)

    messages: list[str] = []
    rows = run_toy2_sweep.run_toy2_sweep_rows(
        base={"base": True},
        label="toy2_adapter_unit",
        settings=settings,
        update_rules=["fermi_imitation"],
        seeds=[1, 2],
        epochs=1,
        payoff_transform="linear",
        exploration_epsilon=0.0,
        reputation_observation_mode="none",
        skip_rd=False,
        config_dir=tmp_path / "configs",
        progress=messages.append,
    )

    assert rows == [
        {"kind": "case", "seed": 1, "policy_rule": "fermi_imitation"},
        {"kind": "case", "seed": 2, "policy_rule": "fermi_imitation"},
        {"kind": "rd", "seed": 1, "policy_rule": "rd_well_mixed"},
    ]
    assert calls == [("case", 1), ("case", 2), ("rd", 1)]
    assert messages == ["case seed=1", "case seed=2", "rd seed=1"]


def test_toy2_sweep_summary_includes_policy_prior_columns(tmp_path: Path) -> None:
    regime = stag_hunt_regime()
    row = result_row(
        label="policy_prior_test",
        regime=regime,
        update_rule="neural_policy",
        mixer="none",
        peer_rule="none",
        seed=1,
        initial_action_probability=0.6,
        policy_prior_mode="match_p0",
        policy_prior_action_probability=0.6,
        local_update_rule="counterfactual_advantage",
        neural_peer_mode="well_mixed",
        interaction_mode="well_mixed_resampled",
        decision_mode="argmax",
        action_temperature=0.5,
        decision_calibration_mode="none",
        decision_calibration_strength=4.0,
        decision_threshold=None,
        revision_rate=0.1,
        alpha=0.0,
        selection_strength=1.0,
        policy_temperature=1.0,
        payoff_transform="linear",
        exploration_epsilon=0.0,
        learning_enabled=False,
        reputation_decay=0.9,
        reputation_temperature=1.0,
        reputation_noise=0.0,
        reputation_observation_mode="none",
        mobility_enabled=False,
        run_dir=tmp_path / "run",
        final_action_rate=0.6,
        final_mean_payoff=2.0,
        final_mean_policy_action_probability=0.6,
        final_mean_reputation=0.55,
        final_reputation_dispersion=0.1,
        final_mobility_rate=0.0,
        final_mean_mobility_gain=0.0,
        domain_action_components=1,
        domain_largest_action_cluster_fraction=0.5,
        final_fragmentation_components=1,
    )
    summary_path = tmp_path / "summary.csv"

    write_summary_csv(summary_path, [row])
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = next(reader)

    grouped = build_grouped_summary([row])

    assert "policy_prior_mode" in reader.fieldnames
    assert "policy_prior_action_probability" in reader.fieldnames
    assert "coordination_threshold" in reader.fieldnames
    assert "local_update_rule" in reader.fieldnames
    assert "neural_peer_mode" in reader.fieldnames
    assert "interaction_mode" in reader.fieldnames
    assert "decision_mode" in reader.fieldnames
    assert "action_temperature" in reader.fieldnames
    assert "decision_calibration_mode" in reader.fieldnames
    assert "decision_calibration_strength" in reader.fieldnames
    assert "decision_threshold" in reader.fieldnames
    assert "reputation_decay" in reader.fieldnames
    assert "reputation_temperature" in reader.fieldnames
    assert "reputation_noise" in reader.fieldnames
    assert "reputation_observation_mode" in reader.fieldnames
    assert "mobility_enabled" in reader.fieldnames
    assert "final_mean_reputation" in reader.fieldnames
    assert "final_reputation_dispersion" in reader.fieldnames
    assert "final_mobility_rate" in reader.fieldnames
    assert "final_mean_mobility_gain" in reader.fieldnames
    assert written["policy_prior_mode"] == "match_p0"
    assert float(written["policy_prior_action_probability"]) == pytest.approx(0.6)
    assert float(written["coordination_threshold"]) == pytest.approx(0.0)
    assert written["local_update_rule"] == "counterfactual_advantage"
    assert written["neural_peer_mode"] == "well_mixed"
    assert written["interaction_mode"] == "well_mixed_resampled"
    assert written["decision_mode"] == "argmax"
    assert float(written["action_temperature"]) == pytest.approx(0.5)
    assert written["decision_calibration_mode"] == "none"
    assert float(written["decision_calibration_strength"]) == pytest.approx(4.0)
    assert written["decision_threshold"] == ""
    assert float(written["final_mean_reputation"]) == pytest.approx(0.55)
    assert float(written["final_mobility_rate"]) == pytest.approx(0.0)
    assert "policy_prior_mode" in grouped.columns
    assert "policy_prior_action_probability" in grouped.columns
    assert "coordination_threshold" in grouped.columns
    assert "local_update_rule" in grouped.columns
    assert "neural_peer_mode" in grouped.columns
    assert "interaction_mode" in grouped.columns
    assert "decision_mode" in grouped.columns
    assert "action_temperature" in grouped.columns
    assert "decision_calibration_mode" in grouped.columns
    assert "decision_calibration_strength" in grouped.columns
    assert "decision_threshold" in grouped.columns
    assert "exploration_epsilon" in grouped.columns
    assert "reputation_decay" in grouped.columns
    assert "reputation_temperature" in grouped.columns
    assert "reputation_noise" in grouped.columns
    assert "reputation_observation_mode" in grouped.columns
    assert "mobility_enabled" in grouped.columns
    assert "reputation_mean" in grouped.columns
    assert "mobility_rate_mean" in grouped.columns
    assert "mobility_gain_mean" in grouped.columns
    assert grouped.loc[0, "policy_prior_mode"] == "match_p0"
    assert grouped.loc[0, "policy_prior_action_probability"] == pytest.approx(0.6)
    assert grouped.loc[0, "coordination_threshold"] == pytest.approx(0.0)
    assert grouped.loc[0, "local_update_rule"] == "counterfactual_advantage"
    assert grouped.loc[0, "neural_peer_mode"] == "well_mixed"
    assert grouped.loc[0, "interaction_mode"] == "well_mixed_resampled"
    assert grouped.loc[0, "decision_mode"] == "argmax"
    assert grouped.loc[0, "action_temperature"] == pytest.approx(0.5)
    assert grouped.loc[0, "decision_calibration_mode"] == "none"
    assert grouped.loc[0, "decision_calibration_strength"] == pytest.approx(4.0)
    assert grouped.loc[0, "decision_threshold"] == ""
    assert grouped.loc[0, "reputation_mean"] == pytest.approx(0.55)
    assert grouped.loc[0, "mobility_rate_mean"] == pytest.approx(0.0)


def test_toy2_sweep_non_neural_uses_canonical_neural_only_options() -> None:
    specs = parse_policy_prior_specs(["default", "match_p0", "0.5"])
    local_update_rules = [
        "sampled_policy_gradient",
        "counterfactual_advantage",
    ]
    neural_peer_modes = ["spatial", "well_mixed"]
    interaction_modes = ["spatial", "well_mixed_resampled"]
    temperatures = [0.5, 1.0, 2.0]
    action_temperatures = [1.0, 0.5]
    action_selection_modes = ["sampled", "argmax"]
    decision_calibrations = ["none", "payoff_threshold"]
    calibration_strengths = [1.0, 2.0, 4.0]

    assert effective_policy_prior_specs("neural_policy", specs) == specs
    assert (
        effective_local_update_rules(
            "neural_policy",
            local_update_rules,
        )
        == local_update_rules
    )
    assert effective_policy_temperatures("neural_policy", temperatures) == temperatures
    assert (
        effective_action_temperatures("neural_policy", action_temperatures)
        == action_temperatures
    )
    assert effective_action_temperatures(
        "neural_policy",
        action_temperatures,
        decision_mode="argmax",
    ) == [1.0]
    assert effective_neural_peer_modes("neural_policy", neural_peer_modes) == (
        neural_peer_modes
    )
    assert effective_interaction_modes("neural_policy", interaction_modes) == (
        interaction_modes
    )
    assert (
        effective_action_selection_modes(
            "neural_policy",
            action_selection_modes,
        )
        == action_selection_modes
    )
    assert effective_decision_calibrations(
        "neural_policy",
        decision_calibrations,
        decision_mode="sampled",
    ) == decision_calibrations
    assert effective_decision_calibrations(
        "neural_policy",
        decision_calibrations,
        decision_mode="argmax",
    ) == ["none"]
    assert effective_calibration_strengths(
        "neural_policy",
        calibration_strengths,
        decision_mode="sampled",
        decision_calibration_mode="payoff_threshold",
    ) == calibration_strengths
    assert effective_calibration_strengths(
        "neural_policy",
        calibration_strengths,
        decision_mode="sampled",
        decision_calibration_mode="none",
    ) == [4.0]
    assert effective_interaction_modes("fermi_imitation", interaction_modes) == (
        interaction_modes
    )
    assert effective_interaction_modes("reputation_imitation", interaction_modes) == (
        interaction_modes
    )

    assert effective_policy_prior_specs("fermi_imitation", specs) == [specs[0]]
    assert effective_policy_prior_specs("reputation_imitation", specs) == [specs[0]]
    assert effective_local_update_rules(
        "fermi_imitation",
        local_update_rules,
    ) == ["sampled_policy_gradient"]
    assert effective_local_update_rules(
        "reputation_imitation",
        local_update_rules,
    ) == ["sampled_policy_gradient"]
    assert effective_policy_temperatures("fermi_imitation", temperatures) == [1.0]
    assert effective_policy_temperatures("reputation_imitation", temperatures) == [
        1.0
    ]
    assert effective_action_temperatures("fermi_imitation", action_temperatures) == [
        1.0
    ]
    assert effective_neural_peer_modes("fermi_imitation", neural_peer_modes) == [
        "spatial"
    ]
    assert effective_action_selection_modes(
        "fermi_imitation",
        action_selection_modes,
    ) == ["sampled"]
    assert effective_decision_calibrations(
        "fermi_imitation",
        decision_calibrations,
    ) == ["none"]
    assert effective_calibration_strengths(
        "fermi_imitation",
        calibration_strengths,
    ) == [4.0]
