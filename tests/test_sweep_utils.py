from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from neural_abm.sweep import (
    CoordinationSweepPoint,
    SweepRunArtifacts,
    SweepOutputSpec,
    add_common_sweep_args,
    apply_config_updates,
    build_grouped_summary,
    build_result_row,
    build_sweep_row,
    effective_alphas,
    effective_peer_rules,
    effective_thresholds,
    ensure_supported_coordination,
    extract_attr_values,
    extract_domain_metrics,
    extract_sweep_run_fields,
    iter_coordination_sweep,
    iter_parameter_grid,
    make_sweep_output_helpers,
    non_overwriting_path,
    prepare_sweep_case,
    read_final_aggregate_metrics,
    resolve_attr_path,
    resolve_sweep_output_paths,
    run_case_points,
    run_explicit_sweep_cases,
    run_explicit_sweep_from_args,
    run_point_sweep_from_args,
    run_sweep_cases,
    run_sweep_from_args,
    safe_float,
    set_config_value,
    supports_sweep_coordination,
    toy_display_name,
    write_prepared_sweep_case_config,
    write_sweep_case_config,
    write_sweep_outputs,
    write_summary_csv,
)


def minimal_case_base() -> dict[str, object]:
    return {
        "run": {"name": "base", "seed": 99},
        "simulation": {"epochs": 10},
        "model": {
            "coordination": {
                "mixer": "none",
                "peer_rule": "none",
                "alpha": 0.0,
                "threshold": 0.0,
            }
        },
    }


def test_safe_float_uses_filesystem_safe_tokens() -> None:
    assert safe_float(0.25) == "0p25"
    assert safe_float(-0.1) == "m0p1"
    assert safe_float(1.0) == "1"


def test_sweep_output_spec_groups_output_contract_fields() -> None:
    spec = SweepOutputSpec(
        summary_fields=["label", "value"],
        group_fields=["label"],
        aggregations={"value_mean": ("value", "mean")},
        metric_keys=["value"],
        config_value_paths={"resolved_value": "config.value"},
        result_value_paths={"result_value": "result.value"},
    )

    assert spec.summary_fields == ["label", "value"]
    assert spec.group_fields == ["label"]
    assert spec.aggregations == {"value_mean": ("value", "mean")}
    assert spec.metric_keys == ["value"]
    assert spec.config_value_paths == {"resolved_value": "config.value"}
    assert spec.result_value_paths == {"result_value": "result.value"}


def test_effective_coordination_sweep_values() -> None:
    assert effective_alphas("none", [0.25, 0.5]) == [0.0]
    assert effective_alphas("output_average", [0.25, 0.5]) == [0.25, 0.5]
    assert effective_peer_rules("none", None) == ["none"]
    assert effective_peer_rules("output_average", None) == ["output_similarity"]
    assert effective_peer_rules("none", ["none", "output_similarity"]) == ["none"]
    assert effective_peer_rules(
        "output_average",
        ["none", "output_similarity"],
    ) == ["none", "output_similarity"]
    assert effective_thresholds("none", "none", [0.4, 0.8]) == [0.0]
    assert effective_thresholds(
        "output_average",
        "output_similarity",
        [0.4, 0.8],
    ) == [0.4, 0.8]


def test_add_common_sweep_args_sets_defaults() -> None:
    parser = argparse.ArgumentParser()
    add_common_sweep_args(
        parser,
        base_config=Path("configs/base.yaml"),
        default_label="unit_label",
        toy_name="Toy Unit",
        threshold_argument="--coordination-thresholds",
    )

    args = parser.parse_args([])

    assert args.base_config == Path("configs/base.yaml")
    assert args.label == "unit_label"
    assert args.seeds == [1, 2, 3, 4, 5]
    assert args.epochs is None
    assert args.mixers == ["none", "output_average"]
    assert args.peer_rules is None
    assert args.alphas == [0.0, 0.25, 0.5]
    assert args.coordination_thresholds == [0.0]
    assert args.config_dir is None
    assert args.results_dir == Path("experiments/results")


def test_add_common_sweep_args_accepts_script_specific_defaults() -> None:
    parser = argparse.ArgumentParser()
    add_common_sweep_args(
        parser,
        base_config=Path("configs/base.yaml"),
        default_label="unit_label",
        toy_name="Toy Unit",
        default_seeds=(7, 8),
        default_epochs=50,
        default_alphas=None,
        default_config_dir=Path("configs/generated"),
        peer_rule_choices=("none", "bounded_confidence", "output_similarity"),
        peer_rules_help="custom peer rule help",
        threshold_argument="--coordination-thresholds",
    )

    args = parser.parse_args([])
    parsed = parser.parse_args(["--peer-rules", "bounded_confidence"])

    assert args.seeds == [7, 8]
    assert args.epochs == 50
    assert args.alphas is None
    assert args.config_dir == Path("configs/generated")
    assert args.coordination_thresholds == [0.0]
    assert parsed.peer_rules == ["bounded_confidence"]


def test_iter_coordination_sweep_expands_supported_points() -> None:
    points = iter_coordination_sweep(
        "toy8",
        ["none", "output_average"],
        None,
        [0.0, 0.25],
        [0.0, 0.8],
    )

    assert points == [
        CoordinationSweepPoint(
            mixer="none",
            peer_rule="none",
            alpha=0.0,
            threshold=0.0,
        ),
        CoordinationSweepPoint(
            mixer="output_average",
            peer_rule="output_similarity",
            alpha=0.0,
            threshold=0.0,
        ),
        CoordinationSweepPoint(
            mixer="output_average",
            peer_rule="output_similarity",
            alpha=0.0,
            threshold=0.8,
        ),
        CoordinationSweepPoint(
            mixer="output_average",
            peer_rule="output_similarity",
            alpha=0.25,
            threshold=0.0,
        ),
        CoordinationSweepPoint(
            mixer="output_average",
            peer_rule="output_similarity",
            alpha=0.25,
            threshold=0.8,
        ),
    ]


def test_iter_coordination_sweep_allows_social_default_override() -> None:
    points = iter_coordination_sweep(
        "toy3",
        ["output_average"],
        None,
        [0.25],
        [0.0],
        social_default="bounded_confidence",
    )

    assert points == [
        CoordinationSweepPoint(
            mixer="output_average",
            peer_rule="bounded_confidence",
            alpha=0.25,
            threshold=0.0,
        ),
    ]


def test_iter_parameter_grid_preserves_nested_loop_order() -> None:
    assert iter_parameter_grid(
        {
            "outer": [1, 2],
            "inner": ["a", "b"],
        }
    ) == [
        {"outer": 1, "inner": "a"},
        {"outer": 1, "inner": "b"},
        {"outer": 2, "inner": "a"},
        {"outer": 2, "inner": "b"},
    ]


def test_iter_parameter_grid_handles_empty_and_zero_dimensions() -> None:
    assert iter_parameter_grid({}) == [{}]
    assert iter_parameter_grid({"value": []}) == []


def test_supported_coordination_guard() -> None:
    assert supports_sweep_coordination("toy8", "output_average", "output_similarity")
    ensure_supported_coordination("toy8", "output_average", "output_similarity")

    with pytest.raises(
        ValueError,
        match="Unsupported Async Event ABM coordination: output_average/bounded_confidence",
    ):
        ensure_supported_coordination(
            "toy8",
            "output_average",
            "bounded_confidence",
        )

    assert toy_display_name("toy10") == "Market Ecology Network"
    assert toy_display_name("custom") == "custom"


def test_prepare_sweep_case_sets_common_case_fields() -> None:
    base = minimal_case_base()

    prepared = prepare_sweep_case(
        base=base,
        toy="toy8",
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        seed=3,
        epochs=7,
        coordination_threshold=0.8,
    )

    assert prepared.effective_alpha == pytest.approx(0.25)
    assert prepared.effective_threshold == pytest.approx(0.8)
    assert prepared.case["run"]["seed"] == 3
    assert prepared.case["simulation"]["epochs"] == 7
    assert prepared.case["model"]["coordination"] == {
        "mixer": "output_average",
        "peer_rule": "output_similarity",
        "alpha": 0.25,
        "threshold": 0.8,
    }
    assert base["run"] == {"name": "base", "seed": 99}
    assert base["simulation"] == {"epochs": 10}


def test_prepare_sweep_case_zeroes_inactive_coordination_values() -> None:
    prepared = prepare_sweep_case(
        base=minimal_case_base(),
        toy="toy8",
        mixer="none",
        peer_rule="none",
        alpha=0.5,
        seed=2,
        epochs=None,
        coordination_threshold=0.8,
    )

    assert prepared.effective_alpha == pytest.approx(0.0)
    assert prepared.effective_threshold == pytest.approx(0.0)
    assert prepared.case["simulation"]["epochs"] == 10
    assert prepared.case["model"]["coordination"]["alpha"] == pytest.approx(0.0)
    assert prepared.case["model"]["coordination"]["threshold"] == pytest.approx(0.0)


def test_write_sweep_case_config_uses_run_name_and_seed(tmp_path: Path) -> None:
    case = minimal_case_base()
    case["run"] = {"name": "unit_case", "seed": 4}

    config_path = write_sweep_case_config(case, tmp_path / "configs")

    assert config_path == tmp_path / "configs" / "unit_case_seed04.yaml"
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == case


def test_write_prepared_sweep_case_config_applies_name_updates_and_mutation(
    tmp_path: Path,
) -> None:
    prepared = prepare_sweep_case(
        base=minimal_case_base(),
        toy="toy8",
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        seed=3,
        epochs=5,
        coordination_threshold=0.8,
    )

    def mutate_case(case: dict[str, object]) -> None:
        case["mutated"] = True

    config_path = write_prepared_sweep_case_config(
        prepared,
        run_name="prepared_unit",
        config_dir=tmp_path / "configs",
        updates={"model.coordination.threshold": 0.9},
        mutate_case=mutate_case,
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config_path == tmp_path / "configs" / "prepared_unit_seed03.yaml"
    assert raw["run"]["name"] == "prepared_unit"
    assert raw["run"]["seed"] == 3
    assert raw["simulation"]["epochs"] == 5
    assert raw["model"]["coordination"]["threshold"] == pytest.approx(0.9)
    assert raw["mutated"] is True


def test_apply_config_updates_sets_nested_mapping_and_list_values() -> None:
    config = {
        "domain": {"graph": {"k": 4}},
        "model": {
            "agents": {
                "groups": [
                    {"fraction": 0.5},
                    {"learning_rate": 0.1},
                ]
            }
        },
    }

    apply_config_updates(
        config,
        {
            "domain.graph.k": 6,
            "model.agents.groups.0.fraction": 0.4,
            "model.agents.groups.1.learning_rate": 0.2,
        },
    )

    assert config["domain"]["graph"]["k"] == 6
    assert config["model"]["agents"]["groups"][0]["fraction"] == pytest.approx(0.4)
    assert config["model"]["agents"]["groups"][1]["learning_rate"] == pytest.approx(
        0.2
    )


def test_set_config_value_rejects_non_integer_list_indices() -> None:
    config = {"items": [{"value": 1}]}

    with pytest.raises(
        TypeError,
        match="must be an integer list index",
    ):
        set_config_value(config, "items.first.value", 2)


def test_build_result_row_applies_common_aliases_and_string_fields(
    tmp_path: Path,
) -> None:
    row = build_result_row(
        [
            "label",
            "coordination_mixer",
            "coordination_peer_rule",
            "run_dir",
            "domain_value",
        ],
        {
            "label": "case",
            "mixer": "output_average",
            "peer_rule": "output_similarity",
            "run_dir": tmp_path / "run",
            "domain_value": 1.25,
        },
    )

    assert row == {
        "label": "case",
        "coordination_mixer": "output_average",
        "coordination_peer_rule": "output_similarity",
        "run_dir": str(tmp_path / "run"),
        "domain_value": 1.25,
    }


def test_build_result_row_supports_custom_aliases() -> None:
    row = build_result_row(
        ["label", "initial_distribution"],
        {"label": "case", "distribution_label": "biased"},
        aliases={"initial_distribution": "distribution_label"},
    )

    assert row == {
        "label": "case",
        "initial_distribution": "biased",
    }


def test_read_final_aggregate_metrics_returns_final_numeric_values(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "aggregate_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "mean_reputation",
                "reputation_dispersion",
                "mobility_rate",
                "mean_mobility_gain",
                "ignored",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                "mean_reputation": 0.1,
                "reputation_dispersion": 0.2,
                "mobility_rate": "",
                "mean_mobility_gain": 0.3,
                "ignored": 9.0,
            }
        )
        writer.writerow(
            {
                "epoch": 1,
                "mean_reputation": 0.4,
                "reputation_dispersion": 0.5,
                "mobility_rate": 0.6,
                "mean_mobility_gain": "",
                "ignored": 10.0,
            }
        )

    assert read_final_aggregate_metrics(run_dir) == {
        "mean_reputation": pytest.approx(0.4),
        "reputation_dispersion": pytest.approx(0.5),
        "mobility_rate": pytest.approx(0.6),
    }
    assert read_final_aggregate_metrics(run_dir, ["ignored"]) == {
        "ignored": pytest.approx(10.0),
    }


def test_read_final_aggregate_metrics_handles_missing_or_empty_files(
    tmp_path: Path,
) -> None:
    assert read_final_aggregate_metrics(tmp_path / "missing") == {}

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "aggregate_metrics.csv").write_text(
        "mean_reputation\n",
        encoding="utf-8",
    )

    assert read_final_aggregate_metrics(run_dir) == {}


def test_write_summary_csv_preserves_fields_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "summary.csv"
    write_summary_csv(
        path,
        [{"label": "case", "value": 1.5, "extra": "ignored"}],
        ["label", "missing", "value"],
    )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    assert reader.fieldnames == ["label", "missing", "value"]
    assert row == {"label": "case", "missing": "", "value": "1.5"}


def test_non_overwriting_path_selects_next_available_suffix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.csv"
    path.write_text("existing", encoding="utf-8")
    (tmp_path / "summary_01.csv").write_text("existing", encoding="utf-8")

    assert non_overwriting_path(path) == tmp_path / "summary_02.csv"
    assert non_overwriting_path(tmp_path / "unused.csv") == tmp_path / "unused.csv"


def test_resolve_sweep_output_paths_builds_optional_markdown_and_avoids_collisions(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    plain = resolve_sweep_output_paths(
        results_dir,
        "unit",
        grouped_markdown=True,
    )

    assert plain.summary_path == results_dir / "unit_summary.csv"
    assert plain.grouped_summary_path == results_dir / "unit_grouped_summary.csv"
    assert plain.grouped_markdown_path == results_dir / "unit_grouped_summary.md"

    results_dir.mkdir()
    plain.summary_path.write_text("existing", encoding="utf-8")
    plain.grouped_summary_path.write_text("existing", encoding="utf-8")
    assert plain.grouped_markdown_path is not None
    plain.grouped_markdown_path.write_text("existing", encoding="utf-8")

    resolved = resolve_sweep_output_paths(
        results_dir,
        "unit",
        grouped_markdown=True,
        avoid_overwrite=True,
    )

    assert resolved.summary_path == results_dir / "unit_summary_01.csv"
    assert resolved.grouped_summary_path == (
        results_dir / "unit_grouped_summary_01.csv"
    )
    assert resolved.grouped_markdown_path == (
        results_dir / "unit_grouped_summary_01.md"
    )


def test_build_grouped_summary_uses_shared_pandas_shape() -> None:
    grouped = build_grouped_summary(
        [
            {"label": "case", "alpha": 0.25, "seed": 1, "value": 1.0},
            {"label": "case", "alpha": 0.25, "seed": 2, "value": 3.0},
        ],
        ["label", "alpha"],
        {
            "seeds": ("seed", "nunique"),
            "value_mean": ("value", "mean"),
            "value_std": ("value", "std"),
        },
    )

    assert list(grouped.columns) == [
        "label",
        "alpha",
        "seeds",
        "value_mean",
        "value_std",
    ]
    assert grouped["label"].iloc[0] == "case"
    assert grouped["alpha"].iloc[0] == pytest.approx(0.25)
    assert grouped["seeds"].iloc[0] == 2
    assert grouped["value_mean"].iloc[0] == pytest.approx(2.0)
    assert grouped["value_std"].iloc[0] == pytest.approx(2.0**0.5)


def test_build_grouped_summary_handles_empty_rows() -> None:
    grouped = build_grouped_summary(
        [],
        ["label"],
        {"seeds": ("seed", "nunique")},
    )

    assert grouped.empty


def test_write_sweep_outputs_writes_summary_and_grouped_csvs(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.csv"
    grouped_path = tmp_path / "grouped" / "summary.csv"

    write_sweep_outputs(
        summary_path=summary_path,
        grouped_summary_path=grouped_path,
        rows=[
            {"label": "case", "alpha": 0.25, "seed": 1, "value": 1.0},
            {"label": "case", "alpha": 0.25, "seed": 2, "value": 3.0},
        ],
        summary_fields=["label", "alpha", "seed", "value"],
        group_fields=["label", "alpha"],
        aggregations={
            "seeds": ("seed", "nunique"),
            "value_mean": ("value", "mean"),
        },
    )

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_reader = csv.DictReader(handle)
        summary_rows = list(summary_reader)
    with grouped_path.open("r", encoding="utf-8", newline="") as handle:
        grouped_reader = csv.DictReader(handle)
        grouped_rows = list(grouped_reader)

    assert summary_reader.fieldnames == ["label", "alpha", "seed", "value"]
    assert len(summary_rows) == 2
    assert grouped_reader.fieldnames == ["label", "alpha", "seeds", "value_mean"]
    assert grouped_rows == [
        {
            "label": "case",
            "alpha": "0.25",
            "seeds": "2",
            "value_mean": "2.0",
        }
    ]


def test_make_sweep_output_helpers_preserves_script_compatibility(
    tmp_path: Path,
) -> None:
    spec = SweepOutputSpec(
        summary_fields=["label", "seed", "value"],
        group_fields=["label"],
        aggregations={"value_mean": ("value", "mean")},
        metric_keys=[],
    )
    rows = [
        {"label": "case", "seed": 1, "value": 1.0, "extra": "ignored"},
        {"label": "case", "seed": 2, "value": 3.0, "extra": "ignored"},
    ]
    write_summary, build_grouped = make_sweep_output_helpers(spec)
    path = tmp_path / "summary.csv"

    write_summary(path, rows)
    grouped = build_grouped(rows)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)

    assert reader.fieldnames == ["label", "seed", "value"]
    assert written[0] == {"label": "case", "seed": "1", "value": "1.0"}
    assert list(grouped.columns) == ["label", "value_mean"]
    assert grouped.loc[0, "value_mean"] == pytest.approx(2.0)


def test_extract_domain_metrics_selects_requested_keys() -> None:
    result = SimpleNamespace(
        domain_metrics={
            "domain_first": 1.0,
            "domain_second": 2,
            "domain_unused": "ignored",
        }
    )

    assert extract_domain_metrics(
        result,
        ["domain_second", "domain_first"],
    ) == {
        "domain_second": 2,
        "domain_first": 1.0,
    }


def test_extract_domain_metrics_raises_for_missing_keys() -> None:
    result = SimpleNamespace(domain_metrics={"domain_present": 1.0})

    with pytest.raises(KeyError, match="domain_missing"):
        extract_domain_metrics(result, ["domain_missing"])


def test_extract_sweep_run_fields_returns_common_row_kwargs(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        coordination=SimpleNamespace(
            mixer="output_average",
            peer_rule="output_similarity",
            alpha=0.25,
            threshold=0.8,
        ),
        run=SimpleNamespace(seed=3),
        simulation=SimpleNamespace(epochs=5),
    )
    result = SimpleNamespace(
        run_dir=tmp_path / "run",
        final_fragmentation_components=2,
    )

    assert extract_sweep_run_fields(config, result) == {
        "mixer": "output_average",
        "peer_rule": "output_similarity",
        "alpha": 0.25,
        "coordination_threshold": 0.8,
        "seed": 3,
        "epochs": 5,
        "run_dir": tmp_path / "run",
        "final_fragmentation_components": 2,
    }


def test_resolve_attr_path_reads_nested_objects_and_mappings() -> None:
    source = SimpleNamespace(
        environment=SimpleNamespace(rate=0.2),
        nested={"value": SimpleNamespace(name="ok")},
    )

    assert resolve_attr_path(source, "environment.rate") == pytest.approx(0.2)
    assert resolve_attr_path(source, "nested.value.name") == "ok"


def test_extract_attr_values_maps_row_fields_to_paths() -> None:
    source = SimpleNamespace(
        game=SimpleNamespace(win_payoff=1.5, loss_payoff=-1.0),
        policy=SimpleNamespace(exploration_std=0.02),
    )

    assert extract_attr_values(
        source,
        {
            "win_payoff": "game.win_payoff",
            "loss_payoff": "game.loss_payoff",
            "exploration_std": "policy.exploration_std",
        },
    ) == {
        "win_payoff": 1.5,
        "loss_payoff": -1.0,
        "exploration_std": 0.02,
    }


def test_build_sweep_row_combines_base_config_run_and_metric_fields(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        coordination=SimpleNamespace(
            mixer="output_average",
            peer_rule="output_similarity",
            alpha=0.25,
            threshold=0.8,
        ),
        run=SimpleNamespace(seed=3),
        simulation=SimpleNamespace(epochs=5),
        policy=SimpleNamespace(exploration_std=0.02),
    )
    result = SimpleNamespace(
        run_dir=tmp_path / "run",
        final_fragmentation_components=2,
        final_value=5.0,
        domain_metrics={"domain_value": 4.0},
    )

    row = build_sweep_row(
        lambda **kwargs: dict(kwargs),
        base_fields={"label": "case"},
        config=config,
        result=result,
        metric_keys=["domain_value"],
        config_value_paths={"exploration_std": "policy.exploration_std"},
        result_value_paths={"final_value": "final_value"},
    )

    assert row == {
        "label": "case",
        "exploration_std": 0.02,
        "mixer": "output_average",
        "peer_rule": "output_similarity",
        "alpha": 0.25,
        "coordination_threshold": 0.8,
        "seed": 3,
        "epochs": 5,
        "run_dir": tmp_path / "run",
        "final_fragmentation_components": 2,
        "domain_value": 4.0,
        "final_value": 5.0,
    }


def test_run_case_points_collects_rows_and_progress_in_order() -> None:
    messages: list[str] = []

    rows = run_case_points(
        points=[1, 2, 3],
        run_point=lambda point: ({"point": point}, f"finished {point}"),
        progress=messages.append,
    )

    assert rows == [{"point": 1}, {"point": 2}, {"point": 3}]
    assert messages == ["finished 1", "finished 2", "finished 3"]


def test_run_point_sweep_from_args_loads_base_and_writes_outputs(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump({"base": True}), encoding="utf-8")
    spec = SweepOutputSpec(
        summary_fields=["label", "case", "seed", "value"],
        group_fields=["case"],
        aggregations={"value_mean": ("value", "mean")},
        metric_keys=[],
    )
    args = SimpleNamespace(
        base_config=base_path,
        label="point",
        config_dir=tmp_path / "configs",
        results_dir=tmp_path / "results",
    )
    args.results_dir.mkdir()
    (args.results_dir / "point_summary.csv").write_text(
        "existing",
        encoding="utf-8",
    )
    (args.results_dir / "point_grouped_summary.csv").write_text(
        "existing",
        encoding="utf-8",
    )
    (args.results_dir / "point_grouped_summary.md").write_text(
        "existing",
        encoding="utf-8",
    )

    def build_rows(
        base: dict[str, object],
        parsed_args: SimpleNamespace,
        label: str,
        config_dir: Path,
        progress: object,
    ) -> list[dict[str, object]]:
        assert base == {"base": True}
        assert parsed_args is args
        assert label == "point"
        assert config_dir == tmp_path / "configs"
        assert progress == messages.append
        messages.append("rows built")
        return [{"label": label, "case": "a", "seed": 1, "value": 2.0}]

    markdown_calls: list[tuple[Path, str, list[str]]] = []

    def write_markdown(path: Path, label: str, grouped: object) -> None:
        markdown_calls.append((path, label, list(grouped["case"])))
        path.write_text(f"# {label}", encoding="utf-8")

    messages: list[str] = []
    artifacts = run_point_sweep_from_args(
        args=args,
        output_spec=spec,
        rows_builder=build_rows,
        grouped_markdown_writer=write_markdown,
        avoid_overwrite=True,
        progress=messages.append,
    )

    assert artifacts.rows == [{"label": "point", "case": "a", "seed": 1, "value": 2.0}]
    assert artifacts.config_dir == tmp_path / "configs"
    assert artifacts.summary_path == args.results_dir / "point_summary_01.csv"
    assert artifacts.grouped_summary_path == (
        args.results_dir / "point_grouped_summary_01.csv"
    )
    assert artifacts.grouped_markdown_path == (
        args.results_dir / "point_grouped_summary_01.md"
    )
    assert artifacts.summary_path.exists()
    assert artifacts.grouped_summary_path.exists()
    assert artifacts.grouped_markdown_path.exists()
    assert markdown_calls == [
        (artifacts.grouped_markdown_path, "point", ["a"]),
    ]
    assert messages[-3:] == [
        f"summary={artifacts.summary_path}",
        f"grouped_summary={artifacts.grouped_summary_path}",
        f"grouped_markdown={artifacts.grouped_markdown_path}",
    ]


def test_run_point_sweep_from_args_can_write_nabm_metadata_markdown(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump({"base": True}), encoding="utf-8")
    spec = SweepOutputSpec(
        summary_fields=["label", "case", "seed", "value"],
        group_fields=["case"],
        aggregations={"value_mean": ("value", "mean")},
        metric_keys=[],
    )
    args = SimpleNamespace(
        base_config=base_path,
        label="point",
        config_dir=tmp_path / "configs",
        results_dir=tmp_path / "results",
    )

    def build_rows(
        base: dict[str, object],
        parsed_args: SimpleNamespace,
        label: str,
        config_dir: Path,
        progress: object,
    ) -> list[dict[str, object]]:
        del parsed_args, config_dir, progress
        assert base == {"base": True}
        return [{"label": label, "case": "a", "seed": 1, "value": 2.0}]

    def write_markdown(path: Path, label: str, grouped: object) -> None:
        del grouped
        path.write_text(f"# {label}", encoding="utf-8")

    artifacts = run_point_sweep_from_args(
        args=args,
        output_spec=spec,
        rows_builder=build_rows,
        grouped_markdown_writer=write_markdown,
        toy="toy2",
        progress=None,
    )

    assert artifacts.rows[0]["nabm_status"] == "full"
    assert "rd_well_mixed" in str(artifacts.rows[0]["reference_policies"])
    with artifacts.summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["nabm_status"] == "full"
    assert artifacts.grouped_markdown_path is not None
    markdown = artifacts.grouped_markdown_path.read_text(encoding="utf-8")
    assert "## NABM Claim Metadata" in markdown
    assert "rd_well_mixed" in markdown


def test_run_sweep_cases_executes_domain_coordination_seed_grid(
    tmp_path: Path,
) -> None:
    write_calls: list[dict[str, object]] = []

    def write_case_config(**kwargs: object) -> Path:
        write_calls.append(dict(kwargs))
        return tmp_path / f"case_{kwargs['domain_param']}_{kwargs['seed']}.yaml"

    def load_config(config_path: Path) -> SimpleNamespace:
        call = write_calls[-1]
        return SimpleNamespace(
            coordination=SimpleNamespace(
                mixer=call["mixer"],
                peer_rule=call["peer_rule"],
                alpha=call["alpha"],
                threshold=call["coordination_threshold"],
            ),
            run=SimpleNamespace(
                name=config_path.stem,
                seed=call["seed"],
            ),
            simulation=SimpleNamespace(epochs=call["epochs"]),
            resolved=SimpleNamespace(domain_param=call["domain_param"]),
        )

    def run_case(*, config: SimpleNamespace, config_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            run_dir=tmp_path / "runs" / config.run.name,
            final_fragmentation_components=1,
            domain_metrics={"domain_metric": config.run.seed + 0.5},
        )

    messages: list[str] = []
    rows = run_sweep_cases(
        label="unit",
        base={"base": True},
        config_dir=tmp_path / "configs",
        domain_points=[
            {"domain_param": "a"},
            {"domain_param": "b"},
        ],
        coordination_points=[
            CoordinationSweepPoint(
                mixer="output_average",
                peer_rule="output_similarity",
                alpha=0.25,
                threshold=0.8,
            )
        ],
        seeds=[1, 2],
        epochs=3,
        write_case_config=write_case_config,
        load_config=load_config,
        run_case=run_case,
        row_builder=lambda **kwargs: dict(kwargs),
        metric_keys=["domain_metric"],
        config_value_paths={"resolved_domain_param": "resolved.domain_param"},
        progress=messages.append,
    )

    assert [
        (call["domain_param"], call["seed"])
        for call in write_calls
    ] == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]
    assert rows[0]["label"] == "unit"
    assert rows[0]["domain_param"] == "a"
    assert rows[0]["mixer"] == "output_average"
    assert rows[0]["peer_rule"] == "output_similarity"
    assert rows[0]["alpha"] == pytest.approx(0.25)
    assert rows[0]["coordination_threshold"] == pytest.approx(0.8)
    assert rows[0]["seed"] == 1
    assert rows[0]["epochs"] == 3
    assert rows[0]["resolved_domain_param"] == "a"
    assert rows[0]["domain_metric"] == pytest.approx(1.5)
    assert messages[0].startswith("finished case_a_1 seed=1 run_dir=")


def test_run_sweep_cases_can_skip_domain_coordination_pairs(
    tmp_path: Path,
) -> None:
    write_calls: list[dict[str, object]] = []

    def write_case_config(**kwargs: object) -> Path:
        write_calls.append(dict(kwargs))
        return tmp_path / f"case_{kwargs['domain_param']}_{kwargs['seed']}.yaml"

    def load_config(config_path: Path) -> SimpleNamespace:
        call = write_calls[-1]
        return SimpleNamespace(
            coordination=SimpleNamespace(
                mixer=call["mixer"],
                peer_rule=call["peer_rule"],
                alpha=call["alpha"],
                threshold=call["coordination_threshold"],
            ),
            run=SimpleNamespace(name=config_path.stem, seed=call["seed"]),
            simulation=SimpleNamespace(epochs=call["epochs"]),
        )

    def run_case(*, config: SimpleNamespace, config_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            run_dir=tmp_path / "runs" / config.run.name,
            final_fragmentation_components=1,
            domain_metrics={"domain_metric": 1.0},
        )

    rows = run_sweep_cases(
        label="unit",
        base={},
        config_dir=tmp_path / "configs",
        domain_points=[
            {"domain_param": "skip"},
            {"domain_param": "run"},
        ],
        coordination_points=[
            CoordinationSweepPoint(
                mixer="output_average",
                peer_rule="output_similarity",
                alpha=0.25,
                threshold=0.8,
            )
        ],
        seeds=[1],
        epochs=3,
        write_case_config=write_case_config,
        load_config=load_config,
        run_case=run_case,
        row_builder=lambda **kwargs: dict(kwargs),
        metric_keys=["domain_metric"],
        should_run=lambda domain, _coordination: domain["domain_param"] == "run",
        progress=None,
    )

    assert [call["domain_param"] for call in write_calls] == ["run"]
    assert rows[0]["domain_param"] == "run"


def test_run_explicit_sweep_cases_executes_paired_points(
    tmp_path: Path,
) -> None:
    write_calls: list[dict[str, object]] = []

    def write_case_config(**kwargs: object) -> Path:
        write_calls.append(dict(kwargs))
        return tmp_path / f"case_{kwargs['case']}_{kwargs['seed']}.yaml"

    def load_config(config_path: Path) -> SimpleNamespace:
        call = write_calls[-1]
        return SimpleNamespace(
            coordination=SimpleNamespace(
                mixer=call["mixer"],
                peer_rule=call["peer_rule"],
                alpha=call["alpha"],
                threshold=call["coordination_threshold"],
            ),
            run=SimpleNamespace(name=config_path.stem, seed=call["seed"]),
            simulation=SimpleNamespace(epochs=call["epochs"]),
            agents=SimpleNamespace(init_mode=call["init_mode"]),
        )

    def run_case(*, config: SimpleNamespace, config_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            run_dir=tmp_path / "runs" / config.run.name,
            final_fragmentation_components=1,
            domain_metrics={"domain_metric": config.run.seed + 1.0},
        )

    rows = run_explicit_sweep_cases(
        label="unit",
        base={},
        config_dir=tmp_path / "configs",
        sweep_points=[
            {
                "case": "paired",
                "mixer": "latent_average",
                "peer_rule": "latent_similarity",
                "init_mode": "same_init",
                "alpha": 0.25,
                "coordination_threshold": 0.8,
            }
        ],
        seeds=[1, 2],
        epochs=3,
        write_case_config=write_case_config,
        load_config=load_config,
        run_case=run_case,
        row_builder=lambda **kwargs: dict(kwargs),
        metric_keys=["domain_metric"],
        config_value_paths={"model_init_mode": "agents.init_mode"},
        progress=None,
    )

    assert [(call["case"], call["seed"]) for call in write_calls] == [
        ("paired", 1),
        ("paired", 2),
    ]
    assert rows[0]["case"] == "paired"
    assert rows[0]["mixer"] == "latent_average"
    assert rows[0]["model_init_mode"] == "same_init"
    assert rows[0]["domain_metric"] == pytest.approx(2.0)


def test_run_explicit_sweep_from_args_writes_outputs_and_markdown_hook(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump({"base": True}), encoding="utf-8")
    spec = SweepOutputSpec(
        summary_fields=[
            "label",
            "case",
            "model_init_mode",
            "coordination_mixer",
            "coordination_peer_rule",
            "alpha",
            "coordination_threshold",
            "seed",
            "epochs",
            "run_dir",
            "domain_metric",
            "final_fragmentation_components",
        ],
        group_fields=["case"],
        aggregations={"domain_metric_mean": ("domain_metric", "mean")},
        metric_keys=["domain_metric"],
        config_value_paths={"model_init_mode": "agents.init_mode"},
    )
    args = SimpleNamespace(
        base_config=base_path,
        label="explicit",
        config_dir=tmp_path / "configs",
        results_dir=tmp_path / "results",
        seeds=[1],
        epochs=2,
    )

    def build_sweep_points(
        base: dict[str, object],
        parsed_args: SimpleNamespace,
    ) -> list[dict[str, object]]:
        assert base == {"base": True}
        assert parsed_args.label == "explicit"
        return [
            {
                "case": "paired",
                "mixer": "latent_average",
                "peer_rule": "latent_similarity",
                "init_mode": "same_init",
                "alpha": 0.25,
                "coordination_threshold": 0.8,
            }
        ]

    write_calls: list[dict[str, object]] = []

    def write_case_config(**kwargs: object) -> Path:
        write_calls.append(dict(kwargs))
        return tmp_path / f"case_{kwargs['case']}_{kwargs['seed']}.yaml"

    def load_config(config_path: Path) -> SimpleNamespace:
        call = write_calls[-1]
        return SimpleNamespace(
            coordination=SimpleNamespace(
                mixer=call["mixer"],
                peer_rule=call["peer_rule"],
                alpha=call["alpha"],
                threshold=call["coordination_threshold"],
            ),
            run=SimpleNamespace(name=config_path.stem, seed=call["seed"]),
            simulation=SimpleNamespace(epochs=call["epochs"]),
            agents=SimpleNamespace(init_mode=call["init_mode"]),
        )

    def run_case(*, config: SimpleNamespace, config_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            run_dir=tmp_path / "runs" / config.run.name,
            final_fragmentation_components=1,
            domain_metrics={"domain_metric": 2.5},
        )

    markdown_calls: list[tuple[Path, str, list[str]]] = []

    def write_markdown(path: Path, label: str, grouped: object) -> None:
        markdown_calls.append((path, label, list(grouped["case"])))
        path.write_text(f"# {label}", encoding="utf-8")

    messages: list[str] = []
    artifacts = run_explicit_sweep_from_args(
        args=args,
        output_spec=spec,
        sweep_points_builder=build_sweep_points,
        write_case_config=write_case_config,
        load_config=load_config,
        run_case=run_case,
        row_builder=lambda **kwargs: build_result_row(spec.summary_fields, kwargs),
        grouped_markdown_writer=write_markdown,
        progress=messages.append,
    )

    grouped_markdown_path = tmp_path / "results" / "explicit_grouped_summary.md"
    assert artifacts.config_dir == tmp_path / "configs"
    assert artifacts.summary_path == tmp_path / "results" / "explicit_summary.csv"
    assert artifacts.grouped_summary_path == (
        tmp_path / "results" / "explicit_grouped_summary.csv"
    )
    assert artifacts.grouped_markdown_path == grouped_markdown_path
    assert write_calls[0]["config_dir"] == tmp_path / "configs"
    assert artifacts.rows[0]["model_init_mode"] == "same_init"
    assert artifacts.summary_path.exists()
    assert artifacts.grouped_summary_path.exists()
    assert grouped_markdown_path.exists()
    assert markdown_calls == [(grouped_markdown_path, "explicit", ["paired"])]
    assert messages[-3:] == [
        f"summary={artifacts.summary_path}",
        f"grouped_summary={artifacts.grouped_summary_path}",
        f"grouped_markdown={grouped_markdown_path}",
    ]


def test_run_sweep_from_args_loads_base_runs_cases_and_writes_outputs(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump({"base": True}), encoding="utf-8")
    spec = SweepOutputSpec(
        summary_fields=[
            "label",
            "domain_param",
            "coordination_mixer",
            "coordination_peer_rule",
            "alpha",
            "coordination_threshold",
            "seed",
            "epochs",
            "run_dir",
            "domain_metric",
            "final_fragmentation_components",
        ],
        group_fields=["label", "domain_param"],
        aggregations={"domain_metric_mean": ("domain_metric", "mean")},
        metric_keys=["domain_metric"],
    )
    args = SimpleNamespace(
        base_config=base_path,
        label="unit",
        config_dir=None,
        results_dir=tmp_path / "results",
        mixers=["output_average"],
        peer_rules=None,
        alphas=[0.25],
        seeds=[1],
        epochs=3,
    )

    def build_domain_points(
        base: dict[str, object],
        parsed_args: SimpleNamespace,
    ) -> list[dict[str, object]]:
        assert base == {"base": True}
        assert parsed_args.label == "unit"
        return [{"domain_param": "a"}]

    def write_case_config(**kwargs: object) -> Path:
        return tmp_path / f"case_{kwargs['domain_param']}_{kwargs['seed']}.yaml"

    def load_config(config_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            coordination=SimpleNamespace(
                mixer="output_average",
                peer_rule="output_similarity",
                alpha=0.25,
                threshold=0.8,
            ),
            run=SimpleNamespace(name=config_path.stem, seed=1),
            simulation=SimpleNamespace(epochs=3),
        )

    def run_case(*, config: SimpleNamespace, config_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            run_dir=tmp_path / "runs" / config.run.name,
            final_fragmentation_components=1,
            domain_metrics={"domain_metric": 2.5},
        )

    messages: list[str] = []
    artifacts = run_sweep_from_args(
        args=args,
        toy="toy8",
        output_spec=spec,
        domain_points_builder=build_domain_points,
        thresholds=[0.8],
        write_case_config=write_case_config,
        load_config=load_config,
        run_case=run_case,
        row_builder=lambda **kwargs: build_result_row(spec.summary_fields, kwargs),
        progress=messages.append,
    )

    assert isinstance(artifacts, SweepRunArtifacts)
    assert artifacts.config_dir == Path("experiments/configs/generated/unit")
    assert artifacts.summary_path == tmp_path / "results" / "unit_summary.csv"
    assert artifacts.grouped_summary_path == (
        tmp_path / "results" / "unit_grouped_summary.csv"
    )
    assert artifacts.grouped_markdown_path is None
    assert artifacts.rows[0]["domain_metric"] == pytest.approx(2.5)
    assert artifacts.rows[0]["nabm_status"] == "compatible"
    assert artifacts.rows[0]["reference_policies"] == "none"
    assert artifacts.summary_path.exists()
    assert artifacts.grouped_summary_path.exists()
    with artifacts.summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_reader = csv.DictReader(handle)
        assert summary_reader.fieldnames is not None
        assert "nabm_status" in summary_reader.fieldnames
        assert "neural_role" in summary_reader.fieldnames
        summary_rows = list(summary_reader)
    assert summary_rows[0]["nabm_status"] == "compatible"
    with artifacts.grouped_summary_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        grouped_rows = list(csv.DictReader(handle))
    assert grouped_rows[0]["nabm_status"] == "compatible"
    assert messages[-2:] == [
        f"summary={artifacts.summary_path}",
        f"grouped_summary={artifacts.grouped_summary_path}",
    ]
