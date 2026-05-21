from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
import yaml

from binary_config_helpers import toy3_config
from neural_abm.config import load_toy3_config
from neural_abm.toy_opinion import (
    aggregate_metrics,
    deffuant_pair_update,
    edge_disagreements,
    hk_update_opinions,
    rewire_disagreeing_edges,
    run_toy3,
    select_neural_peer_ids,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_toy3_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_toy3_sweep", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_toy3_sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_toy3_sweep
SPEC.loader.exec_module(run_toy3_sweep)

build_grouped_summary = run_toy3_sweep.build_grouped_summary
result_row = run_toy3_sweep.result_row
write_case_config = run_toy3_sweep.write_case_config
write_summary_csv = run_toy3_sweep.write_summary_csv


def tiny_config_dict(
    tmp_path: Path,
    update_rule: str = "hk",
    mixer: str = "none",
    peer_rule: str | None = None,
    rewiring_enabled: bool = False,
    rewiring_rate: float = 0.0,
) -> dict:
    resolved_peer_rule = (
        peer_rule
        if peer_rule is not None
        else "bounded_confidence"
        if mixer == "output_average"
        else "none"
    )
    return toy3_config(
        {
            "run": {
                "name": f"tiny_toy3_{update_rule}_{mixer}",
                "seed": 7,
                "output_dir": str(tmp_path / "runs"),
            },
            "simulation": {
                "epochs": 2,
                "sync_mode": "synchronous",
                "device": "cpu",
            },
            "environment": {
                "opinion_min": -1.0,
                "opinion_max": 1.0,
                "initial_opinion_mode": "two_clusters",
                "cluster_centers": [-0.4, 0.4],
                "cluster_std": 0.02,
            },
            "dynamics": {
                "update_rule": update_rule,
                "confidence_threshold": 0.35,
                "influence_rate": 1.0,
                "deffuant_mu": 0.5,
                "neural_delta_scale": 0.25,
                "neural_learning_rate": 0.01,
            },
            "agents": {
                "count": 8,
                "init_mode": "same_init",
                "model": {
                    "input_dim": 6,
                    "hidden_dim": 8,
                    "output_dim": 1,
                    "activation": "relu",
                },
                "optimizer": {
                    "name": "adam",
                    "learning_rate": 0.01,
                },
            },
            "graph": {
                "type": "watts_strogatz",
                "k": 2,
                "rewire_probability": 0.0,
            },
            "social": {
                "mixer": mixer,
                "peer_rule": resolved_peer_rule,
                "alpha": 0.25 if mixer == "output_average" else 0.0,
                "threshold": 0.8 if resolved_peer_rule == "output_similarity" else 0.0,
            },
            "rewiring": {
                "enabled": rewiring_enabled,
                "threshold": 0.7,
                "rate": rewiring_rate,
                "candidate_pool_size": 8,
            },
            "logging": {
                "micro_state": True,
                "interval": 1,
                "aggregate_metrics": True,
                "probe_predictions": False,
                "probe_prediction_interval": 1,
            },
        }
    )


def write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "toy3.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_toy3_baseline_config_loads() -> None:
    config = load_toy3_config(
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "configs"
        / "toy3_opinion_rewiring_baseline.yaml"
    )

    assert config.agents.count == 100
    assert config.policy.update_rule == "hk"
    assert config.policy.confidence_threshold == pytest.approx(0.35)
    assert config.rewiring.enabled is False


def test_toy3_rejects_invalid_confidence_threshold(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["dynamics"]["confidence_threshold"] = 3.0
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="confidence_threshold"):
        load_toy3_config(path)


def test_toy3_rejects_invalid_opinion_bounds(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["environment"]["opinion_min"] = 1.0
    raw["environment"]["opinion_max"] = -1.0
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="opinion_min"):
        load_toy3_config(path)


def test_hk_update_moves_toward_compatible_neighbor_mean() -> None:
    opinions = np.asarray([0.0, 0.2, 0.9], dtype=np.float64)
    neighbors = [[1, 2], [0], [0]]

    updated, peer_ids = hk_update_opinions(
        opinions=opinions,
        neighbors=neighbors,
        confidence_threshold=0.3,
        influence_rate=1.0,
    )

    assert peer_ids == [[1], [0], []]
    assert updated[0] == pytest.approx(0.1)
    assert updated[1] == pytest.approx(0.1)
    assert updated[2] == pytest.approx(0.9)


def test_hk_incompatible_neighbors_do_not_influence() -> None:
    opinions = np.asarray([-0.8, 0.8], dtype=np.float64)
    neighbors = [[1], [0]]

    updated, peer_ids = hk_update_opinions(
        opinions=opinions,
        neighbors=neighbors,
        confidence_threshold=0.3,
        influence_rate=1.0,
    )

    assert peer_ids == [[], []]
    assert updated.tolist() == pytest.approx(opinions.tolist())


def test_deffuant_pair_update_conserves_symmetric_pair_mean() -> None:
    opinion_i, opinion_j = deffuant_pair_update(
        -0.2,
        0.4,
        confidence_threshold=0.7,
        mu=0.25,
    )

    assert opinion_i + opinion_j == pytest.approx(0.2)
    assert opinion_i == pytest.approx(-0.05)
    assert opinion_j == pytest.approx(0.25)


def test_rewiring_preserves_simple_graph_invariants() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 2), (1, 3), (4, 5)])
    opinions = np.asarray([-0.9, -0.85, 0.9, 0.85, -0.88, 0.88])

    stats = rewire_disagreeing_edges(
        graph=graph,
        opinions=opinions,
        threshold=1.0,
        rate=1.0,
        candidate_pool_size=10,
        rng=np.random.default_rng(3),
    )

    assert stats.rewired_edge_count > 0
    assert nx.number_of_selfloops(graph) == 0
    assert graph.number_of_edges() == len(
        set(tuple(sorted(edge)) for edge in graph.edges())
    )


def test_rewiring_reduces_or_preserves_high_disagreement_edges() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 2), (1, 3)])
    opinions = np.asarray([-0.9, -0.85, 0.9, 0.85, -0.88, 0.88])
    graph.add_nodes_from(range(len(opinions)))
    before = sum(
        disagreement > 1.0 for disagreement in edge_disagreements(graph, opinions)
    )

    rewire_disagreeing_edges(
        graph=graph,
        opinions=opinions,
        threshold=1.0,
        rate=1.0,
        candidate_pool_size=10,
        rng=np.random.default_rng(1),
    )
    after = sum(
        disagreement > 1.0 for disagreement in edge_disagreements(graph, opinions)
    )

    assert before == 2
    assert after <= before


def test_aggregate_rewiring_rate_uses_considered_edge_count(tmp_path: Path) -> None:
    config = load_toy3_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    graph = nx.Graph()
    graph.add_nodes_from(range(config.agents.count))
    graph.add_edges_from([(0, 1), (2, 3)])
    opinions = np.zeros(config.agents.count, dtype=np.float64)
    peer_ids = [[] for _ in range(config.agents.count)]

    row = aggregate_metrics(
        config=config,
        epoch=1,
        opinions=opinions,
        graph=graph,
        peer_ids=peer_ids,
        rewired_edge_count=1,
        cumulative_rewired_edge_count=1,
        considered_edge_count=4,
    )

    assert row["domain_realized_rewiring_rate"] == pytest.approx(0.25)


def test_toy3_output_similarity_peer_selection_uses_acceptance_probs() -> None:
    opinions = np.asarray([0.0, 0.1, 0.9], dtype=np.float64)
    neighbors = [[1, 2], [0, 2], [0, 1]]
    acceptance_probs = torch.as_tensor([0.1, 0.2, 0.9], dtype=torch.float32)

    peer_ids = select_neural_peer_ids(
        opinions=opinions,
        neighbors=neighbors,
        peer_rule="output_similarity",
        confidence_threshold=0.2,
        output_similarity_threshold=0.8,
        acceptance_probs=acceptance_probs,
    )

    assert peer_ids == [[1], [0], []]


def test_toy3_no_rewiring_when_disabled(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, rewiring_enabled=False, rewiring_rate=1.0)
    config_path = write_config(tmp_path, raw)

    result = run_toy3(config=load_toy3_config(config_path), config_path=config_path)

    assert result.domain_metrics["domain_cumulative_rewired_edge_count"] == 0


@pytest.mark.parametrize(
    ("update_rule", "mixer", "peer_rule"),
    [
        ("hk", "none", None),
        ("deffuant", "none", None),
        ("neural_policy", "output_average", "bounded_confidence"),
        ("neural_policy", "output_average", "output_similarity"),
    ],
)
def test_toy3_runner_smoke_writes_expected_outputs(
    tmp_path: Path,
    update_rule: str,
    mixer: str,
    peer_rule: str | None,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        update_rule=update_rule,
        mixer=mixer,
        peer_rule=peer_rule,
    )
    config_path = write_config(tmp_path, raw)

    result = run_toy3(config=load_toy3_config(config_path), config_path=config_path)

    assert result.run_dir.exists()
    for filename in [
        "config.yaml",
        "resolved_config.yaml",
        "metadata.json",
        "aggregate_metrics.csv",
        "micro_state.csv",
        "summary.json",
    ]:
        assert (result.run_dir / filename).exists()

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))
    assert aggregate_rows
    final = aggregate_rows[-1]
    assert 0.0 <= float(final["domain_polarization_index"]) <= 1.0
    assert int(final["domain_opinion_cluster_count"]) >= 1
    assert float(final["domain_mean_edge_disagreement"]) >= 0.0

    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)
    assert micro_rows
    for field in [
        "domain_opinion",
        "peer_count",
        "component_id",
        "domain_edge_disagreement",
        "revised",
        "domain_rewired",
    ]:
        assert field in micro_reader.fieldnames


def test_toy3_sweep_parse_args_preserves_legacy_common_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_toy3_sweep.py"])

    args = run_toy3_sweep.parse_args()

    assert args.base_config == Path(
        "experiments/configs/toy3_opinion_rewiring_baseline.yaml"
    )
    assert args.label == "toy3_opinion_rewiring_sweep_seeds01_05"
    assert args.seeds == [1, 2, 3, 4, 5]
    assert args.epochs is None
    assert args.mixers == ["none", "output_average"]
    assert args.peer_rules is None
    assert args.alpha == pytest.approx(0.25)
    assert args.alphas is None
    assert args.thresholds == [0.0]
    assert args.config_dir is None
    assert args.results_dir == Path("experiments/results")


def test_toy3_sweep_parse_args_accepts_common_alphas_and_peer_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_toy3_sweep.py",
            "--peer-rules",
            "bounded_confidence",
            "output_similarity",
            "--alphas",
            "0.1",
            "0.3",
            "--thresholds",
            "0.2",
        ],
    )

    args = run_toy3_sweep.parse_args()

    assert args.peer_rules == ["bounded_confidence", "output_similarity"]
    assert args.alphas == [0.1, 0.3]
    assert args.thresholds == [0.2]


def test_toy3_sweep_summary_includes_required_metrics(tmp_path: Path) -> None:
    base = tiny_config_dict(tmp_path)
    config_path = write_case_config(
        base=base,
        label="toy3_sweep_test",
        update_rule="hk",
        mixer="none",
        seed=1,
        confidence_threshold=0.25,
        rewiring_rate=0.5,
        rewiring_threshold=0.7,
        alpha=0.25,
        epochs=1,
        config_dir=tmp_path / "configs",
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["model"]["policy"]["confidence_threshold"] == pytest.approx(0.25)
    assert raw["domain"]["rewiring"]["enabled"] is True

    row = result_row(
        label="toy3_sweep_test",
        update_rule="hk",
        mixer="none",
        peer_rule="none",
        seed=1,
        confidence_threshold=0.25,
        rewiring_enabled=True,
        rewiring_threshold=0.7,
        rewiring_rate=0.5,
        alpha=0.0,
        run_dir=tmp_path / "run",
        domain_final_opinion_mean=0.0,
        domain_final_polarization_index=0.4,
        domain_final_opinion_cluster_count=2,
        domain_final_mean_edge_disagreement=0.3,
        final_fragmentation_components=1,
        domain_final_largest_connected_component_fraction=1.0,
        domain_cumulative_rewired_edge_count=4,
    )
    summary_path = tmp_path / "summary.csv"
    write_summary_csv(summary_path, [row])

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = next(reader)

    grouped = build_grouped_summary([row])

    assert "domain_final_polarization_index" in reader.fieldnames
    assert "domain_final_opinion_cluster_count" in reader.fieldnames
    assert "domain_rewiring_rate" in reader.fieldnames
    assert "domain_final_mean_edge_disagreement" in reader.fieldnames
    assert "coordination_threshold" in reader.fieldnames
    assert float(written["domain_final_polarization_index"]) == pytest.approx(0.4)
    assert int(written["domain_final_opinion_cluster_count"]) == 2
    assert float(written["domain_rewiring_rate"]) == pytest.approx(0.5)
    assert float(written["domain_final_mean_edge_disagreement"]) == pytest.approx(0.3)
    assert "final_polarization_index_mean" in grouped.columns
    assert "final_opinion_cluster_count_mean" in grouped.columns
    assert "final_mean_edge_disagreement_mean" in grouped.columns


def test_toy3_sweep_writes_output_similarity_peer_rule(tmp_path: Path) -> None:
    base = tiny_config_dict(tmp_path, update_rule="neural_policy")
    config_path = write_case_config(
        base=base,
        label="toy3_sweep_output_similarity_test",
        update_rule="neural_policy",
        mixer="output_average",
        peer_rule="output_similarity",
        seed=1,
        confidence_threshold=0.25,
        rewiring_rate=0.0,
        rewiring_threshold=0.7,
        alpha=0.25,
        epochs=1,
        config_dir=tmp_path / "configs",
    )

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = load_toy3_config(config_path)

    assert raw["model"]["coordination"]["peer_rule"] == "output_similarity"
    assert config.coordination.peer_rule == "output_similarity"
