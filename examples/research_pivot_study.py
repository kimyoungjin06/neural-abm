"""Replicated science-of-science PIVOT study through neural_abm.api_lite.

The study asks under which scientific environments researcher pivots become
productive rather than merely frequent. It compares four environments through
seed-paired replicates:

- ``baseline``: no explicit pivot support.
- ``interdisciplinary_seed_grants``: funding, bridge ties, and reduced
  reputation risk raise both pivot pressure and structural fit.
- ``hot_field_hype``: attention and peer-success signals raise pivot pressure
  while its smaller structural signals do not provide broad support.
- ``hype_with_support``: both signal families at once.

Hypotheses:

- H1: seed grants raise the productive pivot rate over baseline.
- H2: hype raises the pivot rate while the productive share of pivots drops.
- H3: support layered on hype recovers part of the productive share.

Hype susceptibility is heterogeneous: attention and peer-success signals are
scaled per researcher by ``0.6 * (1 - resource_security) + 0.4 * openness``,
so the specified hype signal is designed to weigh resource-insecure and open
researchers more heavily. Seed-grant support is program-level and applies
uniformly. These are postulated stylized mechanisms behind H2, not a measured
subgroup result or an empirically identified causal decomposition.

Each replicate samples a fresh researcher population and stage-assortative
base network. Scenario-independent keyed streams align the population, base
network, and agent-step shocks across paired scenarios; topology interventions
are added through a separate stream. Run ``--quick`` for a fast smoke and
``--sweeps`` to add the sensitivity sweeps used by the case-study figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from neural_abm.api_lite import (
    BoundedScalarScenarioSpec,
    ReplicationSpec,
    ScenarioDefinition,
    ScenarioReplicateContext,
    run_replicated_bounded_scalar_scenarios,
)

STAGES: tuple[str, ...] = ("early", "mid", "senior", "bridge")
STAGE_WEIGHTS: tuple[float, ...] = (0.35, 0.30, 0.20, 0.15)

# Stable numeric component identifiers keep exogenous draws aligned across
# scenarios without relying on Python's process-randomized string hash.
RNG_COMPONENTS: dict[str, int] = {
    "population": 1,
    "base_network": 2,
    "bridge_intervention": 3,
    "local_noise": 4,
}

# Beta(a, b) parameters per stage for the structural attributes.
STAGE_ATTRIBUTES: dict[str, dict[str, tuple[float, float]]] = {
    "early": {
        "skill_distance": (4.0, 4.0),
        "resource_security": (2.5, 5.0),
        "network_support": (2.5, 5.0),
        "reputation_risk": (4.0, 4.0),
        "openness": (6.0, 2.5),
    },
    "mid": {
        "skill_distance": (4.0, 4.5),
        "resource_security": (4.0, 4.0),
        "network_support": (4.0, 4.0),
        "reputation_risk": (4.0, 4.5),
        "openness": (4.5, 3.5),
    },
    "senior": {
        "skill_distance": (3.5, 5.0),
        "resource_security": (6.0, 2.5),
        "network_support": (6.0, 2.5),
        "reputation_risk": (2.5, 6.0),
        "openness": (3.0, 4.5),
    },
    "bridge": {
        "skill_distance": (2.5, 6.0),
        "resource_security": (4.0, 4.0),
        "network_support": (5.5, 3.0),
        "reputation_risk": (3.0, 5.0),
        "openness": (6.5, 2.0),
    },
}


@dataclass
class Researcher:
    agent_id: int
    career_stage: str
    skill_distance: float
    resource_security: float
    network_support: float
    field_opportunity: float
    reputation_risk: float
    openness: float
    pivot_readiness: float
    funding_signal: float = 0.0
    attention_signal: float = 0.0
    peer_success_signal: float = 0.0
    pivoted: bool = False
    productive_pivot: bool = False

    @property
    def pivot_pressure(self) -> float:
        # Push-pull structure: insecurity and attention push the desire to
        # pivot, while resources and networks mostly shape productive_fit.
        pressure = (
            0.28 * self.field_opportunity
            + 0.15 * self.openness
            + 0.12 * (1.0 - self.resource_security)
            + 0.12 * self.network_support
            + 0.20 * self.funding_signal
            + 0.25 * self.attention_signal
            + 0.18 * self.peer_success_signal
            - 0.22 * self.skill_distance
            - 0.18 * self.reputation_risk
            + 0.05
        )
        return float(np.clip(pressure, 0.0, 1.0))

    @property
    def productive_fit(self) -> float:
        fit = (
            0.35 * self.field_opportunity
            + 0.30 * self.network_support
            + 0.25 * self.resource_security
            + 0.10 * self.openness
            - 0.25 * self.skill_distance
            - 0.15 * self.reputation_risk
        )
        return float(np.clip(fit, 0.0, 1.0))


@dataclass(frozen=True)
class StudyConfig:
    agent_count: int = 120
    steps: int = 8
    replicates: int = 100
    base_seed: int = 20260715
    neighbor_count: int = 4
    within_stage_probability: float = 0.6
    local_alpha: float = 0.45
    local_noise_scale: float = 0.02
    social_alpha: float = 0.40
    peer_similarity_threshold: float = 0.68
    pivot_threshold: float = 0.34
    productive_threshold: float = 0.40
    success_min_delta: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "agent_count",
            "steps",
            "replicates",
            "base_seed",
            "neighbor_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.agent_count < 2:
            raise ValueError("agent_count must be >= 2")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.replicates < 1:
            raise ValueError("replicates must be >= 1")
        if self.base_seed < 0:
            raise ValueError("base_seed must be >= 0")
        if not 0 <= self.neighbor_count < self.agent_count:
            raise ValueError(
                "neighbor_count must satisfy 0 <= neighbor_count < agent_count"
            )
        for name in (
            "within_stage_probability",
            "local_alpha",
            "social_alpha",
            "peer_similarity_threshold",
            "pivot_threshold",
            "productive_threshold",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise TypeError(f"{name} must be a finite real number")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("local_noise_scale", "success_min_delta"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise TypeError(f"{name} must be a finite real number")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")


SEED_GRANT_PARAMETERS: dict[str, float] = {
    "field_signal": 0.04,
    "resource_signal": 0.10,
    "bridge_signal": 0.10,
    "funding_signal": 0.25,
    "peer_success_signal": 0.08,
    "reputation_penalty": -0.05,
    "extra_bridge_tie_probability": 0.5,
}

# Hype is modeled as attention exceeding real opportunity: strong attention
# and peer-success signals, a small real field signal, no bridge ties, and a
# modest resource/reputation penalty. Several terms therefore affect fit; the
# model does not identify attention susceptibility as a unique cause.
HOT_FIELD_HYPE_PARAMETERS: dict[str, float] = {
    "field_signal": 0.06,
    "resource_signal": -0.05,
    "attention_signal": 0.55,
    "peer_success_signal": 0.35,
    "reputation_penalty": 0.10,
}


def study_scenarios(grant_scale: float = 1.0) -> tuple[ScenarioDefinition, ...]:
    if (
        isinstance(grant_scale, bool)
        or not isinstance(grant_scale, Real)
        or not np.isfinite(grant_scale)
        or grant_scale < 0.0
    ):
        raise ValueError("grant_scale must be finite and >= 0")
    # The topology intervention is part of the dose. At scale zero every
    # grant mechanism, including the extra-bridge probability, is absent.
    grants = {key: value * grant_scale for key, value in SEED_GRANT_PARAMETERS.items()}
    combined = dict(HOT_FIELD_HYPE_PARAMETERS)
    for key, value in grants.items():
        combined[key] = combined.get(key, 0.0) + value
    return (
        ScenarioDefinition(
            name="baseline",
            description="No explicit interdisciplinary pivot support.",
        ),
        ScenarioDefinition(
            name="interdisciplinary_seed_grants",
            description=(
                "Small grants, bridge ties, and reduced reputation risk raise "
                "pivot pressure and structural fit together."
            ),
            parameters=grants,
        ),
        ScenarioDefinition(
            name="hot_field_hype",
            description=(
                "A visible hot-field signal raises attention and peer-success "
                "pressure without adding bridge ties or resources."
            ),
            parameters=dict(HOT_FIELD_HYPE_PARAMETERS),
        ),
        ScenarioDefinition(
            name="hype_with_support",
            description=(
                "Hot-field attention combined with seed-grant support signals."
            ),
            parameters=combined,
        ),
    )


def _parameter(context: ScenarioReplicateContext, key: str) -> float:
    return float(context.scenario.parameters.get(key, 0.0))


def component_rng(
    base_seed: int,
    replicate: int,
    component: str,
    *indices: int,
) -> np.random.Generator:
    """Return one deterministic, scenario-independent exogenous RNG stream."""

    try:
        component_id = RNG_COMPONENTS[component]
    except KeyError as exc:
        raise ValueError(f"unknown RNG component: {component}") from exc
    return np.random.default_rng(
        np.random.SeedSequence((base_seed, replicate, component_id, *indices))
    )


def _sample_bounded(
    rng: np.random.Generator,
    beta_params: tuple[float, float],
    shift: float = 0.0,
) -> float:
    return float(np.clip(rng.beta(*beta_params) + shift, 0.0, 1.0))


def make_researchers(
    context: ScenarioReplicateContext,
    config: StudyConfig,
) -> list[Researcher]:
    # Scenario parameters transform the same underlying population draws.
    rng = component_rng(config.base_seed, context.replicate, "population")
    stages = rng.choice(STAGES, size=config.agent_count, p=STAGE_WEIGHTS)
    researchers: list[Researcher] = []
    for agent_id, stage in enumerate(stages):
        attributes = STAGE_ATTRIBUTES[str(stage)]
        resource_security = _sample_bounded(
            rng,
            attributes["resource_security"],
            shift=_parameter(context, "resource_signal"),
        )
        openness = _sample_bounded(rng, attributes["openness"])
        # Hype signals are attention-driven and land hardest on researchers
        # with insecure resources; grant signals are program-level and uniform.
        hype_susceptibility = float(
            np.clip(0.6 * (1.0 - resource_security) + 0.4 * openness, 0.0, 1.0)
        )
        researchers.append(
            Researcher(
                agent_id=agent_id,
                career_stage=str(stage),
                skill_distance=_sample_bounded(rng, attributes["skill_distance"]),
                resource_security=resource_security,
                network_support=_sample_bounded(
                    rng,
                    attributes["network_support"],
                    shift=_parameter(context, "bridge_signal"),
                ),
                field_opportunity=_sample_bounded(
                    rng,
                    (4.0, 4.0),
                    shift=_parameter(context, "field_signal"),
                ),
                reputation_risk=_sample_bounded(
                    rng,
                    attributes["reputation_risk"],
                    shift=_parameter(context, "reputation_penalty"),
                ),
                openness=openness,
                pivot_readiness=float(np.clip(rng.beta(2.0, 4.0), 0.0, 0.9)),
                funding_signal=_parameter(context, "funding_signal"),
                attention_signal=(
                    _parameter(context, "attention_signal") * hype_susceptibility
                ),
                peer_success_signal=(
                    _parameter(context, "peer_success_signal") * hype_susceptibility
                ),
            )
        )
    return researchers


def build_stage_assortative_network(
    researchers: list[Researcher],
    context: ScenarioReplicateContext,
    config: StudyConfig,
) -> list[list[int]]:
    # The base graph and the optional bridge intervention have separate
    # streams. This makes the base graph identical across paired scenarios.
    rng = component_rng(config.base_seed, context.replicate, "base_network")
    bridge_rng = component_rng(
        config.base_seed,
        context.replicate,
        "bridge_intervention",
    )
    count = len(researchers)
    by_stage: dict[str, list[int]] = {stage: [] for stage in STAGES}
    for researcher in researchers:
        by_stage[researcher.career_stage].append(researcher.agent_id)
    bridge_ids = by_stage["bridge"]

    neighbors: list[list[int]] = []
    for researcher in researchers:
        chosen: set[int] = set()
        same_stage = [
            other
            for other in by_stage[researcher.career_stage]
            if other != researcher.agent_id
        ]
        while len(chosen) < config.neighbor_count:
            if same_stage and rng.random() < config.within_stage_probability:
                candidate = int(rng.choice(same_stage))
            else:
                candidate = int(rng.integers(count))
            if candidate != researcher.agent_id:
                chosen.add(candidate)
        eligible_bridges = [
            candidate for candidate in bridge_ids if candidate != researcher.agent_id
        ]
        if eligible_bridges:
            bridge_draw = float(bridge_rng.random())
            candidate = int(bridge_rng.choice(eligible_bridges))
            if bridge_draw < _parameter(context, "extra_bridge_tie_probability"):
                chosen.add(candidate)
        neighbors.append(sorted(chosen))
    return neighbors


def run_study(
    config: StudyConfig,
    *,
    scenarios: tuple[ScenarioDefinition, ...] | None = None,
    social_alpha: float | None = None,
    replicates: int | None = None,
) -> dict[str, Any]:
    spec = BoundedScalarScenarioSpec(
        research_question=(
            "under_which_environments_do_researcher_pivots_become_productive"
        ),
        state_field="pivot_readiness",
        channel_name="pivot_readiness",
        transition_label="pivoted = pivot_readiness >= pivot_threshold",
        commit_mode="domain_pivot_threshold",
        outcome_field="productive_pivot_rate",
        success_direction="increase",
        success_min_delta=config.success_min_delta,
        steps=config.steps,
        peer_similarity_threshold=config.peer_similarity_threshold,
        social_alpha=config.social_alpha if social_alpha is None else social_alpha,
    )
    replication = ReplicationSpec(
        replicates=config.replicates if replicates is None else replicates,
        base_seed=config.base_seed,
    )
    local_step_by_agent: dict[tuple[str, int, int], int] = {}

    def make_agents(context: ScenarioReplicateContext) -> list[Researcher]:
        return make_researchers(context, config)

    def build_neighbors(
        agents: list[Researcher],
        context: ScenarioReplicateContext,
    ) -> list[list[int]]:
        return build_stage_assortative_network(agents, context, config)

    def local_update(
        agent: Researcher,
        context: ScenarioReplicateContext,
    ) -> float:
        before = float(agent.pivot_readiness)
        key = (context.scenario.name, context.replicate, agent.agent_id)
        step = local_step_by_agent.get(key, 0)
        local_step_by_agent[key] = step + 1
        noise_rng = component_rng(
            config.base_seed,
            context.replicate,
            "local_noise",
            step,
            agent.agent_id,
        )
        noise = float(noise_rng.normal(0.0, config.local_noise_scale))
        agent.pivot_readiness = float(
            np.clip(
                (1.0 - config.local_alpha) * before
                + config.local_alpha * agent.pivot_pressure
                + noise,
                0.0,
                1.0,
            )
        )
        return abs(agent.pivot_readiness - before)

    def domain_transition(
        agent: Researcher,
        mixed_readiness: float,
        context: ScenarioReplicateContext,
    ) -> dict[str, Any]:
        del context
        agent.pivot_readiness = float(np.clip(mixed_readiness, 0.0, 1.0))
        agent.pivoted = agent.pivot_readiness >= config.pivot_threshold
        agent.productive_pivot = (
            agent.pivoted and agent.productive_fit >= config.productive_threshold
        )
        return {
            "pivoted": agent.pivoted,
            "productive_pivot": agent.productive_pivot,
        }

    def micro_fields(
        agent: Researcher,
        context: ScenarioReplicateContext,
    ) -> dict[str, Any]:
        return {
            "career_stage": agent.career_stage,
            "scenario": context.scenario.name,
            "replicate": context.replicate,
            "skill_distance": agent.skill_distance,
            "resource_security": agent.resource_security,
            "network_support": agent.network_support,
            "field_opportunity": agent.field_opportunity,
            "reputation_risk": agent.reputation_risk,
            "pivot_pressure": agent.pivot_pressure,
            "productive_fit": agent.productive_fit,
        }

    def aggregate_fields(
        agents: list[Researcher],
        context: ScenarioReplicateContext,
    ) -> dict[str, Any]:
        del context
        count = len(agents)
        pivots = sum(agent.pivoted for agent in agents)
        productive = sum(agent.productive_pivot for agent in agents)
        return {
            "pivot_rate": pivots / count,
            "productive_pivot_rate": productive / count,
            "failed_pivot_rate": (pivots - productive) / count,
            "productive_share_of_pivots": productive / pivots if pivots else 0.0,
        }

    return run_replicated_bounded_scalar_scenarios(
        scenarios=study_scenarios() if scenarios is None else scenarios,
        spec=spec,
        replication=replication,
        make_agents=make_agents,
        build_neighbors=build_neighbors,
        local_update=local_update,
        domain_transition=domain_transition,
        micro_fields=micro_fields,
        aggregate_fields=aggregate_fields,
    ).to_dict()


def run_sensitivity_sweeps(config: StudyConfig, replicates: int) -> dict[str, Any]:
    grant_scales = (0.0, 0.5, 1.0, 1.5, 2.0)
    social_alphas = (0.0, 0.2, 0.4, 0.6, 0.8)

    grant_rows: list[dict[str, Any]] = []
    for scale in grant_scales:
        result = run_study(
            config,
            scenarios=study_scenarios(grant_scale=scale),
            replicates=replicates,
        )
        comparisons = {row["scenario"]: row for row in result["comparisons"]}
        grant_rows.append(
            {
                "grant_scale": scale,
                "seed_grants": comparisons["interdisciplinary_seed_grants"],
                "hype_with_support": comparisons["hype_with_support"],
            }
        )

    social_rows: list[dict[str, Any]] = []
    for alpha in social_alphas:
        result = run_study(config, social_alpha=alpha, replicates=replicates)
        scenario_summaries = {
            name: {
                "pivot_rate": summary["aggregate_summaries"]["pivot_rate"],
                "productive_pivot_rate": summary["aggregate_summaries"][
                    "productive_pivot_rate"
                ],
                "productive_share_of_pivots": summary["aggregate_summaries"][
                    "productive_share_of_pivots"
                ],
            }
            for name, summary in result["scenarios"].items()
        }
        social_rows.append(
            {
                "social_alpha": alpha,
                "scenarios": scenario_summaries,
            }
        )

    return {
        "replicates_per_point": replicates,
        "grant_scale_sweep": grant_rows,
        "social_alpha_sweep": social_rows,
    }


def _compact_scenarios(scenarios: dict[str, Any], micro_sample: int) -> dict[str, Any]:
    if isinstance(micro_sample, bool) or not isinstance(micro_sample, Integral):
        raise ValueError("micro_sample must be an integer")
    if micro_sample < 0:
        raise ValueError("micro_sample must be >= 0")
    compact: dict[str, Any] = {}
    for name, summary in scenarios.items():
        entry = dict(summary)
        final = entry.pop("first_replicate_final")
        entry["first_replicate_aggregate"] = final["aggregate_audit"]
        entry["first_replicate_micro_sample"] = final["micro_audit"][:micro_sample]
        entry["first_replicate_micro_sample_size"] = min(
            micro_sample,
            len(final["micro_audit"]),
        )
        compact[name] = entry
    return compact


def paired_outcome_contrast(
    scenarios: dict[str, Any],
    *,
    reference: str,
    comparison: str,
) -> dict[str, Any]:
    """Summarize one direct paired contrast from raw replicate outcomes."""

    reference_values = np.asarray(scenarios[reference]["outcome_values"], dtype=float)
    comparison_values = np.asarray(
        scenarios[comparison]["outcome_values"],
        dtype=float,
    )
    deltas = comparison_values - reference_values
    mean_delta = float(deltas.mean())
    std = float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0
    if deltas.size > 1:
        half_width = 1.959963984540054 * std / np.sqrt(deltas.size)
        mean_ci95 = [mean_delta - half_width, mean_delta + half_width]
        mean_ci_method = "normal_approximation_for_paired_mean"
    else:
        mean_ci95 = None
        mean_ci_method = "unavailable_requires_at_least_2_replicates"
    empirical = [float(value) for value in np.percentile(deltas, [2.5, 97.5])]
    return {
        "contrast": f"{comparison} - {reference}",
        "outcome_field": scenarios[reference]["outcome_field"],
        "replicates": int(deltas.size),
        "mean_delta": round(mean_delta, 6),
        "delta_std": round(std, 6),
        "mean_effect_ci95": (
            None if mean_ci95 is None else [round(value, 6) for value in mean_ci95]
        ),
        "delta_ci95_method": mean_ci_method,
        "delta_empirical_percentile_interval95": [
            round(value, 6) for value in empirical
        ],
        "paired_deltas": [round(float(value), 8) for value in deltas],
    }


def _source_snapshot(
    root: Path,
    additional_source_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Hash the source state that can affect the tracked study payload."""

    candidates = set((root / "src" / "neural_abm").rglob("*.py"))
    candidates.update(
        {
            root / "examples" / "research_pivot_study.py",
            root / "pyproject.toml",
            root / "uv.lock",
            *additional_source_paths,
        }
    )
    paths = sorted(
        (path for path in candidates if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "algorithm": "sha256(sorted repository-relative path NUL bytes NUL)",
        "sha256": digest.hexdigest(),
        "file_count": len(relative_paths),
        "paths": relative_paths,
    }


def reproducibility_metadata(
    *,
    additional_source_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Capture enough runtime provenance to interpret a written artifact."""

    try:
        package_version: str | None = version("neural-abm")
    except PackageNotFoundError:
        package_version = None
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    git_revision = completed.stdout.strip() if completed.returncode == 0 else None
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_revision": git_revision,
        "git_worktree_dirty": (
            bool(status.stdout.strip()) if status.returncode == 0 else None
        ),
        "package_version": package_version,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_snapshot": _source_snapshot(root, additional_source_paths),
        "rng_contract": (
            "keyed SeedSequence streams by base_seed, replicate, component, "
            "step, and agent where applicable"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--sweep-replicates", type=int, default=40)
    parser.add_argument("--micro-sample", type=int, default=3)
    parser.add_argument("--sweeps", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.quick:
        overrides.update({"agent_count": 40, "steps": 4, "replicates": 10})
    if args.agents is not None:
        overrides["agent_count"] = args.agents
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.replicates is not None:
        overrides["replicates"] = args.replicates
    if args.base_seed is not None:
        overrides["base_seed"] = args.base_seed
    config = StudyConfig(**overrides)

    study = run_study(config)
    payload: dict[str, Any] = {
        "status": study["status"],
        "surface": study["surface"],
        "base_surface": study["base_surface"],
        "default_profile": study["default_profile"],
        "torch_loaded": "torch" in sys.modules,
        "config": asdict(config),
        "provenance": reproducibility_metadata(),
        "research_question": study["research_question"],
        "outcome_field": study["outcome_field"],
        "comparisons": study["comparisons"],
        "direct_contrasts": [
            paired_outcome_contrast(
                study["scenarios"],
                reference="interdisciplinary_seed_grants",
                comparison="hot_field_hype",
            )
        ],
        "scenarios": _compact_scenarios(study["scenarios"], args.micro_sample),
    }
    if args.sweeps:
        sweep_replicates = (
            min(args.sweep_replicates, config.replicates)
            if args.quick
            else args.sweep_replicates
        )
        payload["sensitivity"] = run_sensitivity_sweeps(config, sweep_replicates)

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output": str(args.output),
                    "comparisons": payload["comparisons"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
