#!/usr/bin/env python3
"""Run the precommitted Pivot decision-making (persona) LLM agent v1 (N07).

The local LLM role-plays each researcher and decides for itself whether to
pivot now, and, after a failed pivot, whether to stay. ``run`` needs
transformers and a GPU:

    uv run --with 'transformers==5.14.1' --with accelerate \
        python scripts/prototype_pivot_llm_persona_agent_v1.py run ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_persona_agent_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_llm_agent_reference_v1 as n06  # noqa: E402


pilot4 = n06.pilot4
pilot3 = n06.pilot3
pilot1 = n06.pilot1
abm1 = n06.abm1
abm2 = n06.abm2
delayed = n06.delayed
persistence = n06.persistence
study2 = n06.study2
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = n06.FIT_WORLDS
FEATURES = n06.FEATURES
NABM_ARMS = n06.NABM_ARMS
FIELD_ARMS = n06.FIELD_ARMS
NABM_CONDITIONS = n06.NABM_CONDITIONS
PERSONA_ARM = ("persona_agent", "field_public_and_collaborators_in_context")
PERSONA_CONDITION = "persona_agent_qwen3_4b_2507"
CONDITION_IDS = (*NABM_CONDITIONS, PERSONA_CONDITION)
OPENED_SEEDS = n06.OPENED_SEEDS | {20261027, 20261028}
DECISION_LOG_ODDS_THRESHOLD = 0.0


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("persona-agent contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_llm_persona_agent.v1":
        raise ValueError("unexpected persona-agent schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("persona-agent contract is not a precommit design")
    if [row["id"] for row in contract["conditions"]] != list(CONDITION_IDS):
        raise ValueError("conditions changed")
    share = float(contract["exploration_floor"]["reference_share"])
    for row in contract["conditions"]:
        expected = 0.0 if row["id"] == "hand" else share
        if float(row["exploratory_share"]) != expected:
            raise ValueError(f"{row['id']} must use exploratory share {expected}")
    if [tuple(row) for row in contract["abm_arms"]] != list(NABM_ARMS):
        raise ValueError("NABM arms changed")
    if [row["id"] for row in contract["fit_worlds"]["worlds"]] != list(FIT_WORLDS):
        raise ValueError("fit worlds changed")
    if int(contract["runtime"]["batch_size"]) != 1:
        raise ValueError("persona queries run alone so no answer depends on its batch")
    persona = contract["persona"]
    if persona["decision_rule"] != "GREEDY_YES_WHEN_LOGIT_YES_EXCEEDS_LOGIT_NO":
        raise ValueError("the persona decides greedily")
    if persona["calibration"] != "NONE_THE_PERSONA_DECIDES_FOR_ITSELF":
        raise ValueError("the persona is not calibrated")
    config = contract["profiles"]["development"]
    open_steps = int(config["steps"]) - int(config["burn_in_steps"])
    if int(contract["exploration_floor"]["open_steps"]) != open_steps:
        raise ValueError("exploration open steps must match the development profile")
    decisions = contract["decision_contract"]
    if decisions.get("improvement_rule") != pilot1.IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside this study")
    seeds = {int(config["base_seed"]), int(contract["probe_cohort"]["base_seed"])}
    if len(seeds) != 2 or seeds & OPENED_SEEDS:
        raise ValueError("the persona agent needs fresh seed namespaces")
    v2_learners = abm2.load_contract()["batch_learners"]
    for arm_id in abm2.BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v2_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from ABM v2")
    if contract["boundaries"].get("paid_api_calls") != 0:
        raise ValueError("the persona agent makes no paid API calls")


# -- prompts --------------------------------------------------------------------------


def _profile(contract: Mapping[str, Any], features: np.ndarray) -> str:
    return contract["persona"]["profile_template"].format(
        **{name: float(value) for name, value in zip(FEATURES, features, strict=True)}
    )


def social_text(contract: Mapping[str, Any], collaborators: int, pivoted: int) -> str:
    return contract["persona"]["social_template"].format(collaborators=collaborators, pivoted=pivoted)


def pivot_prompt(
    contract: Mapping[str, Any], features: np.ndarray, evidence: str, social: str | None
) -> str:
    parts = [_profile(contract, features), evidence]
    if social is not None:
        parts.append(social)
    parts.append(contract["persona"]["pivot_question"])
    return "\n\n".join(parts)


def stay_prompt(
    contract: Mapping[str, Any], features: np.ndarray, evidence: str, steps_since_entry: int
) -> str:
    status = contract["persona"]["failed_status_template"].format(steps=steps_since_entry)
    return "\n\n".join(
        [_profile(contract, features), evidence, status, contract["persona"]["stay_question"]]
    )


# -- the persona field and adapter ------------------------------------------------------


class PersonaField(n06.LLMField):
    """Asks each researcher, in its own voice, whether to pivot now."""

    def __init__(self, contract: Mapping[str, Any], policy: Any) -> None:
        super().__init__(contract, policy, 0.0)
        self.log_odds_by_agent: dict[int, float] = {}
        self.pivot_questions = 0
        self.stay_questions = 0
        self.pivot_yes = 0
        self.stay_no = 0

    def _needs_query(self, agent: Any) -> bool:
        return agent.agent_id not in self.outputs or not agent.base.pivoted

    def _social(self, agent: Any, pivoted_ids: set[int]) -> str:
        neighbors = list(self.environment.neighbors[agent.agent_id])
        pivoted = sum(other in pivoted_ids for other in neighbors)
        return social_text(self.contract, len(neighbors), pivoted)

    def _compute(self, step: int, agents: Sequence[Any]) -> None:
        events = list(self.environment.public_events)
        pivoted_ids = {other.agent_id for other in self.agents if other.base.pivoted}
        queried = [agent for agent in agents if self._needs_query(agent)]
        prompts = [
            pivot_prompt(
                self.contract,
                agent.features,
                n06.evidence_text(self.contract, events, agent.features, agent.agent_id),
                self._social(agent, pivoted_ids),
            )
            for agent in queried
        ]
        odds = self.policy.log_odds(prompts) if prompts else np.zeros(0)
        for agent, value in zip(queried, odds, strict=True):
            self.log_odds_by_agent[agent.agent_id] = float(value)
            self.outputs[agent.agent_id] = float(1.0 / (1.0 + np.exp(-value)))
        self.agent_steps_queried += len(queried)
        self.step_computed = step

    def decides_to_pivot(self, agent: Any) -> bool:
        self.output_for(agent)
        self.pivot_questions += 1
        decision = self.log_odds_by_agent[agent.agent_id] > DECISION_LOG_ODDS_THRESHOLD
        self.pivot_yes += int(decision)
        return decision

    def decides_to_stay(self, agent: Any, steps_since_entry: int) -> bool:
        events = list(self.environment.public_events)
        prompt = stay_prompt(
            self.contract,
            agent.features,
            n06.evidence_text(self.contract, events, agent.features, agent.agent_id),
            steps_since_entry,
        )
        self.stay_questions += 1
        stay = float(self.policy.log_odds([prompt])[0]) > DECISION_LOG_ODDS_THRESHOLD
        self.stay_no += int(not stay)
        return stay

    def probe_predictions(self, probe_features: np.ndarray) -> np.ndarray:
        events = list(self.environment.public_events)
        prompts = [
            pivot_prompt(self.contract, row, n06.evidence_text(self.contract, events, row, None), None)
            for row in probe_features
        ]
        return 1.0 / (1.0 + np.exp(-self.policy.log_odds(prompts)))


class DecisionCommitAdapter(pilot3.ExplorationCommitAdapter):
    """The exploration adapter with entry and failed-exit decisions behind hooks.

    With ``persona_field`` unset, the hooks are the predecessor's rules exactly.
    """

    persona_field: Any = None

    def entry_ready(self, agent: Any, readiness: float) -> bool:
        if self.persona_field is None:
            return readiness >= self.environment.config.pivot_threshold
        return self.persona_field.decides_to_pivot(agent)

    def _evaluate_failed_exit(self, index: int, agent: Any) -> None:
        if self.persona_field is None:
            super()._evaluate_failed_exit(index, agent)
            return
        if self.states[index] != persistence.ACTIVE_FAILED:
            return
        entry_step = self.entry_steps[index]
        if entry_step is None:
            raise RuntimeError("active failed agent is missing entry_step")
        steps = self.environment.step - entry_step
        if steps < self.minimum_active_steps:
            return
        if self.persona_field.decides_to_stay(agent, steps):
            return
        self.states[index] = persistence.EXITED_OR_REDIRECTED
        self.exit_steps[index] = self.environment.step
        self.exits_this_step.append(agent.agent_id)
        self.continuation_scores_at_exit.append(0.0)

    def commit(self, mix_result: Any) -> Any:
        # Copied from the exploration adapter; only the entry test is a hook.
        config = self.environment.config
        values = np.asarray(mix_result.mixed_values, dtype=np.float64)
        committed: list[int] = []
        losses = [0.0 for _agent in self.agents]
        transitions_open = self.environment.step > config.burn_in_steps
        self.exits_this_step = []
        for index, agent in enumerate(self.agents):
            if agent.base.pivoted:
                self._evaluate_failed_exit(index, agent)
                continue
            before = float(agent.base.pivot_readiness)
            after = float(np.clip(values[index], 0.0, 1.0))
            agent.base.pivot_readiness = after
            losses[index] = abs(after - before)
            committed.append(agent.agent_id)
            ready = transitions_open and self.entry_ready(agent, after)
            exploring = transitions_open and not ready and self.explores(agent.agent_id)
            if transitions_open and (ready or exploring):
                if exploring:
                    self.exploratory_entries += 1
                agent.base.pivoted = True
                agent.base.productive_pivot = (
                    agent.base.productive_fit >= config.productive_threshold
                )
                agent.pivot_step = self.environment.step
                self.entry_steps[index] = self.environment.step
                self.states[index] = (
                    persistence.ACTIVE_PRODUCTIVE
                    if agent.base.productive_pivot
                    else persistence.ACTIVE_FAILED
                )
                self.environment.new_events.append(
                    study2.PivotEvent(
                        agent_id=agent.agent_id,
                        features=agent.features.copy(),
                        productive=agent.base.productive_pivot,
                        step=self.environment.step,
                    )
                )
        return study2.CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=committed,
            losses=losses,
        )


def decision_adapter(epsilon: float, base_seed: int, replicate: int, persona_field: Any) -> type:
    return type(
        "TaskDecisionAdapter",
        (DecisionCommitAdapter,),
        {
            "epsilon": float(epsilon),
            "base_seed": int(base_seed),
            "replicate": int(replicate),
            "created": [],
            "persona_field": persona_field,
        },
    )


def _fixed_rule_run(contract: Mapping[str, Any], config: Any, replicate: int, world_id: str, probe: Any) -> dict[str, Any]:
    return abm2.run_arm(
        contract, config, replicate=replicate, visibility_id="immediate", world_id=world_id,
        arm_id="fixed_rule", evidence_regime="none", probe=probe,
    )


def run_persona_agent(
    contract: Mapping[str, Any], profile: str, replicate: int, world_id: str, field: PersonaField
) -> dict[str, Any]:
    config = abm1.study_config(contract, profile)
    condition = n06._condition(contract, PERSONA_CONDITION)
    share = float(condition["exploratory_share"])
    epsilon = pilot3.epsilon_for(share, config.steps - config.burn_in_steps)
    probe_features, probe_labels = abm2.probe_cohort(contract, replicate, world_id)
    adapter_class = decision_adapter(epsilon, config.base_seed, replicate, field)
    with n06.agent_slot(n06.llm_agent_class(field)), pilot3.adapter_slot(adapter_class):
        run = _fixed_rule_run(contract, config, replicate, world_id, (probe_features, probe_labels))
    if study2.PivotPolicyAgent is not n06.ORIGINAL_AGENT:
        raise RuntimeError("the agent slot was not restored")
    if persistence.PersistenceCommitAdapter is not pilot3.ORIGINAL_ADAPTER:
        raise RuntimeError("the adapter slot was not restored")
    if len(adapter_class.created) != 1 or len(field.agents) != config.agent_count:
        raise RuntimeError("the persona run did not build one adapter and one agent per researcher")
    willingness = field.probe_predictions(probe_features)
    return {
        **run,
        "probe_brier": None,
        "probe_auc": abm2._auc(willingness, probe_labels),
        "probe_mean_willingness": float(np.mean(willingness)),
        "evidence_events_per_learner": float(len(field.environment.public_events)),
        "exploratory_entry_rate": adapter_class.created[0].exploratory_entries / config.agent_count,
        "decided_entry_rate": (
            run["cumulative_entry_rate"] - adapter_class.created[0].exploratory_entries / config.agent_count
        ),
        "exploratory_share": share,
        "epsilon": epsilon,
        "agent_steps_queried": field.agent_steps_queried,
        "pivot_questions": field.pivot_questions,
        "pivot_yes": field.pivot_yes,
        "stay_questions": field.stay_questions,
        "stay_no": field.stay_no,
    }


def check_decision_machinery_nests_the_fixed_rule(
    contract: Mapping[str, Any], profile: str, replicate: int, world_id: str
) -> None:
    """With no persona and no exploration, the decision adapter equals the fixed-rule arm."""

    config = abm1.study_config(contract, profile)
    probe = abm2.probe_cohort(contract, replicate, world_id)
    reference = _fixed_rule_run(contract, config, replicate, world_id, probe)
    field = n06.HandField(contract)
    adapter_class = decision_adapter(0.0, config.base_seed, replicate, None)
    with n06.agent_slot(n06.llm_agent_class(field)), pilot3.adapter_slot(adapter_class):
        nested = _fixed_rule_run(contract, config, replicate, world_id, probe)
    if study2.PivotPolicyAgent is not n06.ORIGINAL_AGENT or (
        persistence.PersistenceCommitAdapter is not pilot3.ORIGINAL_ADAPTER
    ):
        raise RuntimeError("a slot was not restored")
    for key, value in reference.items():
        if nested[key] != value:
            raise RuntimeError(f"the decision machinery does not nest the fixed rule at {key}")


def run_persona(
    contract: Mapping[str, Any], profile: str, policy: Any
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    config = abm1.study_config(contract, profile)
    rows, timing = [], []
    for replicate in range(config.replicates):
        for world_id in FIT_WORLDS:
            field = PersonaField(contract, policy)
            passes = getattr(policy, "forward_passes", 0)
            seconds = getattr(policy, "gpu_seconds", 0.0)
            began = time.time()
            run = run_persona_agent(contract, profile, replicate, world_id, field)
            rows.append(
                {
                    "replicate": replicate,
                    "world_id": world_id,
                    "condition_id": PERSONA_CONDITION,
                    "arm_id": PERSONA_ARM[0],
                    "evidence_regime": PERSONA_ARM[1],
                    **run,
                }
            )
            timing.append(
                {
                    "wall_seconds": time.time() - began,
                    "gpu_seconds": getattr(policy, "gpu_seconds", 0.0) - seconds,
                    "forward_passes": getattr(policy, "forward_passes", 0) - passes,
                }
            )
    return rows, timing


def phase_a(contract: Mapping[str, Any], reader: Any) -> dict[str, Any]:
    """Validity and determinism of the persona readout; no decision rate is used to tune anything."""

    features, _keys = pilot1.design_cohort(contract)
    persona = contract["persona"]
    no_evidence = contract["agent_prompt"]["no_evidence_text"]
    prompts = [
        pivot_prompt(contract, row, no_evidence, social_text(contract, 4, 0)) for row in features
    ]
    odds = reader.log_odds(prompts)
    repeat_count = int(contract["runtime"]["determinism_repeat_prompts"])
    repeated = np.asarray([reader._forward(prompt)[0] for prompt in prompts[:repeat_count]])
    mass = reader.answer_mass(prompts)
    stay = [stay_prompt(contract, row, no_evidence, int(persona["phase_a_stay_steps"])) for row in features[:repeat_count]]
    stay_mass = reader.answer_mass(stay)
    return {
        "profiles": len(prompts),
        "zero_evidence_yes_share": float(np.mean(odds > DECISION_LOG_ODDS_THRESHOLD)),
        "zero_evidence_mean_willingness": float(np.mean(1.0 / (1.0 + np.exp(-odds)))),
        "validity": {
            "median_answer_token_mass": float(np.median(mass)),
            "stay_median_answer_token_mass": float(np.median(stay_mass)),
            "weak_format": bool(np.median(mass) < 0.5 or np.median(stay_mass) < 0.5),
        },
        "determinism": {
            "repeat_prompts": repeat_count,
            "repeat_identical": bool(np.array_equal(odds[:repeat_count], repeated)),
        },
    }


# -- decisions ------------------------------------------------------------------------------


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {
        (int(row["replicate"]), row["world_id"], row["condition_id"], row["arm_id"]): row
        for row in rows
    }
    replicates = sorted({int(row["replicate"]) for row in rows})
    decision = contract["decision_contract"]
    required = int(decision["improved_replicates_at_least"])
    rule = decision["nabm_versus_persona"]
    share = "productive_share_of_entrants"
    persona = (PERSONA_CONDITION, PERSONA_ARM[0])
    nabm_cells, hand_cells, sealing = {}, {}, {}
    for world in FIT_WORLDS:
        persona_precision = n06._values(indexed, replicates, world, *persona, share)
        persona_auc = n06._values(indexed, replicates, world, *persona, "probe_auc")
        decided = n06._values(indexed, replicates, world, *persona, "decided_entry_rate")
        sealing[world] = {
            "mean_decided_entry_rate": float(np.mean(decided)),
            "self_sealing": float(np.mean(decided)) < float(decision["self_sealing"]["decided_entry_below"]),
        }
        for arm in FIELD_ARMS:
            key = f"{world}/{arm}"
            precision_gap = n06._values(indexed, replicates, world, NABM_CONDITIONS[1], arm, share) - persona_precision
            ranking_gap = n06._values(indexed, replicates, world, NABM_CONDITIONS[1], arm, "probe_auc") - persona_auc
            precision_ok = float(np.mean(precision_gap)) >= float(rule["precision_margin"])
            ranking_ok = float(np.mean(ranking_gap)) >= float(rule["ranking_margin"])
            if precision_ok and ranking_ok:
                disposition = "NABM_MATCHES_PERSONA_AGENT"
            elif ranking_ok:
                disposition = "PERSONA_AGENT_BETTER_PRECISION"
            elif precision_ok:
                disposition = "PERSONA_AGENT_BETTER_RANKING"
            else:
                disposition = "PERSONA_AGENT_BETTER_BOTH"
            nabm_cells[key] = {
                "disposition": disposition,
                "nabm_minus_persona_precision": abm1.v2._distribution(precision_gap),
                "nabm_minus_persona_probe_auc": abm1.v2._distribution(ranking_gap),
                "nabm_precision_superior": abm1._judge(
                    precision_gap, float(decision["superiority"]["precision_at_least"]), required
                ),
                "nabm_ranking_superior": abm1._judge(
                    ranking_gap, float(decision["superiority"]["probe_auc_at_least"]), required
                ),
            }
            hand_cells[key] = abm1._judge(
                persona_precision - n06._values(indexed, replicates, world, "hand", arm, share),
                float(decision["precision_versus_hand"]["improvement_at_least"]),
                required,
            )
    matching = sorted(k for k, v in nabm_cells.items() if v["disposition"] == "NABM_MATCHES_PERSONA_AGENT")
    selecting = sorted(k for k, v in hand_cells.items() if v["pass"])
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "nabm_versus_persona": {
            "summary": pilot4._count_label(len(matching), len(nabm_cells), "NABM_MATCHES_PERSONA_AGENT"),
            "matching_cells": matching,
            "cells": nabm_cells,
        },
        "persona_versus_hand_precision": {
            "summary": pilot4._count_label(len(selecting), len(hand_cells), "SELECTS_BETTER"),
            "selecting_cells": selecting,
            "cells": hand_cells,
        },
        "self_sealing_by_world": sealing,
    }


def summarize(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], profile: str, phase: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = (
        "cumulative_entry_rate",
        "exploratory_entry_rate",
        "decided_entry_rate",
        "productive_share_of_entrants",
        "probe_brier",
        "probe_auc",
        "probe_mean_willingness",
        "active_failed_rate",
        "exited_failed_rate",
        "evidence_events_per_learner",
    )
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["world_id"], row["condition_id"], row["arm_id"])].append(row)
    arms = {condition: NABM_ARMS for condition in NABM_CONDITIONS} | {PERSONA_CONDITION: (PERSONA_ARM,)}
    cells = []
    for world in FIT_WORLDS:
        for condition in CONDITION_IDS:
            for arm, evidence in arms[condition]:
                cell_rows = grouped[(world, condition, arm)]
                means = {}
                for metric in metrics:
                    values = [row[metric] for row in cell_rows if row.get(metric) is not None]
                    means[metric] = round(float(np.mean(values)), 6) if values else None
                cells.append(
                    {
                        "world_id": world,
                        "condition_id": condition,
                        "arm_id": arm,
                        "evidence_regime": evidence,
                        "replicates": len(cell_rows),
                        "means": means,
                    }
                )
    persona_rows = [row for row in rows if row["condition_id"] == PERSONA_CONDITION]
    config = abm1.study_config(contract, profile)
    required = int(contract["decision_contract"]["improved_replicates_at_least"])
    totals = {key: int(sum(row[key] for row in persona_rows)) for key in ("pivot_questions", "pivot_yes", "stay_questions", "stay_no")}
    return {
        "schema_version": "pivot_llm_persona_agent_scorecard.v1",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "phase_a": dict(phase),
        "decision_totals": totals,
        "query_profile": {
            "agent_steps_per_run": config.agent_count * config.steps,
            "mean_agent_steps_queried_per_run": (
                float(np.mean([row["agent_steps_queried"] for row in persona_rows])) if persona_rows else None
            ),
        },
        "cells": cells,
        "abm_runs": len(rows),
        "dispositions": (
            {"status": "NOT_EVALUATED_SMOKE_PROFILE"}
            if config.replicates < required
            else decide(rows, contract)
        ),
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot persona LLM-agent scorecard v1",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        "| World | Condition | Arm | Entry | Decided entry | Entrant precision | Probe AUC | Evidence |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        m = cell["means"]
        decided = m["decided_entry_rate"]
        evidence = m["evidence_events_per_learner"]
        auc = m["probe_auc"]
        lines.append(
            f"| {cell['world_id']} | {cell['condition_id']} | {cell['arm_id']} | "
            f"{m['cumulative_entry_rate']:.4f} | {'-' if decided is None else f'{decided:.4f}'} | "
            f"{m['productive_share_of_entrants']:.4f} | {'-' if auc is None else f'{auc:.4f}'} | "
            f"{'-' if evidence is None else f'{evidence:.1f}'} |"
        )
    dispositions = summary["dispositions"]
    if "nabm_versus_persona" in dispositions:
        lines.extend(
            [
                "",
                "## Frozen dispositions",
                "",
                f"- NABM versus persona agent: `{dispositions['nabm_versus_persona']['summary']}`",
                f"- Persona precision versus hand: `{dispositions['persona_versus_hand_precision']['summary']}`",
            ]
        )
    lines.extend(["", "Synthetic worlds only.", ""])
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    precommit: Mapping[str, Any],
    contract_path: Path,
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "runs.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        "scorecard.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "scorecard.md": _markdown(summary),
    }
    for name, content in files.items():
        delayed._write_exclusive(output_dir / name, content)
    receipt = {
        "schema_version": "pivot_llm_persona_agent_receipt.v1",
        "status": "COMPLETE",
        "row_count": len(rows),
        "contract_sha256": pilot1._sha256(contract_path),
        "precommit": dict(precommit),
        "outputs": {name: pilot1._sha256(output_dir / name) for name in files},
        "boundaries": summary["boundaries"],
    }
    delayed._write_exclusive(
        output_dir / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    # Timing is not byte-reproducible, so it sits outside the receipt.
    delayed._write_exclusive(
        output_dir / "cost_timing.json", json.dumps(timing, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def cost_timing(
    nabm_seconds: Sequence[float], persona_timing: Sequence[Mapping[str, float]], reader: Any
) -> dict[str, Any]:
    per_arm_cpu = [seconds / len(NABM_ARMS) for seconds in nabm_seconds]
    gpu = [row["gpu_seconds"] for row in persona_timing]
    passes = [row["forward_passes"] for row in persona_timing]
    return {
        "status": "TIMING_NOT_BYTE_REPRODUCIBLE",
        "nabm_cpu_seconds_per_arm_run": {"median": float(np.median(per_arm_cpu)), "mean": float(np.mean(per_arm_cpu))},
        "persona_agent": {
            "median_gpu_seconds_per_run": float(np.median(gpu)),
            "median_forward_passes_per_run": float(np.median(passes)),
            "total_gpu_seconds": float(np.sum(gpu)),
            "total_forward_passes": int(np.sum(passes)),
            "median_wall_seconds_per_run": float(np.median([row["wall_seconds"] for row in persona_timing])),
            "reader_environment": getattr(reader, "environment", None),
            "weight_shard_sha256": getattr(reader, "weight_shard_sha256", None),
        },
        "machine_load_average": list(os.getloadavg()),
    }


def _plan_payload(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    features = np.full(len(FEATURES), 0.5)
    evidence = contract["agent_prompt"]["no_evidence_text"]
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_sha256": pilot1._sha256(config_path),
        "conditions": list(CONDITION_IDS),
        "example_pivot_prompt": pivot_prompt(contract, features, evidence, social_text(contract, 4, 1)),
        "example_stay_prompt": stay_prompt(contract, features, evidence, 2),
        "source_bindings": delayed.verify_source_bindings(contract),
        "boundaries": contract["boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=("smoke", "development"), default="development")
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    config_path = args.config.resolve()
    contract = load_contract(config_path)
    if args.command == "plan":
        print(json.dumps(_plan_payload(contract, config_path), indent=2, sort_keys=True))
        return
    if args.precommit_sha is None or FULL_SHA_PATTERN.fullmatch(args.precommit_sha) is None:
        parser.error("run requires a full 40-character lowercase --precommit-sha")
    if args.output_dir is None:
        parser.error("run requires --output-dir")
    precommit = delayed.verify_precommit(args.precommit_sha, contract)
    config = abm1.study_config(contract, args.profile)
    for replicate in range(config.replicates):
        check_decision_machinery_nests_the_fixed_rule(contract, args.profile, replicate, "linear_fit")
    prior = n06.load_v4_prior(contract)
    nabm_rows, nabm_seconds = n06.run_nabm(contract, args.profile, prior, workers=args.workers)
    reader = n06.LLMReader(contract)
    phase = phase_a(contract, reader)
    if phase["determinism"]["repeat_identical"] is not True:
        raise RuntimeError("repeated LLM forward passes differ")
    persona_rows, persona_timing = run_persona(contract, args.profile, reader)
    rows = [*nabm_rows, *persona_rows]
    summary = {**summarize(rows, contract, args.profile, phase), "precommit_sha": args.precommit_sha}
    receipt = write_outputs(
        output_dir=args.output_dir.resolve(),
        rows=rows,
        summary=summary,
        precommit=precommit,
        contract_path=config_path,
        timing=cost_timing(nabm_seconds, persona_timing, reader),
    )
    print(json.dumps({"status": receipt["status"], "row_count": receipt["row_count"]}))


if __name__ == "__main__":
    main()
