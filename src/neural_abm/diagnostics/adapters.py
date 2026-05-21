"""Toy-specific interpretation hooks for evidence profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neural_abm.diagnostics.schema import (
    CaseProfile,
    VariantProfile,
    optional_float,
)
from neural_abm.evidence_matrix import MatrixCase, MatrixVariant


class EvidenceProfileAdapter:
    """Base adapter for toy-specific evidence interpretation."""

    name = "generic"

    def supports(self, case: MatrixCase) -> bool:
        del case
        return True

    def variant_details(
        self,
        *,
        case: MatrixCase,
        variant: MatrixVariant,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        del case, variant, rows
        return {}

    def annotate_case(
        self,
        *,
        case: MatrixCase,
        profile: CaseProfile,
    ) -> None:
        del case, profile


class Toy5WavefrontAdapter(EvidenceProfileAdapter):
    """Interpret Toy5 readiness/precommitment evidence without owning core logic."""

    name = "toy5_wavefront"

    def supports(self, case: MatrixCase) -> bool:
        return case.toy == "toy5"

    def variant_details(
        self,
        *,
        case: MatrixCase,
        variant: MatrixVariant,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        del case
        updates = variant.updates
        details: dict[str, Any] = {
            "direction_source": updates.get(
                "model.coordination.precommitment_direction_source",
                _last_nonempty(rows, "precommitment_direction_source"),
            ),
            "readiness_aggregation": updates.get(
                "model.coordination.precommitment_peer_readiness_aggregation",
                _last_nonempty(rows, "precommitment_peer_readiness_aggregation"),
            ),
            "graph_k": updates.get("domain.graph.k"),
            "graph_rewire_probability": updates.get(
                "domain.graph.rewire_probability"
            ),
            "threshold_mode": updates.get("domain.environment.threshold_mode"),
            "heterogeneous_threshold_high": updates.get(
                "domain.environment.heterogeneous_threshold_high"
            ),
        }
        _copy_numeric_row_summary(
            details,
            rows,
            "precommitment_direction_score_mean",
            "direction_score_mean",
        )
        _copy_numeric_row_summary(
            details,
            rows,
            "precommitment_direction_ok_rate",
            "direction_ok_rate",
        )
        _copy_numeric_row_summary(
            details,
            rows,
            "precommitment_peer_readiness_mean",
            "peer_readiness_mean",
        )
        return {key: value for key, value in details.items() if _is_present(value)}

    def annotate_case(
        self,
        *,
        case: MatrixCase,
        profile: CaseProfile,
    ) -> None:
        del case
        variants = profile.variants
        main_variants = [variant for variant in variants if variant.role == "main"]
        diagnostic_variants = [
            variant for variant in variants if variant.role == "diagnostic"
        ]
        best_main = _best_by_hits_then_time(main_variants)
        if best_main is None:
            return

        if _case_name_suggests_no_seed(profile.case):
            profile.notes.append("toy5_no_seed_safety_case")
            failed_diagnostics = [
                variant
                for variant in diagnostic_variants
                if variant.final_ceiling_hits == 0
                and "non_directional" in variant.variant
            ]
            if best_main.final_ceiling_hits == best_main.expected_seed_count and failed_diagnostics:
                profile.notes.append("toy5_direction_gate_separates_self_excitation")

        mean_diagnostics = [
            variant
            for variant in diagnostic_variants
            if variant.details.get("readiness_aggregation") == "mean"
        ]
        max_main = [
            variant
            for variant in main_variants
            if variant.details.get("readiness_aggregation") == "max"
        ]
        if mean_diagnostics and max_main:
            mean_metric = max(
                (
                    value
                    for variant in mean_diagnostics
                    if (value := variant.metric.mean) is not None
                ),
                default=None,
            )
            max_metric = max(
                (
                    value
                    for variant in max_main
                    if (value := variant.metric.mean) is not None
                ),
                default=None,
            )
            if mean_metric is not None and max_metric is not None and max_metric > mean_metric:
                profile.notes.append("toy5_mean_readiness_frontier_stall")

        if any(
            variant.details.get("direction_source")
            == "readiness_augmented_threshold_with_action_anchor"
            for variant in main_variants
        ):
            profile.notes.append("toy5_threshold_aware_direction")

        if best_main.final_ceiling_hits < best_main.expected_seed_count:
            if best_main.metric.mean is not None and best_main.metric.mean >= 90.0:
                profile.issue_codes.append("toy5_frontier_near_miss")
            elif (
                best_main.details.get("direction_score_mean") is not None
                and optional_float(best_main.details["direction_score_mean"]) is not None
                and optional_float(best_main.details["direction_score_mean"]) < 0.0
            ):
                profile.issue_codes.append("toy5_ignition_failure")
            else:
                profile.issue_codes.append("toy5_unclassified_wavefront_failure")


class Toy24BasinRevisionAdapter(EvidenceProfileAdapter):
    """Interpret Toy2/Toy4 basin-credit and revision evidence artifacts."""

    name = "toy24_basin_revision"

    def supports(self, case: MatrixCase) -> bool:
        return case.toy in {"toy2", "toy4"}

    def variant_details(
        self,
        *,
        case: MatrixCase,
        variant: MatrixVariant,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        del case
        updates = variant.updates
        details: dict[str, Any] = {
            "policy_rule": updates.get("model.policy.rule"),
            "local_update_rule": updates.get("model.policy.domain.local_update_rule"),
            "objective_profile": updates.get(
                "model.policy.domain.objective.profile"
            ),
            "basin_credit_enabled": updates.get(
                "model.policy.domain.basin_credit.enabled"
            ),
            "basin_critic": updates.get("model.policy.domain.basin_credit.critic"),
            "basin_credit_method": updates.get(
                "model.policy.domain.basin_credit.credit_method"
            ),
            "basin_objective_weight": updates.get(
                "model.policy.domain.basin_credit.objective_weight"
            ),
            "basin_individual_weight": updates.get(
                "model.policy.domain.basin_credit.individual_weight"
            ),
            "basin_local_social_weight": updates.get(
                "model.policy.domain.basin_credit.local_social_weight"
            ),
            "basin_weight": updates.get("model.policy.domain.basin_credit.basin_weight"),
            "basin_horizon": updates.get("model.policy.domain.basin_credit.horizon"),
            "basin_target": updates.get("model.policy.domain.basin_credit.target_basin"),
            "revision_operator_enabled": updates.get(
                "model.coordination.revision_operator_enabled"
            ),
            "revision_operator_source": updates.get(
                "model.coordination.revision_operator_source"
            ),
            "confidence_weighting": updates.get("model.coordination.confidence_weighting"),
            "confidence_weight_floor": updates.get(
                "model.coordination.confidence_weight_floor"
            ),
            "confidence_tail_floor": updates.get(
                "model.coordination.confidence_tail_floor"
            ),
            "commitment_enabled": updates.get("model.coordination.commitment_enabled"),
            "precommitment_enabled": updates.get(
                "model.coordination.precommitment_enabled"
            ),
            "precommitment_peer_evidence_enabled": updates.get(
                "model.coordination.precommitment_peer_evidence_enabled"
            ),
            "precommitment_peer_evidence_weight": updates.get(
                "model.coordination.precommitment_peer_evidence_weight"
            ),
        }
        _copy_numeric_row_summary(
            details,
            rows,
            "ceiling_gap",
            "ceiling_gap_mean",
        )
        _copy_numeric_row_summary(
            details,
            rows,
            "late_flip_count_after_first_ceiling",
            "late_flip_count_after_first_ceiling_mean",
        )
        _copy_numeric_row_summary(
            details,
            rows,
            "terminal_window_mean_ceiling_metric",
            "terminal_window_mean_ceiling_metric_mean",
        )
        details["ever_ceiling_final_miss_rate"] = _row_bool_rate(
            rows,
            "ever_ceiling_final_miss",
        )
        return {key: value for key, value in details.items() if _is_present(value)}

    def annotate_case(
        self,
        *,
        case: MatrixCase,
        profile: CaseProfile,
    ) -> None:
        del case
        variants = profile.variants
        main_variants = [variant for variant in variants if variant.role == "main"]
        diagnostic_variants = [
            variant for variant in variants if variant.role == "diagnostic"
        ]
        best_main = _best_by_hits_then_time(main_variants)
        if any(_detail_enabled(variant, "basin_credit_enabled") for variant in variants):
            profile.notes.append("toy24_basin_credit_evidence")
        if any(
            _positive_detail(variant, "basin_objective_weight")
            and _positive_detail(variant, "basin_weight")
            for variant in main_variants
        ):
            profile.notes.append("toy24_objective_basin_blend")
        if any(
            _positive_detail(variant, "basin_individual_weight")
            and _positive_detail(variant, "basin_weight")
            and variant.final_ceiling_hits < variant.expected_seed_count
            for variant in diagnostic_variants
        ):
            profile.notes.append("toy24_material_basin_collapse_diagnostic")
        if any(_detail_enabled(variant, "revision_operator_enabled") for variant in variants):
            profile.notes.append("toy24_revision_operator_path")
        if any(
            variant.ever_ceiling_hits > variant.final_ceiling_hits
            for variant in main_variants
        ):
            profile.notes.append("toy24_final_vs_ever_gap")
            profile.issue_codes.append("toy24_final_epoch_hazard")
        if any(_mean_or_zero(variant.late_flip_rate.mean) > 0.0 for variant in main_variants):
            profile.notes.append("toy24_late_flip_hazard")
        if any(
            variant.role == "main"
            and variant.final_ceiling_hits < variant.expected_seed_count
            for variant in variants
        ):
            profile.notes.append("toy24_main_candidate_ceiling_miss")
        if (
            best_main is not None
            and best_main.final_ceiling_hits < best_main.expected_seed_count
        ):
            profile.issue_codes.append("toy24_best_main_ceiling_miss")


def adapter_for_case(case: MatrixCase) -> EvidenceProfileAdapter:
    for adapter in (Toy5WavefrontAdapter(), Toy24BasinRevisionAdapter()):
        if adapter.supports(case):
            return adapter
    return EvidenceProfileAdapter()


def _last_nonempty(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
    for row in reversed(rows):
        value = row.get(field)
        if _is_present(value):
            return value
    return None


def _is_present(value: Any) -> bool:
    return value is not None and value != ""


def _copy_numeric_row_summary(
    details: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    field: str,
    target: str,
) -> None:
    numbers = [
        number
        for row in rows
        if (number := optional_float(row.get(field))) is not None
    ]
    if numbers:
        details[target] = sum(numbers) / len(numbers)


def _row_bool_rate(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [
        _optional_bool(row.get(field))
        for row in rows
        if _optional_bool(row.get(field)) is not None
    ]
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _detail_enabled(variant: VariantProfile, key: str) -> bool:
    value = variant.details.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _positive_detail(variant: VariantProfile, key: str) -> bool:
    value = optional_float(variant.details.get(key))
    return value is not None and value > 0.0


def _mean_or_zero(value: float | None) -> float:
    return 0.0 if value is None else value


def _optional_bool(value: Any) -> bool | None:
    if value == "" or value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _best_by_hits_then_time(variants: Sequence[VariantProfile]) -> VariantProfile | None:
    if not variants:
        return None

    def key(variant: VariantProfile) -> tuple[int, float, float]:
        time = variant.time_to_ceiling.mean
        metric = variant.metric.mean
        return (
            variant.final_ceiling_hits,
            -(time if time is not None else float("inf")),
            metric if metric is not None else float("-inf"),
        )

    return max(variants, key=key)


def _case_name_suggests_no_seed(case_name: str) -> bool:
    return "no_seed" in case_name or "safety" in case_name
