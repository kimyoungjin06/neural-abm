"""Summaries for read-only learned basin critic runtime diagnostics."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASIN_LEARNED_RUN_FIELDS: tuple[str, ...] = (
    "label",
    "case",
    "toy",
    "variant",
    "group",
    "seed",
    "run_dir",
    "final_within_ceiling",
    "time_to_ceiling",
    "diagnostic_status",
    "micro_row_count",
    "finite_pair_count",
    "learned_abstention_rate",
    "learned_non_abstain_rate",
    "prototype_learned_sign_agreement_rate",
    "prototype_learned_sign_conflict_rate",
    "non_abstain_sign_agreement_rate",
    "non_abstain_sign_conflict_rate",
    "prototype_learned_advantage_correlation",
    "prototype_advantage_mean",
    "learned_advantage_mean",
    "prototype_advantage_abs_mean",
    "learned_advantage_abs_mean",
    "advantage_delta_abs_mean",
    "learned_uncertainty_mean",
    "aggregate_training_replay_min_selected_rate_mean",
    "aggregate_training_replay_selected_rate_mean",
    "aggregate_training_replay_weight_mean",
    "aggregate_training_replay_weight_positive_rate_mean",
    "aggregate_training_learned_credit_rate_mean",
    "final_aggregate_training_replay_selection",
    "final_aggregate_training_replay_min_selected_rate",
    "final_aggregate_training_replay_selected_rate",
    "final_aggregate_training_replay_weight_mean",
    "final_aggregate_training_replay_weight_positive_rate",
    "final_aggregate_training_learned_credit_rate",
    "final_aggregate_abstention_rate",
    "final_aggregate_prototype_correlation",
)

BASIN_LEARNED_GROUP_FIELDS: tuple[str, ...] = (
    "label",
    "case",
    "toy",
    "variant",
    "group",
    "run_count",
    "complete_run_count",
    "final_ceiling_hits",
    "mean_time_to_ceiling",
    "finite_pair_count",
    "learned_abstention_rate_mean",
    "learned_non_abstain_rate_mean",
    "prototype_learned_sign_agreement_rate_mean",
    "prototype_learned_sign_conflict_rate_mean",
    "non_abstain_sign_agreement_rate_mean",
    "non_abstain_sign_conflict_rate_mean",
    "prototype_learned_advantage_correlation_mean",
    "advantage_delta_abs_mean",
    "learned_uncertainty_mean",
    "aggregate_training_replay_min_selected_rate_mean",
    "aggregate_training_replay_selected_rate_mean",
    "aggregate_training_replay_weight_mean",
    "aggregate_training_replay_weight_positive_rate_mean",
    "aggregate_training_learned_credit_rate_mean",
)


@dataclass(frozen=True)
class BasinLearnedDiagnosticSummaryResult:
    run_summary_path: Path
    group_summary_path: Path
    markdown_path: Path
    run_rows: list[dict[str, object]]
    group_rows: list[dict[str, object]]


def summarize_basin_learned_diagnostics(
    runs_csv: Path,
    *,
    output_dir: Path = Path("experiments/results/basin_critic"),
    label: str | None = None,
) -> BasinLearnedDiagnosticSummaryResult:
    """Summarize learned-vs-prototype basin diagnostics from run artifacts."""

    runs = pd.read_csv(runs_csv)
    resolved_label = label or _label_from_runs(runs, runs_csv)
    run_rows = [
        summarize_basin_learned_run(row)
        for row in runs.to_dict(orient="records")
    ]
    group_rows = group_basin_learned_summaries(run_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = output_dir / f"{resolved_label}_learned_diagnostic_runs.csv"
    group_summary_path = output_dir / f"{resolved_label}_learned_diagnostic_groups.csv"
    markdown_path = output_dir / f"{resolved_label}_learned_diagnostic_summary.md"
    write_csv(run_summary_path, run_rows, BASIN_LEARNED_RUN_FIELDS)
    write_csv(group_summary_path, group_rows, BASIN_LEARNED_GROUP_FIELDS)
    markdown_path.write_text(
        render_basin_learned_diagnostic_markdown(group_rows),
        encoding="utf-8",
    )
    return BasinLearnedDiagnosticSummaryResult(
        run_summary_path=run_summary_path,
        group_summary_path=group_summary_path,
        markdown_path=markdown_path,
        run_rows=run_rows,
        group_rows=group_rows,
    )


def summarize_basin_learned_run(row: Mapping[str, object]) -> dict[str, object]:
    run_dir = Path(str(row.get("run_dir", "")))
    base = {
        "label": row.get("label", ""),
        "case": row.get("case", ""),
        "toy": row.get("toy", ""),
        "variant": row.get("variant", ""),
        "group": row.get("group", ""),
        "seed": row.get("seed", ""),
        "run_dir": str(run_dir),
        "final_within_ceiling": row.get("final_within_ceiling", ""),
        "time_to_ceiling": row.get("time_to_ceiling", ""),
    }
    aggregate = aggregate_diagnostic_fields(run_dir)
    micro_path = run_dir / "micro_state.csv"
    if not micro_path.exists():
        return {
            **base,
            **_empty_run_metrics(status="missing_micro_state"),
            **_aggregate_run_metric_fields(aggregate),
        }
    micro = pd.read_csv(micro_path)
    required = {
        "domain_basin_action1_advantage",
        "domain_basin_learned_action1_advantage",
        "domain_basin_learned_uncertainty",
        "domain_basin_learned_abstain",
    }
    if not required <= set(micro.columns):
        return {
            **base,
            **_empty_run_metrics(status="missing_learned_fields"),
            **_aggregate_run_metric_fields(aggregate),
            "micro_row_count": len(micro),
        }

    prototype = pd.to_numeric(
        micro["domain_basin_action1_advantage"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    learned = pd.to_numeric(
        micro["domain_basin_learned_action1_advantage"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    uncertainty = pd.to_numeric(
        micro["domain_basin_learned_uncertainty"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    abstain = _bool_series(micro["domain_basin_learned_abstain"]).to_numpy(dtype=bool)
    finite = np.isfinite(prototype) & np.isfinite(learned)
    if not np.any(finite):
        return {
            **base,
            **_empty_run_metrics(status="no_finite_pairs"),
            **_aggregate_run_metric_fields(aggregate),
            "micro_row_count": len(micro),
        }

    prototype_finite = prototype[finite]
    learned_finite = learned[finite]
    abstain_finite = abstain[finite]
    uncertainty_finite = uncertainty[finite]
    agreement = (prototype_finite > 0.0) == (learned_finite > 0.0)
    non_abstain = ~abstain_finite
    return {
        **base,
        "diagnostic_status": "complete",
        "micro_row_count": len(micro),
        "finite_pair_count": int(np.sum(finite)),
        "learned_abstention_rate": float(np.mean(abstain_finite)),
        "learned_non_abstain_rate": float(np.mean(non_abstain)),
        "prototype_learned_sign_agreement_rate": float(np.mean(agreement)),
        "prototype_learned_sign_conflict_rate": float(np.mean(~agreement)),
        "non_abstain_sign_agreement_rate": _masked_mean_or_empty(
            agreement.astype(np.float64),
            non_abstain,
        ),
        "non_abstain_sign_conflict_rate": _masked_mean_or_empty(
            (~agreement).astype(np.float64),
            non_abstain,
        ),
        "prototype_learned_advantage_correlation": _finite_or_empty(
            pearson_or_nan(prototype_finite, learned_finite)
        ),
        "prototype_advantage_mean": float(np.mean(prototype_finite)),
        "learned_advantage_mean": float(np.mean(learned_finite)),
        "prototype_advantage_abs_mean": float(np.mean(np.abs(prototype_finite))),
        "learned_advantage_abs_mean": float(np.mean(np.abs(learned_finite))),
        "advantage_delta_abs_mean": float(
            np.mean(np.abs(prototype_finite - learned_finite))
        ),
        "learned_uncertainty_mean": _finite_mean_or_empty(uncertainty_finite),
        **_aggregate_run_metric_fields(aggregate),
    }


def group_basin_learned_summaries(
    run_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for row in run_rows:
        grouped[
            (
                str(row.get("label", "")),
                str(row.get("case", "")),
                str(row.get("toy", "")),
                str(row.get("variant", "")),
                str(row.get("group", "")),
            )
        ].append(row)
    summaries: list[dict[str, object]] = []
    for (label, case, toy, variant, group), rows in sorted(grouped.items()):
        complete = [row for row in rows if row.get("diagnostic_status") == "complete"]
        summaries.append(
            {
                "label": label,
                "case": case,
                "toy": toy,
                "variant": variant,
                "group": group,
                "run_count": len(rows),
                "complete_run_count": len(complete),
                "final_ceiling_hits": sum(
                    1 for row in rows if _bool_value(row.get("final_within_ceiling"))
                ),
                "mean_time_to_ceiling": _mean_optional(
                    row.get("time_to_ceiling") for row in rows
                ),
                "finite_pair_count": sum(
                    int(float(row.get("finite_pair_count", 0) or 0))
                    for row in complete
                ),
                "learned_abstention_rate_mean": _mean_field(
                    complete,
                    "learned_abstention_rate",
                ),
                "learned_non_abstain_rate_mean": _mean_field(
                    complete,
                    "learned_non_abstain_rate",
                ),
                "prototype_learned_sign_agreement_rate_mean": _mean_field(
                    complete,
                    "prototype_learned_sign_agreement_rate",
                ),
                "prototype_learned_sign_conflict_rate_mean": _mean_field(
                    complete,
                    "prototype_learned_sign_conflict_rate",
                ),
                "non_abstain_sign_agreement_rate_mean": _mean_field(
                    complete,
                    "non_abstain_sign_agreement_rate",
                ),
                "non_abstain_sign_conflict_rate_mean": _mean_field(
                    complete,
                    "non_abstain_sign_conflict_rate",
                ),
                "prototype_learned_advantage_correlation_mean": _mean_field(
                    complete,
                    "prototype_learned_advantage_correlation",
                ),
                "advantage_delta_abs_mean": _mean_field(
                    complete,
                    "advantage_delta_abs_mean",
                ),
                "learned_uncertainty_mean": _mean_field(
                    complete,
                    "learned_uncertainty_mean",
                ),
                "aggregate_training_replay_min_selected_rate_mean": _mean_field(
                    rows,
                    "aggregate_training_replay_min_selected_rate_mean",
                ),
                "aggregate_training_replay_selected_rate_mean": _mean_field(
                    rows,
                    "aggregate_training_replay_selected_rate_mean",
                ),
                "aggregate_training_replay_weight_mean": _mean_field(
                    rows,
                    "aggregate_training_replay_weight_mean",
                ),
                "aggregate_training_replay_weight_positive_rate_mean": _mean_field(
                    rows,
                    "aggregate_training_replay_weight_positive_rate_mean",
                ),
                "aggregate_training_learned_credit_rate_mean": _mean_field(
                    rows,
                    "aggregate_training_learned_credit_rate_mean",
                ),
            }
        )
    return summaries


def aggregate_diagnostic_fields(run_dir: Path) -> dict[str, object]:
    path = run_dir / "aggregate_metrics.csv"
    if not path.exists():
        return {}
    rows = pd.read_csv(path)
    if rows.empty:
        return {}
    final = rows.iloc[-1].to_dict()
    return {
        **final,
        "aggregate_training_replay_min_selected_rate_mean": _mean_optional(
            rows.get("domain_basin_training_replay_min_selected_rate", [])
        ),
        "aggregate_training_replay_selected_rate_mean": _mean_optional(
            rows.get("domain_basin_training_replay_selected_rate", [])
        ),
        "aggregate_training_replay_weight_mean": _mean_optional(
            rows.get("domain_basin_training_replay_weight_mean", [])
        ),
        "aggregate_training_replay_weight_positive_rate_mean": _mean_optional(
            rows.get("domain_basin_training_replay_weight_positive_rate", [])
        ),
        "aggregate_training_learned_credit_rate_mean": _mean_optional(
            rows.get("domain_basin_training_learned_credit_rate", [])
        ),
    }


def _aggregate_run_metric_fields(aggregate: Mapping[str, object]) -> dict[str, object]:
    return {
        "aggregate_training_replay_min_selected_rate_mean": aggregate.get(
            "aggregate_training_replay_min_selected_rate_mean",
            "",
        ),
        "aggregate_training_replay_selected_rate_mean": aggregate.get(
            "aggregate_training_replay_selected_rate_mean",
            "",
        ),
        "aggregate_training_replay_weight_mean": aggregate.get(
            "aggregate_training_replay_weight_mean",
            "",
        ),
        "aggregate_training_replay_weight_positive_rate_mean": aggregate.get(
            "aggregate_training_replay_weight_positive_rate_mean",
            "",
        ),
        "aggregate_training_learned_credit_rate_mean": aggregate.get(
            "aggregate_training_learned_credit_rate_mean",
            "",
        ),
        "final_aggregate_training_replay_selection": aggregate.get(
            "domain_basin_training_replay_selection",
            "",
        ),
        "final_aggregate_training_replay_min_selected_rate": aggregate.get(
            "domain_basin_training_replay_min_selected_rate",
            "",
        ),
        "final_aggregate_training_replay_selected_rate": aggregate.get(
            "domain_basin_training_replay_selected_rate",
            "",
        ),
        "final_aggregate_training_replay_weight_mean": aggregate.get(
            "domain_basin_training_replay_weight_mean",
            "",
        ),
        "final_aggregate_training_replay_weight_positive_rate": aggregate.get(
            "domain_basin_training_replay_weight_positive_rate",
            "",
        ),
        "final_aggregate_training_learned_credit_rate": aggregate.get(
            "domain_basin_training_learned_credit_rate",
            "",
        ),
        "final_aggregate_abstention_rate": aggregate.get(
            "domain_basin_learned_abstention_rate",
            "",
        ),
        "final_aggregate_prototype_correlation": aggregate.get(
            "domain_basin_learned_prototype_advantage_correlation",
            "",
        ),
    }


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_basin_learned_diagnostic_markdown(
    rows: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# Basin Learned Diagnostic Summary",
        "",
        "| Case | Toy | Variant | Group | Runs | Complete | Final Hits | Sign Agree | Non-Abstain Agree | Abstain | Replay Floor | Replay Selected | Replay Weight | Training Learned | Uncertainty |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {toy} | {variant} | {group} | {runs} | {complete} | "
            "{hits} | {agree} | {non_abstain_agree} | {abstain} | "
            "{replay_floor} | {replay_selected} | {replay_weight} | "
            "{training_learned} | {uncertainty} |".format(
                case=row["case"],
                toy=row["toy"],
                variant=row["variant"],
                group=row["group"],
                runs=row["run_count"],
                complete=row["complete_run_count"],
                hits=row["final_ceiling_hits"],
                agree=_format_metric(
                    row.get("prototype_learned_sign_agreement_rate_mean")
                ),
                non_abstain_agree=_format_metric(
                    row.get("non_abstain_sign_agreement_rate_mean")
                ),
                abstain=_format_metric(row.get("learned_abstention_rate_mean")),
                replay_floor=_format_metric(
                    row.get("aggregate_training_replay_min_selected_rate_mean")
                ),
                replay_selected=_format_metric(
                    row.get("aggregate_training_replay_selected_rate_mean")
                ),
                replay_weight=_format_metric(
                    row.get("aggregate_training_replay_weight_mean")
                ),
                training_learned=_format_metric(
                    row.get("aggregate_training_learned_credit_rate_mean")
                ),
                uncertainty=_format_metric(row.get("learned_uncertainty_mean")),
            )
        )
    return "\n".join(lines) + "\n"


def pearson_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if np.sum(mask) < 2:
        return float("nan")
    left_values = left[mask]
    right_values = right[mask]
    if np.std(left_values) <= 1e-12 or np.std(right_values) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _empty_run_metrics(*, status: str) -> dict[str, object]:
    return {
        "diagnostic_status": status,
        "micro_row_count": "",
        "finite_pair_count": "",
        "learned_abstention_rate": "",
        "learned_non_abstain_rate": "",
        "prototype_learned_sign_agreement_rate": "",
        "prototype_learned_sign_conflict_rate": "",
        "non_abstain_sign_agreement_rate": "",
        "non_abstain_sign_conflict_rate": "",
        "prototype_learned_advantage_correlation": "",
        "prototype_advantage_mean": "",
        "learned_advantage_mean": "",
        "prototype_advantage_abs_mean": "",
        "learned_advantage_abs_mean": "",
        "advantage_delta_abs_mean": "",
        "learned_uncertainty_mean": "",
        "aggregate_training_replay_min_selected_rate_mean": "",
        "aggregate_training_replay_selected_rate_mean": "",
        "aggregate_training_replay_weight_mean": "",
        "aggregate_training_replay_weight_positive_rate_mean": "",
        "aggregate_training_learned_credit_rate_mean": "",
        "final_aggregate_training_replay_selection": "",
        "final_aggregate_training_replay_min_selected_rate": "",
        "final_aggregate_training_replay_selected_rate": "",
        "final_aggregate_training_replay_weight_mean": "",
        "final_aggregate_training_replay_weight_positive_rate": "",
        "final_aggregate_training_learned_credit_rate": "",
        "final_aggregate_abstention_rate": "",
        "final_aggregate_prototype_correlation": "",
    }


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _masked_mean_or_empty(values: np.ndarray, mask: np.ndarray) -> float | str:
    if not np.any(mask):
        return ""
    return float(np.mean(values[mask]))


def _finite_mean_or_empty(values: np.ndarray) -> float | str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ""
    return float(np.mean(finite))


def _finite_or_empty(value: float) -> float | str:
    return float(value) if math.isfinite(float(value)) else ""


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float | str:
    return _mean_optional(row.get(field) for row in rows)


def _mean_optional(values: Sequence[object] | object) -> float | str:
    finite: list[float] = []
    for value in values:
        if value in {"", None}:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    if not finite:
        return ""
    return float(math.fsum(finite) / len(finite))


def _format_metric(value: object) -> str:
    if value in {"", None}:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.6g}"


def _label_from_runs(runs: pd.DataFrame, runs_csv: Path) -> str:
    if "label" in runs.columns and not runs.empty:
        labels = sorted(str(value) for value in runs["label"].dropna().unique())
        if len(labels) == 1:
            return labels[0]
    stem = runs_csv.stem
    return stem.removesuffix("_runs")
