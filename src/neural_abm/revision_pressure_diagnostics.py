"""Offline revision-pressure diagnostics for binary NABM evidence runs."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPOCH_REVISION_PRESSURE_FIELDS = [
    "label",
    "case",
    "toy",
    "variant",
    "group",
    "seed",
    "epoch",
    "time_to_ceiling",
    "final_within_ceiling",
    "action_rate",
    "mean_policy_action_probability_post_local",
    "mean_policy_action_probability_post_social",
    "policy_action_probability_post_social_gt_0p7_rate",
    "domain_revision_advantage_mean",
    "domain_revision_advantage_positive_rate",
    "revision_pressure_proxy",
    "revision_pressure_active",
    "social_revision_pressure_proxy",
    "policy_readiness_proxy",
    "pressure_policy_gap",
    "pressure_action_gap",
    "realized_revision_rate",
    "precommitment_rate",
    "precommitment_mean_evidence",
    "commitment_rate",
]


RUN_REVISION_PRESSURE_FIELDS = [
    "label",
    "case",
    "toy",
    "variant",
    "group",
    "seed",
    "time_to_ceiling",
    "final_within_ceiling",
    "ever_reached_ceiling",
    "window_epoch_count",
    "first_revision_pressure_epoch",
    "first_policy_readiness_epoch",
    "first_action_response_epoch",
    "pressure_to_policy_lag",
    "pressure_to_action_lag",
    "early_revision_pressure_mean",
    "early_revision_pressure_active_rate",
    "early_social_revision_pressure_mean",
    "early_policy_readiness_mean",
    "early_pressure_policy_gap_mean",
    "early_pressure_action_gap_mean",
    "early_action_rate_mean",
    "early_realized_revision_rate_mean",
    "early_precommitment_rate_mean",
    "early_commitment_rate_mean",
    "revision_framing_flags",
]


@dataclass(frozen=True)
class RevisionPressureDiagnosticResult:
    """Output paths and in-memory rows for a revision-pressure diagnostic run."""

    epoch_path: Path
    run_path: Path
    markdown_path: Path
    epoch_rows: list[dict[str, Any]]
    run_rows: list[dict[str, Any]]


def write_revision_pressure_diagnostics(
    *,
    runs_csv: Path,
    output_prefix: Path,
    early_epochs: int = 10,
    pressure_threshold: float = 0.05,
    policy_threshold: float = 0.7,
    policy_ready_rate_threshold: float = 0.5,
    action_threshold: float = 0.9,
) -> RevisionPressureDiagnosticResult:
    """Write offline diagnostics for the revision-operator framing.

    The diagnostic is intentionally counterfactual: it does not train a revision
    operator. It asks whether domain/objective pressure appears before policy
    readiness or action response in existing evidence runs.
    """

    run_inputs = _read_csv_rows(runs_csv)
    epoch_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for run_input in run_inputs:
        run_dir_value = run_input.get("run_dir")
        if not run_dir_value:
            continue
        aggregate_path = Path(str(run_dir_value)) / "aggregate_metrics.csv"
        aggregate_rows = _read_csv_rows(aggregate_path)
        derived_epochs = _derive_epoch_rows(
            run_input,
            aggregate_rows,
            pressure_threshold=pressure_threshold,
            policy_threshold=policy_threshold,
            action_threshold=action_threshold,
        )
        epoch_rows.extend(derived_epochs)
        run_rows.append(
            _summarize_run(
                run_input,
                derived_epochs,
                early_epochs=early_epochs,
                pressure_threshold=pressure_threshold,
                policy_threshold=policy_threshold,
                policy_ready_rate_threshold=policy_ready_rate_threshold,
                action_threshold=action_threshold,
            )
        )

    epoch_path = output_prefix.with_name(f"{output_prefix.name}_epoch_diagnostics.csv")
    run_path = output_prefix.with_name(f"{output_prefix.name}_run_diagnostics.csv")
    markdown_path = output_prefix.with_name(f"{output_prefix.name}_summary.md")
    _write_csv(epoch_path, EPOCH_REVISION_PRESSURE_FIELDS, epoch_rows)
    _write_csv(run_path, RUN_REVISION_PRESSURE_FIELDS, run_rows)
    _write_markdown(
        markdown_path,
        runs_csv=runs_csv,
        run_rows=run_rows,
        early_epochs=early_epochs,
        pressure_threshold=pressure_threshold,
        policy_threshold=policy_threshold,
        policy_ready_rate_threshold=policy_ready_rate_threshold,
        action_threshold=action_threshold,
    )
    return RevisionPressureDiagnosticResult(
        epoch_path=epoch_path,
        run_path=run_path,
        markdown_path=markdown_path,
        epoch_rows=epoch_rows,
        run_rows=run_rows,
    )


def _derive_epoch_rows(
    run_input: dict[str, str],
    aggregate_rows: list[dict[str, str]],
    *,
    pressure_threshold: float,
    policy_threshold: float,
    action_threshold: float,
) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    for aggregate in aggregate_rows:
        epoch = _optional_int(aggregate.get("epoch"))
        if epoch is None:
            continue
        action_rate = _optional_float(aggregate.get("action_rate"))
        post_local_prob = _optional_float(
            aggregate.get("mean_policy_action_probability_post_local")
        )
        post_social_prob = _optional_float(
            aggregate.get("mean_policy_action_probability_post_social")
        )
        post_social_gt_0p7 = _optional_float(
            aggregate.get("policy_action_probability_post_social_gt_0p7_rate")
        )
        advantage = _revision_advantage_mean(aggregate)
        positive_rate = _revision_advantage_positive_rate(aggregate)
        pressure = _revision_pressure_proxy(advantage, positive_rate)
        social_pressure = _positive_delta(post_social_prob, post_local_prob)
        policy_readiness = _policy_readiness_proxy(
            post_social_prob,
            post_social_gt_0p7,
            policy_threshold=policy_threshold,
        )
        row = {
            "label": run_input.get("label", ""),
            "case": run_input.get("case", ""),
            "toy": run_input.get("toy", ""),
            "variant": run_input.get("variant", ""),
            "group": run_input.get("group", ""),
            "seed": run_input.get("seed", ""),
            "epoch": epoch,
            "time_to_ceiling": run_input.get("time_to_ceiling", ""),
            "final_within_ceiling": run_input.get("final_within_ceiling", ""),
            "action_rate": action_rate,
            "mean_policy_action_probability_post_local": post_local_prob,
            "mean_policy_action_probability_post_social": post_social_prob,
            "policy_action_probability_post_social_gt_0p7_rate": post_social_gt_0p7,
            "domain_revision_advantage_mean": advantage,
            "domain_revision_advantage_positive_rate": positive_rate,
            "revision_pressure_proxy": pressure,
            "revision_pressure_active": (
                pressure is not None and pressure >= pressure_threshold
            ),
            "social_revision_pressure_proxy": social_pressure,
            "policy_readiness_proxy": policy_readiness,
            "pressure_policy_gap": _pressure_gap(
                pressure,
                post_social_prob,
                threshold=policy_threshold,
            ),
            "pressure_action_gap": _pressure_gap(
                pressure,
                action_rate,
                threshold=action_threshold,
            ),
            "realized_revision_rate": _optional_float(
                aggregate.get("realized_revision_rate")
            ),
            "precommitment_rate": _optional_float(aggregate.get("precommitment_rate")),
            "precommitment_mean_evidence": _optional_float(
                aggregate.get("precommitment_mean_evidence")
            ),
            "commitment_rate": _optional_float(aggregate.get("commitment_rate")),
        }
        derived.append(row)
    return derived


def _summarize_run(
    run_input: dict[str, str],
    epoch_rows: list[dict[str, Any]],
    *,
    early_epochs: int,
    pressure_threshold: float,
    policy_threshold: float,
    policy_ready_rate_threshold: float,
    action_threshold: float,
) -> dict[str, Any]:
    time_to_ceiling = _optional_float(run_input.get("time_to_ceiling"))
    window_rows = [
        row
        for row in epoch_rows
        if int(row["epoch"]) > 0
        and int(row["epoch"]) <= early_epochs
        and (time_to_ceiling is None or int(row["epoch"]) < time_to_ceiling)
    ]
    if not window_rows:
        window_rows = [
            row
            for row in epoch_rows
            if int(row["epoch"]) > 0 and int(row["epoch"]) <= early_epochs
        ]

    first_pressure = _first_epoch(
        epoch_rows,
        lambda row: _ge(row.get("revision_pressure_proxy"), pressure_threshold),
    )
    first_policy = _first_epoch(
        epoch_rows,
        lambda row: _policy_ready(
            row,
            policy_threshold=policy_threshold,
            policy_ready_rate_threshold=policy_ready_rate_threshold,
        ),
    )
    first_action = _first_epoch(
        epoch_rows,
        lambda row: _ge(row.get("action_rate"), action_threshold),
    )
    summary = {
        "label": run_input.get("label", ""),
        "case": run_input.get("case", ""),
        "toy": run_input.get("toy", ""),
        "variant": run_input.get("variant", ""),
        "group": run_input.get("group", ""),
        "seed": run_input.get("seed", ""),
        "time_to_ceiling": run_input.get("time_to_ceiling", ""),
        "final_within_ceiling": run_input.get("final_within_ceiling", ""),
        "ever_reached_ceiling": run_input.get("ever_reached_ceiling", ""),
        "window_epoch_count": len(window_rows),
        "first_revision_pressure_epoch": first_pressure,
        "first_policy_readiness_epoch": first_policy,
        "first_action_response_epoch": first_action,
        "pressure_to_policy_lag": _epoch_lag(first_pressure, first_policy),
        "pressure_to_action_lag": _epoch_lag(first_pressure, first_action),
        "early_revision_pressure_mean": _window_mean(
            window_rows,
            "revision_pressure_proxy",
        ),
        "early_revision_pressure_active_rate": _active_rate(
            window_rows,
            "revision_pressure_proxy",
            threshold=pressure_threshold,
        ),
        "early_social_revision_pressure_mean": _window_mean(
            window_rows,
            "social_revision_pressure_proxy",
        ),
        "early_policy_readiness_mean": _window_mean(
            window_rows,
            "policy_readiness_proxy",
        ),
        "early_pressure_policy_gap_mean": _window_mean(
            window_rows,
            "pressure_policy_gap",
        ),
        "early_pressure_action_gap_mean": _window_mean(
            window_rows,
            "pressure_action_gap",
        ),
        "early_action_rate_mean": _window_mean(window_rows, "action_rate"),
        "early_realized_revision_rate_mean": _window_mean(
            window_rows,
            "realized_revision_rate",
        ),
        "early_precommitment_rate_mean": _window_mean(
            window_rows,
            "precommitment_rate",
        ),
        "early_commitment_rate_mean": _window_mean(window_rows, "commitment_rate"),
    }
    summary["revision_framing_flags"] = ";".join(
        _revision_framing_flags(
            summary,
            pressure_threshold=pressure_threshold,
        )
    )
    return summary


def _revision_framing_flags(
    summary: dict[str, Any],
    *,
    pressure_threshold: float,
) -> list[str]:
    flags: list[str] = []
    first_pressure = _optional_int(summary.get("first_revision_pressure_epoch"))
    first_policy = _optional_int(summary.get("first_policy_readiness_epoch"))
    first_action = _optional_int(summary.get("first_action_response_epoch"))
    pressure_mean = _optional_float(summary.get("early_revision_pressure_mean"))
    pressure_active = _optional_float(
        summary.get("early_revision_pressure_active_rate")
    )
    pressure_policy_gap = _optional_float(
        summary.get("early_pressure_policy_gap_mean")
    )
    pressure_action_gap = _optional_float(
        summary.get("early_pressure_action_gap_mean")
    )
    revision_rate = _optional_float(summary.get("early_realized_revision_rate_mean"))

    if pressure_mean is None or pressure_mean < pressure_threshold:
        flags.append("weak_revision_pressure")
    if first_pressure is not None and first_policy is not None:
        if first_pressure < first_policy:
            flags.append("revision_pressure_precedes_policy")
        elif first_pressure > first_policy:
            flags.append("policy_precedes_revision_pressure")
    if first_pressure is not None and first_action is not None:
        if first_pressure < first_action:
            flags.append("revision_pressure_precedes_action")
        elif first_pressure > first_action:
            flags.append("action_precedes_revision_pressure")
    if pressure_policy_gap is not None and pressure_policy_gap >= 0.01:
        flags.append("pressure_policy_translation_lag")
    if pressure_action_gap is not None and pressure_action_gap >= 0.02:
        flags.append("pressure_action_translation_lag")
    if revision_rate is not None and revision_rate < 0.5:
        flags.append("low_realized_revision_rate")
    if (
        pressure_active is not None
        and pressure_active >= 0.5
        and str(summary.get("final_within_ceiling", "")).lower() != "true"
    ):
        flags.append("pressure_not_sufficient_for_ceiling")
    if not flags:
        flags.append("no_revision_pressure_support")
    return flags


def _write_markdown(
    path: Path,
    *,
    runs_csv: Path,
    run_rows: list[dict[str, Any]],
    early_epochs: int,
    pressure_threshold: float,
    policy_threshold: float,
    policy_ready_rate_threshold: float,
    action_threshold: float,
) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["case"]), str(row["variant"]), str(row["group"]))].append(row)
    lines = [
        "# Revision-Pressure Diagnostics",
        "",
        f"Input runs: `{runs_csv}`",
        f"Early window: epochs 1-{early_epochs} before ceiling",
        f"Pressure threshold: `{pressure_threshold}`",
        f"Policy threshold: `{policy_threshold}`",
        f"Policy-ready rate threshold: `{policy_ready_rate_threshold}`",
        f"Action threshold: `{action_threshold}`",
        "",
        "## Group Summary",
        "",
        (
            "| Case | Variant | Group | Runs | Final Hits | Mean TtC | "
            "Top Flags | First Pressure | First Policy | First Action | "
            "Pressure->Policy | Pressure->Action | Pressure | Active | "
            "Policy Gap | Action Gap | Precommit | Commit |"
        ),
        (
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for (case, variant, group), rows in sorted(grouped.items()):
        final_hits = sum(
            1 for row in rows if str(row.get("final_within_ceiling", "")).lower() == "true"
        )
        flag_counts = Counter(
            flag
            for row in rows
            for flag in str(row.get("revision_framing_flags", "")).split(";")
            if flag
        )
        top_flags = ", ".join(
            f"{flag}:{count}" for flag, count in flag_counts.most_common(3)
        )
        lines.append(
            "| {case} | {variant} | {group} | {runs} | {hits} | {ttc} | "
            "{flags} | {pressure_epoch} | {policy_epoch} | {action_epoch} | "
            "{pressure_policy_lag} | {pressure_action_lag} | {pressure} | "
            "{active} | {policy_gap} | {action_gap} | {precommit} | {commit} |".format(
                case=case,
                variant=variant,
                group=group,
                runs=len(rows),
                hits=final_hits,
                ttc=_format_number(
                    _mean(_optional_float(row.get("time_to_ceiling")) for row in rows)
                ),
                flags=top_flags,
                pressure_epoch=_format_number(
                    _mean(
                        _optional_float(row.get("first_revision_pressure_epoch"))
                        for row in rows
                    )
                ),
                policy_epoch=_format_number(
                    _mean(
                        _optional_float(row.get("first_policy_readiness_epoch"))
                        for row in rows
                    )
                ),
                action_epoch=_format_number(
                    _mean(
                        _optional_float(row.get("first_action_response_epoch"))
                        for row in rows
                    )
                ),
                pressure_policy_lag=_format_number(
                    _mean(
                        _optional_float(row.get("pressure_to_policy_lag"))
                        for row in rows
                    )
                ),
                pressure_action_lag=_format_number(
                    _mean(
                        _optional_float(row.get("pressure_to_action_lag"))
                        for row in rows
                    )
                ),
                pressure=_format_number(
                    _mean(
                        _optional_float(row.get("early_revision_pressure_mean"))
                        for row in rows
                    )
                ),
                active=_format_number(
                    _mean(
                        _optional_float(row.get("early_revision_pressure_active_rate"))
                        for row in rows
                    )
                ),
                policy_gap=_format_number(
                    _mean(
                        _optional_float(row.get("early_pressure_policy_gap_mean"))
                        for row in rows
                    )
                ),
                action_gap=_format_number(
                    _mean(
                        _optional_float(row.get("early_pressure_action_gap_mean"))
                        for row in rows
                    )
                ),
                precommit=_format_number(
                    _mean(
                        _optional_float(row.get("early_precommitment_rate_mean"))
                        for row in rows
                    )
                ),
                commit=_format_number(
                    _mean(
                        _optional_float(row.get("early_commitment_rate_mean"))
                        for row in rows
                    )
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Run Details",
            "",
            (
                "| Case | Variant | Seed | TtC | Final Hit | Flags | "
                "Pressure | Policy | Action | P->Policy | P->Action |"
            ),
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        run_rows,
        key=lambda item: (
            str(item["case"]),
            str(item["variant"]),
            int(str(item["seed"]) or 0),
        ),
    ):
        lines.append(
            "| {case} | {variant} | {seed} | {ttc} | {hit} | {flags} | "
            "{pressure} | {policy} | {action} | {p_policy} | {p_action} |".format(
                case=row["case"],
                variant=row["variant"],
                seed=row["seed"],
                ttc=row.get("time_to_ceiling", ""),
                hit=row.get("final_within_ceiling", ""),
                flags=row.get("revision_framing_flags", ""),
                pressure=row.get("first_revision_pressure_epoch", ""),
                policy=row.get("first_policy_readiness_epoch", ""),
                action=row.get("first_action_response_epoch", ""),
                p_policy=row.get("pressure_to_policy_lag", ""),
                p_action=row.get("pressure_to_action_lag", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _revision_advantage_mean(row: dict[str, str]) -> float | None:
    training = _optional_float(row.get("domain_basin_training_effective_advantage_mean"))
    if training is not None:
        return training
    effective = _optional_float(row.get("domain_effective_advantage_mean"))
    if effective is not None:
        return effective
    return _optional_float(row.get("domain_basin_action1_advantage_mean"))


def _revision_advantage_positive_rate(row: dict[str, str]) -> float | None:
    training = _optional_float(
        row.get("domain_basin_training_effective_advantage_positive_rate")
    )
    if training is not None:
        return training
    effective = _optional_float(row.get("domain_effective_advantage_positive_rate"))
    if effective is not None:
        return effective
    return _optional_float(row.get("domain_basin_action1_advantage_positive_rate"))


def _revision_pressure_proxy(
    advantage: float | None,
    positive_rate: float | None,
) -> float | None:
    if advantage is None:
        return None
    pressure = max(0.0, advantage)
    if positive_rate is not None:
        pressure *= max(0.0, min(1.0, positive_rate))
    return pressure


def _policy_readiness_proxy(
    post_social_prob: float | None,
    post_social_gt_0p7: float | None,
    *,
    policy_threshold: float,
) -> float | None:
    if post_social_gt_0p7 is not None:
        return post_social_gt_0p7
    if post_social_prob is None:
        return None
    if post_social_prob <= 0.5:
        return 0.0
    return max(0.0, min(1.0, (post_social_prob - 0.5) / (policy_threshold - 0.5)))


def _policy_ready(
    row: dict[str, Any],
    *,
    policy_threshold: float,
    policy_ready_rate_threshold: float,
) -> bool:
    gt_rate = _optional_float(row.get("policy_action_probability_post_social_gt_0p7_rate"))
    if gt_rate is not None and gt_rate >= policy_ready_rate_threshold:
        return True
    mean_prob = _optional_float(row.get("mean_policy_action_probability_post_social"))
    return mean_prob is not None and mean_prob >= policy_threshold


def _pressure_gap(
    pressure: float | None,
    value: float | None,
    *,
    threshold: float,
) -> float | None:
    if pressure is None or value is None:
        return None
    return pressure * max(0.0, threshold - value)


def _positive_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return max(0.0, left - right)


def _epoch_lag(start: int | str, end: int | str) -> int | str:
    start_epoch = _optional_int(start)
    end_epoch = _optional_int(end)
    if start_epoch is None or end_epoch is None:
        return ""
    return end_epoch - start_epoch


def _active_rate(rows: list[dict[str, Any]], field: str, *, threshold: float) -> float | str:
    numbers = [_optional_float(row.get(field)) for row in rows]
    numbers = [number for number in numbers if number is not None]
    if not numbers:
        return ""
    return sum(1 for number in numbers if number >= threshold) / len(numbers)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)


def _ge(value: Any, threshold: float) -> bool:
    number = _optional_float(value)
    return number is not None and number >= threshold


def _first_epoch(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> int | str:
    for row in rows:
        epoch = int(row["epoch"])
        if epoch > 0 and predicate(row):
            return epoch
    return ""


def _window_mean(rows: list[dict[str, Any]], field: str) -> float | str:
    return _mean(_optional_float(row.get(field)) for row in rows)


def _mean(values: Iterable[float | None]) -> float | str:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return ""
    return float(sum(numbers) / len(numbers))


def _format_number(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"
