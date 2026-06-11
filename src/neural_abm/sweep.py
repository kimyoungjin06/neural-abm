"""Common helpers for toy sweep scripts."""

from __future__ import annotations

import argparse
import csv
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from neural_abm.capabilities import (
    NABM_ARTIFACT_FIELDS,
    supports_coordination,
    sweep_capability_metadata,
    toy_display_name as capability_toy_display_name,
)


@dataclass(frozen=True)
class PreparedSweepCase:
    case: dict[str, Any]
    effective_alpha: float
    effective_threshold: float


@dataclass(frozen=True)
class CoordinationSweepPoint:
    mixer: str
    peer_rule: str
    alpha: float
    threshold: float


@dataclass(frozen=True)
class SweepOutputSpec:
    summary_fields: Sequence[str]
    group_fields: Sequence[str]
    aggregations: Mapping[str, tuple[str, str]]
    metric_keys: Sequence[str]
    config_value_paths: Mapping[str, str] | None = None
    result_value_paths: Mapping[str, str] | None = None


@dataclass(frozen=True)
class SweepOutputPaths:
    summary_path: Path
    grouped_summary_path: Path
    grouped_markdown_path: Path | None = None


@dataclass(frozen=True)
class SweepRunArtifacts:
    rows: list[dict[str, Any]]
    config_dir: Path
    summary_path: Path
    grouped_summary_path: Path
    grouped_markdown_path: Path | None = None


DEFAULT_FINAL_AGGREGATE_METRIC_KEYS = (
    "mean_reputation",
    "reputation_dispersion",
    "mobility_rate",
    "mean_mobility_gain",
)


def safe_float(value: float) -> str:
    """Format a float for stable filesystem-safe run labels."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def effective_alphas(mixer: str, alphas: Sequence[float]) -> list[float]:
    if mixer == "none":
        return [0.0]
    return list(alphas)


def effective_peer_rules(
    mixer: str,
    peer_rules: Sequence[str] | None,
    *,
    social_default: str = "output_similarity",
) -> list[str]:
    if peer_rules is None:
        return [social_default] if mixer == "output_average" else ["none"]
    return [
        peer_rule
        for peer_rule in peer_rules
        if mixer != "none" or peer_rule == "none"
    ]


def effective_thresholds(
    mixer: str,
    peer_rule: str,
    thresholds: Sequence[float],
    *,
    threshold_peer_rule: str = "output_similarity",
) -> list[float]:
    if mixer == "output_average" and peer_rule == threshold_peer_rule:
        return list(thresholds)
    return [0.0]


def add_common_sweep_args(
    parser: argparse.ArgumentParser,
    *,
    base_config: Path,
    default_label: str,
    toy_name: str,
    default_seeds: Sequence[int] = (1, 2, 3, 4, 5),
    default_epochs: int | None = None,
    default_alphas: Sequence[float] | None = (0.0, 0.25, 0.5),
    default_config_dir: Path | None = None,
    peer_rule_choices: Sequence[str] = ("none", "output_similarity"),
    epochs_help: str = "Optional epochs per run; defaults to the base config value.",
    peer_rules_help: str | None = None,
    threshold_argument: str = "--thresholds",
) -> None:
    parser.add_argument(
        "--base-config",
        type=Path,
        default=base_config,
        help=f"Base {toy_name} YAML config.",
    )
    parser.add_argument(
        "--label",
        default=default_label,
        help="Sweep label used for generated configs and summaries.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(default_seeds),
        help="Seeds for each sweep point.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_epochs,
        help=epochs_help,
    )
    parser.add_argument(
        "--mixers",
        nargs="+",
        default=["none", "output_average"],
        choices=["none", "output_average"],
        help="Coordination mixers to sweep.",
    )
    parser.add_argument(
        "--peer-rules",
        nargs="+",
        default=None,
        choices=list(peer_rule_choices),
        help=peer_rules_help
        or (
            "Optional peer rules to sweep. Defaults to none for mixer=none and "
            "output_similarity for output_average."
        ),
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None if default_alphas is None else list(default_alphas),
        help="Social influence strengths for output_average.",
    )
    parser.add_argument(
        threshold_argument,
        type=float,
        nargs="+",
        default=[0.0],
        help="Output-similarity thresholds to sweep.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=default_config_dir,
        help="Directory for generated configs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Directory for summary CSV files.",
    )


def iter_coordination_sweep(
    toy: str,
    mixers: Sequence[str],
    peer_rules: Sequence[str] | None,
    alphas: Sequence[float],
    thresholds: Sequence[float],
    *,
    social_default: str = "output_similarity",
) -> list[CoordinationSweepPoint]:
    points: list[CoordinationSweepPoint] = []
    for mixer in mixers:
        for peer_rule in effective_peer_rules(
            mixer,
            peer_rules,
            social_default=social_default,
        ):
            if not supports_sweep_coordination(toy, mixer, peer_rule):
                continue
            for alpha in effective_alphas(mixer, alphas):
                for threshold in effective_thresholds(mixer, peer_rule, thresholds):
                    points.append(
                        CoordinationSweepPoint(
                            mixer=mixer,
                            peer_rule=peer_rule,
                            alpha=alpha,
                            threshold=threshold,
                        )
                    )
    return points


def iter_parameter_grid(
    dimensions: Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    names = list(dimensions)
    value_sets = [list(dimensions[name]) for name in names]
    return [
        dict(zip(names, values, strict=True))
        for values in product(*value_sets)
    ]


def toy_display_name(toy: str) -> str:
    try:
        return capability_toy_display_name(toy)
    except KeyError:
        pass
    if toy.startswith("toy") and toy[3:].isdigit():
        return f"Toy {int(toy[3:])}"
    return toy


def supports_sweep_coordination(toy: str, mixer: str, peer_rule: str) -> bool:
    return supports_coordination(toy, mixer, peer_rule)


def ensure_supported_coordination(toy: str, mixer: str, peer_rule: str) -> None:
    if not supports_sweep_coordination(toy, mixer, peer_rule):
        raise ValueError(
            f"Unsupported {toy_display_name(toy)} coordination: {mixer}/{peer_rule}"
        )


def prepare_sweep_case(
    *,
    base: dict[str, Any],
    toy: str,
    mixer: str,
    peer_rule: str,
    alpha: float,
    seed: int,
    epochs: int | None,
    coordination_threshold: float = 0.0,
) -> PreparedSweepCase:
    ensure_supported_coordination(toy, mixer, peer_rule)

    case = deepcopy(base)
    effective_alpha = alpha if mixer == "output_average" else 0.0
    effective_threshold = (
        coordination_threshold
        if mixer == "output_average" and peer_rule == "output_similarity"
        else 0.0
    )

    case["run"]["seed"] = seed
    if epochs is not None:
        case["simulation"]["epochs"] = epochs

    coordination = case["model"]["coordination"]
    coordination["mixer"] = mixer
    coordination["peer_rule"] = peer_rule
    coordination["alpha"] = effective_alpha
    coordination["threshold"] = effective_threshold

    return PreparedSweepCase(
        case=case,
        effective_alpha=effective_alpha,
        effective_threshold=effective_threshold,
    )


def write_sweep_case_config(case: dict[str, Any], config_dir: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    seed = int(case["run"]["seed"])
    config_path = config_dir / f"{case['run']['name']}_seed{seed:02d}.yaml"
    config_path.write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    return config_path


def write_prepared_sweep_case_config(
    prepared: PreparedSweepCase,
    *,
    run_name: str,
    config_dir: Path,
    updates: Mapping[str, Any] | None = None,
    mutate_case: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    case = prepared.case
    case["run"]["name"] = run_name
    if updates is not None:
        apply_config_updates(case, updates)
    if mutate_case is not None:
        mutate_case(case)
    return write_sweep_case_config(case, config_dir)


def set_config_value(
    config: MutableMapping[str, Any],
    path: str,
    value: Any,
) -> None:
    if not path:
        raise ValueError("Config update path must not be empty")

    parts = path.split(".")
    current: Any = config
    for part in parts[:-1]:
        if isinstance(current, MutableMapping):
            current = current[part]
        elif isinstance(current, MutableSequence):
            current = current[_path_list_index(part, path)]
        else:
            raise TypeError(f"Path segment {part!r} in {path!r} is not settable")

    final_part = parts[-1]
    if isinstance(current, MutableMapping):
        current[final_part] = value
    elif isinstance(current, MutableSequence):
        current[_path_list_index(final_part, path)] = value
    else:
        raise TypeError(f"Path segment {final_part!r} in {path!r} is not settable")


def _path_list_index(part: str, path: str) -> int:
    try:
        return int(part)
    except ValueError as exc:
        raise TypeError(
            f"Path segment {part!r} in {path!r} must be an integer list index"
        ) from exc


def apply_config_updates(
    config: MutableMapping[str, Any],
    updates: Mapping[str, Any],
) -> None:
    for path, value in updates.items():
        set_config_value(config, path, value)


def build_result_row(
    fieldnames: Sequence[str],
    values: Mapping[str, Any],
    *,
    aliases: Mapping[str, str] | None = None,
    stringify_fields: Sequence[str] = ("run_dir",),
) -> dict[str, object]:
    field_aliases = {
        "coordination_mixer": "mixer",
        "coordination_peer_rule": "peer_rule",
    }
    if aliases is not None:
        field_aliases.update(aliases)

    string_fields = set(stringify_fields)
    row: dict[str, object] = {}
    for field in fieldnames:
        value_key = field_aliases.get(field, field)
        value = values[value_key]
        row[field] = str(value) if field in string_fields else value
    return row


def read_final_aggregate_metrics(
    run_dir: Path,
    keys: Sequence[str] = DEFAULT_FINAL_AGGREGATE_METRIC_KEYS,
    *,
    filename: str = "aggregate_metrics.csv",
) -> dict[str, float]:
    path = run_dir / filename
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    final = rows[-1]
    metrics: dict[str, float] = {}
    for key in keys:
        value = final.get(key, "")
        if value == "":
            continue
        metrics[key] = float(value)
    return metrics


def write_summary_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    resolved_fieldnames = _with_present_nabm_fields(fieldnames, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in resolved_fieldnames})


def non_overwriting_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unused path for {path}")


def resolve_sweep_output_paths(
    results_dir: Path,
    label: str,
    *,
    grouped_markdown: bool = False,
    avoid_overwrite: bool = False,
) -> SweepOutputPaths:
    summary_path = results_dir / f"{label}_summary.csv"
    grouped_summary_path = results_dir / f"{label}_grouped_summary.csv"
    grouped_markdown_path = (
        results_dir / f"{label}_grouped_summary.md" if grouped_markdown else None
    )
    if not avoid_overwrite:
        return SweepOutputPaths(
            summary_path=summary_path,
            grouped_summary_path=grouped_summary_path,
            grouped_markdown_path=grouped_markdown_path,
        )

    return SweepOutputPaths(
        summary_path=non_overwriting_path(summary_path),
        grouped_summary_path=non_overwriting_path(grouped_summary_path),
        grouped_markdown_path=(
            non_overwriting_path(grouped_markdown_path)
            if grouped_markdown_path is not None
            else None
        ),
    )


def build_grouped_summary(
    rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
    aggregations: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    resolved_group_fields = _with_present_nabm_fields(group_fields, rows)
    return (
        frame.groupby(resolved_group_fields, dropna=False, as_index=False)
        .agg(**dict(aggregations))
        .fillna("")
    )


def write_summary_csv_for_spec(
    path: Path,
    rows: Sequence[dict[str, Any]],
    output_spec: SweepOutputSpec,
) -> None:
    write_summary_csv(path, rows, output_spec.summary_fields)


def build_grouped_summary_for_spec(
    rows: Sequence[dict[str, Any]],
    output_spec: SweepOutputSpec,
) -> pd.DataFrame:
    return build_grouped_summary(
        rows,
        output_spec.group_fields,
        output_spec.aggregations,
    )


def make_sweep_output_helpers(
    output_spec: SweepOutputSpec,
) -> tuple[
    Callable[[Path, Sequence[dict[str, Any]]], None],
    Callable[[Sequence[dict[str, Any]]], pd.DataFrame],
]:
    def write_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
        write_summary_csv_for_spec(path, rows, output_spec)

    def build_grouped(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
        return build_grouped_summary_for_spec(rows, output_spec)

    return write_summary, build_grouped


def write_sweep_outputs(
    *,
    summary_path: Path,
    grouped_summary_path: Path,
    rows: Sequence[dict[str, Any]],
    summary_fields: Sequence[str],
    group_fields: Sequence[str],
    aggregations: Mapping[str, tuple[str, str]],
    grouped_markdown_path: Path | None = None,
    grouped_markdown_writer: Callable[[Path, str, pd.DataFrame], None] | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    write_summary_csv(summary_path, rows, summary_fields)
    grouped_summary_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = build_grouped_summary(rows, group_fields, aggregations)
    grouped.to_csv(grouped_summary_path, index=False)
    if grouped_markdown_writer is not None:
        if grouped_markdown_path is None:
            raise ValueError("grouped_markdown_path is required for Markdown output")
        if label is None:
            raise ValueError("label is required for Markdown output")
        grouped_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        grouped_markdown_writer(grouped_markdown_path, label, grouped)
        _append_nabm_markdown(grouped_markdown_path, rows)
    return grouped


def extract_domain_metrics(result: Any, keys: Sequence[str]) -> dict[str, Any]:
    metrics = result.domain_metrics
    return {key: metrics[key] for key in keys}


def extract_sweep_run_fields(config: Any, result: Any) -> dict[str, Any]:
    return {
        "mixer": config.coordination.mixer,
        "peer_rule": config.coordination.peer_rule,
        "alpha": config.coordination.alpha,
        "coordination_threshold": config.coordination.threshold,
        "seed": config.run.seed,
        "epochs": config.simulation.epochs,
        "run_dir": result.run_dir,
        "final_fragmentation_components": result.final_fragmentation_components,
    }


def extract_attr_values(source: Any, paths: Mapping[str, str]) -> dict[str, Any]:
    return {key: resolve_attr_path(source, path) for key, path in paths.items()}


def resolve_attr_path(source: Any, path: str) -> Any:
    value = source
    for part in path.split("."):
        if isinstance(value, Mapping):
            value = value[part]
        else:
            value = getattr(value, part)
    return value


def build_sweep_row(
    row_builder: Callable[..., dict[str, Any]],
    *,
    base_fields: Mapping[str, Any],
    config: Any,
    result: Any,
    metric_keys: Sequence[str],
    config_value_paths: Mapping[str, str] | None = None,
    result_value_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    row_values = dict(base_fields)
    if config_value_paths is not None:
        row_values.update(extract_attr_values(config, config_value_paths))
    row_values.update(extract_sweep_run_fields(config, result))
    row_values.update(extract_domain_metrics(result, metric_keys))
    if result_value_paths is not None:
        row_values.update(extract_attr_values(result, result_value_paths))
    row = row_builder(**row_values)
    result_toy = getattr(result, "toy", None)
    if result_toy is None:
        return row
    return with_sweep_capability_metadata(row, result_toy)


def with_sweep_capability_metadata(row: Mapping[str, Any], toy: str) -> dict[str, Any]:
    return {**dict(row), **sweep_capability_metadata(toy)}


def _with_present_nabm_fields(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    fields = list(fieldnames)
    present_fields = {
        field
        for row in rows
        for field in NABM_ARTIFACT_FIELDS
        if field in row
    }
    for field in NABM_ARTIFACT_FIELDS:
        if field in present_fields and field not in fields:
            fields.append(field)
    return fields


def _append_nabm_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows or not all(field in rows[0] for field in NABM_ARTIFACT_FIELDS):
        return
    unique_rows = {
        tuple(str(row.get(field, "")) for field in NABM_ARTIFACT_FIELDS)
        for row in rows
    }
    lines = [
        "",
        "## NABM Claim Metadata",
        "",
        "| NABM Status | Neural Role | Social Channels | Reference Policies |",
        "| --- | --- | --- | --- |",
    ]
    for status, role, channels, policies in sorted(unique_rows):
        lines.append(
            f"| `{status}` | {role} | `{channels}` | `{policies}` |"
        )
    existing = path.read_text(encoding="utf-8")
    path.write_text(f"{existing}\n" + "\n".join(lines), encoding="utf-8")


def print_sweep_progress(message: str) -> None:
    print(message, flush=True)


def run_sweep_cases(
    *,
    label: str,
    base: dict[str, Any],
    config_dir: Path,
    domain_points: Sequence[Mapping[str, Any]],
    coordination_points: Sequence[CoordinationSweepPoint],
    seeds: Sequence[int],
    epochs: int | None,
    write_case_config: Callable[..., Path],
    load_config: Callable[[Path], Any],
    run_case: Callable[..., Any],
    row_builder: Callable[..., dict[str, Any]],
    metric_keys: Sequence[str],
    config_value_paths: Mapping[str, str] | None = None,
    result_value_paths: Mapping[str, str] | None = None,
    should_run: Callable[[Mapping[str, Any], CoordinationSweepPoint], bool]
    | None = None,
    progress: Callable[[str], None] | None = print_sweep_progress,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain_point in domain_points:
        domain_fields = dict(domain_point)
        for coordination in coordination_points:
            if should_run is not None and not should_run(
                domain_fields,
                coordination,
            ):
                continue
            for seed in seeds:
                config_path = write_case_config(
                    base=base,
                    label=label,
                    **domain_fields,
                    mixer=coordination.mixer,
                    peer_rule=coordination.peer_rule,
                    alpha=coordination.alpha,
                    seed=seed,
                    epochs=epochs,
                    config_dir=config_dir,
                    coordination_threshold=coordination.threshold,
                )
                config = load_config(config_path)
                result = run_case(config=config, config_path=config_path)
                rows.append(
                    build_sweep_row(
                        row_builder,
                        base_fields={"label": label, **domain_fields},
                        config=config,
                        result=result,
                        metric_keys=metric_keys,
                        config_value_paths=config_value_paths,
                        result_value_paths=result_value_paths,
                    )
                )
                if progress is not None:
                    progress(
                        f"finished {config.run.name} seed={seed} "
                        f"run_dir={result.run_dir}"
                    )
    return rows


def run_explicit_sweep_cases(
    *,
    label: str,
    base: dict[str, Any],
    config_dir: Path,
    sweep_points: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    epochs: int | None,
    write_case_config: Callable[..., Path],
    load_config: Callable[[Path], Any],
    run_case: Callable[..., Any],
    row_builder: Callable[..., dict[str, Any]],
    metric_keys: Sequence[str],
    config_value_paths: Mapping[str, str] | None = None,
    result_value_paths: Mapping[str, str] | None = None,
    progress: Callable[[str], None] | None = print_sweep_progress,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sweep_point in sweep_points:
        point_fields = dict(sweep_point)
        for seed in seeds:
            config_path = write_case_config(
                base=base,
                label=label,
                **point_fields,
                seed=seed,
                epochs=epochs,
                config_dir=config_dir,
            )
            config = load_config(config_path)
            result = run_case(config=config, config_path=config_path)
            rows.append(
                build_sweep_row(
                    row_builder,
                    base_fields={"label": label, **point_fields},
                    config=config,
                    result=result,
                    metric_keys=metric_keys,
                    config_value_paths=config_value_paths,
                    result_value_paths=result_value_paths,
                )
            )
            if progress is not None:
                progress(
                    f"finished {config.run.name} seed={seed} "
                    f"run_dir={result.run_dir}"
                )
    return rows


def run_case_points(
    *,
    points: Iterable[Any],
    run_point: Callable[[Any], tuple[dict[str, Any], str]],
    progress: Callable[[str], None] | None = print_sweep_progress,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in points:
        row, message = run_point(point)
        rows.append(row)
        if progress is not None:
            progress(message)
    return rows


def run_sweep_from_args(
    *,
    args: Any,
    toy: str,
    output_spec: SweepOutputSpec,
    domain_points_builder: Callable[[dict[str, Any], Any], Sequence[Mapping[str, Any]]],
    thresholds: Sequence[float],
    write_case_config: Callable[..., Path],
    load_config: Callable[[Path], Any],
    run_case: Callable[..., Any],
    row_builder: Callable[..., dict[str, Any]],
    alphas: Sequence[float] | None = None,
    coordination_social_default: str = "output_similarity",
    should_run: Callable[[Mapping[str, Any], CoordinationSweepPoint], bool]
    | None = None,
    avoid_overwrite: bool = False,
    progress: Callable[[str], None] | None = print_sweep_progress,
) -> SweepRunArtifacts:
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config_dir = args.config_dir or Path("experiments/configs/generated") / args.label
    alpha_values = args.alphas if alphas is None else alphas
    rows = run_sweep_cases(
        label=args.label,
        base=base,
        config_dir=config_dir,
        domain_points=domain_points_builder(base, args),
        coordination_points=iter_coordination_sweep(
            toy,
            args.mixers,
            args.peer_rules,
            alpha_values,
            thresholds,
            social_default=coordination_social_default,
        ),
        seeds=args.seeds,
        epochs=args.epochs,
        write_case_config=write_case_config,
        load_config=load_config,
        run_case=run_case,
        row_builder=row_builder,
        metric_keys=output_spec.metric_keys,
        config_value_paths=output_spec.config_value_paths,
        result_value_paths=output_spec.result_value_paths,
        should_run=should_run,
        progress=progress,
    )
    rows = [with_sweep_capability_metadata(row, toy) for row in rows]

    output_paths = resolve_sweep_output_paths(
        args.results_dir,
        args.label,
        avoid_overwrite=avoid_overwrite,
    )
    write_sweep_outputs(
        summary_path=output_paths.summary_path,
        grouped_summary_path=output_paths.grouped_summary_path,
        rows=rows,
        summary_fields=output_spec.summary_fields,
        group_fields=output_spec.group_fields,
        aggregations=output_spec.aggregations,
    )
    if progress is not None:
        progress(f"summary={output_paths.summary_path}")
        progress(f"grouped_summary={output_paths.grouped_summary_path}")

    return SweepRunArtifacts(
        rows=rows,
        config_dir=config_dir,
        summary_path=output_paths.summary_path,
        grouped_summary_path=output_paths.grouped_summary_path,
        grouped_markdown_path=output_paths.grouped_markdown_path,
    )


def run_explicit_sweep_from_args(
    *,
    args: Any,
    output_spec: SweepOutputSpec,
    sweep_points_builder: Callable[[dict[str, Any], Any], Sequence[Mapping[str, Any]]],
    write_case_config: Callable[..., Path],
    load_config: Callable[[Path], Any],
    run_case: Callable[..., Any],
    row_builder: Callable[..., dict[str, Any]],
    label: str | None = None,
    grouped_markdown_writer: Callable[[Path, str, pd.DataFrame], None] | None = None,
    avoid_overwrite: bool = False,
    toy: str | None = None,
    progress: Callable[[str], None] | None = print_sweep_progress,
) -> SweepRunArtifacts:
    resolved_label = label if label is not None else args.label
    if not resolved_label:
        raise ValueError("Sweep label is required")

    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config_dir = args.config_dir or Path("experiments/configs/generated") / resolved_label
    rows = run_explicit_sweep_cases(
        label=resolved_label,
        base=base,
        config_dir=config_dir,
        sweep_points=sweep_points_builder(base, args),
        seeds=args.seeds,
        epochs=args.epochs,
        write_case_config=write_case_config,
        load_config=load_config,
        run_case=run_case,
        row_builder=row_builder,
        metric_keys=output_spec.metric_keys,
        config_value_paths=output_spec.config_value_paths,
        result_value_paths=output_spec.result_value_paths,
        progress=progress,
    )
    if toy is not None:
        rows = [with_sweep_capability_metadata(row, toy) for row in rows]

    output_paths = resolve_sweep_output_paths(
        args.results_dir,
        resolved_label,
        grouped_markdown=grouped_markdown_writer is not None,
        avoid_overwrite=avoid_overwrite,
    )
    write_sweep_outputs(
        summary_path=output_paths.summary_path,
        grouped_summary_path=output_paths.grouped_summary_path,
        rows=rows,
        summary_fields=output_spec.summary_fields,
        group_fields=output_spec.group_fields,
        aggregations=output_spec.aggregations,
        grouped_markdown_path=output_paths.grouped_markdown_path,
        grouped_markdown_writer=grouped_markdown_writer,
        label=resolved_label,
    )
    if progress is not None:
        progress(f"summary={output_paths.summary_path}")
        progress(f"grouped_summary={output_paths.grouped_summary_path}")
        if output_paths.grouped_markdown_path is not None:
            progress(f"grouped_markdown={output_paths.grouped_markdown_path}")

    return SweepRunArtifacts(
        rows=rows,
        config_dir=config_dir,
        summary_path=output_paths.summary_path,
        grouped_summary_path=output_paths.grouped_summary_path,
        grouped_markdown_path=output_paths.grouped_markdown_path,
    )


def run_point_sweep_from_args(
    *,
    args: Any,
    output_spec: SweepOutputSpec,
    rows_builder: Callable[
        [dict[str, Any], Any, str, Path, Callable[[str], None] | None],
        list[dict[str, Any]],
    ],
    grouped_markdown_writer: Callable[[Path, str, pd.DataFrame], None] | None = None,
    avoid_overwrite: bool = False,
    toy: str | None = None,
    progress: Callable[[str], None] | None = print_sweep_progress,
) -> SweepRunArtifacts:
    if not args.label:
        raise ValueError("Sweep label is required")

    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config_dir = args.config_dir or Path("experiments/configs/generated") / args.label
    rows = rows_builder(base, args, args.label, config_dir, progress)
    if toy is not None:
        rows = [with_sweep_capability_metadata(row, toy) for row in rows]
    output_paths = resolve_sweep_output_paths(
        args.results_dir,
        args.label,
        grouped_markdown=grouped_markdown_writer is not None,
        avoid_overwrite=avoid_overwrite,
    )
    write_sweep_outputs(
        summary_path=output_paths.summary_path,
        grouped_summary_path=output_paths.grouped_summary_path,
        rows=rows,
        summary_fields=output_spec.summary_fields,
        group_fields=output_spec.group_fields,
        aggregations=output_spec.aggregations,
        grouped_markdown_path=output_paths.grouped_markdown_path,
        grouped_markdown_writer=grouped_markdown_writer,
        label=args.label,
    )
    if progress is not None:
        progress(f"summary={output_paths.summary_path}")
        progress(f"grouped_summary={output_paths.grouped_summary_path}")
        if output_paths.grouped_markdown_path is not None:
            progress(f"grouped_markdown={output_paths.grouped_markdown_path}")

    return SweepRunArtifacts(
        rows=rows,
        config_dir=config_dir,
        summary_path=output_paths.summary_path,
        grouped_summary_path=output_paths.grouped_summary_path,
        grouped_markdown_path=output_paths.grouped_markdown_path,
    )
