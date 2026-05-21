"""Data structures and helpers for evidence profiling."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NumericSummary:
    n: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    std: float | None


@dataclass
class VariantProfile:
    variant: str
    group: str
    role: str
    status: str
    expected_seed_count: int
    observed_seed_count: int
    final_ceiling_hits: int
    ever_ceiling_hits: int
    missing_seeds: list[int] = field(default_factory=list)
    failed_seeds: list[int] = field(default_factory=list)
    metric: NumericSummary = field(
        default_factory=lambda: NumericSummary(
            n=0,
            mean=None,
            minimum=None,
            maximum=None,
            std=None,
        )
    )
    time_to_ceiling: NumericSummary = field(
        default_factory=lambda: NumericSummary(
            n=0,
            mean=None,
            minimum=None,
            maximum=None,
            std=None,
        )
    )
    terminal_ceiling_rate: NumericSummary = field(
        default_factory=lambda: NumericSummary(
            n=0,
            mean=None,
            minimum=None,
            maximum=None,
            std=None,
        )
    )
    late_flip_rate: NumericSummary = field(
        default_factory=lambda: NumericSummary(
            n=0,
            mean=None,
            minimum=None,
            maximum=None,
            std=None,
        )
    )
    details: dict[str, Any] = field(default_factory=dict)
    issue_codes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class CaseProfile:
    case: str
    toy: str
    status: str
    primary_metric: str
    baseline_group: str
    nabm_group: str
    main_group: str
    best_main_variant: str | None
    best_baseline_variant: str | None
    variants: list[VariantProfile]
    issue_codes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class EvidenceProfile:
    label: str
    status: str
    passed: bool | None
    manifest_path: str
    runs_path: str
    gate_summary_path: str
    cases: list[CaseProfile]
    input_validation: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def optional_float(value: Any) -> float | None:
    if value == "" or value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def optional_int(value: Any) -> int | None:
    number = optional_float(value)
    if number is None:
        return None
    return int(number)


def optional_bool(value: Any) -> bool | None:
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


def numeric_summary(values: Iterable[Any]) -> NumericSummary:
    numbers = [number for value in values if (number := optional_float(value)) is not None]
    if not numbers:
        return NumericSummary(n=0, mean=None, minimum=None, maximum=None, std=None)
    return NumericSummary(
        n=len(numbers),
        mean=float(statistics.fmean(numbers)),
        minimum=float(min(numbers)),
        maximum=float(max(numbers)),
        std=float(statistics.stdev(numbers)) if len(numbers) > 1 else 0.0,
    )


def format_number(value: Any) -> str:
    number = optional_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"
