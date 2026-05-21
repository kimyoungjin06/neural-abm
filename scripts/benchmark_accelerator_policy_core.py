#!/usr/bin/env python
"""Benchmark per-agent loop inference against batched MLP policy inference."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Callable

import torch

from neural_abm.accelerator import (
    BatchedMLPParameters,
    BatchedMLPPolicyCache,
    batched_mlp_policy_probs,
    resolve_torch_device,
)
from neural_abm.toy_pd import PolicyMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: cpu, cuda, cuda:0, mps, or auto.",
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=[32, 128, 512],
        help="Agent counts to benchmark.",
    )
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_seconds(
    fn: Callable[[], torch.Tensor],
    *,
    repeats: int,
    device: torch.device,
) -> float:
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return time.perf_counter() - start


def warmup(fn: Callable[[], torch.Tensor], *, repeats: int, device: torch.device) -> None:
    for _ in range(repeats):
        fn()
    synchronize(device)


def make_models(
    *,
    agent_count: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    device: torch.device,
) -> list[PolicyMLP]:
    models = []
    for agent_id in range(agent_count):
        torch.manual_seed(10_000 + agent_id)
        models.append(
            PolicyMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
            ).to(device)
        )
    return models


@torch.no_grad()
def run_case(
    *,
    agent_count: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    warmup_repeats: int,
    repeats: int,
    device: torch.device,
) -> dict[str, object]:
    models = make_models(
        agent_count=agent_count,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        device=device,
    )
    observations = torch.randn(agent_count, input_dim, device=device)
    params = BatchedMLPParameters.from_models(models, device=device)
    cache = BatchedMLPPolicyCache.from_models(models, device=device)

    def per_agent_loop() -> torch.Tensor:
        return torch.stack(
            [
                torch.softmax(model(observations[index].unsqueeze(0)), dim=-1)[0]
                for index, model in enumerate(models)
            ],
            dim=0,
        )

    def batched_cached() -> torch.Tensor:
        return params.probabilities(observations)

    def batched_policy_cache() -> torch.Tensor:
        return cache.probabilities(observations)

    def batched_cache_refresh() -> torch.Tensor:
        cache.refresh(models)
        return cache.probabilities(observations)

    def batched_restack() -> torch.Tensor:
        return batched_mlp_policy_probs(models, observations, device=device)

    warmup(per_agent_loop, repeats=warmup_repeats, device=device)
    warmup(batched_cached, repeats=warmup_repeats, device=device)
    warmup(batched_policy_cache, repeats=warmup_repeats, device=device)
    warmup(batched_cache_refresh, repeats=warmup_repeats, device=device)
    warmup(batched_restack, repeats=warmup_repeats, device=device)

    loop_seconds = timed_seconds(per_agent_loop, repeats=repeats, device=device)
    cached_seconds = timed_seconds(batched_cached, repeats=repeats, device=device)
    policy_cache_seconds = timed_seconds(
        batched_policy_cache,
        repeats=repeats,
        device=device,
    )
    refresh_seconds = timed_seconds(
        batched_cache_refresh,
        repeats=repeats,
        device=device,
    )
    restack_seconds = timed_seconds(batched_restack, repeats=repeats, device=device)

    return {
        "device": str(device),
        "agent_count": agent_count,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "repeats": repeats,
        "loop_seconds": loop_seconds,
        "batched_cached_seconds": cached_seconds,
        "batched_policy_cache_seconds": policy_cache_seconds,
        "batched_cache_refresh_seconds": refresh_seconds,
        "batched_restack_seconds": restack_seconds,
        "cached_speedup": loop_seconds / cached_seconds if cached_seconds else "",
        "policy_cache_speedup": (
            loop_seconds / policy_cache_seconds if policy_cache_seconds else ""
        ),
        "cache_refresh_speedup": (
            loop_seconds / refresh_seconds if refresh_seconds else ""
        ),
        "restack_speedup": loop_seconds / restack_seconds if restack_seconds else "",
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_torch_device(args.device)
    rows = [
        run_case(
            agent_count=agent_count,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            warmup_repeats=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        for agent_count in args.agent_counts
    ]
    if args.output is not None:
        write_rows(args.output, rows)
        print(args.output)
        return
    fieldnames = list(rows[0])
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[field]) for field in fieldnames))


if __name__ == "__main__":
    main()
