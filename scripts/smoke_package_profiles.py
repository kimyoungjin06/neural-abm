"""Smoke-test package dependency profiles from a built wheel.

This script is intentionally separate from pytest because the torch/research/full
profiles may install large dependencies. Run it before release packaging changes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ("default", "torch", "research", "full")

SMOKE_CODE = {
    "default": r"""
import importlib.abc
import importlib.util
import json
import sys
from importlib.metadata import distribution

import numpy as np

torch_installed = importlib.util.find_spec("torch") is not None
default_requires = [
    requirement
    for requirement in distribution("neural-abm").requires or []
    if "extra ==" not in requirement
]
torch_required_by_default = any(
    requirement.lower().startswith("torch") for requirement in default_requires
)

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch import blocked for default profile smoke")
        return None

sys.meta_path.insert(0, BlockTorch())

import neural_abm
import neural_abm.api_lite as api_lite

channel = api_lite.SocialChannel(
    name="readiness",
    kind=api_lite.SCALAR_PROBABILITY_CHANNEL,
    commit_mode="readiness_commit",
)
mixed = api_lite.mix_scalar_probabilities(
    values=np.asarray([0.0, 1.0], dtype=np.float64),
    peer_ids=[[1], [0]],
    alpha=0.5,
    channel=channel.name,
    commit_mode=channel.commit_mode,
)
commit = api_lite.CommitReport.from_mix_result(mixed, committed_agent_ids=[0, 1])

class Adapter:
    def update(self, scale):
        return api_lite.LocalUpdateReport(losses=[scale], active_agent_ids=[0])

local_report = api_lite.NABMLocalStep(Adapter()).run(0.25)
try:
    api_lite.SocialChannel(
        name="tensor",
        kind="tensor",
        commit_mode="tensor_commit",
    )
except ValueError:
    rejected_tensor_channel = True
else:
    rejected_tensor_channel = False

print(json.dumps({
    "profile": "default",
    "version": neural_abm.__version__,
    "default_requires": default_requires,
    "torch_required_by_default": torch_required_by_default,
    "torch_installed": torch_installed,
    "torch_loaded": "torch" in sys.modules,
    "api_lite_exports": len(api_lite.__all__),
    "commit_channel": commit.channel,
    "lite_social_channel_kinds": list(api_lite.LITE_SOCIAL_CHANNEL_KINDS),
    "local_losses": local_report.losses,
    "rejected_tensor_channel": rejected_tensor_channel,
    "social_mix_values": mixed.mixed_values.tolist(),
    "toy_catalog_count": len(api_lite.toy_catalog()),
    "taxonomy_binary_probability": list(
        api_lite.toys_by_taxonomy("output_family", "binary_probability")
    ),
    "toy10_display_name": api_lite.toy_display_name("toy10"),
}, sort_keys=True))
""",
    "torch": r"""
import json
import torch
from neural_abm.api import NABMUnit, SocialBlock, SocialChannel

print(json.dumps({
    "profile": "torch",
    "torch_version": torch.__version__.split("+")[0],
    "exports": [NABMUnit.__name__, SocialBlock.__name__, SocialChannel.__name__],
}, sort_keys=True))
""",
    "research": r"""
import json
import networkx
import numpy
import pandas
import pyarrow
import scipy
import sklearn
import torch
import tqdm

from neural_abm.config import Toy6Config
from neural_abm.evidence_matrix import EvidenceManifest
from neural_abm.toy_categorical import run_toy6

print(json.dumps({
    "profile": "research",
    "imports": [
        networkx.__name__,
        numpy.__name__,
        pandas.__name__,
        pyarrow.__name__,
        scipy.__name__,
        sklearn.__name__,
        torch.__name__,
        tqdm.__name__,
        Toy6Config.__name__,
        EvidenceManifest.__name__,
        run_toy6.__name__,
    ],
}, sort_keys=True))
""",
    "full": r"""
import json
import matplotlib
import networkx
import pandas
import pyarrow
import scipy
import sklearn
import torch
import tqdm

from neural_abm.api import NABMUnit
from neural_abm.config import Toy10Config
from neural_abm.toy_market import run_toy10

print(json.dumps({
    "profile": "full",
    "imports": [
        matplotlib.__name__,
        networkx.__name__,
        pandas.__name__,
        pyarrow.__name__,
        scipy.__name__,
        sklearn.__name__,
        torch.__name__,
        tqdm.__name__,
        NABMUnit.__name__,
        Toy10Config.__name__,
        run_toy10.__name__,
    ],
}, sort_keys=True))
""",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the wheel and smoke-test package dependency profiles.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=DEFAULT_PROFILES,
        default=list(DEFAULT_PROFILES),
        help="Profiles to smoke-test.",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        help="Existing wheel to test. If omitted, uv build creates one.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="neural-abm-profile-smoke.") as temp:
        temp_path = Path(temp)
        wheel = (args.wheel.resolve() if args.wheel else _build_wheel(temp_path))
        cache_dir = temp_path / "uv-cache"
        results = [
            _run_profile(wheel=wheel, profile=profile, cache_dir=cache_dir)
            for profile in args.profiles
        ]
    print(json.dumps({"wheel": str(wheel), "profiles": results}, indent=2))


def _build_wheel(output_dir: Path) -> Path:
    subprocess.run(
        ["uv", "build", "--out-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
    )
    wheels = sorted(output_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"uv build did not produce a wheel in {output_dir}")
    return wheels[0]


def _run_profile(wheel: Path, profile: str, cache_dir: Path) -> dict[str, object]:
    requirement = str(wheel) if profile == "default" else f"{wheel}[{profile}]"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--with",
            requirement,
            "python",
            "-c",
            SMOKE_CODE[profile],
        ],
        cwd=wheel.parent,
        env={**os.environ, "UV_CACHE_DIR": str(cache_dir)},
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"profile {profile!r} did not print a JSON payload")
    return json.loads(lines[-1])


if __name__ == "__main__":
    main()
