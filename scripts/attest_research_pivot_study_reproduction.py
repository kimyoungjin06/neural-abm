"""Attest that the tracked Study 1 and Study 2 results reproduce from this tree.

The tracked results under docs/case-studies/researcher-pivot/data are bound by
hash in frozen contracts and in the release inspector, so their provenance is
never refreshed in place. Their claim-ledger tests accept the source snapshot
of a later reproduction instead: a record under experiments/evidence/results
named research_pivot_study_reproduction_<date>.json.

This script regenerates both studies from the current tree, checks that each
matches its tracked file in everything but provenance, and writes such a
record. Run it after the last change to src/neural_abm, pyproject.toml,
uv.lock, or the study script, and commit the record with that change.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "case-studies" / "researcher-pivot" / "data"
STUDIES = {
    "study_results.json": ["examples/research_pivot_study.py", "--sweeps"],
    "learning_study_results.json": ["examples/research_pivot_learning_study.py"],
}


def _without_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "provenance"}


def attest(purpose: str) -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()
    studies: dict[str, Any] = {}
    commands: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name, command in STUDIES.items():
            output = Path(scratch) / name
            subprocess.run([sys.executable, *command, "--output", str(output)], cwd=ROOT, check=True)
            tracked_path = DATA / name
            tracked = json.loads(tracked_path.read_text(encoding="utf-8"))
            regenerated = json.loads(output.read_text(encoding="utf-8"))
            identical = _without_provenance(tracked) == _without_provenance(regenerated)
            if not identical:
                raise RuntimeError(f"{name} no longer reproduces its tracked result")
            provenance = regenerated["provenance"]
            studies[name] = {
                "tracked_path": tracked_path.relative_to(ROOT).as_posix(),
                "tracked_sha256": hashlib.sha256(tracked_path.read_bytes()).hexdigest(),
                "non_provenance_identical": identical,
                "regenerated_source_snapshot": provenance["source_snapshot"],
                "python": provenance["python"],
                "numpy": provenance["numpy"],
                "package_version": provenance["package_version"],
            }
            commands[name] = "python " + " ".join(command) + " --output <tmp>"
    return {
        "schema_version": "research_pivot_study_reproduction.v1",
        "date": datetime.date.today().isoformat(),
        "purpose": purpose,
        "regenerated_at_commit": head,
        "regenerated_from_worktree": "the commit above plus the changes committed with this record",
        "commands": commands,
        "studies": studies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args()
    record = attest(args.purpose)
    args.output.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in study.items() if k != "regenerated_source_snapshot"}
                      for name, study in record["studies"].items()}, indent=1))


if __name__ == "__main__":
    main()
