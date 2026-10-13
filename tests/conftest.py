"""Collection rules for tests that cannot run from a clean clone.

Two registries, both explicit:

- ``machine_local_tests.txt`` lists tests that read machine-local artifacts:
  private outputs under ``/ssd`` or the git-ignored ``experiments/runs``.
  They run on the workstation and are skipped where ``/ssd`` is absent, as in
  CI. Many of their files are frozen precommit paths, so the rule lives here
  rather than in each file.
- ``HISTORICAL_BINDING_DRIFT`` lists tests of frozen prototypes whose source
  bindings name package files that a later accepted decision changed. The
  prototype correctly refuses to run at HEAD; reproducing it needs its freeze
  commit. Each entry is a strict xfail, so it fails loudly if the binding ever
  holds again.
"""

from __future__ import annotations

from pathlib import Path

import pytest


TESTS = Path(__file__).resolve().parent
MACHINE_LOCAL_ROOT = Path("/ssd")
MACHINE_LOCAL_TESTS = frozenset(
    line.strip()
    for line in (TESTS / "machine_local_tests.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
)
HISTORICAL_BINDING_DRIFT = {
    "tests/test_pivot_delayed_feedback_learning_prototype_v1.py::test_source_bindings_and_exclusive_output_receipt": (
        "the frozen prototype (2026-08-07) binds src/neural_abm/api_lite.py, which "
        "Decision 0016 changed at 29bd59c; rerun it from its freeze commit"
    ),
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    machine_local_absent = not MACHINE_LOCAL_ROOT.is_dir()
    for item in items:
        if machine_local_absent and item.nodeid in MACHINE_LOCAL_TESTS:
            item.add_marker(
                pytest.mark.skip(reason="reads machine-local artifacts absent here")
            )
        reason = HISTORICAL_BINDING_DRIFT.get(item.nodeid)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=reason, raises=RuntimeError, strict=True))
