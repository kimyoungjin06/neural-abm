"""Collection rules for tests that cannot run from a clean clone.

Five registries, all explicit:

- ``duckdb_overlay_tests.txt`` lists test modules that import the
  repository-only DuckDB overlay, which is not a declared dependency. They are
  not collected where ``duckdb`` is not installed, as in CI.
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
- ``PYTHON_VERSION_SENSITIVE`` lists tests whose frozen code behaves
  differently below a Python version. They are strict xfails there.
- ``HOST_SENSITIVE`` lists tests that compare last-bit floating-point results
  recorded on the workstation, whose NumPy has no AVX-512 dispatch. On hosts
  where NumPy dispatches AVX-512 kernels, they are non-strict xfails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


TESTS = Path(__file__).resolve().parent
MACHINE_LOCAL_ROOT = Path("/ssd")


def _registry(name: str) -> frozenset[str]:
    lines = (TESTS / name).read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip() for line in lines if line.strip() and not line.startswith("#"))


MACHINE_LOCAL_TESTS = _registry("machine_local_tests.txt")
DUCKDB_OVERLAY_TESTS = _registry("duckdb_overlay_tests.txt")
collect_ignore = (
    sorted(Path(path).name for path in DUCKDB_OVERLAY_TESTS)
    if importlib.util.find_spec("duckdb") is None
    else []
)
HISTORICAL_BINDING_DRIFT = {
    "tests/test_pivot_delayed_feedback_learning_prototype_v1.py::test_source_bindings_and_exclusive_output_receipt": (
        "the frozen prototype (2026-08-07) binds src/neural_abm/api_lite.py, which "
        "Decision 0016 changed at 29bd59c; rerun it from its freeze commit"
    ),
}


PYTHON_VERSION_SENSITIVE = {
    "tests/test_openalex_work_side_author_history_fixture_v2.py::test_deep_json_yaml_and_overlong_integer_are_controlled_validation_failures": (
        (3, 12),
        "below Python 3.12 the json decoder raises RecursionError on deep nesting "
        "before the frozen verifier's own depth check, so its controlled failure "
        "carries a different message",
    ),
}
HOST_SENSITIVE = {
    "tests/test_pivot_matched_marginal_stop_result_v1.py::test_recorded_scalar_drift_reproduces_from_frozen_v1_code": (
        "the recorded last-bit drift comes from NumPy without AVX-512 dispatch; "
        "AVX-512 kernels change which replicates differ"
    ),
}


def _numpy_dispatches_avx512() -> bool:
    try:
        from numpy._core._multiarray_umath import __cpu_features__
    except ImportError:
        return False
    return bool(__cpu_features__.get("AVX512F"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    machine_local_absent = not MACHINE_LOCAL_ROOT.is_dir()
    avx512 = _numpy_dispatches_avx512()
    for item in items:
        if machine_local_absent and item.nodeid in MACHINE_LOCAL_TESTS:
            item.add_marker(
                pytest.mark.skip(reason="reads machine-local artifacts absent here")
            )
        reason = HISTORICAL_BINDING_DRIFT.get(item.nodeid)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=reason, raises=RuntimeError, strict=True))
        version_rule = PYTHON_VERSION_SENSITIVE.get(item.nodeid)
        if version_rule is not None and sys.version_info < version_rule[0]:
            item.add_marker(pytest.mark.xfail(reason=version_rule[1], strict=True))
        reason = HOST_SENSITIVE.get(item.nodeid)
        if reason is not None and avx512:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
