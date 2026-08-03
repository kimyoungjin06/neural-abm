from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_pivot_feedback_g_minus_1_readiness as verifier  # noqa: E402


REGISTRY_PATH = ROOT / "experiments/configs/pivot_feedback_g_minus_1_readiness_v0.yaml"


def _registry() -> dict[str, Any]:
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write_registry(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _pin_test_registry(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
) -> None:
    monkeypatch.setattr(
        verifier,
        "EXPECTED_REGISTRY_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _set_nested(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = target
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        assert isinstance(child, dict)
        current = child
    current[parts[-1]] = value


def _operator_receipts(tmp_path: Path, raw: dict[str, Any]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for binding in raw["passed_operator_control_receipts"]:
        document: dict[str, Any] = {
            "schema_version": binding["required_schema_version"],
            "status": binding["required_status"],
        }
        for dotted_path, value in binding["required_fields"].items():
            _set_nested(document, dotted_path, value)
        payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        receipt_dir = tmp_path / binding["component_id"]
        receipt_dir.mkdir()
        path = receipt_dir / binding["expected_basename"]
        path.write_bytes(payload)
        binding["sha256"] = hashlib.sha256(payload).hexdigest()
        binding["maximum_bytes"] = max(len(payload), 1)
        overrides[binding["component_id"]] = path.resolve()
    return overrides


def test_registry_is_exactly_fail_closed_and_lists_three_missing_blockers() -> None:
    raw = _registry()

    assert set(raw) == verifier.EXPECTED_TOP_LEVEL_KEYS
    assert raw["schema_version"] == verifier.REGISTRY_SCHEMA
    assert hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest() == (
        verifier.EXPECTED_REGISTRY_SHA256
    )
    assert raw["status"] == "G_MINUS_1_PENDING"
    assert raw["gate_state"] == {
        "g_minus_1_data_readiness": "PENDING",
        "pass_g_minus_1": False,
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "execution_authorized": False,
    }
    assert raw["safety_boundary"]["source_payload_access_allowed"] is False
    assert raw["safety_boundary"]["production_scan_allowed"] is False
    assert raw["safety_boundary"]["artifact_mutation_allowed"] is False

    missing = raw["missing_required_components"]
    assert {item["component_id"] for item in missing} == (
        verifier.EXPECTED_MISSING_COMPONENTS
    )
    assert all(item["production_receipt_sha256"] is None for item in missing)
    assert all(item["promotion_allowed_by_this_registry"] is False for item in missing)
    assert all(item["blocks_g0"] is True for item in missing)

    blocking = raw["blocking_repository_evidence"]
    assert {item["component_id"] for item in blocking} == (
        verifier.EXPECTED_BLOCKING_REPOSITORY_COMPONENTS
    )
    assert len(missing) + len(blocking) == 3
    assert blocking == []
    relation = next(
        item
        for item in raw["passed_repository_evidence"]
        if item["component_id"] == "work_topic_relation"
    )
    assert relation["required_status"] == (
        "PASS_CURRENT_SNAPSHOT_WORK_TOPIC_RELATION_COMPONENT_"
        "WITH_PRIMARY_EXACT_SCORE_ADJUDICATION"
    )
    assert relation["required_fields"]["adjudication.unresolved_group_count"] == 0
    assert (
        relation["required_fields"][
            "independent_verification.canonical_relation_payload_all_exact"
        ]
        is True
    )
    canonical = next(
        item
        for item in raw["passed_repository_evidence"]
        if item["component_id"] == "canonical_full"
    )
    assert canonical["required_fields"]["identifier_accounting.retained_rows"] == (
        492293206
    )
    assert (
        canonical["required_fields"]["independent_verification.parquet_validation"]
        == "full"
    )

    author = next(
        item
        for item in raw["passed_repository_evidence"]
        if item["component_id"] == "author_history"
    )
    assert author["required_schema_version"] == (
        "openalex_g1_author_history_production_full_summary.v1"
    )
    assert author["required_status"] == (
        "PASS_CURRENT_SNAPSHOT_WORK_SIDE_AUTHOR_TOKEN_LINKED_RETROSPECTIVE_"
        "HISTORY_ACCOUNTING_AND_SENSITIVITY_COMPONENT_AUTHOR_ENTITY_AND_"
        "PAST_OBSERVABILITY_UNVERIFIED"
    )
    assert author["required_fields"]["scope.production_component_pass"] is True
    assert author["required_fields"]["work_event_accounting.unmatched_authorship_slots"] == 0
    assert author["required_fields"]["gate_status.author_history"] == (
        "PASS_BOUNDED_AUTHOR_TOKEN_ACCOUNTING_ONLY"
    )
    assert author["nonpromoting_synthetic_core_evidence"] == (
        verifier.EXPECTED_AUTHOR_SYNTHETIC_EVIDENCE
    )
    citation = next(
        item
        for item in raw["passed_repository_evidence"]
        if item["component_id"] == "citation_chronology"
    )
    assert citation["required_schema_version"] == (
        "openalex_g1_citation_chronology_production_full_summary.v1"
    )
    assert citation["required_fields"]["pair_accounting.unique_pair_rows"] == (
        3042536160
    )
    assert citation["required_fields"][
        "independent_verification.full_deduplicated_row_scan"
    ] is True
    cutoff = next(
        item
        for item in raw["passed_repository_evidence"]
        if item["component_id"] == "cutoff_primitives"
    )
    assert cutoff["required_fields"]["work_metadata.rows"] == 492293206
    assert cutoff["required_fields"][
        "independent_verification.full_work_metadata_row_scan"
    ] is True
    assert cutoff["required_fields"]["gate_status.cutoff_panel"] == (
        "PENDING_G0_RESOLVED_FINAL_PANEL_SUCCESSOR"
    )
    cutoff_missing = next(
        item for item in missing if item["component_id"] == "cutoff_panel"
    )
    assert cutoff_missing["current_state"] == (
        "MISSING_G0_RESOLVED_FINAL_PANEL_SUCCESSOR"
    )
    evidence = author["nonpromoting_synthetic_core_evidence"]
    assert evidence["config_sha256"] == (
        "f0359f9b9ab7c2e5a51b1e760f94c70d51e544dc7f594d8e0512a9cc8ff216c2"
    )
    assert evidence["validator_sha256"] == (
        "6a95a633b67604da803f7405ef5ddfd20986b00518b4ebf9bae07dbffb9661de"
    )
    assert evidence["pinned_evidence_class"] == "PINNED_CHECKED_FIXTURE"
    assert evidence["generic_evidence_class"] == "UNATTESTED_GENERIC_FIXTURE"
    assert evidence["pinned_fixture_promotes_component"] is False
    assert evidence["generic_fixture_promotes_component"] is False


def test_checked_in_repository_evidence_verifies_without_operator_paths() -> None:
    report = verifier.verify_registry(REGISTRY_PATH, repo_root=ROOT)

    assert report["verification_status"] == (
        "PASS_FAIL_CLOSED_REGISTRY_REPOSITORY_EVIDENCE_ONLY"
    )
    assert report["status"] == "G_MINUS_1_PENDING"
    assert report["pass_g_minus_1"] is False
    assert report["execution_authorized"] is False
    assert report["source_payloads_opened"] == 0
    assert report["production_scans_started"] == 0
    assert len(report["repository_evidence"]) == 10
    assert all(
        item["verification"] == "PASS_EXACT_REPOSITORY_EVIDENCE"
        for item in report["repository_evidence"]
    )
    assert report["blocking_repository_evidence"] == []
    relation = next(
        item
        for item in report["repository_evidence"]
        if item["component_id"] == "work_topic_relation"
    )
    assert relation["sha256"] == (
        "76de9e32c627eafb491b0be25c342445adc9e6fcdbca040bc48d25cbb19652bc"
    )
    canonical = next(
        item
        for item in report["repository_evidence"]
        if item["component_id"] == "canonical_full"
    )
    assert canonical["sha256"] == (
        "b02fc7f38632f698a41689312e0eba9a4a70b71cdfb03ed69e1ba0a6586db565"
    )
    author = next(
        item
        for item in report["repository_evidence"]
        if item["component_id"] == "author_history"
    )
    assert author["sha256"] == (
        "125fa24dfc8c59c76867e60e7431f72f325bc6341fc84c28fad111a66dbfbb86"
    )
    citation = next(
        item
        for item in report["repository_evidence"]
        if item["component_id"] == "citation_chronology"
    )
    assert citation["sha256"] == (
        "79cfca7ce217a25e682ef617c26d42d234fe84f5f4a0fafdae19d7d16c99893a"
    )
    cutoff = next(
        item
        for item in report["repository_evidence"]
        if item["component_id"] == "cutoff_primitives"
    )
    assert cutoff["sha256"] == (
        "874698fa70edf98d62e73a7f5cb7e3d0b15ecd653535781916af4832e7ded12e"
    )
    synthetic = report["nonpromoting_synthetic_evidence"]
    assert synthetic["component_id"] == "author_history"
    runtime_matches = (
        verifier._current_runtime_fingerprint()
        == (verifier.EXPECTED_AUTHOR_SYNTHETIC_EVIDENCE["attested_runtime_fingerprint"])
    )
    if runtime_matches:
        assert synthetic["verification"] == (
            "PASS_EXACT_CONFIG_PINNED_NONPROMOTING_SYNTHETIC_ATTESTATION"
        )
        assert synthetic["canonical_stdout_reproduced"] is True
        assert synthetic["current_canonical_stdout_bytes"] == 6806
        assert synthetic["current_canonical_stdout_sha256"] == (
            "ee2ca45756f09b130f6ad00fe7b56219eaf9083429d80eb657424212c5381ef7"
        )
    else:
        assert synthetic["verification"] == "NOT_REATTESTED_RUNTIME_MISMATCH"
        assert synthetic["canonical_stdout_reproduced"] is False
    assert [item["role"] for item in synthetic["artifacts"]] == [
        "config",
        "validator",
        "fixture",
    ]
    assert synthetic["bounded_synthetic_validator_runs"] == 1
    assert synthetic["fixture_evidence_class"] == "PINNED_CHECKED_FIXTURE"
    assert synthetic["parent_evidence_class"] == (
        "CONFIG_PINNED_SYNTHETIC_PARENT_STUBS"
    )
    assert synthetic["generic_report_status"] == (
        "VALID_UNATTESTED_GENERIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_V2_MECHANICS_ONLY"
    )
    assert synthetic["generic_evidence_class"] == "UNATTESTED_GENERIC_FIXTURE"
    assert synthetic["generic_fixture_promotes_component"] is False
    assert synthetic["attested_canonical_stdout_bytes"] == 6806
    assert synthetic["attested_canonical_stdout_sha256"] == (
        "ee2ca45756f09b130f6ad00fe7b56219eaf9083429d80eb657424212c5381ef7"
    )
    assert synthetic["current_runtime_fingerprint"] == (
        verifier._current_runtime_fingerprint()
    )
    foundations = synthetic["foundation_verification"]
    assert len(foundations) == 6
    assert foundations == sorted(
        foundations, key=lambda item: item["repository_relative_path"]
    )
    assert all(item["verified_against_config"] is True for item in foundations)
    assert synthetic["production_status"] == "PRODUCTION_NOT_RUN"
    assert synthetic["production_source_interface_used"] is False
    assert synthetic["production_input_accessed"] is False
    assert synthetic["production_parity_established"] is False
    assert synthetic["g_minus_1_data_readiness"] == "PENDING"
    assert synthetic["g0_measurement_freeze"] == "PENDING"
    assert synthetic["promotes_component"] is False
    assert all(
        item["verification"] == "NOT_REVERIFIED_NO_EXPLICIT_PATH"
        for item in report["operator_control_receipts"]
    )


def test_author_attestation_uses_exact_isolated_fixed_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    original_run = verifier.subprocess.run

    def recording_run(command: list[str], **kwargs: Any) -> Any:
        calls.append((command, kwargs))
        return original_run(command, **kwargs)

    monkeypatch.setattr(verifier.subprocess, "run", recording_run)

    verifier.verify_registry(REGISTRY_PATH, repo_root=ROOT)

    assert len(calls) == 1
    command, kwargs = calls[0]
    evidence = verifier.EXPECTED_AUTHOR_SYNTHETIC_EVIDENCE
    assert command == [
        sys.executable,
        "-I",
        str((ROOT / evidence["validator_path"]).resolve()),
        "--config",
        str((ROOT / evidence["config_path"]).resolve()),
        "--synthetic-fixture",
        str((ROOT / evidence["fixture_path"]).resolve()),
    ]
    assert kwargs == {
        "cwd": ROOT.resolve(),
        "stdin": verifier.subprocess.DEVNULL,
        "capture_output": True,
        "timeout": verifier.AUTHOR_VALIDATOR_TIMEOUT_SECONDS,
        "shell": False,
        "check": False,
    }


def test_runtime_mismatch_is_reported_without_cross_runtime_stdout_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = verifier.subprocess.run
    simulated_runtime = {
        **verifier._current_runtime_fingerprint(),
        "python_version": "3.11.simulated",
        "unicodedata_unidata_version": "14.0.simulated",
    }

    def simulated_run(command: list[str], **kwargs: Any) -> Any:
        completed = original_run(command, **kwargs)
        report = json.loads(completed.stdout)
        report["inputs"]["runtime_fingerprint"] = simulated_runtime
        stdout = (
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        return verifier.subprocess.CompletedProcess(command, 0, stdout, b"")

    monkeypatch.setattr(verifier.subprocess, "run", simulated_run)
    monkeypatch.setattr(
        verifier, "_current_runtime_fingerprint", lambda: simulated_runtime
    )

    report = verifier.verify_registry(REGISTRY_PATH, repo_root=ROOT)
    synthetic = report["nonpromoting_synthetic_evidence"]

    assert synthetic["verification"] == "NOT_REATTESTED_RUNTIME_MISMATCH"
    assert synthetic["canonical_stdout_reproduced"] is False
    assert synthetic["current_runtime_fingerprint"] == simulated_runtime
    assert (
        synthetic["current_canonical_stdout_sha256"]
        != (synthetic["attested_canonical_stdout_sha256"])
    )
    assert report["missing_required_component_count"] == 3
    assert report["blocking_required_component_count"] == 0
    assert report["total_blocking_component_count"] == 3
    assert report["status"] == "G_MINUS_1_PENDING"
    assert report["pass_g_minus_1"] is False


def test_generic_author_report_cannot_satisfy_pinned_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = verifier.subprocess.run

    def generic_run(command: list[str], **kwargs: Any) -> Any:
        completed = original_run(command, **kwargs)
        report = json.loads(completed.stdout)
        report["status"] = (
            "VALID_UNATTESTED_GENERIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_V2_MECHANICS_ONLY"
        )
        report["evidence_class"] = "UNATTESTED_GENERIC_FIXTURE"
        report["scope"]["fixture_evidence_class"] = "UNATTESTED_GENERIC_FIXTURE"
        report["scope"]["parent_evidence_class"] = (
            "SELF_DECLARED_UNVERIFIED_SYNTHETIC_PARENT_STUBS"
        )
        report["scope"]["production_input_accessed"] = None
        stdout = (
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        return verifier.subprocess.CompletedProcess(command, 0, stdout, b"")

    monkeypatch.setattr(verifier.subprocess, "run", generic_run)

    with pytest.raises(
        verifier.ReadinessError, match="report is not the pinned evidence class"
    ):
        verifier.verify_registry(REGISTRY_PATH, repo_root=ROOT)


def test_author_attestation_timeout_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout_run(command: list[str], **kwargs: Any) -> Any:
        raise verifier.subprocess.TimeoutExpired(
            command, verifier.AUTHOR_VALIDATOR_TIMEOUT_SECONDS
        )

    monkeypatch.setattr(verifier.subprocess, "run", timeout_run)

    with pytest.raises(verifier.ReadinessError, match="validator timed out"):
        verifier.verify_registry(REGISTRY_PATH, repo_root=ROOT)


def test_default_verification_ignores_receipt_environment_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _registry()
    for binding in raw["passed_operator_control_receipts"]:
        monkeypatch.setenv(binding["path_env"], "/must/not/be/opened.json")

    report = verifier.verify_registry(REGISTRY_PATH, repo_root=ROOT)

    assert report["verification_status"] == (
        "PASS_FAIL_CLOSED_REGISTRY_REPOSITORY_EVIDENCE_ONLY"
    )
    assert all(
        item["verification"] == "NOT_REVERIFIED_NO_EXPLICIT_PATH"
        for item in report["operator_control_receipts"]
    )


def test_exact_operator_control_receipts_still_cannot_promote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _registry()
    overrides = _operator_receipts(tmp_path, raw)
    registry_path = _write_registry(tmp_path, raw)
    _pin_test_registry(monkeypatch, registry_path)

    report = verifier.verify_registry(
        registry_path,
        repo_root=ROOT,
        receipt_overrides=overrides,
        require_operator_receipts=True,
    )

    assert report["verification_status"] == (
        "PASS_FAIL_CLOSED_REGISTRY_AND_ALL_BOUND_CONTROL_EVIDENCE"
    )
    assert report["status"] == "G_MINUS_1_PENDING"
    assert report["pass_g_minus_1"] is False
    assert report["execution_authorized"] is False
    assert report["missing_required_component_count"] == 3
    assert report["blocking_required_component_count"] == 0
    assert report["total_blocking_component_count"] == 3
    assert all(
        item["receipt_bound"] is False for item in report["missing_required_components"]
    )
    assert all(
        item["verification"] == "PASS_EXACT_OPERATOR_CONTROL_RECEIPT"
        for item in report["operator_control_receipts"]
    )


def test_repository_hash_tampering_fails_closed(tmp_path: Path) -> None:
    raw = _registry()
    raw["passed_repository_evidence"][0]["sha256"] = "0" * 64
    registry_path = _write_registry(tmp_path, raw)

    with pytest.raises(
        verifier.ReadinessError,
        match="versioned G-1 readiness registry file SHA-256 mismatch",
    ):
        verifier.verify_registry(registry_path, repo_root=ROOT)


def test_repository_path_escape_is_rejected_as_registry_drift(tmp_path: Path) -> None:
    raw = _registry()
    raw["passed_repository_evidence"][0]["relative_path"] = "../outside.json"
    registry_path = _write_registry(tmp_path, raw)

    with pytest.raises(
        verifier.ReadinessError,
        match="versioned G-1 readiness registry file SHA-256 mismatch",
    ):
        verifier.verify_registry(registry_path, repo_root=ROOT)


def test_operator_receipt_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _registry()
    overrides = _operator_receipts(tmp_path, raw)
    registry_path = _write_registry(tmp_path, raw)
    _pin_test_registry(monkeypatch, registry_path)
    overrides["current_content_baseline"].write_bytes(b"{}\n")

    with pytest.raises(verifier.ReadinessError, match="SHA-256 mismatch"):
        verifier.verify_registry(
            registry_path,
            repo_root=ROOT,
            receipt_overrides=overrides,
            require_operator_receipts=True,
        )


@pytest.mark.parametrize(
    "mutation",
    ["coherent_evidence_substitution", "claim_boundary", "missing_metadata"],
)
def test_any_evidence_or_claim_contract_drift_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    raw = _registry()
    if mutation == "coherent_evidence_substitution":
        binding = raw["passed_repository_evidence"][0]
        replacement = raw["passed_repository_evidence"][1]
        binding.update(
            {
                "relative_path": replacement["relative_path"],
                "sha256": replacement["sha256"],
                "required_schema_version": replacement["required_schema_version"],
                "required_status": replacement["required_status"],
                "required_fields": replacement["required_fields"],
                "claim_scope": replacement["claim_scope"],
            }
        )
    elif mutation == "claim_boundary":
        raw["claim_boundary"] = "A broader but still pending claim."
    elif mutation == "missing_metadata":
        raw["missing_required_components"][0].pop("expected_basename")
    else:  # pragma: no cover - parametrization owns the values
        raise AssertionError(mutation)
    registry_path = _write_registry(tmp_path, raw)

    with pytest.raises(
        verifier.ReadinessError,
        match="versioned G-1 readiness registry file SHA-256 mismatch",
    ):
        verifier.verify_registry(registry_path, repo_root=ROOT)


def test_coherent_author_synthetic_evidence_substitution_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _registry()
    author = next(
        item
        for item in raw["passed_repository_evidence"]
        if item["component_id"] == "author_history"
    )
    evidence = author["nonpromoting_synthetic_core_evidence"]
    evidence["config_path"] = evidence["fixture_path"]
    evidence["config_sha256"] = evidence["fixture_sha256"]
    registry_path = _write_registry(tmp_path, raw)
    _pin_test_registry(monkeypatch, registry_path)

    with pytest.raises(
        verifier.ReadinessError,
        match="author synthetic evidence contract differs",
    ):
        verifier.verify_registry(registry_path, repo_root=ROOT)


def test_operator_receipts_can_be_mandatory_without_source_access() -> None:
    with pytest.raises(verifier.ReadinessError, match="required but unavailable"):
        verifier.verify_registry(
            REGISTRY_PATH,
            repo_root=ROOT,
            receipt_overrides={},
            require_operator_receipts=True,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("status", "PASS_G_MINUS_1"),
        ("gate_state.pass_g_minus_1", True),
        ("gate_state.execution_authorized", True),
        ("safety_boundary.source_payload_access_allowed", True),
        ("safety_boundary.production_scan_allowed", True),
    ],
)
def test_any_promotion_or_source_access_mutation_is_rejected(
    tmp_path: Path,
    path: str,
    value: Any,
) -> None:
    raw = _registry()
    _set_nested(raw, path, value)
    registry_path = _write_registry(tmp_path, raw)

    with pytest.raises(
        verifier.ReadinessError,
        match="versioned G-1 readiness registry file SHA-256 mismatch",
    ):
        verifier.verify_registry(registry_path, repo_root=ROOT)


def test_cli_surface_has_no_production_or_write_arguments() -> None:
    destinations = {action.dest for action in verifier.build_parser()._actions}
    source = (SCRIPTS / "verify_pivot_feedback_g_minus_1_readiness.py").read_text(
        encoding="utf-8"
    )

    assert destinations == {
        "help",
        "registry",
        "repo_root",
        "receipt",
        "require_operator_receipts",
    }
    assert "--source-root" not in source
    assert "--execute" not in source
    assert "--output" not in source


def test_cli_failure_report_remains_pending(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = verifier.main(
        [
            "--registry",
            str(REGISTRY_PATH),
            "--repo-root",
            str(ROOT),
            "--require-operator-receipts",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert report["verification_status"] == (
        "BLOCKED_INVALID_REGISTRY_OR_CONTROL_EVIDENCE"
    )
    assert report["status"] == "G_MINUS_1_PENDING"
    assert report["pass_g_minus_1"] is False
    assert report["execution_authorized"] is False


def test_cli_missing_repo_root_fails_as_pending_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_root = tmp_path / "missing"
    exit_code = verifier.main(
        ["--registry", str(REGISTRY_PATH), "--repo-root", str(missing_root)]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert report["verification_status"] == (
        "BLOCKED_INVALID_REGISTRY_OR_CONTROL_EVIDENCE"
    )
    assert report["status"] == "G_MINUS_1_PENDING"
    assert report["pass_g_minus_1"] is False
    assert report["execution_authorized"] is False
    assert report["error"] == "repository evidence is unavailable"
