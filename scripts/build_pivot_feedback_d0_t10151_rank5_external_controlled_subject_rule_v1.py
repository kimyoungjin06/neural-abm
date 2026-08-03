#!/usr/bin/env python3
"""Replay the rank-5 post-diagnostic external controlled-subject rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata
from xml.etree import ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-external-controlled-subject-rule-v1-20260730"
)
STATUS = "PASS_EXACT_NINE_POST_DIAGNOSTIC_ACCEPT_ONLY_EXTERNAL_RULE_REPLAY"
TARGET_DESCRIPTORS = {"Hepatitis C", "Hepacivirus"}
TITLE_MARKER = re.compile(
    r"(?<![a-z0-9])hcv(?:-ab)?(?![a-z0-9])|"
    r"(?<![a-z0-9])hepatitis\\s+c(?![a-z0-9])|"
    r"(?<![a-z0-9])hepacivirus(?![a-z0-9])"
)


class ExternalControlledSubjectReplayError(RuntimeError):
    """Fail-closed retrospective replay error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalControlledSubjectReplayError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _repo(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _bound(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "bound root")
    path = root / str(binding[basename_key])
    require(path.is_file() and not path.is_symlink(), f"bound file: {path.name}")
    require(_sha256(path) == binding[sha_key], f"bound SHA: {path.name}")
    return path


def _top_level(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    path = (Path("/ssd") / str(binding[basename_key])).resolve(strict=True)
    require(path.parent == Path("/ssd").resolve(), "top-level binding")
    require(path.is_file() and not path.is_symlink(), f"top-level file: {path.name}")
    require(_sha256(path) == binding[sha_key], f"top-level SHA: {path.name}")
    return path


def _normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = " ".join(normalized.split())
    return normalized.rstrip(" .,:;!?")


def _major(value: str | None) -> bool:
    return str(value or "").upper() == "Y"


def _pubmed_records(path: Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(path).getroot()
    records: dict[str, dict[str, Any]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext("./MedlineCitation/PMID") or "").strip()
        title_node = article.find("./MedlineCitation/Article/ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        subjects: list[dict[str, Any]] = []
        for heading in article.findall("./MedlineCitation/MeshHeadingList/MeshHeading"):
            descriptor = heading.find("DescriptorName")
            if descriptor is None:
                continue
            subjects.append(
                {
                    "descriptor": "".join(descriptor.itertext()).strip(),
                    "major_topic": _major(descriptor.get("MajorTopicYN")),
                    "qualifiers": [
                        {
                            "major_topic": _major(qualifier.get("MajorTopicYN")),
                            "name": "".join(qualifier.itertext()).strip(),
                        }
                        for qualifier in heading.findall("QualifierName")
                    ],
                }
            )
        require(pmid and pmid not in records, "duplicate or empty PMID")
        records[pmid] = {"title": title, "controlled_subjects": subjects}
    return records


def _has_major_target(subjects: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    matched: list[str] = []
    major = False
    for subject in subjects:
        descriptor = str(subject.get("descriptor", ""))
        if descriptor not in TARGET_DESCRIPTORS:
            continue
        matched.append(descriptor)
        qualifiers = subject.get("qualifiers", [])
        require(isinstance(qualifiers, list), "qualifiers")
        if bool(subject.get("major_topic")) or any(
            bool(_mapping(qualifier, "qualifier").get("major_topic"))
            for qualifier in qualifiers
        ):
            major = True
    return major, sorted(set(matched))


def load_config(path: Path) -> dict[str, Any]:
    config = _yaml(path.resolve(strict=True))
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_config.v1",
        "config schema",
    )
    require(
        config.get("status")
        == "FROZEN_EXACT_NINE_POST_DIAGNOSTIC_EXTERNAL_RULE_REPLAY_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_doc = _yaml(_repo(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    rule = _mapping(config.get("accept_only_rule"), "rule")
    require(rule.get("retrospective_post_diagnostic") is True, "retrospective label")
    require(rule.get("out_of_sample_validation") is False, "out of sample")
    require(rule.get("absence_action") == "REMAIN_UNRESOLVED", "absence action")
    require(rule.get("negative_override") == "FORBIDDEN", "negative override")
    require(set(rule.get("controlled_descriptors", [])) == TARGET_DESCRIPTORS, "descriptors")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(authorization.get("exact_nine_rule_replay") == "AUTHORIZED", "replay")
    for key in (
        "production_disposition",
        "selection_certificate",
        "deterministic_reselection",
        "new_api_request",
        "production_pivot_classification",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config


def load_inputs(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    evidence_binding = _mapping(frozen.get("external_evidence"), "external evidence")
    evidence_path = _bound(evidence_binding, "evidence_basename", "evidence_sha256")
    pubmed_path = _bound(evidence_binding, "raw_pubmed_basename", "raw_pubmed_sha256")
    evidence_receipt_path = _bound(evidence_binding, "receipt_basename", "receipt_sha256")
    evidence_verification_path = _top_level(
        evidence_binding, "verification_basename", "verification_sha256"
    )
    require(_json(evidence_receipt_path).get("status") == evidence_binding["receipt_status"], "evidence receipt")
    require(_json(evidence_verification_path).get("status") == evidence_binding["verification_status"], "evidence verification")
    evidence_table = pq.read_table(evidence_path)
    require(evidence_table.num_rows == evidence_binding["evidence_rows"] == 14, "evidence rows")
    evidence_rows = evidence_table.to_pylist()
    pubmed = _pubmed_records(pubmed_path)
    require(len(pubmed) == evidence_binding["pubmed_rows"] == 8, "PubMed rows")

    provider_binding = _mapping(frozen.get("cross_provider"), "cross provider")
    decisions_path = _bound(provider_binding, "decisions_basename", "decisions_sha256")
    candidate_path = _bound(provider_binding, "candidate_basename", "candidate_sha256")
    provider_receipt_path = _bound(provider_binding, "receipt_basename", "receipt_sha256")
    provider_verification_path = _top_level(
        provider_binding, "verification_basename", "verification_sha256"
    )
    require(_json(provider_receipt_path).get("status") == provider_binding["receipt_status"], "provider receipt")
    require(_json(provider_verification_path).get("status") == provider_binding["verification_status"], "provider verification")
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        decisions = list(csv.DictReader(handle))
    require(len(decisions) == provider_binding["decision_rows"] == 11, "decision rows")
    candidate = _json(candidate_path)
    require(candidate.get("final_accept_rows") == 30, "candidate accepts")
    require(candidate.get("final_reject_rows") == 25, "candidate rejects")
    require(candidate.get("final_unresolved_rows") == 9, "candidate unresolved")

    prior = _mapping(frozen.get("prior_closeout"), "prior closeout")
    prior_receipt_path = _bound(prior, "receipt_basename", "receipt_sha256")
    prior_verification_path = _top_level(prior, "verification_basename", "verification_sha256")
    prior_receipt = _json(prior_receipt_path)
    require(prior_receipt.get("status") == prior["receipt_status"], "prior receipt")
    require(_json(prior_verification_path).get("status") == prior["verification_status"], "prior verification")
    require(
        prior_receipt.get("rank5_counts")
        == {
            "accept": prior["accept_rows"],
            "reject": prior["reject_rows"],
            "unresolved": prior["unresolved_rows"],
        },
        "prior counts",
    )
    hashes = {
        "evidence_sha256": _sha256(evidence_path),
        "raw_pubmed_sha256": _sha256(pubmed_path),
        "evidence_receipt_sha256": _sha256(evidence_receipt_path),
        "evidence_verification_sha256": _sha256(evidence_verification_path),
        "provider_decisions_sha256": _sha256(decisions_path),
        "provider_candidate_sha256": _sha256(candidate_path),
        "provider_receipt_sha256": _sha256(provider_receipt_path),
        "provider_verification_sha256": _sha256(provider_verification_path),
        "prior_closeout_receipt_sha256": _sha256(prior_receipt_path),
        "prior_closeout_verification_sha256": _sha256(prior_verification_path),
    }
    return evidence_rows, decisions, candidate, pubmed, hashes


def replay(
    evidence_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, str]],
    pubmed: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence = {str(row["item_id"]): dict(row) for row in evidence_rows}
    require(len(evidence) == len(evidence_rows), "duplicate evidence item")
    reviewed_unresolved = {
        str(row["item_id"])
        for row in decisions
        if row.get("consensus_route") == "EXTERNAL_REMAINS_UNRESOLVED"
    }
    holdouts = {
        item_id for item_id, row in evidence.items() if row.get("review_eligible") is False
    }
    require(reviewed_unresolved.isdisjoint(holdouts), "cohort overlap")
    cohort = reviewed_unresolved | holdouts
    frozen_cohort = _mapping(config.get("replay_cohort"), "cohort")
    require(len(reviewed_unresolved) == frozen_cohort["reviewed_unresolved_rows"] == 6, "reviewed unresolved")
    require(len(holdouts) == frozen_cohort["holdout_rows"] == 3, "holdouts")
    require(len(cohort) == frozen_cohort["total_rows"] == 9, "cohort rows")
    require(cohort == set(frozen_cohort["expected_item_ids"]), "cohort identity")

    rows: list[dict[str, Any]] = []
    for item_id in sorted(cohort):
        row = evidence[item_id]
        pmid = str(row.get("pmid") or "")
        exact_pubmed_link = bool(pmid and pmid in pubmed)
        pubmed_title = str(pubmed[pmid]["title"]) if exact_pubmed_link else ""
        external_title = str(row["external_title"])
        title_agreement = bool(
            exact_pubmed_link
            and _normalize_title(external_title) == _normalize_title(pubmed_title)
        )
        title_has_marker = bool(TITLE_MARKER.search(_normalize_title(external_title)))
        normalized_subjects = json.loads(str(row["controlled_subjects_json"]))
        require(isinstance(normalized_subjects, list), "normalized subjects")
        if exact_pubmed_link:
            require(
                normalized_subjects == pubmed[pmid]["controlled_subjects"],
                f"normalized PubMed subjects: {item_id}",
            )
        major_target, matched_descriptors = _has_major_target(normalized_subjects)
        deterministic_accept = (
            exact_pubmed_link
            and title_agreement
            and title_has_marker
            and major_target
        )
        failed: list[str] = []
        if not exact_pubmed_link:
            failed.append("NO_EXACT_PUBMED_LINK")
        if not title_agreement:
            failed.append("NO_NORMALIZED_CROSS_SOURCE_TITLE_AGREEMENT")
        if not title_has_marker:
            failed.append("NO_EXPLICIT_TARGET_TITLE_MARKER")
        if not major_target:
            failed.append("NO_TARGET_MAJOR_CONTROLLED_SUBJECT")
        rows.append(
            {
                "item_id": item_id,
                "openalex_id": str(row["openalex_id"]),
                "pmid": pmid,
                "external_title": external_title,
                "pubmed_title": pubmed_title,
                "unresolved_origin": (
                    "CROSS_PROVIDER_REVIEWED_UNRESOLVED"
                    if item_id in reviewed_unresolved
                    else "EXTERNAL_EVIDENCE_HOLDOUT"
                ),
                "exact_pubmed_link": exact_pubmed_link,
                "normalized_title_agreement": title_agreement,
                "explicit_target_title_marker": title_has_marker,
                "matched_target_descriptors_json": json.dumps(
                    matched_descriptors, ensure_ascii=False, separators=(",", ":")
                ),
                "target_descriptor_major_topic": major_target,
                "deterministic_accept": deterministic_accept,
                "replay_state": (
                    "ACCEPT_POST_DIAGNOSTIC_EXTERNAL_RULE"
                    if deterministic_accept
                    else "REMAIN_UNRESOLVED"
                ),
                "failed_conditions_json": json.dumps(
                    failed, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
    expected = _mapping(config.get("accept_only_rule"), "rule")
    accepted = {row["item_id"] for row in rows if row["deterministic_accept"]}
    require(accepted == set(expected["expected_accept_item_ids"]), "accepted identity")
    return rows


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def build(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    output = _mapping(config.get("output_contract"), "output")
    output_root = output_root.absolute()
    require(output_root.parent == Path("/ssd"), "output parent")
    require(output_root.name == output["root_basename"], "output basename")
    require(not output_root.exists(), "output exists")
    evidence, decisions, prior_candidate, pubmed, input_hashes = load_inputs(config)
    rows = replay(evidence, decisions, pubmed, config)
    expected = _mapping(config.get("expected_result"), "expected result")
    new_accepts = sum(bool(row["deterministic_accept"]) for row in rows)
    require(new_accepts == expected["new_accept_rows"] == 2, "new accepts")
    require(expected["new_reject_rows"] == 0, "new rejects")
    remaining = len(rows) - new_accepts
    require(remaining == expected["remaining_unresolved_rows"] == 7, "remaining")
    candidate = {
        "schema_version": "pivot_feedback_d0_t10151_rank5_post_diagnostic_candidate.v1",
        "field_id": 27,
        "field_candidate_rank": 5,
        "topic_id": 10151,
        "entry_year": 1993,
        "prior_accept_rows": prior_candidate["final_accept_rows"],
        "prior_reject_rows": prior_candidate["final_reject_rows"],
        "prior_unresolved_rows": prior_candidate["final_unresolved_rows"],
        "new_accept_rows": new_accepts,
        "new_reject_rows": 0,
        "remaining_unresolved_rows": remaining,
        "final_accept_rows": prior_candidate["final_accept_rows"] + new_accepts,
        "final_reject_rows": prior_candidate["final_reject_rows"],
        "final_unresolved_rows": remaining,
        "recent_denominator_rows": 64,
        "pass_threshold_rows": 32,
        "candidate_content_state": "CONTENT_PASS",
        "rule_version": config["accept_only_rule"]["rule_version"],
        "retrospective_post_diagnostic": True,
        "out_of_sample_validation": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "provider_labels_are_ground_truth": False,
        "pubmed_indexing_is_ground_truth": False,
        "production_disposition_created": False,
        "selection_certificate_issued": False,
    }
    for key in (
        "final_accept_rows",
        "final_reject_rows",
        "final_unresolved_rows",
        "recent_denominator_rows",
        "pass_threshold_rows",
        "candidate_content_state",
        "production_disposition_created",
        "selection_certificate_issued",
    ):
        require(candidate[key] == expected[key], f"candidate expected: {key}")

    replay_schema = pa.schema(
        [
            ("item_id", pa.string()),
            ("openalex_id", pa.string()),
            ("pmid", pa.string()),
            ("external_title", pa.string()),
            ("pubmed_title", pa.string()),
            ("unresolved_origin", pa.string()),
            ("exact_pubmed_link", pa.bool_()),
            ("normalized_title_agreement", pa.bool_()),
            ("explicit_target_title_marker", pa.bool_()),
            ("matched_target_descriptors_json", pa.string()),
            ("target_descriptor_major_topic", pa.bool_()),
            ("deterministic_accept", pa.bool_()),
            ("replay_state", pa.string()),
            ("failed_conditions_json", pa.string()),
        ],
        metadata={b"schema_version": str(output["replay_schema_version"]).encode()},
    )
    table = pa.Table.from_pylist(rows, schema=replay_schema)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir="/ssd")
    )
    temporary.chmod(0o700)
    try:
        replay_path = temporary / str(output["replay_basename"])
        pq.write_table(table, replay_path, compression="zstd", use_dictionary=True)
        replay_path.chmod(0o600)
        candidate_path = temporary / str(output["candidate_basename"])
        _write_json(candidate_path, candidate)
        report_path = temporary / str(output["report_basename"])
        report_path.write_text(
            "\n".join(
                [
                    "# External controlled-subject rule replay",
                    "",
                    f"Status: `{STATUS}`",
                    "",
                    "- scope: all 9 externally evidenced unresolved rows",
                    "- deterministic accept-only additions: 2",
                    "- remaining unresolved: 7",
                    "- candidate counts: 32 accept / 25 reject / 7 unresolved",
                    "- candidate state: `CONTENT_PASS`",
                    "- method label: retrospective post-diagnostic amendment",
                    "- out-of-sample validation: false",
                    "- production disposition/certificate: not created",
                    "- human review: `NOT_RUN_BY_DESIGN`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        report_path.chmod(0o600)
        artifacts = {
            "replay": _artifact(replay_path, rows=9),
            "candidate": _artifact(candidate_path),
            "report": _artifact(report_path),
        }
        receipt = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_outputs.v1",
            "status": STATUS,
            "config_sha256": _sha256(config_path),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "input_hashes": input_hashes,
            "artifacts": artifacts,
            "replay_rows": 9,
            "accepted_item_ids": sorted(
                row["item_id"] for row in rows if row["deterministic_accept"]
            ),
            "new_accept_rows": 2,
            "new_reject_rows": 0,
            "remaining_unresolved_rows": 7,
            "candidate_counts": {"accept": 32, "reject": 25, "unresolved": 7},
            "candidate_content_state": "CONTENT_PASS",
            "retrospective_post_diagnostic": True,
            "out_of_sample_validation": False,
            "human_review": "NOT_RUN_BY_DESIGN",
            "provider_labels_used_as_ground_truth": False,
            "pubmed_indexing_used_as_ground_truth": False,
            "network_accessed": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "production_disposition_created": False,
            "selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        }
        _write_json(temporary / str(output["receipt_basename"]), receipt)
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            require(temporary.parent == Path("/ssd"), "temporary parent")
            shutil.rmtree(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(args.config, args.output), indent=2, sort_keys=True))
        return 0
    except (
        ExternalControlledSubjectReplayError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        csv.Error,
        yaml.YAMLError,
        json.JSONDecodeError,
        ET.ParseError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_EXTERNAL_CONTROLLED_SUBJECT_RULE_REPLAY", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
