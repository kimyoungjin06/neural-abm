#!/usr/bin/env python3
"""Independently verify the rank-5 external controlled-subject replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata
from xml.etree import ElementTree as ET

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_v1.yaml"
)
DEFAULT_ROOT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-external-controlled-subject-rule-v1-20260730"
)
DEFAULT_VERIFICATION = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-external-controlled-subject-rule-v1-20260730.verification-v1.json"
)
STATUS = "PASS_INDEPENDENT_EXACT_NINE_EXTERNAL_CONTROLLED_SUBJECT_RULE_REPLAY"
TARGETS = {"Hepatitis C", "Hepacivirus"}
MARKER = re.compile(
    r"(?<![a-z0-9])hcv(?:-ab)?(?![a-z0-9])|"
    r"(?<![a-z0-9])hepatitis\s+c(?![a-z0-9])|"
    r"(?<![a-z0-9])hepacivirus(?![a-z0-9])"
)


class ExternalControlledSubjectVerificationError(RuntimeError):
    """Fail-closed replay verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalControlledSubjectVerificationError(message)


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


def _normal(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split()).rstrip(" .,:;!?")


def _yes(value: str | None) -> bool:
    return str(value or "").upper() == "Y"


def _parse_pubmed(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for article in ET.parse(path).getroot().findall(".//PubmedArticle"):
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
                    "descriptor_major": _yes(descriptor.get("MajorTopicYN")),
                    "qualifier_major": any(
                        _yes(q.get("MajorTopicYN"))
                        for q in heading.findall("QualifierName")
                    ),
                }
            )
        require(pmid and pmid not in records, "PubMed identity")
        records[pmid] = {"title": title, "subjects": subjects}
    return records


def verify(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_config.v1",
        "config schema",
    )
    output = _mapping(config.get("output_contract"), "output")
    root = output_root.resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "root parent")
    require(root.name == output["root_basename"], "root basename")
    require(root.stat().st_mode & 0o077 == 0, "root mode")
    receipt_path = root / str(output["receipt_basename"])
    receipt = _json(receipt_path)
    require(receipt.get("status") == "PASS_EXACT_NINE_POST_DIAGNOSTIC_ACCEPT_ONLY_EXTERNAL_RULE_REPLAY", "receipt status")
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    artifacts = _mapping(receipt.get("artifacts"), "artifacts")
    for label, artifact_value in artifacts.items():
        artifact = _mapping(artifact_value, label)
        path = root / str(artifact["basename"])
        require(path.is_file() and not path.is_symlink(), f"artifact: {label}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {label}")
        require(path.stat().st_mode & 0o077 == 0, f"artifact mode: {label}")

    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    evidence_binding = _mapping(frozen.get("external_evidence"), "evidence")
    evidence_root = (Path("/ssd") / str(evidence_binding["root_basename"])).resolve(strict=True)
    evidence_path = evidence_root / str(evidence_binding["evidence_basename"])
    pubmed_path = evidence_root / str(evidence_binding["raw_pubmed_basename"])
    require(_sha256(evidence_path) == evidence_binding["evidence_sha256"], "evidence SHA")
    require(_sha256(pubmed_path) == evidence_binding["raw_pubmed_sha256"], "PubMed SHA")
    evidence_rows = pq.read_table(evidence_path).to_pylist()
    evidence = {row["item_id"]: row for row in evidence_rows}
    require(len(evidence) == 14, "evidence rows")
    pubmed = _parse_pubmed(pubmed_path)
    require(len(pubmed) == 8, "PubMed rows")

    provider = _mapping(frozen.get("cross_provider"), "provider")
    decisions_path = (
        Path("/ssd") / str(provider["root_basename"]) / str(provider["decisions_basename"])
    )
    require(_sha256(decisions_path) == provider["decisions_sha256"], "decision SHA")
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        decisions = list(csv.DictReader(handle))
    reviewed = {
        row["item_id"]
        for row in decisions
        if row["consensus_route"] == "EXTERNAL_REMAINS_UNRESOLVED"
    }
    holdouts = {item_id for item_id, row in evidence.items() if row["review_eligible"] is False}
    require(reviewed.isdisjoint(holdouts), "cohort overlap")
    cohort = reviewed | holdouts
    expected_ids = set(config["replay_cohort"]["expected_item_ids"])
    require(len(reviewed) == 6 and len(holdouts) == 3, "cohort partitions")
    require(cohort == expected_ids and len(cohort) == 9, "cohort identities")

    independent: dict[str, dict[str, Any]] = {}
    for item_id in sorted(cohort):
        source = evidence[item_id]
        pmid = str(source.get("pmid") or "")
        linked = bool(pmid and pmid in pubmed)
        title_match = bool(
            linked
            and _normal(str(source["external_title"]))
            == _normal(str(pubmed[pmid]["title"]))
        )
        marker = bool(MARKER.search(_normal(str(source["external_title"]))))
        subjects = pubmed[pmid]["subjects"] if linked else []
        matched = sorted(
            {
                subject["descriptor"]
                for subject in subjects
                if subject["descriptor"] in TARGETS
            }
        )
        target_major = any(
            subject["descriptor"] in TARGETS
            and (subject["descriptor_major"] or subject["qualifier_major"])
            for subject in subjects
        )
        independent[item_id] = {
            "exact_pubmed_link": linked,
            "normalized_title_agreement": title_match,
            "explicit_target_title_marker": marker,
            "matched_target_descriptors_json": json.dumps(
                matched, ensure_ascii=False, separators=(",", ":")
            ),
            "target_descriptor_major_topic": target_major,
            "deterministic_accept": linked and title_match and marker and target_major,
        }

    replay_path = root / str(output["replay_basename"])
    replay = pq.read_table(replay_path)
    require(replay.num_rows == 9, "replay rows")
    require(
        (replay.schema.metadata or {}).get(b"schema_version", b"").decode()
        == output["replay_schema_version"],
        "replay schema",
    )
    replay_rows = replay.to_pylist()
    require({row["item_id"] for row in replay_rows} == cohort, "replay identities")
    for row in replay_rows:
        expected = independent[row["item_id"]]
        for key, value in expected.items():
            require(row[key] == value, f"independent replay: {row['item_id']} {key}")
        require(
            row["replay_state"]
            == (
                "ACCEPT_POST_DIAGNOSTIC_EXTERNAL_RULE"
                if expected["deterministic_accept"]
                else "REMAIN_UNRESOLVED"
            ),
            f"replay state: {row['item_id']}",
        )
    accepted = sorted(
        item_id for item_id, row in independent.items() if row["deterministic_accept"]
    )
    require(accepted == sorted(config["accept_only_rule"]["expected_accept_item_ids"]), "accepted identities")
    require(accepted == receipt["accepted_item_ids"], "receipt accepted identities")
    require(receipt.get("candidate_counts") == {"accept": 32, "reject": 25, "unresolved": 7}, "counts")
    require(receipt.get("candidate_content_state") == "CONTENT_PASS", "candidate state")
    require(receipt.get("retrospective_post_diagnostic") is True, "retrospective")
    require(receipt.get("out_of_sample_validation") is False, "out of sample")
    require(receipt.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(receipt.get("provider_labels_used_as_ground_truth") is False, "provider ground truth")
    require(receipt.get("pubmed_indexing_used_as_ground_truth") is False, "PubMed ground truth")
    for key in (
        "production_disposition_created",
        "selection_certificate_issued",
        "deterministic_reselection_executed",
        "outcomes_read",
        "g4_open",
    ):
        require(receipt.get(key) is False, f"closed boundary: {key}")
    candidate_path = root / str(output["candidate_basename"])
    candidate = _json(candidate_path)
    require(candidate.get("final_accept_rows") == 32, "candidate accept")
    require(candidate.get("final_reject_rows") == 25, "candidate reject")
    require(candidate.get("final_unresolved_rows") == 7, "candidate unresolved")
    require(candidate.get("candidate_content_state") == "CONTENT_PASS", "candidate content state")
    require(candidate.get("production_disposition_created") is False, "candidate production")
    return {
        "schema_version": "pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(config_path),
        "receipt_sha256": _sha256(receipt_path),
        "replay_sha256": _sha256(replay_path),
        "candidate_sha256": _sha256(candidate_path),
        "independently_parsed_pubmed_rows": len(pubmed),
        "replay_rows": len(cohort),
        "accepted_item_ids": accepted,
        "new_accept_rows": len(accepted),
        "new_reject_rows": 0,
        "remaining_unresolved_rows": 7,
        "candidate_counts": {"accept": 32, "reject": 25, "unresolved": 7},
        "candidate_content_state": "CONTENT_PASS",
        "retrospective_post_diagnostic": True,
        "out_of_sample_validation": False,
        "production_disposition_created": False,
        "selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "outcomes_read": False,
        "g4_open": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--verification-output", type=Path, default=DEFAULT_VERIFICATION)
    args = parser.parse_args(argv)
    try:
        result = verify(args.config, args.output_root)
        output = args.verification_output.absolute()
        require(output.parent == Path("/ssd"), "verification parent")
        require(not output.exists(), "verification exists")
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.chmod(0o600)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        ExternalControlledSubjectVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        csv.Error,
        yaml.YAMLError,
        json.JSONDecodeError,
        ET.ParseError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_EXTERNAL_CONTROLLED_SUBJECT_RULE_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
