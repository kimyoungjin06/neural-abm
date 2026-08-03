#!/usr/bin/env python3
"""Independently verify the exact rank-5 external evidence packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_external_evidence_v1.yaml"
)
DEFAULT_ROOT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-external-evidence-v1-20260730"
)
STATUS = "PASS_INDEPENDENT_EXACT_14_EXTERNAL_EVIDENCE_EXACT_11_PACKET_VERIFICATION"


class ExternalEvidenceVerificationError(RuntimeError):
    """Fail-closed external evidence verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalEvidenceVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def verify(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_t10151_rank5_external_evidence_config.v1",
        "config schema",
    )
    output = _mapping(config.get("output_contract"), "output")
    output_root = output_root.resolve(strict=True)
    require(output_root.parent == Path("/ssd").resolve(), "output parent")
    require(output_root.name == output["root_basename"], "output name")
    receipt_path = output_root / str(output["receipt_basename"])
    receipt = _json(receipt_path)
    require(
        receipt.get("status")
        == "PASS_EXACT_14_EXTERNAL_EVIDENCE_COLLECTED_EXACT_11_PACKET_PREPARED",
        "receipt status",
    )
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    require(receipt.get("network_requests") == 9, "network requests")
    require(receipt.get("credentials_used") is False, "credentials")
    require(receipt.get("semantic_labels_created") is False, "semantic labels")
    require(receipt.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    artifacts = _mapping(receipt.get("artifacts"), "artifacts")
    for label, artifact_value in artifacts.items():
        artifact = _mapping(artifact_value, label)
        path = output_root / str(artifact["basename"])
        require(path.is_file() and not path.is_symlink(), f"artifact: {label}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {label}")
        require(path.stat().st_mode & 0o077 == 0, f"artifact mode: {label}")

    raw_openalex = _json(output_root / str(output["raw_openalex_basename"]))
    oa_results = raw_openalex.get("results")
    require(isinstance(oa_results, list) and len(oa_results) == 14, "OpenAlex rows")
    oa_ids = {str(row["id"]) for row in oa_results}

    raw_crossref = _json(output_root / str(output["raw_crossref_basename"]))
    require(len(raw_crossref) == 7, "Crossref rows")
    crossref_source = _mapping(
        _mapping(config.get("network_sources"), "sources").get("crossref"), "Crossref"
    )
    expected_dois = {
        str(openalex_id): str(doi).lower()
        for openalex_id, doi in _mapping(
            crossref_source.get("exact_doi_by_openalex_id"), "DOIs"
        ).items()
    }
    require(set(raw_crossref) == set(expected_dois), "Crossref ID set")
    for openalex_id, value in raw_crossref.items():
        require(str(_mapping(value, "Crossref work").get("DOI")).lower() == expected_dois[openalex_id], "Crossref DOI")
    correction = _mapping(_mapping(config.get("review_packet"), "packet").get("title_correction"), "correction")
    corrected_titles = _mapping(raw_crossref[correction["openalex_id"]], "corrected work").get("title")
    require(
        isinstance(corrected_titles, list)
        and corrected_titles[0] == correction["corrected_crossref_title"],
        "corrected Crossref title",
    )

    pubmed_root = ET.parse(output_root / str(output["raw_pubmed_basename"])).getroot()
    pubmed_articles = pubmed_root.findall(".//PubmedArticle")
    require(len(pubmed_articles) == 8, "PubMed rows")
    pmids = {str(article.findtext(".//PMID") or "") for article in pubmed_articles}
    pubmed_source = _mapping(
        _mapping(config.get("network_sources"), "sources").get("pubmed"), "PubMed"
    )
    expected_pmids = {
        str(value)
        for value in _mapping(pubmed_source.get("exact_pmid_by_openalex_id"), "PMIDs").values()
    }
    require(pmids == expected_pmids, "PubMed PMID set")
    by_pmid = {
        str(article.findtext(".//PMID") or ""): article for article in pubmed_articles
    }
    aids_abstract = " ".join(
        "".join(node.itertext())
        for node in by_pmid["3444748"].findall(".//Abstract/AbstractText")
    )
    require("AIDS" in aids_abstract and "HIV" in aids_abstract, "external AIDS abstract")
    hcv_mesh = {
        "".join(node.itertext()).strip()
        for node in by_pmid["1662091"].findall(".//MeshHeading/DescriptorName")
    }
    require({"Hepacivirus", "Hepatitis C"} <= hcv_mesh, "HCV controlled subjects")

    evidence_path = output_root / str(output["evidence_basename"])
    evidence = pq.read_table(evidence_path)
    require(evidence.num_rows == 14, "evidence rows")
    require(
        (evidence.schema.metadata or {}).get(b"schema_version", b"").decode()
        == "pivot_feedback_d0_t10151_rank5_external_evidence.v1",
        "evidence schema",
    )
    evidence_rows = evidence.to_pylist()
    require({str(row["openalex_id"]) for row in evidence_rows} == oa_ids, "evidence ID set")
    require(sum(bool(row["review_eligible"]) for row in evidence_rows) == 11, "review rows")
    require(sum(not bool(row["review_eligible"]) for row in evidence_rows) == 3, "holdout rows")

    packet_path = output_root / str(output["review_packet_basename"])
    packet = _json(packet_path)
    items = packet.get("items")
    require(isinstance(items, list) and len(items) == 11, "packet rows")
    require(len({str(item["item_id"]) for item in items}) == 11, "packet item IDs")
    item_ids = [str(item["item_id"]) for item in items]
    require(
        packet.get("item_ids_sha256") == _sha256_text("\n".join(item_ids) + "\n"),
        "packet ID digest",
    )
    evidence_by_item = {str(row["item_id"]): row for row in evidence_rows}
    for item in items:
        item_id = str(item["item_id"])
        require(item_id in evidence_by_item, "packet/evidence join")
        visible_sha = _sha256_text(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        require(visible_sha == evidence_by_item[item_id]["evidence_sha256"], "visible evidence SHA")
    serialized_packet = packet_path.read_text(encoding="utf-8")
    require("openalex.org/" not in serialized_packet, "OpenAlex identifier leaked")
    require(not re.search(r"10[.][0-9]{4,9}/", serialized_packet), "DOI leaked")
    require('"pmid"' not in serialized_packet.lower(), "PMID key leaked")

    availability_path = output_root / str(output["availability_basename"])
    with availability_path.open(newline="", encoding="utf-8") as handle:
        availability = list(csv.DictReader(handle))
    require(len(availability) == 14, "availability rows")
    require(
        {row["openalex_id"] for row in availability}
        == {str(row["openalex_id"]) for row in evidence_rows},
        "availability ID set",
    )
    return {
        "schema_version": "pivot_feedback_d0_t10151_rank5_external_evidence_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(config_path),
        "producer_sha256": receipt["producer_sha256"],
        "receipt_sha256": _sha256(receipt_path),
        "evidence_sha256": _sha256(evidence_path),
        "review_packet_sha256": _sha256(packet_path),
        "external_rows": 14,
        "review_packet_rows": 11,
        "holdout_rows": 3,
        "crossref_rows": 7,
        "pubmed_rows": 8,
        "semantic_labels_created": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "production_disposition_created": False,
        "selection_certificate_issued": False,
        "outcomes_read": False,
        "g4_open": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--verification-output",
        type=Path,
        default=Path(
            "/ssd/pivot-feedback-d0-t10151-rank5-external-evidence-v1-20260730.verification-v1.json"
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = verify(config_path=args.config, output_root=args.output_root)
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
        ExternalEvidenceVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
        ET.ParseError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_EXTERNAL_EVIDENCE_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
