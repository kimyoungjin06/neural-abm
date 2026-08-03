#!/usr/bin/env python3
"""Collect exact external bibliographic evidence for rank-5 unresolved works."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_external_evidence_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-external-evidence-v1-20260730"
)
CONFIG_SCHEMA = "pivot_feedback_d0_t10151_rank5_external_evidence_config.v1"
PARQUET_SCHEMA = "pivot_feedback_d0_t10151_rank5_external_evidence.v1"
STATUS = "PASS_EXACT_14_EXTERNAL_EVIDENCE_COLLECTED_EXACT_11_PACKET_PREPARED"


class ExternalEvidenceCollectionError(RuntimeError):
    """Fail-closed external evidence collection error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalEvidenceCollectionError(message)


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
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _nested(document: Mapping[str, Any], dotted: str) -> Any:
    value: Any = document
    for part in dotted.split("."):
        require(isinstance(value, Mapping) and part in value, f"missing {dotted}")
        value = value[part]
    return value


def _repo(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _bound(binding: Mapping[str, Any]) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "input root")
    path = root / str(binding["basename"])
    require(path.is_file() and not path.is_symlink(), f"input: {path.name}")
    require(_sha256(path) == binding["sha256"], f"input SHA: {path.name}")
    return path


def validate_config(path: Path) -> dict[str, Any]:
    config = _yaml(path.resolve(strict=True))
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status") == "FROZEN_EXACT_14_EXTERNAL_COLLECTION_AUTHORIZED",
        "config status",
    )
    gate = _mapping(config.get("gate_parent"), "gate")
    gate_doc = _yaml(_repo(gate))
    require(gate_doc.get("schema_version") == gate["required_schema_version"], "gate schema")
    require(gate_doc.get("status") == gate["required_status"], "gate status")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(authorization.get("network_access") == "AUTHORIZED_EXACT_HOSTS_ONLY", "network")
    require(authorization.get("credentials_access") is False, "credentials")
    require(authorization.get("provider_calls") == 0, "provider calls")
    require(authorization.get("semantic_labels_created") == 0, "semantic labels")
    require(authorization.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(authorization.get("production_disposition") == "NOT_AUTHORIZED", "production")
    packet = _mapping(config.get("review_packet"), "packet")
    require(len(packet["exact_review_ids"]) == packet["exact_review_rows"] == 11, "review rows")
    require(len(packet["exact_holdout_ids"]) == packet["exact_holdout_rows"] == 3, "holdout rows")
    require(set(packet["exact_review_ids"]).isdisjoint(packet["exact_holdout_ids"]), "packet overlap")
    require(packet.get("mesh_absence_is_proof") is False, "MeSH absence")
    require(packet.get("external_labels_created") is False, "external labels")
    return config


def _request_bytes(
    url: str,
    *,
    allowed_host: str,
    user_agent: str,
    timeout: int,
) -> bytes:
    require(urlparse(url).scheme == "https", "HTTPS required")
    require(urlparse(url).hostname == allowed_host, f"forbidden host: {urlparse(url).hostname}")
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "*/*"})
    with urlopen(request, timeout=timeout) as response:
        require(response.status == 200, f"HTTP status: {response.status}")
        return response.read()


def _parse_inverted_index(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    by_position: dict[int, str] = {}
    for token, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and position not in by_position:
                by_position[position] = str(token)
    return " ".join(by_position[position] for position in sorted(by_position)).strip()


def _clean_markup(value: Any) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    return " ".join(html.unescape(text).split())


def _input_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = _mapping(config.get("source_inputs"), "inputs")
    recovery_binding = _mapping(source.get("recovery"), "recovery")
    recovery_path = _bound(recovery_binding)
    table = pq.read_table(recovery_path)
    require(table.num_rows == recovery_binding["rows"] == 16, "recovery rows")
    require(
        (table.schema.metadata or {}).get(b"schema_version", b"").decode()
        == recovery_binding["schema_version"],
        "recovery schema",
    )
    decision_binding = _mapping(source.get("abstract_decisions"), "decisions")
    decisions_path = _bound(decision_binding)
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        decisions = list(csv.DictReader(handle))
    require(len(decisions) == decision_binding["rows"] == 5, "abstract decision rows")
    resolved = {
        str(decision_binding["resolved_accept_id"]),
        str(decision_binding["resolved_reject_id"]),
    }
    rows = [row for row in table.to_pylist() if str(row["openalex_id"]) not in resolved]
    require(len(rows) == source["expected_unresolved_rows"] == 14, "unresolved rows")
    ids = sorted(str(row["openalex_id"]) for row in rows)
    require(_sha256_text("\n".join(ids) + "\n") == source["sorted_openalex_ids_sha256"], "ID digest")
    return sorted(rows, key=lambda row: str(row["openalex_id"]))


def fetch_openalex(
    ids: Sequence[str],
    source: Mapping[str, Any],
    *,
    user_agent: str,
    timeout: int,
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    short_ids = [value.rsplit("/", 1)[-1] for value in ids]
    params = {
        "filter": "openalex_id:" + "|".join(short_ids),
        "per-page": "100",
        "select": ",".join(str(value) for value in source["select"]),
    }
    raw = _request_bytes(
        str(source["endpoint"]) + "?" + urlencode(params),
        allowed_host=str(source["allowed_host"]),
        user_agent=user_agent,
        timeout=timeout,
    )
    payload = _mapping(json.loads(raw), "OpenAlex response")
    results = payload.get("results")
    require(isinstance(results, list) and len(results) == source["exact_rows"] == 14, "OpenAlex rows")
    by_id = {str(row["id"]): _mapping(row, "OpenAlex work") for row in results}
    require(set(by_id) == set(ids), "OpenAlex exact ID set")
    return raw, by_id


def fetch_crossref(
    mapping: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    user_agent: str,
    timeout: int,
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    endpoint = str(source["endpoint"]).rstrip("/")
    for openalex_id, doi_value in sorted(mapping.items()):
        doi = str(doi_value)
        raw = _request_bytes(
            endpoint + "/" + quote(doi, safe=""),
            allowed_host=str(source["allowed_host"]),
            user_agent=user_agent,
            timeout=timeout,
        )
        payload = _mapping(json.loads(raw), "Crossref response")
        require(payload.get("status") == "ok", "Crossref status")
        message = _mapping(payload.get("message"), "Crossref message")
        require(str(message.get("DOI", "")).lower() == doi.lower(), "Crossref DOI")
        records[str(openalex_id)] = message
    return _json_bytes(records), records


def fetch_pubmed(
    mapping: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    user_agent: str,
    timeout: int,
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    expected_pmids = {str(value) for value in mapping.values()}
    params = {"db": "pubmed", "id": ",".join(sorted(expected_pmids)), "retmode": "xml"}
    raw = _request_bytes(
        str(source["endpoint"]) + "?" + urlencode(params),
        allowed_host=str(source["allowed_host"]),
        user_agent=user_agent,
        timeout=timeout,
    )
    root = ET.fromstring(raw)
    records_by_pmid: dict[str, dict[str, Any]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = str(article.findtext(".//PMID") or "")
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        abstract_parts = [
            " ".join("".join(node.itertext()).split())
            for node in article.findall(".//Abstract/AbstractText")
        ]
        subjects = []
        for heading in article.findall(".//MeshHeading"):
            descriptor = heading.find("DescriptorName")
            if descriptor is None:
                continue
            subjects.append(
                {
                    "descriptor": "".join(descriptor.itertext()).strip(),
                    "major_topic": descriptor.attrib.get("MajorTopicYN") == "Y",
                    "qualifiers": [
                        {
                            "name": "".join(value.itertext()).strip(),
                            "major_topic": value.attrib.get("MajorTopicYN") == "Y",
                        }
                        for value in heading.findall("QualifierName")
                    ],
                }
            )
        records_by_pmid[pmid] = {
            "pmid": pmid,
            "title": title,
            "abstract": " ".join(part for part in abstract_parts if part).strip(),
            "mesh": subjects,
        }
    require(set(records_by_pmid) == expected_pmids, "PubMed exact PMID set")
    records = {
        str(openalex_id): records_by_pmid[str(pmid)]
        for openalex_id, pmid in mapping.items()
    }
    return raw, records


def normalize(
    rows: Sequence[Mapping[str, Any]],
    *,
    openalex: Mapping[str, Mapping[str, Any]],
    crossref: Mapping[str, Mapping[str, Any]],
    pubmed: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet = _mapping(config.get("review_packet"), "packet")
    review_ids = set(str(value) for value in packet["exact_review_ids"])
    holdout_ids = set(str(value) for value in packet["exact_holdout_ids"])
    require(review_ids | holdout_ids == {str(row["openalex_id"]) for row in rows}, "packet ID set")
    reasons = _mapping(packet.get("eligibility_reasons"), "eligibility reasons")
    seed = str(packet["request_seed"])
    normalized: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    for local in rows:
        openalex_id = str(local["openalex_id"])
        oa = _mapping(openalex[openalex_id], "OpenAlex work")
        cr = _mapping(crossref[openalex_id], "Crossref work") if openalex_id in crossref else {}
        pm = _mapping(pubmed[openalex_id], "PubMed work") if openalex_id in pubmed else {}
        local_title = str(local["title"])
        crossref_titles = cr.get("title") or []
        crossref_title = str(crossref_titles[0]) if crossref_titles else ""
        pubmed_title = str(pm.get("title") or "").rstrip(".")
        oa_title = str(oa.get("title") or "")
        external_title = crossref_title or pubmed_title or oa_title
        if openalex_id == packet["title_correction"]["openalex_id"]:
            require(local_title == packet["title_correction"]["corrupt_local_title"], "corrupt title")
            require(crossref_title == packet["title_correction"]["corrected_crossref_title"], "corrected title")
        current_abstract = _parse_inverted_index(oa.get("abstract_inverted_index"))
        local_abstract = str(local.get("abstract") or "")
        pubmed_abstract = str(pm.get("abstract") or "")
        crossref_abstract = _clean_markup(cr.get("abstract"))
        external_abstract = pubmed_abstract or crossref_abstract
        mesh = pm.get("mesh") if isinstance(pm.get("mesh"), list) else []
        source_names = ["OPENALEX_CURRENT_WORK_RECORD"]
        if cr:
            source_names.append("CROSSREF_EXACT_DOI_RECORD")
        if pm:
            source_names.append("PUBMED_EXACT_PMID_RECORD")
        review_eligible = openalex_id in review_ids
        reason = str(reasons[openalex_id]) if review_eligible else "NO_NEW_DECISION_RELEVANT_EXTERNAL_EVIDENCE"
        item_id = _sha256_text(f"{seed}\0item\0{openalex_id}")
        evidence_visible = {
            "item_id": item_id,
            "title": external_title,
            "language": str(local.get("language") or oa.get("language") or ""),
            "work_type": str(local.get("work_type") or oa.get("type") or ""),
            "local_abstract": local_abstract if openalex_id.endswith("W2336012383") else "",
            "external_abstract": external_abstract,
            "controlled_subjects": mesh,
            "external_evidence_types": source_names,
            "eligibility_reason": reason,
        }
        evidence_sha256 = _sha256_text(
            json.dumps(evidence_visible, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        normalized.append(
            {
                "item_id": item_id,
                "openalex_id": openalex_id,
                "local_title": local_title,
                "external_title": external_title,
                "title_corrected": external_title != local_title,
                "language": evidence_visible["language"],
                "work_type": evidence_visible["work_type"],
                "local_abstract_sha256": _sha256_text(local_abstract),
                "current_openalex_abstract_sha256": _sha256_text(current_abstract),
                "external_abstract_sha256": _sha256_text(external_abstract),
                "external_abstract_words": len(external_abstract.split()),
                "controlled_subjects_json": json.dumps(mesh, ensure_ascii=False, sort_keys=True),
                "source_types_json": json.dumps(source_names, ensure_ascii=False),
                "doi": str(cr.get("DOI") or oa.get("doi") or ""),
                "pmid": str(pm.get("pmid") or ""),
                "open_access_state": str(_mapping(oa.get("open_access"), "open access").get("oa_status") or ""),
                "review_eligible": review_eligible,
                "eligibility_reason": reason,
                "evidence_sha256": evidence_sha256,
            }
        )
        if review_eligible:
            review_items.append(evidence_visible)
    normalized.sort(key=lambda row: row["openalex_id"])
    review_items.sort(key=lambda row: row["item_id"])
    require(len(normalized) == 14, "normalized rows")
    require(len(review_items) == 11, "review packet rows")
    return normalized, {
        "schema_version": "pivot_feedback_d0_t10151_rank5_external_review_packet.v1",
        "items": review_items,
        "item_ids_sha256": _sha256_text(
            "\n".join(item["item_id"] for item in review_items) + "\n"
        ),
        "human_review": "NOT_RUN_BY_DESIGN",
        "semantic_labels_created": False,
    }


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
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


def collect(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = validate_config(config_path)
    output = _mapping(config.get("output_contract"), "output")
    output_root = output_root.absolute()
    require(output_root.parent == Path("/ssd"), "output parent")
    require(output_root.name == output["root_basename"], "output basename")
    require(not output_root.exists(), "output exists")
    rows = _input_rows(config)
    ids = [str(row["openalex_id"]) for row in rows]
    sources = _mapping(config.get("network_sources"), "sources")
    user_agent = str(sources["user_agent"])
    timeout = int(sources["timeout_seconds"])
    require(sources.get("retries") == 0, "retries")
    raw_openalex, openalex = fetch_openalex(
        ids, _mapping(sources.get("openalex"), "OpenAlex"), user_agent=user_agent, timeout=timeout
    )
    crossref_source = _mapping(sources.get("crossref"), "Crossref")
    raw_crossref, crossref = fetch_crossref(
        _mapping(crossref_source.get("exact_doi_by_openalex_id"), "DOI mapping"),
        crossref_source,
        user_agent=user_agent,
        timeout=timeout,
    )
    pubmed_source = _mapping(sources.get("pubmed"), "PubMed")
    raw_pubmed, pubmed = fetch_pubmed(
        _mapping(pubmed_source.get("exact_pmid_by_openalex_id"), "PMID mapping"),
        pubmed_source,
        user_agent=user_agent,
        timeout=timeout,
    )
    normalized, review_packet = normalize(
        rows, openalex=openalex, crossref=crossref, pubmed=pubmed, config=config
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir="/ssd"))
    temporary.chmod(0o700)
    try:
        raw_openalex_path = temporary / str(output["raw_openalex_basename"])
        raw_crossref_path = temporary / str(output["raw_crossref_basename"])
        raw_pubmed_path = temporary / str(output["raw_pubmed_basename"])
        _write(raw_openalex_path, raw_openalex)
        _write(raw_crossref_path, raw_crossref)
        _write(raw_pubmed_path, raw_pubmed)
        evidence_path = temporary / str(output["evidence_basename"])
        table = pa.Table.from_pylist(normalized).replace_schema_metadata(
            {b"schema_version": PARQUET_SCHEMA.encode()}
        )
        pq.write_table(table, evidence_path, compression="zstd")
        evidence_path.chmod(0o600)
        packet_path = temporary / str(output["review_packet_basename"])
        _write(packet_path, _json_bytes(review_packet))
        availability_path = temporary / str(output["availability_basename"])
        with availability_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "openalex_id",
                    "external_title",
                    "title_corrected",
                    "external_abstract_words",
                    "pmid",
                    "doi",
                    "review_eligible",
                    "eligibility_reason",
                ],
            )
            writer.writeheader()
            for row in normalized:
                writer.writerow({key: row[key] for key in writer.fieldnames})
        availability_path.chmod(0o600)
        report_path = temporary / str(output["report_basename"])
        _write(
            report_path,
            (
                "# Rank-5 external evidence collection\n\n"
                f"Status: `{STATUS}`\n\n"
                "- exact unresolved works collected: 14\n"
                "- exact review packet rows: 11\n"
                "- no-new-evidence holdouts: 3\n"
                "- Crossref exact DOI records: 7\n"
                "- PubMed exact PMID records: 8\n"
                "- human review: `NOT_RUN_BY_DESIGN`\n"
                "- semantic labels created: no\n"
            ).encode(),
        )
        artifacts = {
            "evidence": _artifact(evidence_path, rows=14),
            "review_packet": _artifact(packet_path, rows=11),
            "availability": _artifact(availability_path, rows=14),
            "raw_openalex": _artifact(raw_openalex_path, rows=14),
            "raw_crossref": _artifact(raw_crossref_path, rows=7),
            "raw_pubmed": _artifact(raw_pubmed_path, rows=8),
            "report": _artifact(report_path),
        }
        receipt = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_external_evidence_outputs.v1",
            "status": STATUS,
            "config_sha256": _sha256(config_path),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "network_hosts": sorted(
                {
                    str(sources["openalex"]["allowed_host"]),
                    str(sources["crossref"]["allowed_host"]),
                    str(sources["pubmed"]["allowed_host"]),
                }
            ),
            "network_requests": 9,
            "retries": 0,
            "external_rows": 14,
            "review_packet_rows": 11,
            "holdout_rows": 3,
            "crossref_rows": 7,
            "pubmed_rows": 8,
            "artifacts": artifacts,
            "credentials_used": False,
            "provider_calls": 0,
            "semantic_labels_created": False,
            "human_review": "NOT_RUN_BY_DESIGN",
            "production_disposition_created": False,
            "selection_certificate_issued": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        }
        _write(temporary / str(output["receipt_basename"]), _json_bytes(receipt))
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
        print(json.dumps(collect(config_path=args.config, output_root=args.output), indent=2, sort_keys=True))
        return 0
    except (
        ExternalEvidenceCollectionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
        ET.ParseError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_EXTERNAL_EVIDENCE_COLLECTION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
