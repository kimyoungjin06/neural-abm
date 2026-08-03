#!/usr/bin/env python3
"""Recover exact local OpenAlex abstracts for the 16 unresolved rank-5 rows."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_local_abstract_recovery_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-local-abstract-recovery-v1-20260730"
)
CONFIG_SCHEMA = "pivot_feedback_d0_t10151_rank5_local_abstract_recovery_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_t10151_rank5_local_abstract_recovery_outputs.v1"
PARQUET_SCHEMA = "pivot_feedback_d0_t10151_rank5_local_abstract_recovery.v1"
STATUS = "PASS_EXACT_16_LOCAL_OPENALEX_ABSTRACT_RECOVERY_FIVE_ELIGIBLE"


class LocalAbstractRecoveryError(RuntimeError):
    """Fail-closed local abstract recovery error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalAbstractRecoveryError(message)


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


def _ssd_input(binding: Mapping[str, Any]) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "SSD input root")
    path = root / str(binding["basename"])
    require(path.is_file() and not path.is_symlink(), f"input file: {path.name}")
    require(_sha256(path) == binding["sha256"], f"input SHA: {path.name}")
    return path


def _validate_config(path: Path) -> dict[str, Any]:
    config = _yaml(path.resolve(strict=True))
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status") == "FROZEN_EXACT_16_LOCAL_ABSTRACT_RECOVERY_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("gate_parent"), "gate parent")
    parent_doc = _yaml(_repo(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "gate schema")
    require(parent_doc.get("status") == parent["required_status"], "gate status")
    for dotted, expected in _mapping(parent.get("required_fields"), "gate fields").items():
        observed = _nested(parent_doc, dotted)
        require(type(observed) is type(expected) and observed == expected, f"gate field: {dotted}")
    source = _mapping(config.get("openalex_source"), "source")
    require(source.get("full_scan_columns") == ["id"], "scan columns")
    require(source.get("selected_row_group_columns") == ["id", "value"], "selected columns")
    require(source.get("full_value_column_scan") is False, "value scan")
    reference = _mapping(source.get("reconstruction_reference"), "reference")
    reference_path = Path(str(reference["path"])).resolve(strict=True)
    require(_sha256(reference_path) == reference["sha256"], "reconstruction reference SHA")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(authorization.get("network_access") is False, "network")
    require(authorization.get("credentials_access") is False, "credentials")
    require(authorization.get("provider_calls") == 0, "provider calls")
    require(authorization.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(authorization.get("local_semantic_labels") == 0, "semantic labels")
    require(authorization.get("production_disposition") == "NOT_AUTHORIZED", "production")
    require(authorization.get("outcomes_read") is False, "outcomes")
    return config


def _parse_inverted_index(value: Any) -> dict[str, list[int]] | None:
    if value is None:
        return None
    payload = value
    if isinstance(payload, str):
        text = payload.strip()
        if not text or text.lower() == "null":
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, Mapping):
        return None
    parsed: dict[str, list[int]] = {}
    for token, positions in payload.items():
        if not isinstance(positions, list):
            continue
        valid = [int(position) for position in positions if isinstance(position, int)]
        if valid:
            parsed[str(token)] = valid
    return parsed or None


def reconstruct_abstract(value: Any) -> dict[str, Any]:
    inverted = _parse_inverted_index(value)
    if not inverted:
        return {
            "has_abstract": False,
            "abstract": "",
            "token_count": 0,
            "position_count": 0,
            "unique_positions": 0,
            "collisions": 0,
        }
    by_position: dict[int, list[str]] = {}
    token_count = 0
    position_count = 0
    collisions = 0
    for token, positions in inverted.items():
        token_count += 1
        for position in positions:
            if position in by_position:
                collisions += 1
            by_position.setdefault(position, []).append(token)
            position_count += 1
    abstract = " ".join(by_position[position][0] for position in sorted(by_position)).strip()
    return {
        "has_abstract": bool(abstract),
        "abstract": abstract,
        "token_count": token_count,
        "position_count": position_count,
        "unique_positions": len(by_position),
        "collisions": collisions,
    }


def eligibility_state(
    reconstructed: Mapping[str, Any],
    *,
    minimum_words: int,
    maximum_collisions: int,
) -> tuple[str, int]:
    abstract = str(reconstructed["abstract"])
    words = len(abstract.split())
    if not reconstructed["has_abstract"]:
        return "NO_LOCAL_ABSTRACT", words
    if words < minimum_words or int(reconstructed["collisions"]) > maximum_collisions:
        return "FRAGMENT_UNUSABLE", words
    return "ELIGIBLE_ABSTRACT_REVIEW", words


def _selected_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = _mapping(config.get("source_inputs"), "inputs")
    decisions_binding = _mapping(inputs.get("decisions"), "decisions")
    decisions_path = _ssd_input(decisions_binding)
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        decisions = list(csv.DictReader(handle))
    require(len(decisions) == decisions_binding["rows"] == 25, "decision rows")
    selected = [
        row
        for row in decisions
        if row["consensus_route"] == decisions_binding["selected_route"]
    ]
    require(len(selected) == decisions_binding["selected_rows"] == 16, "selected rows")
    ids = sorted(row["openalex_id"] for row in selected)
    digest = _sha256_text("\n".join(ids) + "\n")
    require(digest == decisions_binding["sorted_openalex_ids_sha256"], "selected ID digest")

    sample_binding = _mapping(inputs.get("sample"), "sample")
    sample_path = _ssd_input(sample_binding)
    table = pq.read_table(sample_path)
    require(table.num_rows == sample_binding["rows"] == 96, "sample rows")
    require(
        (table.schema.metadata or {}).get(b"schema_version", b"").decode()
        == sample_binding["schema_version"],
        "sample schema",
    )
    samples = {str(row["openalex_id"]): row for row in table.to_pylist()}
    require(len(samples) == 96, "sample IDs")
    output = []
    for decision in selected:
        sample = samples.get(decision["openalex_id"])
        require(sample is not None, "decision/sample join")
        require(str(sample["title"]) == decision["title"], "decision/sample title")
        output.append(
            {
                "item_id": decision["item_id"],
                "openalex_id": decision["openalex_id"],
                "title": decision["title"],
                "language": str(sample["language"] or ""),
                "work_type": str(sample["work_type_name"]),
            }
        )
    return sorted(output, key=lambda row: row["openalex_id"])


def _source_files(config: Mapping[str, Any], source_root: Path) -> tuple[Path, list[Path]]:
    source = _mapping(config.get("openalex_source"), "source")
    root = source_root.resolve(strict=True)
    require(root.name == source["required_root_id"], "source root ID")
    table_root = root / str(source["table_directory"])
    require(table_root.is_dir() and not table_root.is_symlink(), "source table root")
    pattern = re.compile(str(source["allowed_filename_pattern"]))
    files = sorted(
        [path for path in table_root.glob("*.parquet") if pattern.fullmatch(path.name)],
        key=lambda path: (path.stem.split("-")[0], int(path.stem.split("-")[1])),
    )
    require(len(files) == source["expected_parquet_files"], "source file count")
    footer_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    require(footer_rows == source["expected_footer_rows"], "source footer rows")
    first_names = set(pq.ParquetFile(files[0]).schema_arrow.names)
    require(set(source["required_columns"]) <= first_names, "source columns")
    return table_root, files


def scan_locations(
    files: Sequence[Path],
    target_ids: Sequence[str],
    *,
    workers: int,
) -> dict[str, tuple[Path, int, int]]:
    value_set = pa.array(sorted(target_ids), type=pa.string())

    def scan_file(path: Path) -> tuple[Path, list[str]]:
        identifiers = pq.ParquetFile(path).read(columns=["id"])["id"]
        mask = pc.fill_null(pc.is_in(identifiers, value_set=value_set), False)
        if not pc.any(mask).as_py():
            return path, []
        return path, [str(value) for value in pc.filter(identifiers, mask).to_pylist()]

    hit_files: list[tuple[Path, list[str]]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 16))) as executor:
        for path, hits in executor.map(scan_file, files):
            if hits:
                hit_files.append((path, hits))
    locations: dict[str, tuple[Path, int, int]] = {}
    occurrences: Counter[str] = Counter()
    for path, hits in hit_files:
        parquet = pq.ParquetFile(path)
        local_set = pa.array(hits, type=pa.string())
        for row_group in range(parquet.metadata.num_row_groups):
            identifiers = parquet.read_row_group(row_group, columns=["id"])["id"]
            mask = pc.fill_null(pc.is_in(identifiers, value_set=local_set), False)
            if not pc.any(mask).as_py():
                continue
            for row_index, identifier in enumerate(identifiers.to_pylist()):
                if identifier in hits:
                    occurrences[str(identifier)] += 1
                    locations[str(identifier)] = (path, row_group, row_index)
    require(set(occurrences) == set(target_ids), "target IDs not all found")
    require(all(count == 1 for count in occurrences.values()), "target ID occurrence count")
    return locations


def _recover_rows(
    selected: Sequence[Mapping[str, Any]],
    locations: Mapping[str, tuple[Path, int, int]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    eligibility = _mapping(config.get("eligibility"), "eligibility")
    minimum_words = int(eligibility["minimum_whitespace_words"])
    maximum_collisions = int(eligibility["maximum_position_collisions"])
    output: list[dict[str, Any]] = []
    for source in selected:
        identifier = str(source["openalex_id"])
        path, row_group, _ = locations[identifier]
        table = pq.ParquetFile(path).read_row_group(row_group, columns=["id", "value"])
        matches = table.filter(pc.equal(table["id"], identifier)).to_pylist()
        require(len(matches) == 1, f"selected row group lookup: {identifier}")
        reconstructed = reconstruct_abstract(matches[0]["value"])
        state, words = eligibility_state(
            reconstructed,
            minimum_words=minimum_words,
            maximum_collisions=maximum_collisions,
        )
        output.append(
            {
                **dict(source),
                "source_file": path.name,
                "source_row_group": row_group,
                "has_abstract": bool(reconstructed["has_abstract"]),
                "abstract": str(reconstructed["abstract"]),
                "abstract_sha256": _sha256_text(str(reconstructed["abstract"])),
                "abstract_words": words,
                "token_count": int(reconstructed["token_count"]),
                "position_count": int(reconstructed["position_count"]),
                "unique_positions": int(reconstructed["unique_positions"]),
                "collisions": int(reconstructed["collisions"]),
                "eligibility_state": state,
            }
        )
    return sorted(output, key=lambda row: row["openalex_id"])


def _table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("openalex_id", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("language", pa.string(), nullable=False),
            pa.field("work_type", pa.string(), nullable=False),
            pa.field("source_file", pa.string(), nullable=False),
            pa.field("source_row_group", pa.uint16(), nullable=False),
            pa.field("has_abstract", pa.bool_(), nullable=False),
            pa.field("abstract", pa.string(), nullable=False),
            pa.field("abstract_sha256", pa.string(), nullable=False),
            pa.field("abstract_words", pa.uint32(), nullable=False),
            pa.field("token_count", pa.uint32(), nullable=False),
            pa.field("position_count", pa.uint32(), nullable=False),
            pa.field("unique_positions", pa.uint32(), nullable=False),
            pa.field("collisions", pa.uint32(), nullable=False),
            pa.field("eligibility_state", pa.string(), nullable=False),
        ],
        metadata={b"schema_version": PARQUET_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(list(rows), schema=schema)


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o600)


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def build(
    *,
    config_path: Path = DEFAULT_CONFIG,
    source_root: Path,
    output_root: Path = DEFAULT_OUTPUT,
    workers: int = 12,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _validate_config(config_path)
    contract = _mapping(config.get("output_contract"), "output")
    output_root = output_root.absolute()
    require(output_root.parent == Path("/ssd"), "output parent")
    require(output_root.name == contract["root_basename"], "output root name")
    require(not output_root.exists(), "output root exists")
    selected = _selected_rows(config)
    _, files = _source_files(config, source_root)
    locations = scan_locations(
        files,
        [str(row["openalex_id"]) for row in selected],
        workers=workers,
    )
    rows = _recover_rows(selected, locations, config)
    states = Counter(str(row["eligibility_state"]) for row in rows)
    reconstructable = sum(bool(row["has_abstract"]) for row in rows)
    expected = _mapping(_mapping(config.get("eligibility"), "eligibility").get("expected"), "expected")
    require(len(rows) == expected["source_rows_found"] == 16, "recovery rows")
    require(reconstructable == expected["reconstructable_rows"] == 6, "reconstructable rows")
    require(states["ELIGIBLE_ABSTRACT_REVIEW"] == expected["eligible_review_rows"] == 5, "eligible rows")
    require(states["NO_LOCAL_ABSTRACT"] == expected["missing_rows"] == 10, "missing rows")
    require(states["FRAGMENT_UNUSABLE"] == expected["fragment_rows"] == 1, "fragment rows")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    temporary.chmod(0o700)
    try:
        parquet_path = temporary / str(contract["parquet_basename"])
        pq.write_table(_table(rows), parquet_path, compression="zstd", use_dictionary=True)
        parquet_path.chmod(0o600)
        availability_path = temporary / str(contract["availability_basename"])
        with availability_path.open("x", newline="", encoding="utf-8") as handle:
            fields = [
                "item_id",
                "openalex_id",
                "title",
                "source_file",
                "source_row_group",
                "has_abstract",
                "abstract_words",
                "abstract_sha256",
                "eligibility_state",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in rows)
        availability_path.chmod(0o600)
        report_path = temporary / str(contract["report_basename"])
        report = "\n".join(
            [
                "# Rank-5 local OpenAlex abstract recovery",
                "",
                f"Status: `{STATUS}`",
                "",
                "- exact unresolved IDs found: 16/16",
                "- reconstructable abstracts: 6",
                "- eligible long-form abstracts: 5",
                "- null abstract payloads: 10",
                "- unusable fragments: 1",
                "- semantic labels created: 0",
                "- human review: `NOT_RUN_BY_DESIGN`",
                "",
                "Only ID columns were scanned globally. The value column was read",
                "only from row groups containing the exact frozen IDs.",
                "",
            ]
        )
        _write(report_path, report.encode("utf-8"))
        artifacts = [
            _artifact(parquet_path, rows=16),
            _artifact(availability_path, rows=16),
            _artifact(report_path),
        ]
        receipt = {
            "schema_version": OUTPUT_SCHEMA,
            "status": STATUS,
            "config_sha256": _sha256(config_path),
            "source_root_id": source_root.resolve(strict=True).name,
            "source_table": _mapping(config["openalex_source"], "source")["table_directory"],
            "source_parquet_files_scanned": len(files),
            "source_footer_rows": sum(pq.ParquetFile(path).metadata.num_rows for path in files),
            "global_columns_scanned": ["id"],
            "selected_row_group_columns_read": ["id", "value"],
            "full_value_column_scan": False,
            "recovery": {
                "source_rows_found": len(rows),
                "reconstructable_rows": reconstructable,
                "eligible_review_rows": states["ELIGIBLE_ABSTRACT_REVIEW"],
                "missing_rows": states["NO_LOCAL_ABSTRACT"],
                "fragment_rows": states["FRAGMENT_UNUSABLE"],
                "position_collisions": sum(int(row["collisions"]) for row in rows),
            },
            "network_accessed": False,
            "credentials_accessed": False,
            "provider_calls": 0,
            "human_review": "NOT_RUN_BY_DESIGN",
            "semantic_labels_created": 0,
            "production_disposition_created": False,
            "selection_certificate_issued": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "artifacts": artifacts,
        }
        _write(
            temporary / str(contract["receipt_basename"]),
            _json_bytes(receipt),
        )
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            require(temporary.parent == output_root.parent, "temporary parent")
            shutil.rmtree(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    source_root = args.source_root
    if source_root is None:
        env_name = str(_yaml(args.config)["openalex_source"]["root_env"])
        value = os.environ.get(env_name)
        if not value:
            print(json.dumps({"status": "BLOCKED_LOCAL_ABSTRACT_RECOVERY", "error": f"{env_name} is required"}, sort_keys=True))
            return 2
        source_root = Path(value)
    try:
        result = build(
            config_path=args.config,
            source_root=source_root,
            output_root=args.output,
            workers=args.workers,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (LocalAbstractRecoveryError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_LOCAL_ABSTRACT_RECOVERY", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
