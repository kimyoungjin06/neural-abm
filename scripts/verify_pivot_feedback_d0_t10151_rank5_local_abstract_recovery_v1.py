#!/usr/bin/env python3
"""Independently verify the exact local rank-5 abstract recovery artifact."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

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
DEFAULT_VERIFICATION = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-local-abstract-recovery-v1-20260730."
    "verification-v1.json"
)
STATUS = "PASS_INDEPENDENT_EXACT_16_LOCAL_ABSTRACT_RECOVERY_VERIFICATION"


class LocalAbstractRecoveryVerificationError(RuntimeError):
    """Fail-closed local abstract recovery verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalAbstractRecoveryVerificationError(message)


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


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


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


def _reconstruct(value: Any) -> dict[str, Any]:
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
    collisions = 0
    position_count = 0
    for token, positions in inverted.items():
        for position in positions:
            if position in by_position:
                collisions += 1
            by_position.setdefault(position, []).append(token)
            position_count += 1
    abstract = " ".join(by_position[position][0] for position in sorted(by_position)).strip()
    return {
        "has_abstract": bool(abstract),
        "abstract": abstract,
        "token_count": len(inverted),
        "position_count": position_count,
        "unique_positions": len(by_position),
        "collisions": collisions,
    }


def _state(reconstructed: Mapping[str, Any], minimum_words: int, maximum_collisions: int) -> tuple[str, int]:
    words = len(str(reconstructed["abstract"]).split())
    if not reconstructed["has_abstract"]:
        return "NO_LOCAL_ABSTRACT", words
    if words < minimum_words or int(reconstructed["collisions"]) > maximum_collisions:
        return "FRAGMENT_UNUSABLE", words
    return "ELIGIBLE_ABSTRACT_REVIEW", words


def validate(
    *,
    config_path: Path = DEFAULT_CONFIG,
    source_root: Path,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_t10151_rank5_local_abstract_recovery_config.v1",
        "config schema",
    )
    require(
        config.get("status") == "FROZEN_EXACT_16_LOCAL_ABSTRACT_RECOVERY_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("gate_parent"), "parent")
    parent_path = (REPO_ROOT / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(REPO_ROOT), "parent escapes")
    require(_sha256(parent_path) == parent["sha256"], "parent SHA")
    parent_doc = _yaml(parent_path)
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")

    output_contract = _mapping(config.get("output_contract"), "output")
    output_root = output_root.resolve(strict=True)
    require(output_root.parent == Path("/ssd").resolve(), "output parent")
    require(output_root.name == output_contract["root_basename"], "output root")
    receipt_path = output_root / str(output_contract["receipt_basename"])
    receipt = _json(receipt_path)
    require(
        receipt.get("status")
        == "PASS_EXACT_16_LOCAL_OPENALEX_ABSTRACT_RECOVERY_FIVE_ELIGIBLE",
        "receipt status",
    )
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    artifacts = {
        str(item["basename"]): _mapping(item, "artifact")
        for item in receipt.get("artifacts", [])
    }
    require(
        set(artifacts)
        == {
            output_contract["parquet_basename"],
            output_contract["availability_basename"],
            output_contract["report_basename"],
        },
        "artifact set",
    )
    for basename, artifact in artifacts.items():
        path = output_root / basename
        require(path.is_file() and not path.is_symlink(), f"artifact: {basename}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {basename}")
        require(path.stat().st_mode & 0o077 == 0, f"artifact permissions: {basename}")

    parquet_path = output_root / str(output_contract["parquet_basename"])
    table = pq.read_table(parquet_path)
    require(table.num_rows == 16, "parquet rows")
    require(
        (table.schema.metadata or {}).get(b"schema_version", b"").decode()
        == "pivot_feedback_d0_t10151_rank5_local_abstract_recovery.v1",
        "parquet schema",
    )
    rows = table.to_pylist()
    require(len({row["openalex_id"] for row in rows}) == 16, "parquet IDs")
    ids = sorted(str(row["openalex_id"]) for row in rows)
    inputs = _mapping(config.get("source_inputs"), "inputs")
    decisions = _mapping(inputs.get("decisions"), "decisions")
    require(
        _sha256_text("\n".join(ids) + "\n") == decisions["sorted_openalex_ids_sha256"],
        "ID digest",
    )

    availability_path = output_root / str(output_contract["availability_basename"])
    with availability_path.open(newline="", encoding="utf-8") as handle:
        availability = list(csv.DictReader(handle))
    require(len(availability) == 16, "availability rows")
    by_id = {str(row["openalex_id"]): row for row in rows}
    for record in availability:
        row = by_id.get(record["openalex_id"])
        require(row is not None, "availability join")
        for key in (
            "item_id",
            "title",
            "source_file",
            "abstract_sha256",
            "eligibility_state",
        ):
            require(str(row[key]) == record[key], f"availability field: {key}")

    source = _mapping(config.get("openalex_source"), "source")
    source_root = source_root.resolve(strict=True)
    require(source_root.name == source["required_root_id"], "source root ID")
    table_root = source_root / str(source["table_directory"])
    require(table_root.is_dir() and not table_root.is_symlink(), "source table")
    pattern = re.compile(str(source["allowed_filename_pattern"]))
    files = [
        path
        for path in table_root.glob("*.parquet")
        if pattern.fullmatch(path.name)
    ]
    require(len(files) == source["expected_parquet_files"] == 493, "source files")
    footer_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    require(footer_rows == source["expected_footer_rows"] == 492293207, "source rows")
    reference = _mapping(source.get("reconstruction_reference"), "reference")
    require(_sha256(Path(str(reference["path"])).resolve(strict=True)) == reference["sha256"], "reference SHA")

    eligibility = _mapping(config.get("eligibility"), "eligibility")
    minimum_words = int(eligibility["minimum_whitespace_words"])
    maximum_collisions = int(eligibility["maximum_position_collisions"])
    recomputed_states: Counter[str] = Counter()
    for row in rows:
        source_file = table_root / str(row["source_file"])
        require(source_file.is_file() and pattern.fullmatch(source_file.name) is not None, "source file")
        parquet = pq.ParquetFile(source_file)
        row_group = int(row["source_row_group"])
        require(0 <= row_group < parquet.metadata.num_row_groups, "source row group")
        selected = parquet.read_row_group(row_group, columns=["id", "value"])
        matches = selected.filter(pc.equal(selected["id"], row["openalex_id"])).to_pylist()
        require(len(matches) == 1, f"source row lookup: {row['openalex_id']}")
        reconstructed = _reconstruct(matches[0]["value"])
        state, words = _state(reconstructed, minimum_words, maximum_collisions)
        recomputed_states[state] += 1
        expected_values = {
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
        for key, expected in expected_values.items():
            require(row[key] == expected, f"reconstructed field: {key}")

    expected = _mapping(eligibility.get("expected"), "expected")
    require(recomputed_states["ELIGIBLE_ABSTRACT_REVIEW"] == expected["eligible_review_rows"] == 5, "eligible rows")
    require(recomputed_states["NO_LOCAL_ABSTRACT"] == expected["missing_rows"] == 10, "missing rows")
    require(recomputed_states["FRAGMENT_UNUSABLE"] == expected["fragment_rows"] == 1, "fragment rows")
    require(sum(bool(row["has_abstract"]) for row in rows) == expected["reconstructable_rows"] == 6, "reconstructable rows")
    recovery = _mapping(receipt.get("recovery"), "receipt recovery")
    require(recovery.get("source_rows_found") == 16, "receipt rows")
    require(recovery.get("eligible_review_rows") == 5, "receipt eligible")
    require(receipt.get("full_value_column_scan") is False, "full value scan")
    require(receipt.get("network_accessed") is False, "network")
    require(receipt.get("provider_calls") == 0, "provider calls")
    require(receipt.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(receipt.get("semantic_labels_created") == 0, "semantic labels")
    require(receipt.get("production_disposition_created") is False, "production")
    require(receipt.get("outcomes_read") is False, "outcomes")
    return {
        "schema_version": "pivot_feedback_d0_t10151_rank5_local_abstract_recovery_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(config_path),
        "producer_receipt_sha256": _sha256(receipt_path),
        "recovery_parquet_sha256": _sha256(parquet_path),
        "source_rows_verified": 16,
        "reconstructable_rows": 6,
        "eligible_review_rows": 5,
        "missing_rows": 10,
        "fragment_rows": 1,
        "human_review": "NOT_RUN_BY_DESIGN",
        "semantic_labels_created": 0,
        "production_disposition_authorized": False,
        "outcomes_read": False,
        "g4_open": False,
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verification-output", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    source_root = args.source_root
    if source_root is None:
        env_name = str(_yaml(args.config)["openalex_source"]["root_env"])
        value = os.environ.get(env_name)
        if not value:
            print(json.dumps({"status": "BLOCKED_LOCAL_ABSTRACT_RECOVERY_VERIFICATION", "error": f"{env_name} is required"}, sort_keys=True))
            return 2
        source_root = Path(value)
    try:
        result = validate(
            config_path=args.config,
            source_root=source_root,
            output_root=args.output,
        )
        if not args.check_only:
            verification = args.verification_output.absolute()
            require(verification.parent == Path("/ssd"), "verification parent")
            require(not verification.exists(), "verification exists")
            _write(verification, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (LocalAbstractRecoveryVerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_LOCAL_ABSTRACT_RECOVERY_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
