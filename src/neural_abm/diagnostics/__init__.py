"""Read-only diagnostics for generated evidence artifacts."""

from __future__ import annotations

from typing import Any

__all__ = [
    "EvidenceProfileIndexOutput",
    "EvidenceProfileOutput",
    "build_evidence_profile_index",
    "profile_evidence_artifacts",
]


def __getattr__(name: str) -> Any:
    if name in {"EvidenceProfileOutput", "profile_evidence_artifacts"}:
        from neural_abm.diagnostics.evidence_profile import (
            EvidenceProfileOutput,
            profile_evidence_artifacts,
        )

        return {
            "EvidenceProfileOutput": EvidenceProfileOutput,
            "profile_evidence_artifacts": profile_evidence_artifacts,
        }[name]
    if name in {"EvidenceProfileIndexOutput", "build_evidence_profile_index"}:
        from neural_abm.diagnostics.profile_index import (
            EvidenceProfileIndexOutput,
            build_evidence_profile_index,
        )

        return {
            "EvidenceProfileIndexOutput": EvidenceProfileIndexOutput,
            "build_evidence_profile_index": build_evidence_profile_index,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
