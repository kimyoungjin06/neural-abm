"""Print a compact torch-free first-run summary for clone-first users."""

from __future__ import annotations

import json
import sys
from typing import Any

from neural_abm.api_lite import toy_catalog, toy_display_name, toys_by_taxonomy


def build_first_run_summary() -> dict[str, Any]:
    catalog = tuple(toy_catalog())
    binary_probability_toys = tuple(
        toys_by_taxonomy("output_family", "binary_probability")
    )
    parity_coverage_toys = tuple(
        toys_by_taxonomy("evidence_role", "parity_coverage")
    )

    return {
        "status": "ok",
        "surface": "neural_abm.api_lite",
        "default_profile": "torch-free",
        "toy_count": len(catalog),
        "recommended_first_toys": [
            {"toy": toy, "display_name": toy_display_name(toy)}
            for toy in binary_probability_toys[:3]
        ],
        "binary_probability_toy_count": len(binary_probability_toys),
        "parity_coverage_toy_count": len(parity_coverage_toys),
        "next_example": "examples/toy_catalog.py",
        "torch_loaded": "torch" in sys.modules,
    }


def main() -> None:
    print(json.dumps(build_first_run_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
