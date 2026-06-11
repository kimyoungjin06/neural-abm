"""Print a torch-free feature catalog for the built-in toy family."""

from __future__ import annotations

import json
import sys
from typing import Any

from neural_abm.api_lite import toy_catalog, toy_taxonomy_metadata, toys_by_taxonomy


def build_catalog_summary() -> dict[str, Any]:
    catalog = list(toy_catalog())
    return {
        "toy_count": len(catalog),
        "catalog": catalog,
        "binary_probability_toys": list(
            toys_by_taxonomy("output_family", "binary_probability")
        ),
        "parity_coverage_toys": list(
            toys_by_taxonomy("evidence_role", "parity_coverage")
        ),
        "market_ecology": toy_taxonomy_metadata("toy10"),
        "torch_loaded": "torch" in sys.modules,
    }


def main() -> None:
    print(json.dumps(build_catalog_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
