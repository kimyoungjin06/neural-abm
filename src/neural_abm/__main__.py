"""Small package entry point."""

from __future__ import annotations

from neural_abm import __version__


def main() -> None:
    print(f"neural-abm {__version__}")


if __name__ == "__main__":
    main()
