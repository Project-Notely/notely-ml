#!/usr/bin/env python3
"""Formatting scripts for Poetry."""

import sys
from subprocess import run


def main() -> int:
    """Format code with ruff."""
    print("🚀 Formatting code with ruff...")
    # Pass any extra arguments to ruff format
    return run(
        ["ruff", "format", "app/", "tests/", "scripts/", *sys.argv[1:]]
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
