#!/usr/bin/env python3
"""Linting scripts for Poetry."""

import sys
from subprocess import run


def main() -> int:
    """Run linting with ruff."""
    print("🚀 Running linter...")
    cmd = ["ruff", "check", "app/", "tests/", "scripts/"]

    # Pass any extra arguments to ruff check
    cmd.extend(sys.argv[1:])

    exit_code = run(cmd).returncode
    if exit_code == 0:
        print("✅ Linting passed!")
    else:
        print("❌ Linting failed.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
