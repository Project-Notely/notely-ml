#!/usr/bin/env python3
"""Testing scripts for Poetry."""

import sys
from subprocess import run


def main() -> int:
    """Run tests with pytest."""
    print("🚀 Running tests...")
    # Pass any extra arguments to pytest
    return run(["pytest", *sys.argv[1:]]).returncode


if __name__ == "__main__":
    sys.exit(main())
