#!/usr/bin/env python3
"""Type checking scripts for Poetry."""

import sys
from subprocess import run


def main() -> int:
    """Run type checking with mypy."""
    print("🚀 Running type checker...")

    # Run mypy on the main app directory
    mypy_cmd = [
        "mypy",
        "app/",
        "--show-error-codes",
        "--show-error-context",
        "--pretty",
        *sys.argv[1:],
    ]

    exit_code = run(mypy_cmd).returncode

    if exit_code == 0:
        print("✅ Type checking passed!")
    else:
        print("❌ Type checking failed. Fix the issues above.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
