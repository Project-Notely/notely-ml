#!/usr/bin/env python3
"""Linting scripts for Poetry."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str = "") -> int:
    """Run a command and return the exit code."""
    if description:
        print(f"🔍 {description}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return 1


def main() -> int:
    """Run linting with ruff."""
    print("🚀 Running linter...")

    # Run ruff linting
    ruff_cmd = ["ruff", "check", "app/", "tests/", "scripts/"]

    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        ruff_cmd.append("--fix")
        description = "Running ruff with auto-fix"
    else:
        description = "Running ruff linter"

    exit_code = run_command(ruff_cmd, description)

    if exit_code == 0:
        print("✅ Linting passed!")
    else:
        print("❌ Linting failed. Use 'poetry run lint --fix' to auto-fix issues.")

    return exit_code


def lint_diff() -> int:
    """Lint only changed files."""
    print("🚀 Running linter on changed files...")

    # Get changed files
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        changed_files = [
            f
            for f in result.stdout.strip().split("\n")
            if f.endswith(".py") and Path(f).exists()
        ]
    except subprocess.CalledProcessError:
        print("❌ Failed to get changed files from git")
        return 1

    if not changed_files:
        print("✅ No Python files changed")
        return 0

    print(f"🔍 Linting {len(changed_files)} changed files...")

    cmd = ["ruff", "check"] + changed_files

    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        cmd.append("--fix")

    return run_command(cmd, "Running ruff on changed files")


if __name__ == "__main__":
    sys.exit(main())
