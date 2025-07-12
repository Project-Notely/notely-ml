#!/usr/bin/env python3
"""Formatting scripts for Poetry using ruff only."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str = "") -> int:
    """Run a command and return the exit code."""
    if description:
        print(f"✨ {description}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return 1


def main() -> int:
    """Format code with ruff only (replaces black and isort)."""
    print("🚀 Formatting code with ruff...")

    # Use ruff for formatting (replaces black and isort)
    ruff_format_cmd = ["ruff", "format", "app/", "tests/", "scripts/"]
    ruff_format_exit = run_command(ruff_format_cmd, "Formatting code with ruff")

    # Use ruff for auto-fixing linting issues
    ruff_fix_cmd = ["ruff", "check", "--fix", "app/", "tests/", "scripts/"]
    ruff_fix_exit = run_command(ruff_fix_cmd, "Auto-fixing issues with ruff")

    if ruff_format_exit == 0 and ruff_fix_exit == 0:
        print("✅ Code formatting completed successfully!")
        return 0
    else:
        print("❌ Some formatting tools failed")
        return 1


def check() -> int:
    """Check if code is properly formatted without making changes."""
    print("🚀 Checking code formatting...")

    # Check ruff formatting
    ruff_format_cmd = [
        "ruff",
        "format",
        "app/",
        "tests/",
        "scripts/",
        "--check",
        "--diff",
    ]
    ruff_format_exit = run_command(ruff_format_cmd, "Checking ruff formatting")

    # Check ruff linting
    ruff_check_cmd = ["ruff", "check", "app/", "tests/", "scripts/"]
    ruff_check_exit = run_command(ruff_check_cmd, "Checking ruff linting")

    if ruff_format_exit == 0 and ruff_check_exit == 0:
        print("✅ Code is properly formatted!")
        return 0
    else:
        print("❌ Code formatting issues found. Run 'poetry run format' to fix them.")
        return 1


def format_file() -> int:
    """Format a specific file."""
    if len(sys.argv) < 2:
        print("Usage: poetry run format-file <file_path>")
        return 1

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return 1

    if not file_path.endswith(".py"):
        print(f"❌ Not a Python file: {file_path}")
        return 1

    print(f"🚀 Formatting {file_path} with ruff...")

    # Format with ruff
    ruff_format_exit = run_command(["ruff", "format", file_path], "Formatting code")

    # Auto-fix with ruff
    ruff_fix_exit = run_command(
        ["ruff", "check", "--fix", file_path], "Auto-fixing issues"
    )

    if ruff_format_exit == 0 and ruff_fix_exit == 0:
        print(f"✅ {file_path} formatted successfully!")
        return 0
    else:
        print(f"❌ Failed to format {file_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
