#!/usr/bin/env python3
"""Formatting scripts for Poetry."""

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
    """Format code with black and isort."""
    print("🚀 Formatting code...")

    # First run isort to sort imports
    isort_cmd = ["isort", "app/", "tests/", "scripts/"]
    isort_exit = run_command(isort_cmd, "Sorting imports with isort")

    # Then run black to format code
    black_cmd = ["black", "app/", "tests/", "scripts/"]
    black_exit = run_command(black_cmd, "Formatting code with black")

    # Run ruff formatting as well
    ruff_cmd = ["ruff", "format", "app/", "tests/", "scripts/"]
    ruff_exit = run_command(ruff_cmd, "Formatting code with ruff")

    if isort_exit == 0 and black_exit == 0 and ruff_exit == 0:
        print("✅ Code formatting completed successfully!")
        return 0
    else:
        print("❌ Some formatting tools failed")
        return 1


def check() -> int:
    """Check if code is properly formatted without making changes."""
    print("🚀 Checking code formatting...")

    # Check isort
    isort_cmd = ["isort", "app/", "tests/", "scripts/", "--check-only", "--diff"]
    isort_exit = run_command(isort_cmd, "Checking import sorting")

    # Check black
    black_cmd = ["black", "app/", "tests/", "scripts/", "--check", "--diff"]
    black_exit = run_command(black_cmd, "Checking code formatting")

    # Check ruff formatting
    ruff_cmd = ["ruff", "format", "app/", "tests/", "scripts/", "--check", "--diff"]
    ruff_exit = run_command(ruff_cmd, "Checking ruff formatting")

    if isort_exit == 0 and black_exit == 0 and ruff_exit == 0:
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

    print(f"🚀 Formatting {file_path}...")

    # Format with isort
    isort_exit = run_command(["isort", file_path], "Sorting imports")

    # Format with black
    black_exit = run_command(["black", file_path], "Formatting code")

    # Format with ruff
    ruff_exit = run_command(["ruff", "format", file_path], "Ruff formatting")

    if isort_exit == 0 and black_exit == 0 and ruff_exit == 0:
        print(f"✅ {file_path} formatted successfully!")
        return 0
    else:
        print(f"❌ Failed to format {file_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
