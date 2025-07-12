#!/usr/bin/env python3
"""Type checking scripts for Poetry."""

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
    """Run type checking with mypy."""
    print("🚀 Running type checker...")

    # Run mypy on the main app directory
    mypy_cmd = [
        "mypy",
        "app/",
        "--show-error-codes",
        "--show-error-context",
        "--pretty",
    ]

    exit_code = run_command(mypy_cmd, "Running mypy type checker")

    if exit_code == 0:
        print("✅ Type checking passed!")
    else:
        print("❌ Type checking failed. Fix the issues above.")

    return exit_code


def check_file() -> int:
    """Type check a specific file."""
    if len(sys.argv) < 2:
        print("Usage: poetry run type-check-file <file_path>")
        return 1

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return 1

    if not file_path.endswith(".py"):
        print(f"❌ Not a Python file: {file_path}")
        return 1

    print(f"🚀 Type checking {file_path}...")

    cmd = ["mypy", file_path, "--show-error-codes", "--show-error-context", "--pretty"]

    exit_code = run_command(cmd, f"Type checking {file_path}")

    if exit_code == 0:
        print(f"✅ {file_path} type checking passed!")
    else:
        print(f"❌ {file_path} type checking failed.")

    return exit_code


def check_strict() -> int:
    """Run type checking with strict mode."""
    print("🚀 Running strict type checker...")

    cmd = [
        "mypy",
        "app/",
        "--strict",
        "--show-error-codes",
        "--show-error-context",
        "--pretty",
    ]

    exit_code = run_command(cmd, "Running mypy in strict mode")

    if exit_code == 0:
        print("✅ Strict type checking passed!")
    else:
        print("❌ Strict type checking failed.")

    return exit_code


def install_types() -> int:
    """Install missing type stubs."""
    print("🚀 Installing missing type stubs...")

    cmd = ["mypy", "--install-types", "--non-interactive"]

    exit_code = run_command(cmd, "Installing type stubs")

    if exit_code == 0:
        print("✅ Type stubs installed successfully!")
    else:
        print("❌ Failed to install type stubs.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
