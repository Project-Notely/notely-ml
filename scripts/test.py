#!/usr/bin/env python3
"""Test runner scripts for Poetry."""

import subprocess
import sys


def run_command(cmd: list[str], description: str = "") -> int:
    """Run a command and return the exit code."""
    if description:
        print(f"🧪 {description}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return 1


def main() -> int:
    """Run tests without coverage."""
    print("🚀 Running tests...")

    cmd = ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

    # Add any additional arguments passed to the script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    return run_command(cmd, "Running pytest")


def main_with_coverage() -> int:
    """Run tests with coverage report."""
    print("🚀 Running tests with coverage...")

    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=app",
        "--cov-report=term-missing",
        "--cov-report=html:output/htmlcov",
    ]

    # Add any additional arguments passed to the script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    exit_code = run_command(cmd, "Running pytest with coverage")

    if exit_code == 0:
        print("\n📊 Coverage report generated in output/htmlcov/")

    return exit_code


def run_specific_test() -> int:
    """Run a specific test file or test function."""
    if len(sys.argv) < 2:
        print("Usage: poetry run test-specific <test_path>")
        print(
            "Example: poetry run test-specific tests/repositories/drawing_repository.py::TestDrawingModels"
        )
        return 1

    test_path = sys.argv[1]

    cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short"]

    return run_command(cmd, f"Running specific test: {test_path}")


if __name__ == "__main__":
    sys.exit(main())
