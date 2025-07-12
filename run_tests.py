#!/usr/bin/env python3
"""Simple test runner script for the notely-ml project.

This script provides convenient commands to run different types of tests.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'=' * 60}")
    print(f"🧪 {description}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED")
        return result.returncode
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return 1


def main():
    """Main test runner function."""
    project_root = Path(__file__).parent

    commands = {
        "models": {
            "cmd": [
                "python3",
                "-m",
                "pytest",
                "tests/repositories/drawing_repository.py::TestDrawingModels",
                "-v",
            ],
            "desc": "Running Drawing Models Tests",
        },
        "repository": {
            "cmd": [
                "python3",
                "-m",
                "pytest",
                "tests/repositories/drawing_repository.py::TestDrawingRepository",
                "-v",
            ],
            "desc": "Running Drawing Repository Tests",
        },
        "all": {
            "cmd": [
                "python3",
                "-m",
                "pytest",
                "tests/repositories/drawing_repository.py",
                "-v",
            ],
            "desc": "Running All Drawing Tests",
        },
        "coverage": {
            "cmd": [
                "python3",
                "-m",
                "pytest",
                "tests/repositories/drawing_repository.py",
                "-v",
                "--cov=app",
                "--cov-report=term-missing",
            ],
            "desc": "Running Tests with Coverage Report",
        },
    }

    if len(sys.argv) < 2:
        print("🚀 Notely-ML Test Runner")
        print("\nUsage: python run_tests.py <test_type>")
        print("\nAvailable test types:")
        for key, value in commands.items():
            print(f"  {key:<10} - {value['desc']}")
        print("\nExample: python run_tests.py all")
        return 0

    test_type = sys.argv[1].lower()

    if test_type not in commands:
        print(f"❌ Unknown test type: {test_type}")
        print(f"Available types: {', '.join(commands.keys())}")
        return 1

    # Change to project directory
    import os

    os.chdir(project_root)

    # Run the selected test
    cmd_info = commands[test_type]
    exit_code = run_command(cmd_info["cmd"], cmd_info["desc"])

    print(f"\n{'=' * 60}")
    if exit_code == 0:
        print("🎉 All tests completed successfully!")
    else:
        print("💥 Some tests failed. Check the output above.")
    print(f"{'=' * 60}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
