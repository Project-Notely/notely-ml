#!/usr/bin/env python3
"""Development setup scripts for Poetry."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str = "") -> int:
    """Run a command and return the exit code."""
    if description:
        print(f"🔧 {description}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return 1


def main() -> int:
    """Set up development environment."""
    print("🚀 Setting up development environment...")

    # Install pre-commit hooks
    precommit_exit = setup_precommit()

    # Install mypy type stubs
    types_exit = install_types()

    # Create necessary directories
    dirs_exit = create_directories()

    if precommit_exit == 0 and types_exit == 0 and dirs_exit == 0:
        print("✅ Development environment setup completed successfully!")
        print_usage_info()
        return 0
    else:
        print("❌ Some setup steps failed")
        return 1


def setup_precommit() -> int:
    """Set up pre-commit hooks."""
    print("🔧 Setting up pre-commit hooks...")

    # Create .pre-commit-config.yaml if it doesn't exist
    precommit_config = Path(".pre-commit-config.yaml")
    if not precommit_config.exists():
        create_precommit_config()

    # Install pre-commit hooks
    cmd = ["pre-commit", "install"]
    return run_command(cmd, "Installing pre-commit hooks")


def create_precommit_config() -> None:
    """Create .pre-commit-config.yaml file."""
    config_content = """# Pre-commit configuration
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-docstring-first

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/psf/black
    rev: 24.0.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        exclude: ^(tests/|scripts/)
"""

    with open(".pre-commit-config.yaml", "w") as f:
        f.write(config_content)

    print("✅ Created .pre-commit-config.yaml")


def install_types() -> int:
    """Install mypy type stubs."""
    print("🔧 Installing mypy type stubs...")

    cmd = ["mypy", "--install-types", "--non-interactive", "app/"]
    return run_command(cmd, "Installing type stubs")


def create_directories() -> int:
    """Create necessary directories."""
    print("🔧 Creating necessary directories...")

    directories = [
        "output",
        "output/htmlcov",
        "logs",
        "data",
        ".mypy_cache",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    return 0


def print_usage_info() -> None:
    """Print usage information."""
    print("\n" + "=" * 60)
    print("🎉 Development Environment Ready!")
    print("=" * 60)
    print("\n📋 Available Poetry Commands:")
    print("  poetry run test        - Run tests")
    print("  poetry run test-cov    - Run tests with coverage")
    print("  poetry run lint        - Run linting")
    print("  poetry run lint --fix  - Run linting with auto-fix")
    print("  poetry run format      - Format code")
    print("  poetry run format-check - Check code formatting")
    print("  poetry run type-check  - Run type checking")
    print("\n🔧 Development Workflow:")
    print("  1. Make your changes")
    print("  2. Run: poetry run format")
    print("  3. Run: poetry run lint")
    print("  4. Run: poetry run type-check")
    print("  5. Run: poetry run test")
    print("  6. Commit your changes (pre-commit hooks will run)")
    print("\n🪝 Pre-commit hooks are now installed and will run automatically!")
    print("=" * 60)


def clean() -> int:
    """Clean development artifacts."""
    print("🧹 Cleaning development artifacts...")

    # Directories to clean
    clean_dirs = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".coverage",
        "htmlcov",
        "output/htmlcov",
        "*.egg-info",
        ".ruff_cache",
    ]

    for pattern in clean_dirs:
        if "*" in pattern:
            # Use shell command for glob patterns
            cmd = ["sh", "-c", f"rm -rf {pattern}"]
        else:
            # Use rm -rf for specific directories
            cmd = ["rm", "-rf", pattern]

        run_command(cmd, f"Cleaning {pattern}")

    print("✅ Development artifacts cleaned!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
