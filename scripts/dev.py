#!/usr/bin/env python3
"""Development server script."""

import sys
from subprocess import run


def main() -> int:
    """Run the uvicorn development server."""
    port = 8888
    print(f"🚀 Starting development server at http://0.0.0.0:{port}")
    cmd = [
        "uvicorn",
        "app.main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    return run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
