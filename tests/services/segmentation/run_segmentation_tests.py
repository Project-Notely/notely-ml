"""
Comprehensive test runner for segmentation service tests
"""
import subprocess
import sys
import time
from pathlib import Path
import argparse


def run_tests(test_type="all", verbose=False, coverage=False):
    """
    Run segmentation tests
    
    Args:
        test_type: Type of tests to run ("unit", "integration", "performance", "all")
        verbose: Whether to run in verbose mode
        coverage: Whether to generate coverage report
    """
    
    base_cmd = ["python", "-m", "pytest"]
    
    # Base test directory
    test_dir = Path(__file__).parent
    
    # Configure test selection
    if test_type == "unit":
        test_files = [
            "test_segmentation_service.py",
            "test_segmentation_models.py",
            "test_segmentation_controller.py"
        ]
    elif test_type == "integration":
        test_files = ["test_segmentation_integration.py"]
    elif test_type == "performance":
        test_files = ["test_segmentation_performance.py"]
    elif test_type == "all":
        test_files = [
            "test_segmentation_service.py",
            "test_segmentation_models.py", 
            "test_segmentation_controller.py",
            "test_segmentation_integration.py",
            "test_segmentation_performance.py"
        ]
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Build command
    cmd = base_cmd.copy()
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=app.services.segmentation_service", 
                   "--cov=app.models.segmentation_models",
                   "--cov=app.controllers.segmentation_controller",
                   "--cov-report=html",
                   "--cov-report=term-missing"])
    
    # Add specific markers for performance tests
    if test_type == "performance":
        cmd.extend(["-m", "performance or stress or limit or benchmark"])
    elif test_type != "all":
        cmd.extend(["-m", "not performance and not stress and not limit and not benchmark"])
    
    # Add test files
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            cmd.append(str(test_path))
        else:
            print(f"Warning: Test file {test_file} not found")
    
    # Add additional pytest options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "-ra",  # Show all test results summary
    ])
    
    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=False)
        end_time = time.time()
        
        print("-" * 80)
        print(f"Tests completed in {end_time - start_time:.2f} seconds")
        
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_quick_smoke_tests():
    """Run a quick smoke test suite"""
    print("Running quick smoke tests...")
    
    test_dir = Path(__file__).parent
    cmd = [
        "python", "-m", "pytest",
        str(test_dir / "test_segmentation_models.py::TestBoundingBox::test_valid_bounding_box"),
        str(test_dir / "test_segmentation_service.py::TestSegmentationService::test_init"),
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running smoke tests: {e}")
        return False


def run_test_discovery():
    """Discover and list all available tests"""
    print("Discovering tests...")
    
    test_dir = Path(__file__).parent
    cmd = [
        "python", "-m", "pytest",
        "--collect-only", 
        "-q",
        str(test_dir)
    ]
    
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"Error discovering tests: {e}")


def check_test_dependencies():
    """Check if required test dependencies are installed"""
    required_packages = [
        "pytest",
        "pytest-asyncio", 
        "pytest-cov",
        "pillow",
        "numpy",
        "psutil",
        "fastapi",
        "unstructured"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All required packages are installed")
        return True


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run segmentation service tests")
    
    parser.add_argument(
        "test_type", 
        nargs="?", 
        default="all",
        choices=["all", "unit", "integration", "performance", "smoke", "discover"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run tests in verbose mode"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true", 
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check test dependencies"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_test_dependencies():
            sys.exit(1)
        return
    
    # Handle special test types
    if args.test_type == "smoke":
        success = run_quick_smoke_tests()
        sys.exit(0 if success else 1)
    
    if args.test_type == "discover":
        run_test_discovery()
        return
    
    # Check dependencies before running tests
    if not check_test_dependencies():
        print("\nPlease install missing dependencies before running tests.")
        sys.exit(1)
    
    # Run tests
    success = run_tests(
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 