#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 dbjwhs
"""
CTest Runner for Inference Systems Laboratory

A comprehensive test runner that manages CTest execution across all project modules.
Provides unified test execution, reporting, and filtering capabilities.

Usage:
    python3 python_tool/run_tests.py                    # Run all tests
    python3 python_tool/run_tests.py --module common    # Run specific module tests  
    python3 python_tool/run_tests.py --list             # List available tests
    python3 python_tool/run_tests.py --verbose          # Verbose output
    python3 python_tool/run_tests.py --filter Result    # Run tests matching pattern
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class TestResult:
    """Represents the result of a test execution."""
    name: str
    module: str
    passed: bool
    duration: float
    output: str = ""
    error_output: str = ""


@dataclass
class ModuleTestSuite:
    """Represents a test suite for a project module."""
    name: str
    path: Path
    test_count: int
    has_tests: bool


class CTestRunner:
    """Manages CTest execution across project modules."""
    
    def __init__(self, build_dir: str = "build"):
        self.project_root = Path(__file__).parent.parent
        self.build_dir = self.project_root / build_dir
        self.modules = ["common", "engines", "distributed", "performance", "integration", "experiments"]
        
        if not self.build_dir.exists():
            raise RuntimeError(f"Build directory not found: {self.build_dir}")
    
    def discover_test_suites(self) -> List[ModuleTestSuite]:
        """Discover available test suites in each module."""
        suites = []
        
        for module in self.modules:
            module_path = self.build_dir / module
            if not module_path.exists():
                continue
                
            # Check if module has CTestTestfile.cmake
            ctest_file = module_path / "CTestTestfile.cmake"
            if not ctest_file.exists():
                continue
            
            # Get test count from ctest
            try:
                result = subprocess.run(
                    ["ctest", "--test-dir", str(module_path), "--show-only"],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
                
                if result.returncode == 0:
                    # Parse test count from output
                    lines = result.stdout.strip().split('\n')
                    total_line = [l for l in lines if l.startswith("Total Tests:")]
                    test_count = int(total_line[0].split(":")[1].strip()) if total_line else 0
                    
                    suites.append(ModuleTestSuite(
                        name=module,
                        path=module_path,
                        test_count=test_count,
                        has_tests=test_count > 0
                    ))
            except Exception:
                # Module exists but ctest failed - create placeholder
                suites.append(ModuleTestSuite(
                    name=module,
                    path=module_path, 
                    test_count=0,
                    has_tests=False
                ))
        
        return suites
    
    def list_tests(self, module_filter: Optional[str] = None) -> Dict[str, List[str]]:
        """List all available tests, optionally filtered by module."""
        test_map = {}
        suites = self.discover_test_suites()
        
        for suite in suites:
            if module_filter and suite.name != module_filter:
                continue
                
            if not suite.has_tests:
                test_map[suite.name] = []
                continue
            
            try:
                result = subprocess.run(
                    ["ctest", "--test-dir", str(suite.path), "--show-only"],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
                
                if result.returncode == 0:
                    # Parse test names from output
                    lines = result.stdout.strip().split('\n')
                    test_lines = [l.strip() for l in lines if l.strip().startswith("Test #")]
                    tests = [l.split(":", 1)[1].strip() for l in test_lines if ":" in l]
                    test_map[suite.name] = tests
                else:
                    test_map[suite.name] = []
            except Exception:
                test_map[suite.name] = []
        
        return test_map
    
    def run_module_tests(self, module: str, verbose: bool = False, 
                        test_filter: Optional[str] = None) -> List[TestResult]:
        """Run tests for a specific module."""
        module_path = self.build_dir / module
        if not module_path.exists():
            return []
        
        ctest_cmd = ["ctest", "--test-dir", str(module_path)]
        
        if verbose:
            ctest_cmd.append("--verbose")
        else:
            ctest_cmd.append("--output-on-failure")
        
        if test_filter:
            ctest_cmd.extend(["-R", test_filter])
        
        start_time = time.time()
        
        try:
            if verbose:
                # In verbose mode, let output stream to console
                result = subprocess.run(
                    ctest_cmd,
                    cwd=str(self.project_root)
                )
                # Create a result-like object for compatibility
                class VerboseResult:
                    def __init__(self, returncode):
                        self.returncode = returncode
                        self.stdout = ""
                        self.stderr = ""
                result = VerboseResult(result.returncode)
            else:
                result = subprocess.run(
                    ctest_cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
            
            duration = time.time() - start_time
            
            # Parse results - simplified for now
            # In a full implementation, we'd parse individual test results
            return [TestResult(
                name=f"{module}_tests",
                module=module,
                passed=result.returncode == 0,
                duration=duration,
                output=result.stdout,
                error_output=result.stderr
            )]
            
        except Exception as e:
            return [TestResult(
                name=f"{module}_tests",
                module=module,
                passed=False,
                duration=time.time() - start_time,
                error_output=str(e)
            )]
    
    def run_all_tests(self, verbose: bool = False, 
                     test_filter: Optional[str] = None,
                     parallel: bool = False) -> List[TestResult]:
        """Run all tests across all modules."""
        all_results = []
        suites = self.discover_test_suites()
        
        # Filter to only suites with tests
        active_suites = [s for s in suites if s.has_tests]
        
        if not active_suites:
            print("No test suites found with tests")
            return []
        
        for suite in active_suites:
            print(f"Running {suite.name} tests ({suite.test_count} tests)...")
            results = self.run_module_tests(suite.name, verbose, test_filter)
            all_results.extend(results)
        
        return all_results
    
    def print_test_summary(self, results: List[TestResult]):
        """Print a summary of test results."""
        if not results:
            print("No tests were run")
            return
        
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        total_time = sum(r.duration for r in results)
        
        print(f"\n{'='*50}")
        print(f"Test Summary")
        print(f"{'='*50}")
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Success rate: {(passed/total)*100:.1f}%")
        
        if failed > 0:
            print(f"\nFailed tests:")
            for result in results:
                if not result.passed:
                    print(f"  - {result.module}::{result.name}")
                    if result.error_output:
                        # Show first few lines of error
                        error_lines = result.error_output.split('\n')[:3]
                        for line in error_lines:
                            print(f"    {line}")
    
    def save_results_json(self, results: List[TestResult], output_file: str):
        """Save test results to JSON file."""
        data = {
            "timestamp": time.time(),
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "results": [
                {
                    "name": r.name,
                    "module": r.module,
                    "passed": r.passed,
                    "duration": r.duration,
                    "output": r.output if r.passed else "",
                    "error": r.error_output if not r.passed else ""
                }
                for r in results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CTest runner for Inference Systems Laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run all tests
  %(prog)s --module common           # Run common module tests
  %(prog)s --list                    # List all available tests
  %(prog)s --filter Result           # Run tests matching 'Result'
  %(prog)s --verbose                 # Verbose output
  %(prog)s --save-json results.json  # Save results to JSON
        """
    )
    
    parser.add_argument("--module", "-m",
                       help="Run tests for specific module only")
    
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available tests and exit")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose test output")
    
    parser.add_argument("--filter", "-f",
                       help="Run only tests matching this pattern (regex)")
    
    parser.add_argument("--build-dir", "-b", default="build",
                       help="Build directory path (default: build)")
    
    parser.add_argument("--save-json", 
                       help="Save results to JSON file")
    
    parser.add_argument("--parallel", action="store_true",
                       help="Run module tests in parallel (experimental)")
    
    args = parser.parse_args()
    
    try:
        runner = CTestRunner(args.build_dir)
        
        if args.list:
            print("Available tests:")
            test_map = runner.list_tests(args.module)
            for module, tests in test_map.items():
                print(f"\n{module} ({len(tests)} tests):")
                for test in tests:
                    print(f"  - {test}")
            return 0
        
        # Run tests
        if args.module:
            results = runner.run_module_tests(
                args.module, args.verbose, args.filter
            )
        else:
            results = runner.run_all_tests(
                args.verbose, args.filter, args.parallel
            )
        
        # Print summary
        runner.print_test_summary(results)
        
        # Save JSON if requested
        if args.save_json:
            runner.save_results_json(results, args.save_json)
        
        # Exit with error code if any tests failed
        failed_count = sum(1 for r in results if not r.passed)
        return 1 if failed_count > 0 else 0
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
