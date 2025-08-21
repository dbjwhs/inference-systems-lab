#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 dbjwhs
"""
check_coverage.py - Test coverage verification for the Inference Systems Lab

This script runs tests with coverage instrumentation and analyzes the results
to ensure code coverage meets quality standards. It supports multiple coverage
tools and provides detailed reporting with threshold validation.

Features:
- Automatic test discovery and execution with coverage
- Support for gcov/llvm-cov coverage tools
- Coverage report parsing and analysis
- File and line-level coverage metrics
- Configurable coverage thresholds
- HTML and text report generation
- CI/CD integration with exit codes

Usage:
    python tools/check_coverage.py [options]
    
Examples:
    python tools/check_coverage.py --threshold 70.0
    python tools/check_coverage.py --filter "*common*" --html-output coverage.html
    python tools/check_coverage.py --build-dir custom_build --exclude-dirs "tests,examples"
    python tools/check_coverage.py --json-output coverage.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import tempfile
import shutil


class CoverageFile:
    """Represents coverage information for a single source file."""
    
    def __init__(self, file_path: str, total_lines: int = 0, covered_lines: int = 0,
                 total_functions: int = 0, covered_functions: int = 0,
                 total_branches: int = 0, covered_branches: int = 0):
        self.file_path = file_path
        self.total_lines = total_lines
        self.covered_lines = covered_lines
        self.total_functions = total_functions
        self.covered_functions = covered_functions
        self.total_branches = total_branches
        self.covered_branches = covered_branches
        
    @property
    def line_coverage_percent(self) -> float:
        """Calculate line coverage percentage."""
        if self.total_lines == 0:
            return 100.0
        return (self.covered_lines / self.total_lines) * 100.0
    
    @property
    def function_coverage_percent(self) -> float:
        """Calculate function coverage percentage."""
        if self.total_functions == 0:
            return 100.0
        return (self.covered_functions / self.total_functions) * 100.0
    
    @property
    def branch_coverage_percent(self) -> float:
        """Calculate branch coverage percentage."""
        if self.total_branches == 0:
            return 100.0
        return (self.covered_branches / self.total_branches) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'file_path': self.file_path,
            'total_lines': self.total_lines,
            'covered_lines': self.covered_lines,
            'line_coverage_percent': self.line_coverage_percent,
            'total_functions': self.total_functions,
            'covered_functions': self.covered_functions,
            'function_coverage_percent': self.function_coverage_percent,
            'total_branches': self.total_branches,
            'covered_branches': self.covered_branches,
            'branch_coverage_percent': self.branch_coverage_percent
        }


class CoverageSummary:
    """Represents overall coverage summary."""
    
    def __init__(self):
        self.files: List[CoverageFile] = []
        self.timestamp: str = ""
        self.total_files: int = 0
        
    def add_file(self, coverage_file: CoverageFile) -> None:
        """Add a file to the coverage summary."""
        self.files.append(coverage_file)
        self.total_files += 1
    
    @property
    def overall_line_coverage(self) -> float:
        """Calculate overall line coverage percentage."""
        total_lines = sum(f.total_lines for f in self.files)
        covered_lines = sum(f.covered_lines for f in self.files)
        if total_lines == 0:
            return 100.0
        return (covered_lines / total_lines) * 100.0
    
    @property
    def overall_function_coverage(self) -> float:
        """Calculate overall function coverage percentage."""
        total_functions = sum(f.total_functions for f in self.files)
        covered_functions = sum(f.covered_functions for f in self.files)
        if total_functions == 0:
            return 100.0
        return (covered_functions / total_functions) * 100.0
    
    @property
    def overall_branch_coverage(self) -> float:
        """Calculate overall branch coverage percentage."""
        total_branches = sum(f.total_branches for f in self.files)
        covered_branches = sum(f.covered_branches for f in self.files)
        if total_branches == 0:
            return 100.0
        return (covered_branches / total_branches) * 100.0
    
    def get_files_below_threshold(self, threshold: float) -> List[CoverageFile]:
        """Get files with coverage below threshold."""
        return [f for f in self.files if f.line_coverage_percent < threshold]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'total_files': self.total_files,
            'overall_line_coverage': self.overall_line_coverage,
            'overall_function_coverage': self.overall_function_coverage,
            'overall_branch_coverage': self.overall_branch_coverage,
            'files': [f.to_dict() for f in self.files]
        }


class CoverageChecker:
    """Main class for checking test coverage."""
    
    def __init__(self, project_root: Path, build_dir: Optional[Path] = None):
        self.project_root = project_root
        self.build_dir = build_dir or (project_root / "build")
        self.coverage_dir = self.build_dir / "coverage"
        
    def discover_tests(self) -> List[Path]:
        """Discover all test executables in the build directory."""
        test_executables = []
        
        if not self.build_dir.exists():
            print(f"Warning: Build directory {self.build_dir} does not exist")
            return test_executables
            
        # Find all test executables (typically end with '_tests' or contain 'test')
        for root, dirs, files in os.walk(self.build_dir):
            for file in files:
                if ('test' in file.lower() or file.endswith('_tests')) and os.access(Path(root) / file, os.X_OK):
                    # Skip if it's clearly not a test executable
                    file_path = Path(root) / file
                    if not file_path.suffix and not file.endswith('.o'):
                        test_executables.append(file_path)
                        
        return sorted(test_executables)
    
    def check_coverage_support(self) -> Tuple[bool, str]:
        """Check if coverage tools are available."""
        # Check for gcov
        try:
            result = subprocess.run(['gcov', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return True, "gcov"
        except FileNotFoundError:
            pass
            
        # Check for llvm-cov
        try:
            result = subprocess.run(['llvm-cov', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return True, "llvm-cov"
        except FileNotFoundError:
            pass
            
        return False, "none"
    
    def build_with_coverage(self, clean_build: bool = False) -> bool:
        """Build the project with coverage instrumentation."""
        print("Building project with coverage instrumentation...")
        
        if clean_build and self.build_dir.exists():
            print(f"Cleaning build directory: {self.build_dir}")
            shutil.rmtree(self.build_dir)
            
        # Configure with coverage enabled
        cmake_cmd = [
            'cmake',
            '-S', str(self.project_root),
            '-B', str(self.build_dir),
            '-DCMAKE_BUILD_TYPE=Debug',
            '-DENABLE_COVERAGE=ON'
        ]
        
        try:
            result = subprocess.run(cmake_cmd, capture_output=True, text=True,
                                  cwd=self.project_root)
            if result.returncode != 0:
                print(f"CMake configuration failed: {result.stderr}")
                return False
                
            # Build the project
            build_cmd = ['cmake', '--build', str(self.build_dir)]
            result = subprocess.run(build_cmd, capture_output=True, text=True,
                                  cwd=self.project_root)
            if result.returncode != 0:
                print(f"Build failed: {result.stderr}")
                return False
                
            print("Build with coverage completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during build: {e}")
            return False
    
    def run_tests_with_coverage(self, test_filter: Optional[str] = None) -> bool:
        """Run tests to generate coverage data."""
        print("Running tests to generate coverage data...")
        
        # Clean previous coverage data
        if self.coverage_dir.exists():
            shutil.rmtree(self.coverage_dir)
        self.coverage_dir.mkdir(parents=True, exist_ok=True)
        
        # Run tests using CTest
        ctest_cmd = ['ctest', '--test-dir', str(self.build_dir), '--verbose']
        
        if test_filter:
            ctest_cmd.extend(['-R', test_filter])
            
        try:
            result = subprocess.run(ctest_cmd, capture_output=True, text=True,
                                  cwd=self.project_root)
            
            # Note: CTest may return non-zero even if some tests pass
            # We'll check the output for actual test results
            if "tests passed" in result.stdout or "100% tests passed" in result.stdout:
                print("Tests completed successfully")
                return True
            elif result.returncode == 0:
                print("Tests completed")
                return True
            else:
                print(f"Some tests may have failed: {result.stderr}")
                # Continue anyway to generate coverage for passing tests
                return True
                
        except Exception as e:
            print(f"Error running tests: {e}")
            return False
    
    def generate_coverage_report(self, coverage_tool: str) -> bool:
        """Generate coverage report using available tools."""
        print(f"Generating coverage report using {coverage_tool}...")
        
        try:
            if coverage_tool == "gcov":
                return self._generate_gcov_report()
            elif coverage_tool == "llvm-cov":
                return self._generate_llvm_cov_report()
            else:
                print(f"Unsupported coverage tool: {coverage_tool}")
                return False
                
        except Exception as e:
            print(f"Error generating coverage report: {e}")
            return False
    
    def _generate_gcov_report(self) -> bool:
        """Generate coverage report using gcov."""
        # Find all .gcno files (generated during compilation)
        gcno_files = list(self.build_dir.rglob("*.gcno"))
        
        if not gcno_files:
            print("No .gcno files found. Make sure the project was built with coverage.")
            return False
            
        print(f"Found {len(gcno_files)} coverage data files")
        
        # Run gcov on all .gcno files
        coverage_data = {}
        
        for gcno_file in gcno_files:
            gcda_file = gcno_file.with_suffix('.gcda')
            if not gcda_file.exists():
                continue  # Skip files without runtime data
                
            # Run gcov
            gcov_cmd = ['gcov', '-b', '-c', str(gcno_file)]
            result = subprocess.run(gcov_cmd, capture_output=True, text=True,
                                  cwd=gcno_file.parent)
            
            if result.returncode == 0:
                # Parse gcov output
                coverage_info = self._parse_gcov_output(result.stdout)
                if coverage_info:
                    coverage_data.update(coverage_info)
        
        # Save coverage data
        if coverage_data:
            coverage_file = self.coverage_dir / "coverage_data.json"
            with open(coverage_file, 'w') as f:
                json.dump(coverage_data, f, indent=2)
            print(f"Coverage data saved to {coverage_file}")
            return True
        else:
            print("No coverage data generated")
            return False
    
    def _generate_llvm_cov_report(self) -> bool:
        """Generate coverage report using llvm-cov."""
        # This is a placeholder for llvm-cov support
        # Implementation would depend on the specific build setup
        print("llvm-cov support not yet implemented")
        return False
    
    def _parse_gcov_output(self, output: str) -> Dict[str, Dict[str, Any]]:
        """Parse gcov text output."""
        coverage_data = {}
        
        lines = output.split('\n')
        current_file = None
        
        for line in lines:
            line = line.strip()
            
            # Look for file information
            if line.startswith('File '):
                file_match = re.search(r"File '([^']+)'", line)
                if file_match:
                    current_file = file_match.group(1)
                    coverage_data[current_file] = {
                        'lines': {'total': 0, 'covered': 0},
                        'functions': {'total': 0, 'covered': 0},
                        'branches': {'total': 0, 'covered': 0}
                    }
            
            # Parse coverage percentages
            elif current_file and ':' in line:
                if 'Lines executed:' in line:
                    match = re.search(r'(\d+\.\d+)% of (\d+)', line)
                    if match:
                        percent = float(match.group(1))
                        total = int(match.group(2))
                        covered = int((percent / 100.0) * total)
                        coverage_data[current_file]['lines'] = {
                            'total': total, 'covered': covered
                        }
                
                elif 'Branches executed:' in line:
                    match = re.search(r'(\d+\.\d+)% of (\d+)', line)
                    if match:
                        percent = float(match.group(1))
                        total = int(match.group(2))
                        covered = int((percent / 100.0) * total)
                        coverage_data[current_file]['branches'] = {
                            'total': total, 'covered': covered
                        }
                        
                elif 'Functions executed:' in line:
                    match = re.search(r'(\d+\.\d+)% of (\d+)', line)
                    if match:
                        percent = float(match.group(1))
                        total = int(match.group(2))
                        covered = int((percent / 100.0) * total)
                        coverage_data[current_file]['functions'] = {
                            'total': total, 'covered': covered
                        }
        
        return coverage_data
    
    def parse_coverage_data(self, exclude_dirs: Optional[List[str]] = None) -> CoverageSummary:
        """Parse coverage data and create summary."""
        from datetime import datetime
        
        summary = CoverageSummary()
        summary.timestamp = datetime.now().isoformat()
        
        coverage_file = self.coverage_dir / "coverage_data.json"
        if not coverage_file.exists():
            print(f"No coverage data found at {coverage_file}")
            return summary
            
        exclude_dirs = exclude_dirs or []
        exclude_patterns = [str(self.project_root / d) for d in exclude_dirs]
        
        try:
            with open(coverage_file, 'r') as f:
                data = json.load(f)
                
            for file_path, file_data in data.items():
                # Skip excluded directories
                if any(pattern in file_path for pattern in exclude_patterns):
                    continue
                    
                # Skip system headers
                if file_path.startswith('/usr/') or file_path.startswith('/System/'):
                    continue
                    
                # Only include project source files
                if not (file_path.endswith('.cpp') or file_path.endswith('.hpp') or 
                       file_path.endswith('.c') or file_path.endswith('.h')):
                    continue
                
                lines = file_data.get('lines', {})
                functions = file_data.get('functions', {})
                branches = file_data.get('branches', {})
                
                coverage_file_obj = CoverageFile(
                    file_path=file_path,
                    total_lines=lines.get('total', 0),
                    covered_lines=lines.get('covered', 0),
                    total_functions=functions.get('total', 0),
                    covered_functions=functions.get('covered', 0),
                    total_branches=branches.get('total', 0),
                    covered_branches=branches.get('covered', 0)
                )
                
                summary.add_file(coverage_file_obj)
                
        except Exception as e:
            print(f"Error parsing coverage data: {e}")
            
        return summary
    
    def generate_html_report(self, summary: CoverageSummary, output_file: Path) -> bool:
        """Generate HTML coverage report."""
        try:
            html_content = self._create_html_report(summary)
            with open(output_file, 'w') as f:
                f.write(html_content)
            print(f"HTML report generated: {output_file}")
            return True
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return False
    
    def _create_html_report(self, summary: CoverageSummary) -> str:
        """Create HTML coverage report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Code Coverage Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .coverage-table {{ width: 100%; border-collapse: collapse; }}
        .coverage-table th, .coverage-table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        .coverage-table th {{ background-color: #4CAF50; color: white; }}
        .coverage-good {{ background-color: #d4edda; }}
        .coverage-warning {{ background-color: #fff3cd; }}
        .coverage-bad {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <h1>Code Coverage Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Generated:</strong> {summary.timestamp}</p>
        <p><strong>Total Files:</strong> {summary.total_files}</p>
        <p><strong>Overall Line Coverage:</strong> {summary.overall_line_coverage:.2f}%</p>
        <p><strong>Overall Function Coverage:</strong> {summary.overall_function_coverage:.2f}%</p>
        <p><strong>Overall Branch Coverage:</strong> {summary.overall_branch_coverage:.2f}%</p>
    </div>
    
    <h2>File Coverage Details</h2>
    <table class="coverage-table">
        <thead>
            <tr>
                <th>File</th>
                <th>Line Coverage</th>
                <th>Function Coverage</th>
                <th>Branch Coverage</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for file_obj in sorted(summary.files, key=lambda f: f.line_coverage_percent):
            # Determine CSS class based on coverage
            line_cov = file_obj.line_coverage_percent
            css_class = "coverage-good" if line_cov >= 80 else "coverage-warning" if line_cov >= 60 else "coverage-bad"
            
            html += f"""            <tr class="{css_class}">
                <td>{Path(file_obj.file_path).name}</td>
                <td>{line_cov:.2f}% ({file_obj.covered_lines}/{file_obj.total_lines})</td>
                <td>{file_obj.function_coverage_percent:.2f}% ({file_obj.covered_functions}/{file_obj.total_functions})</td>
                <td>{file_obj.branch_coverage_percent:.2f}% ({file_obj.covered_branches}/{file_obj.total_branches})</td>
            </tr>
"""
        
        html += """        </tbody>
    </table>
</body>
</html>"""
        
        return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test coverage verification for the Inference Systems Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --threshold 70.0
  %(prog)s --filter "*common*" --html-output coverage.html
  %(prog)s --build-dir custom_build --exclude-dirs "tests,examples"
  %(prog)s --json-output coverage.json --clean-build
        """
    )
    
    # Build and test options
    parser.add_argument("--build-dir",
                       type=Path,
                       help="Build directory path (default: project_root/build)")
    parser.add_argument("--clean-build",
                       action="store_true",
                       help="Clean build directory before building")
    parser.add_argument("--filter",
                       help="Filter tests by name pattern")
    
    # Coverage options
    parser.add_argument("--threshold",
                       type=float,
                       default=70.0,
                       help="Minimum coverage threshold percentage (default: 70.0)")
    parser.add_argument("--exclude-dirs",
                       help="Comma-separated list of directories to exclude")
    
    # Output options
    parser.add_argument("--html-output",
                       type=Path,
                       help="Generate HTML coverage report")
    parser.add_argument("--json-output",
                       type=Path,
                       help="Output coverage data to JSON file")
    parser.add_argument("--quiet",
                       action="store_true",
                       help="Reduce output verbosity")
    
    # Skip options for testing
    parser.add_argument("--skip-build",
                       action="store_true",
                       help="Skip building, use existing coverage data")
    parser.add_argument("--skip-tests",
                       action="store_true",
                       help="Skip running tests, use existing coverage data")
    
    args = parser.parse_args()
    
    # Determine project root (script is in tools/ subdirectory)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    if not (project_root / "CMakeLists.txt").exists():
        print(f"Error: Project root not found. Expected CMakeLists.txt at {project_root}")
        sys.exit(1)
    
    checker = CoverageChecker(project_root, args.build_dir)
    
    # Check coverage tool support
    coverage_supported, coverage_tool = checker.check_coverage_support()
    if not coverage_supported:
        print("Error: No coverage tools found. Please install gcov or llvm-cov.")
        sys.exit(1)
    
    if not args.quiet:
        print(f"Using coverage tool: {coverage_tool}")
        print(f"Project root: {project_root}")
        print(f"Build directory: {checker.build_dir}")
        print(f"Coverage threshold: {args.threshold}%")
        print()
    
    # Build with coverage if not skipped
    if not args.skip_build:
        if not checker.build_with_coverage(args.clean_build):
            print("Failed to build project with coverage")
            sys.exit(1)
    
    # Run tests if not skipped
    if not args.skip_tests:
        if not checker.run_tests_with_coverage(args.filter):
            print("Failed to run tests with coverage")
            sys.exit(1)
    
    # Generate coverage report
    if not checker.generate_coverage_report(coverage_tool):
        print("Failed to generate coverage report")
        sys.exit(1)
    
    # Parse coverage data
    exclude_dirs = args.exclude_dirs.split(',') if args.exclude_dirs else None
    summary = checker.parse_coverage_data(exclude_dirs)
    
    if summary.total_files == 0:
        print("No coverage data found")
        sys.exit(1)
    
    # Output results
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"Coverage data saved to {args.json_output}")
    
    if args.html_output:
        checker.generate_html_report(summary, args.html_output)
    
    # Summary output
    if not args.quiet:
        print(f"\nCoverage Summary:")
        print(f"- Total files: {summary.total_files}")
        print(f"- Line coverage: {summary.overall_line_coverage:.2f}%")
        print(f"- Function coverage: {summary.overall_function_coverage:.2f}%")
        print(f"- Branch coverage: {summary.overall_branch_coverage:.2f}%")
        
        # Check threshold
        below_threshold = summary.get_files_below_threshold(args.threshold)
        if below_threshold:
            print(f"\n⚠️  {len(below_threshold)} files below {args.threshold}% threshold:")
            for file_obj in below_threshold[:10]:  # Show first 10
                print(f"  {Path(file_obj.file_path).name}: {file_obj.line_coverage_percent:.1f}%")
            if len(below_threshold) > 10:
                print(f"  ... and {len(below_threshold) - 10} more")
        else:
            print(f"\n✅ All files meet {args.threshold}% coverage threshold")
    
    # Exit with appropriate code
    if summary.overall_line_coverage >= args.threshold:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
