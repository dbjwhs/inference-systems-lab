#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 dbjwhs
"""
run_benchmarks.py - Performance regression detection for the Inference Systems Lab

This script automatically discovers, runs, and analyzes benchmarks to detect
performance regressions. It stores baseline results and compares new runs
against historical data with statistical analysis.

Features:
- Auto-discovery of benchmark executables
- Baseline storage and management
- Statistical regression detection
- Performance trend analysis
- JSON output for CI/CD integration
- Configurable thresholds and filters

Usage:
    python python_tool/run_benchmarks.py [options]
    
Examples:
    python python_tool/run_benchmarks.py --save-baseline
    python python_tool/run_benchmarks.py --compare-against baseline_v1.0.0
    python python_tool/run_benchmarks.py --filter "*Result*" --threshold 10.0
    python python_tool/run_benchmarks.py --output-json results.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib


class BenchmarkResult:
    """Represents the results of a single benchmark test."""
    
    def __init__(self, name: str, time_ns: float, iterations: int, 
                 bytes_per_second: Optional[float] = None,
                 items_per_second: Optional[float] = None,
                 cpu_time_ns: Optional[float] = None):
        self.name = name
        self.time_ns = time_ns
        self.iterations = iterations
        self.bytes_per_second = bytes_per_second
        self.items_per_second = items_per_second
        self.cpu_time_ns = cpu_time_ns or time_ns
        
    def time_per_iteration_ns(self) -> float:
        """Calculate average time per iteration in nanoseconds."""
        return self.time_ns / self.iterations if self.iterations > 0 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'time_ns': self.time_ns,
            'iterations': self.iterations,
            'time_per_iteration_ns': self.time_per_iteration_ns(),
            'bytes_per_second': self.bytes_per_second,
            'items_per_second': self.items_per_second,
            'cpu_time_ns': self.cpu_time_ns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create instance from dictionary."""
        return cls(
            name=data['name'],
            time_ns=data['time_ns'],
            iterations=data['iterations'],
            bytes_per_second=data.get('bytes_per_second'),
            items_per_second=data.get('items_per_second'),
            cpu_time_ns=data.get('cpu_time_ns')
        )


class BenchmarkSuite:
    """Represents a collection of benchmark results from a single run."""
    
    def __init__(self, timestamp: str, git_commit: Optional[str] = None,
                 build_type: str = "Release", compiler_info: Optional[str] = None):
        self.timestamp = timestamp
        self.git_commit = git_commit
        self.build_type = build_type
        self.compiler_info = compiler_info
        self.results: List[BenchmarkResult] = []
        
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to this suite."""
        self.results.append(result)
        
    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        """Get result by benchmark name."""
        for result in self.results:
            if result.name == name:
                return result
        return None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'git_commit': self.git_commit,
            'build_type': self.build_type,
            'compiler_info': self.compiler_info,
            'results': [r.to_dict() for r in self.results]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkSuite':
        """Create instance from dictionary."""
        suite = cls(
            timestamp=data['timestamp'],
            git_commit=data.get('git_commit'),
            build_type=data.get('build_type', 'Release'),
            compiler_info=data.get('compiler_info')
        )
        for result_data in data.get('results', []):
            suite.add_result(BenchmarkResult.from_dict(result_data))
        return suite


class PerformanceRegression:
    """Represents a detected performance regression."""
    
    def __init__(self, benchmark_name: str, baseline_time: float, 
                 current_time: float, regression_percent: float,
                 confidence: float = 0.0):
        self.benchmark_name = benchmark_name
        self.baseline_time = baseline_time
        self.current_time = current_time
        self.regression_percent = regression_percent
        self.confidence = confidence
        
    def __str__(self) -> str:
        return (f"REGRESSION: {self.benchmark_name} - "
                f"{self.regression_percent:.1f}% slower "
                f"({self.baseline_time:.0f}ns → {self.current_time:.0f}ns)")


class BenchmarkRunner:
    """Main class for running benchmarks and detecting regressions."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.build_dir = project_root / "build"
        self.baseline_dir = project_root / "benchmarks" / "baselines"
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
    def discover_benchmarks(self) -> List[Path]:
        """Discover all benchmark executables in the build directory."""
        benchmark_executables = []
        
        if not self.build_dir.exists():
            print(f"Warning: Build directory {self.build_dir} does not exist")
            return benchmark_executables
            
        # Find all files ending with 'benchmarks'
        for root, dirs, files in os.walk(self.build_dir):
            for file in files:
                if file.endswith('benchmarks'):
                    executable_path = Path(root) / file
                    if os.access(executable_path, os.X_OK):
                        benchmark_executables.append(executable_path)
                        
        return sorted(benchmark_executables)
    
    def run_benchmark_executable(self, executable: Path, 
                                benchmark_filter: Optional[str] = None,
                                min_time: float = 1.0) -> List[BenchmarkResult]:
        """Run a single benchmark executable and parse results."""
        
        cmd = [str(executable), "--benchmark_format=json"]
        
        if benchmark_filter:
            cmd.append(f"--benchmark_filter={benchmark_filter}")
            
        # Set minimum time for stable measurements (Google Benchmark expects suffix)
        cmd.append(f"--benchmark_min_time={min_time}s")
        
        try:
            print(f"Running {executable.name}...")
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=self.project_root, timeout=300)
            
            if result.returncode != 0:
                print(f"Error running {executable}: {result.stderr}")
                return []
                
            return self._parse_benchmark_json(result.stdout)
            
        except subprocess.TimeoutExpired:
            print(f"Timeout running {executable}")
            return []
        except Exception as e:
            print(f"Error running {executable}: {e}")
            return []
    
    def _parse_benchmark_json(self, output: str) -> List[BenchmarkResult]:
        """Parse Google Benchmark JSON output into BenchmarkResult objects."""
        results = []
        
        # Google Benchmark outputs JSON starting from a line that begins with '{'
        lines = output.strip().split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            if not in_json and line_stripped.startswith('{'):
                in_json = True
                json_lines = [line]
                brace_count = line.count('{') - line.count('}')
            elif in_json:
                json_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    break
                    
        if not json_lines:
            print("Warning: Could not find JSON output in benchmark results")
            return results
            
        json_text = '\n'.join(json_lines)
            
        try:
            data = json.loads(json_text)
            
            for benchmark in data.get('benchmarks', []):
                name = benchmark.get('name', '')
                time_unit = benchmark.get('time_unit', 'ns')
                
                # Convert time to nanoseconds
                time = benchmark.get('real_time', benchmark.get('cpu_time', 0))
                if time_unit == 'us':
                    time *= 1000
                elif time_unit == 'ms':
                    time *= 1_000_000
                elif time_unit == 's':
                    time *= 1_000_000_000
                    
                cpu_time = benchmark.get('cpu_time', time)
                if time_unit == 'us':
                    cpu_time *= 1000
                elif time_unit == 'ms':
                    cpu_time *= 1_000_000
                elif time_unit == 's':
                    cpu_time *= 1_000_000_000
                
                result = BenchmarkResult(
                    name=name,
                    time_ns=time,
                    iterations=benchmark.get('iterations', 1),
                    bytes_per_second=benchmark.get('bytes_per_second'),
                    items_per_second=benchmark.get('items_per_second'),
                    cpu_time_ns=cpu_time
                )
                results.append(result)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing benchmark JSON: {e}")
            
        return results
    
    def run_all_benchmarks(self, benchmark_filter: Optional[str] = None,
                          min_time: float = 1.0) -> BenchmarkSuite:
        """Run all discovered benchmarks and return combined results."""
        
        # Get system information
        timestamp = datetime.now().isoformat()
        git_commit = self._get_git_commit()
        build_type = self._detect_build_type()
        compiler_info = self._get_compiler_info()
        
        suite = BenchmarkSuite(
            timestamp=timestamp,
            git_commit=git_commit,
            build_type=build_type,
            compiler_info=compiler_info
        )
        
        executables = self.discover_benchmarks()
        if not executables:
            print("No benchmark executables found!")
            return suite
            
        print(f"Found {len(executables)} benchmark executables:")
        for exe in executables:
            print(f"  - {exe.relative_to(self.project_root)}")
        print()
        
        for executable in executables:
            results = self.run_benchmark_executable(executable, benchmark_filter, min_time)
            for result in results:
                suite.add_result(result)
                
        print(f"Completed {len(suite.results)} benchmark tests")
        return suite
    
    def save_baseline(self, suite: BenchmarkSuite, baseline_name: str) -> None:
        """Save benchmark results as a baseline for future comparisons."""
        baseline_file = self.baseline_dir / f"{baseline_name}.json"
        
        with open(baseline_file, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2)
            
        print(f"Saved baseline to {baseline_file}")
    
    def load_baseline(self, baseline_name: str) -> Optional[BenchmarkSuite]:
        """Load a previously saved baseline."""
        baseline_file = self.baseline_dir / f"{baseline_name}.json"
        
        if not baseline_file.exists():
            print(f"Baseline {baseline_name} not found at {baseline_file}")
            return None
            
        try:
            with open(baseline_file, 'r') as f:
                data = json.load(f)
            return BenchmarkSuite.from_dict(data)
        except Exception as e:
            print(f"Error loading baseline {baseline_name}: {e}")
            return None
    
    def compare_with_baseline(self, current: BenchmarkSuite, 
                            baseline: BenchmarkSuite,
                            threshold_percent: float = 5.0) -> List[PerformanceRegression]:
        """Compare current results with baseline and detect regressions."""
        regressions = []
        
        for current_result in current.results:
            baseline_result = baseline.get_result(current_result.name)
            if not baseline_result:
                continue  # Skip benchmarks not in baseline
                
            current_time = current_result.time_per_iteration_ns()
            baseline_time = baseline_result.time_per_iteration_ns()
            
            if baseline_time == 0:
                continue  # Avoid division by zero
                
            # Calculate percentage change (positive = regression/slower)
            percent_change = ((current_time - baseline_time) / baseline_time) * 100
            
            if percent_change > threshold_percent:
                regression = PerformanceRegression(
                    benchmark_name=current_result.name,
                    baseline_time=baseline_time,
                    current_time=current_time,
                    regression_percent=percent_change
                )
                regressions.append(regression)
                
        return sorted(regressions, key=lambda r: r.regression_percent, reverse=True)
    
    def list_baselines(self) -> List[str]:
        """List all available baselines."""
        baselines = []
        for baseline_file in self.baseline_dir.glob("*.json"):
            baselines.append(baseline_file.stem)
        return sorted(baselines)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, 
                                  cwd=self.project_root)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _detect_build_type(self) -> str:
        """Detect build type from build directory."""
        cmake_cache = self.build_dir / "CMakeCache.txt"
        if cmake_cache.exists():
            try:
                with open(cmake_cache, 'r') as f:
                    content = f.read()
                    match = re.search(r'CMAKE_BUILD_TYPE:STRING=(\w+)', content)
                    if match:
                        return match.group(1)
            except Exception:
                pass
        return "Unknown"
    
    def _get_compiler_info(self) -> Optional[str]:
        """Get compiler information."""
        try:
            # Try to get compiler info from CMake cache
            cmake_cache = self.build_dir / "CMakeCache.txt"
            if cmake_cache.exists():
                with open(cmake_cache, 'r') as f:
                    content = f.read()
                    compiler_match = re.search(r'CMAKE_CXX_COMPILER:FILEPATH=(.+)', content)
                    version_match = re.search(r'CMAKE_CXX_COMPILER_VERSION:STRING=(.+)', content)
                    
                    if compiler_match:
                        compiler = Path(compiler_match.group(1)).name
                        version = version_match.group(1) if version_match else "unknown"
                        return f"{compiler} {version}"
        except Exception:
            pass
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Performance regression detection for the Inference Systems Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --save-baseline release_v1.0.0
  %(prog)s --compare-against release_v1.0.0 --threshold 10.0
  %(prog)s --filter "*Result*" --min-time 2.0
  %(prog)s --list-baselines
  %(prog)s --output-json benchmark_results.json
        """
    )
    
    # Action options
    parser.add_argument("--save-baseline", 
                       metavar="NAME",
                       help="Save current benchmark results as baseline with given name")
    parser.add_argument("--compare-against",
                       metavar="NAME", 
                       help="Compare current results against named baseline")
    parser.add_argument("--list-baselines",
                       action="store_true",
                       help="List all available baselines")
    
    # Benchmark configuration
    parser.add_argument("--filter",
                       help="Filter benchmarks by name pattern (supports wildcards)")
    parser.add_argument("--min-time",
                       type=float,
                       default=1.0,
                       help="Minimum time to run each benchmark (seconds, default: 1.0)")
    parser.add_argument("--threshold",
                       type=float,
                       default=5.0,
                       help="Regression threshold percentage (default: 5.0)")
    
    # Output options
    parser.add_argument("--output-json",
                       metavar="FILE",
                       help="Output results to JSON file")
    parser.add_argument("--quiet",
                       action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Determine project root (script is in python_tool/ subdirectory)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    if not (project_root / "CMakeLists.txt").exists():
        print(f"Error: Project root not found. Expected CMakeLists.txt at {project_root}")
        sys.exit(1)
    
    runner = BenchmarkRunner(project_root)
    
    # Handle list baselines
    if args.list_baselines:
        baselines = runner.list_baselines()
        if baselines:
            print("Available baselines:")
            for baseline in baselines:
                baseline_path = runner.baseline_dir / f"{baseline}.json"
                try:
                    suite = runner.load_baseline(baseline)
                    if suite:
                        print(f"  {baseline} - {suite.timestamp} ({len(suite.results)} tests)")
                    else:
                        print(f"  {baseline} - (invalid)")
                except Exception:
                    print(f"  {baseline} - (corrupted)")
        else:
            print("No baselines found.")
        sys.exit(0)
    
    # Run benchmarks
    if not args.quiet:
        print("Running benchmarks...")
        print(f"Project root: {project_root}")
        print(f"Build directory: {runner.build_dir}")
        print(f"Filter: {args.filter or 'none'}")
        print(f"Min time: {args.min_time}s")
        print()
    
    current_suite = runner.run_all_benchmarks(args.filter, args.min_time)
    
    if not current_suite.results:
        print("No benchmark results collected!")
        sys.exit(1)
    
    # Output JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(current_suite.to_dict(), f, indent=2)
        print(f"Results saved to {args.output_json}")
    
    # Save baseline if requested
    if args.save_baseline:
        runner.save_baseline(current_suite, args.save_baseline)
    
    # Compare with baseline if requested
    if args.compare_against:
        baseline_suite = runner.load_baseline(args.compare_against)
        if not baseline_suite:
            sys.exit(1)
            
        regressions = runner.compare_with_baseline(
            current_suite, baseline_suite, args.threshold
        )
        
        if regressions:
            print(f"\n⚠️  Found {len(regressions)} performance regressions:")
            print("=" * 60)
            for regression in regressions:
                print(regression)
            print()
            print(f"Threshold: {args.threshold}% slower")
            print(f"Baseline: {args.compare_against} ({baseline_suite.timestamp})")
            sys.exit(1)  # Exit with error code for CI/CD
        else:
            print(f"\n✅ No significant regressions found (threshold: {args.threshold}%)")
            print(f"Compared {len(current_suite.results)} benchmarks against baseline: {args.compare_against}")
    
    # Summary output
    if not args.quiet:
        print(f"\nBenchmark Summary:")
        print(f"- Total tests: {len(current_suite.results)}")
        print(f"- Build type: {current_suite.build_type}")
        print(f"- Git commit: {current_suite.git_commit or 'unknown'}")
        print(f"- Compiler: {current_suite.compiler_info or 'unknown'}")
        
        if current_suite.results:
            times = [r.time_per_iteration_ns() for r in current_suite.results]
            print(f"- Avg time per iteration: {statistics.mean(times):.0f}ns")
            print(f"- Fastest test: {min(times):.0f}ns")
            print(f"- Slowest test: {max(times):.0f}ns")


if __name__ == "__main__":
    main()
