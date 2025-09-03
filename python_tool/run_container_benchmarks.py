#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 dbjwhs
#
# This software is provided "as is" without warranty of any kind, express or implied.
# The authors are not liable for any damages arising from the use of this software.

"""
Container Benchmarking Script for Inference Systems Lab

This script runs comprehensive performance benchmarks comparing our custom containers
against standard library implementations. It saves results to JSON for analysis and
generates performance reports.

Usage:
    python3 python_tool/run_container_benchmarks.py [--save-baseline NAME] [--compare-against NAME]
    python3 python_tool/run_container_benchmarks.py --quick  # Run quick benchmarks
    python3 python_tool/run_container_benchmarks.py --report # Generate markdown report
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ContainerBenchmarkRunner:
    """Manages container performance benchmarking and analysis"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.benchmarks_dir = project_root / "benchmarks" / "baselines" / "containers"
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)
        self.executable_path = None
        self._find_benchmark_executable()

    def _find_benchmark_executable(self) -> None:
        """Find container benchmarks executable in build directories"""
        possible_paths = [
            self.project_root / "build" / "common" / "container_benchmarks",
            self.project_root / "cmake-build-debug" / "common" / "container_benchmarks",
            self.project_root / "cmake-build-release" / "common" / "container_benchmarks",
        ]

        for path in possible_paths:
            if path.exists() and os.access(path, os.X_OK):
                self.executable_path = path
                break

        if not self.executable_path:
            print("❌ Container benchmarks executable not found")
            print("Please build the project first:")
            print("  mkdir build && cd build && cmake .. && make container_benchmarks")
            sys.exit(1)

    def run_benchmarks(self, quick: bool = False, filter_pattern: str = "") -> Dict:
        """Run container benchmarks and return parsed results"""
        print(f"🏃 Running container benchmarks using: {self.executable_path}")

        args = [str(self.executable_path)]
        
        if quick:
            args.extend(["--benchmark_min_time=0.1s", "--benchmark_repetitions=3"])
        else:
            args.extend(["--benchmark_min_time=0.5s", "--benchmark_repetitions=5"])

        if filter_pattern:
            args.extend([f"--benchmark_filter={filter_pattern}"])

        args.extend([
            "--benchmark_format=json",
            "--benchmark_out_format=json",
            "--benchmark_display_aggregates_only=true",
        ])

        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=300, cwd=self.executable_path.parent
            )
            
            if result.returncode != 0:
                print(f"❌ Benchmark execution failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return {}

            # Parse JSON output (benchmark results are in stdout)
            output_lines = result.stdout.strip().split('\n')
            json_line = None
            
            for line in output_lines:
                if line.startswith('{') and '"benchmarks"' in line:
                    json_line = line
                    break

            if not json_line:
                print("❌ Could not find JSON output in benchmark results")
                print("Raw output:")
                print(result.stdout)
                return {}

            benchmark_data = json.loads(json_line)
            return benchmark_data

        except subprocess.TimeoutExpired:
            print("❌ Benchmark execution timed out")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse benchmark JSON: {e}")
            return {}
        except Exception as e:
            print(f"❌ Unexpected error running benchmarks: {e}")
            return {}

    def save_baseline(self, name: str, results: Dict) -> None:
        """Save benchmark results as a named baseline"""
        baseline_file = self.benchmarks_dir / f"{name}.json"
        
        metadata = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "system_info": results.get("context", {}),
            "results": results
        }

        with open(baseline_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved baseline '{name}' to {baseline_file}")

    def load_baseline(self, name: str) -> Optional[Dict]:
        """Load a saved baseline"""
        baseline_file = self.benchmarks_dir / f"{name}.json"
        
        if not baseline_file.exists():
            print(f"❌ Baseline '{name}' not found at {baseline_file}")
            return None

        with open(baseline_file, 'r') as f:
            return json.load(f)

    def compare_baselines(self, current: Dict, baseline_name: str) -> None:
        """Compare current results against a saved baseline"""
        baseline = self.load_baseline(baseline_name)
        if not baseline:
            return

        print(f"\n📊 Comparing against baseline '{baseline_name}'")
        print("=" * 80)

        current_benchmarks = {b["name"]: b for b in current.get("benchmarks", [])}
        baseline_benchmarks = {b["name"]: b for b in baseline["results"].get("benchmarks", [])}

        for name, current_bench in current_benchmarks.items():
            if name in baseline_benchmarks:
                baseline_bench = baseline_benchmarks[name]
                
                current_time = current_bench.get("cpu_time", 0)
                baseline_time = baseline_bench.get("cpu_time", 0)
                
                if baseline_time > 0:
                    improvement = ((baseline_time - current_time) / baseline_time) * 100
                    
                    if improvement > 5:
                        print(f"🟢 {name}: {improvement:+.1f}% faster")
                    elif improvement < -5:
                        print(f"🔴 {name}: {improvement:+.1f}% slower")
                    else:
                        print(f"🟡 {name}: {improvement:+.1f}% (similar)")

    def generate_report(self, results: Dict) -> str:
        """Generate a markdown performance report"""
        report = []
        report.append("# Container Performance Benchmark Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if "context" in results:
            context = results["context"]
            report.append("## System Information")
            report.append(f"- **CPU**: {context.get('host_name', 'Unknown')}")
            report.append(f"- **Date**: {context.get('date', 'Unknown')}")
            report.append(f"- **Library Version**: {context.get('library_build_type', 'Unknown')}")
            report.append("")

        report.append("## Benchmark Results\n")
        
        # Group benchmarks by container type
        container_groups = {}
        for benchmark in results.get("benchmarks", []):
            name = benchmark["name"]
            container_type = name.split("_")[1] if "_" in name else "Other"
            
            if container_type not in container_groups:
                container_groups[container_type] = []
            container_groups[container_type].append(benchmark)

        for container_type, benchmarks in container_groups.items():
            report.append(f"### {container_type} Performance")
            report.append("| Benchmark | Time (ns) | Throughput | Notes |")
            report.append("|-----------|-----------|------------|-------|")
            
            for bench in benchmarks:
                name = bench["name"]
                cpu_time = bench.get("cpu_time", 0)
                
                # Extract throughput info from counters
                throughput = ""
                if "items_per_second" in bench:
                    throughput = f"{bench['items_per_second'] / 1e6:.1f}M items/s"
                elif "bytes_per_second" in bench:
                    throughput = f"{bench['bytes_per_second'] / 1e9:.1f} GB/s"
                
                notes = "Custom implementation" if "MemoryPool" in name or "LockFree" in name or "Ring" in name or "Tensor" in name else "Standard library"
                
                report.append(f"| {name} | {cpu_time:,.0f} | {throughput} | {notes} |")
            
            report.append("")

        report.append("## Performance Analysis\n")
        report.append("### Key Findings:")
        
        # Simple analysis - compare custom vs std implementations
        custom_faster = []
        std_faster = []
        
        benchmarks = results.get("benchmarks", [])
        for i in range(0, len(benchmarks) - 1, 2):
            if i + 1 < len(benchmarks):
                bench1 = benchmarks[i]
                bench2 = benchmarks[i + 1]
                
                custom_bench = bench1 if any(x in bench1["name"] for x in ["MemoryPool", "RingBuffer", "LockFree", "Tensor"]) else bench2
                std_bench = bench2 if custom_bench == bench1 else bench1
                
                if custom_bench["cpu_time"] < std_bench["cpu_time"]:
                    speedup = std_bench["cpu_time"] / custom_bench["cpu_time"]
                    custom_faster.append(f"{custom_bench['name']}: {speedup:.1f}x faster than std")
                else:
                    slowdown = custom_bench["cpu_time"] / std_bench["cpu_time"]
                    std_faster.append(f"{std_bench['name']}: {slowdown:.1f}x faster than custom")

        if custom_faster:
            report.append("✅ **Custom containers outperform standard library in:**")
            for item in custom_faster:
                report.append(f"- {item}")
            report.append("")

        if std_faster:
            report.append("⚠️ **Standard library outperforms custom containers in:**")
            for item in std_faster:
                report.append(f"- {item}")
            report.append("")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Run container performance benchmarks")
    parser.add_argument("--save-baseline", help="Save results as named baseline")
    parser.add_argument("--compare-against", help="Compare against saved baseline")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks")
    parser.add_argument("--filter", help="Filter benchmarks by pattern")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    runner = ContainerBenchmarkRunner(project_root)

    if args.report:
        # Load the most recent baseline for report generation
        latest_baseline = None
        for baseline_file in runner.benchmarks_dir.glob("*.json"):
            if latest_baseline is None or baseline_file.stat().st_mtime > latest_baseline.stat().st_mtime:
                latest_baseline = baseline_file

        if latest_baseline:
            with open(latest_baseline, 'r') as f:
                baseline_data = json.load(f)
            
            report = runner.generate_report(baseline_data["results"])
            report_file = runner.benchmarks_dir / "performance_report.md"
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"📄 Generated performance report: {report_file}")
            print("\nReport preview:")
            print(report[:500] + "..." if len(report) > 500 else report)
        else:
            print("❌ No baseline data found. Run benchmarks first.")
        return

    # Run benchmarks
    results = runner.run_benchmarks(quick=args.quick, filter_pattern=args.filter or "")
    
    if not results:
        print("❌ No benchmark results obtained")
        return

    print(f"\n✅ Completed {len(results.get('benchmarks', []))} benchmarks")

    # Save baseline if requested
    if args.save_baseline:
        runner.save_baseline(args.save_baseline, results)

    # Compare against baseline if requested
    if args.compare_against:
        runner.compare_baselines(results, args.compare_against)

    # Always generate a current report
    report = runner.generate_report(results)
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(report)


if __name__ == "__main__":
    main()
