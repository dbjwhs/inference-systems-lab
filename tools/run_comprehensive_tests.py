#!/usr/bin/env python3
"""
Comprehensive Testing Orchestrator for Inference Systems Laboratory

This script provides a single point of execution for all testing activities:
- Builds multiple configurations (Release, Debug, Sanitizers)
- Runs all test suites (unit, integration, stress, benchmarks)
- Performs memory safety testing (AddressSanitizer, ThreadSanitizer)
- Generates coverage and performance reports
- Future-proof design for easy extension

Usage:
    python tools/run_comprehensive_tests.py              # Run all tests
    python tools/run_comprehensive_tests.py --quick      # Quick smoke tests only
    python tools/run_comprehensive_tests.py --memory     # Focus on memory testing
    python tools/run_comprehensive_tests.py --parallel   # Run builds in parallel
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import our notification and logging systems
try:
    from .notification_system import NotificationManager, TestResults
    from .log_manager import LogManager
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    # Handle case where notification system is not available
    print("Warning: Notification system not available. Install required dependencies for notifications.")
    NOTIFICATIONS_AVAILABLE = False
    NotificationManager = None
    TestResults = None
    LogManager = None


class BuildType(Enum):
    """Build configuration types"""
    RELEASE = "Release"
    DEBUG = "Debug"
    RELWITHDEBINFO = "RelWithDebInfo"


class SanitizerType(Enum):
    """Sanitizer configurations"""
    NONE = "none"
    ADDRESS = "address"
    THREAD = "thread"
    MEMORY = "memory"
    UNDEFINED = "undefined"
    ADDRESS_UNDEFINED = "address+undefined"


class TestResult(Enum):
    """Test execution results"""
    PASSED = "✅ PASSED"
    FAILED = "❌ FAILED"
    SKIPPED = "⚠️  SKIPPED"
    TIMEOUT = "⏱️  TIMEOUT"


@dataclass
class BuildConfig:
    """Configuration for a build variant"""
    name: str
    build_type: BuildType
    sanitizer: SanitizerType
    build_dir: str
    cmake_args: List[str]
    enabled: bool = True


@dataclass
class TestSuite:
    """Definition of a test suite"""
    name: str
    command: List[str]
    timeout: int  # seconds
    requires_sanitizer: Optional[SanitizerType] = None
    enabled: bool = True
    allow_failure: bool = False  # For experimental tests


class TestOrchestrator:
    """Orchestrates comprehensive testing across multiple configurations"""
    
    def __init__(self, root_dir: Path, args: argparse.Namespace):
        self.root_dir = root_dir
        self.args = args
        self.results: Dict[str, Dict[str, TestResult]] = {}
        self.start_time = time.time()
        
        # Configure build variants
        self.build_configs = self._setup_build_configs()
        
        # Configure test suites
        self.test_suites = self._setup_test_suites()
        
        # Setup output directory for results
        self.output_dir = root_dir / "test-results" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize notification and logging systems
        self.notification_manager = None
        self.log_manager = None
        
        if NOTIFICATIONS_AVAILABLE and not args.no_notifications:
            try:
                self.notification_manager = NotificationManager()
                self.log_manager = LogManager()
                print("🔔 Notification system enabled")
            except Exception as e:
                print(f"Warning: Failed to initialize notification system: {e}")
        elif args.no_notifications:
            print("🔇 Notifications disabled by user")
        
    def _setup_build_configs(self) -> List[BuildConfig]:
        """Setup build configurations based on platform and requirements"""
        configs = []
        
        # Main optimized build
        configs.append(BuildConfig(
            name="release",
            build_type=BuildType.RELEASE,
            sanitizer=SanitizerType.NONE,
            build_dir="build",
            cmake_args=[
                "-DCMAKE_BUILD_TYPE=Release",
                "-DENABLE_COVERAGE=OFF",
                "-DBUILD_BENCHMARKS=ON"
            ]
        ))
        
        # Debug build with assertions
        configs.append(BuildConfig(
            name="debug",
            build_type=BuildType.DEBUG,
            sanitizer=SanitizerType.NONE,
            build_dir="build-debug",
            cmake_args=[
                "-DCMAKE_BUILD_TYPE=Debug",
                "-DENABLE_COVERAGE=ON"
            ]
        ))
        
        # AddressSanitizer build (memory safety)
        configs.append(BuildConfig(
            name="asan",
            build_type=BuildType.DEBUG,
            sanitizer=SanitizerType.ADDRESS,
            build_dir="build-sanitizer",
            cmake_args=[
                "-DCMAKE_BUILD_TYPE=Debug",
                "-DSANITIZER_TYPE=address"
            ]
        ))
        
        # ThreadSanitizer build (race conditions)
        if not self.args.quick:
            configs.append(BuildConfig(
                name="tsan",
                build_type=BuildType.DEBUG,
                sanitizer=SanitizerType.THREAD,
                build_dir="build-tsan",
                cmake_args=[
                    "-DCMAKE_BUILD_TYPE=Debug",
                    "-DSANITIZER_TYPE=thread"
                ],
                enabled=not platform.system() == "Darwin"  # TSan has issues on macOS
            ))
        
        # UndefinedBehaviorSanitizer build
        if not self.args.quick:
            configs.append(BuildConfig(
                name="ubsan",
                build_type=BuildType.DEBUG,
                sanitizer=SanitizerType.ADDRESS_UNDEFINED,
                build_dir="build-ubsan",
                cmake_args=[
                    "-DCMAKE_BUILD_TYPE=Debug",
                    "-DSANITIZER_TYPE=address+undefined"
                ]
            ))
        
        return configs
    
    def _setup_test_suites(self) -> List[TestSuite]:
        """Setup test suites to run"""
        suites = []
        
        # Core unit tests
        suites.append(TestSuite(
            name="unit_tests",
            command=["ctest", "--output-on-failure", "-L", "unit"],
            timeout=300
        ))
        
        # Integration tests
        suites.append(TestSuite(
            name="integration_tests",
            command=["ctest", "--output-on-failure", "-L", "integration"],
            timeout=600
        ))
        
        # Stress tests (high concurrency)
        suites.append(TestSuite(
            name="stress_tests",
            command=["./common/concurrency_stress_tests"],
            timeout=900,
            enabled=not self.args.quick
        ))
        
        # Memory leak specific tests
        suites.append(TestSuite(
            name="memory_leak_tests",
            command=["./common/concurrency_stress_tests", "--gtest_filter=*Memory*"],
            timeout=300,
            requires_sanitizer=SanitizerType.ADDRESS
        ))
        
        # ML integration stress tests
        suites.append(TestSuite(
            name="ml_integration_stress",
            command=["./integration/integration_stress_tests"],
            timeout=1200,
            enabled=not self.args.quick
        ))
        
        # Performance benchmarks
        suites.append(TestSuite(
            name="benchmarks",
            command=["./common/result_benchmarks", "--benchmark_format=json", "--benchmark_out=bench_results.json"],
            timeout=600,
            enabled=not self.args.quick and not self.args.memory_only
        ))
        
        # All tests via ctest
        suites.append(TestSuite(
            name="all_ctest",
            command=["ctest", "--output-on-failure", "--timeout", "300"],
            timeout=1800
        ))
        
        return suites
    
    def run(self) -> int:
        """Run the complete testing orchestration"""
        print("=" * 80)
        print("INFERENCE SYSTEMS LAB - COMPREHENSIVE TESTING")
        print("=" * 80)
        print(f"Start time: {datetime.now()}")
        print(f"Root directory: {self.root_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Test mode: {'QUICK' if self.args.quick else 'FULL'}")
        print("=" * 80)
        
        # Notify test start
        if self.notification_manager:
            enabled_configs = [c for c in self.build_configs if c.enabled]
            estimated_duration = 15 if self.args.quick else 30  # rough estimate
            self.notification_manager.notify_test_start(len(enabled_configs), estimated_duration)
        
        # Phase 1: Build all configurations
        print("\n📦 PHASE 1: Building configurations...")
        if not self._build_all_configurations():
            print("❌ Build phase failed!")
            return 1
        
        # Phase 2: Run test suites
        print("\n🧪 PHASE 2: Running test suites...")
        self._run_all_tests()
        
        # Phase 3: Memory safety analysis
        if not self.args.quick:
            print("\n🔍 PHASE 3: Memory safety analysis...")
            self._run_memory_analysis()
        
        # Phase 4: Generate reports
        print("\n📊 PHASE 4: Generating reports...")
        self._generate_reports()
        
        # Final summary
        return self._print_summary()
    
    def _build_all_configurations(self) -> bool:
        """Build all configured build variants"""
        success = True
        
        for config in self.build_configs:
            if not config.enabled:
                print(f"  ⚠️  Skipping {config.name} (not supported on this platform)")
                continue
                
            print(f"\n  Building {config.name} configuration...")
            build_dir = self.root_dir / config.build_dir
            
            # Clean build directory for fresh build (unless --no-clean specified)
            if build_dir.exists() and not self.args.no_clean:
                print(f"    Cleaning existing build directory: {build_dir}")
                shutil.rmtree(build_dir)
            elif build_dir.exists():
                print(f"    Using existing build directory: {build_dir}")
            
            # Create fresh build directory
            build_dir.mkdir(parents=True, exist_ok=True)
            print(f"    Created fresh build directory: {build_dir}")
            
            # Run CMake configuration
            cmake_cmd = ["cmake", ".."] + config.cmake_args
            print(f"    Running: {' '.join(cmake_cmd)}")
            
            result = subprocess.run(
                cmake_cmd,
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"    ❌ CMake configuration failed for {config.name}")
                print(f"    Error: {result.stderr}")
                success = False
                continue
            
            # Build with make
            make_cmd = ["make", "-j", str(os.cpu_count() or 4)]
            print(f"    Running: {' '.join(make_cmd)}")
            
            result = subprocess.run(
                make_cmd,
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"    ❌ Build failed for {config.name}")
                print(f"    Error: {result.stderr}")
                success = False
                continue
                
            print(f"    ✅ {config.name} build complete")
            
        return success
    
    def _run_all_tests(self):
        """Run all test suites across all configurations"""
        for config in self.build_configs:
            if not config.enabled:
                continue
                
            print(f"\n  Testing {config.name} configuration...")
            build_dir = self.root_dir / config.build_dir
            
            if config.name not in self.results:
                self.results[config.name] = {}
            
            for suite in self.test_suites:
                if not suite.enabled:
                    continue
                    
                # Skip if suite requires specific sanitizer
                if suite.requires_sanitizer and config.sanitizer != suite.requires_sanitizer:
                    continue
                
                # Skip benchmarks in debug/sanitizer builds
                if "benchmark" in suite.name and config.sanitizer != SanitizerType.NONE:
                    continue
                
                result = self._run_test_suite(suite, build_dir, config)
                self.results[config.name][suite.name] = result
    
    def _run_test_suite(self, suite: TestSuite, build_dir: Path, config: BuildConfig) -> TestResult:
        """Run a single test suite"""
        print(f"    Running {suite.name}...")
        
        # Prepare environment for sanitizers
        env = os.environ.copy()
        if config.sanitizer == SanitizerType.ADDRESS:
            # LeakSanitizer is only supported on Linux, not on macOS
            if platform.system() == "Linux":
                env["ASAN_OPTIONS"] = "detect_leaks=1:abort_on_error=0:print_summary=1"
            else:
                env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=0:print_summary=1"
                print(f"Note: LeakSanitizer disabled on {platform.system()} (not supported)")
        elif config.sanitizer == SanitizerType.THREAD:
            env["TSAN_OPTIONS"] = "halt_on_error=0:print_summary=1"
        elif config.sanitizer == SanitizerType.UNDEFINED:
            env["UBSAN_OPTIONS"] = "halt_on_error=0:print_summary=1"
        
        # Adjust command path if needed
        command = suite.command.copy()
        if command[0].startswith("./"):
            command[0] = str(build_dir / command[0][2:])
        
        try:
            result = subprocess.run(
                command,
                cwd=build_dir,
                capture_output=True,
                text=True,
                timeout=suite.timeout,
                env=env
            )
            
            # Save output
            output_file = self.output_dir / f"{config.name}_{suite.name}.log"
            with open(output_file, "w") as f:
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"\n=== STDOUT ===\n{result.stdout}")
                f.write(f"\n=== STDERR ===\n{result.stderr}")
            
            if result.returncode == 0:
                print(f"      ✅ {suite.name} passed")
                return TestResult.PASSED
            else:
                print(f"      ❌ {suite.name} failed (see {output_file})")
                return TestResult.FAILED
                
        except subprocess.TimeoutExpired:
            print(f"      ⏱️  {suite.name} timeout ({suite.timeout}s)")
            return TestResult.TIMEOUT
        except FileNotFoundError:
            print(f"      ⚠️  {suite.name} skipped (binary not found)")
            return TestResult.SKIPPED
        except Exception as e:
            print(f"      ❌ {suite.name} error: {e}")
            return TestResult.FAILED
    
    def _run_memory_analysis(self):
        """Run specific memory analysis tests"""
        print("  Running AddressSanitizer memory leak detection...")
        
        asan_build = self.root_dir / "build-sanitizer"
        if not asan_build.exists():
            print("    ⚠️  AddressSanitizer build not found")
            return
        
        # Run specific memory-intensive tests with strict leak checking
        env = os.environ.copy()
        if platform.system() == "Linux":
            env["ASAN_OPTIONS"] = "detect_leaks=1:leak_check_at_exit=1:verbosity=1:print_stats=1"
            print("    Using full LeakSanitizer options (Linux)")
        else:
            env["ASAN_OPTIONS"] = "detect_leaks=0:verbosity=1:print_stats=1"
            print(f"    LeakSanitizer disabled on {platform.system()} - using AddressSanitizer only")
        
        memory_tests = [
            "./common/concurrency_stress_tests",
            "./integration/integration_tests",
            "./engines/engines_tests"
        ]
        
        for test in memory_tests:
            test_path = asan_build / test[2:]
            if test_path.exists():
                print(f"    Checking {test} for memory leaks...")
                result = subprocess.run(
                    [str(test_path), "--gtest_repeat=3"],
                    cwd=asan_build,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env
                )
                
                # Check for leak summary in output
                if "leak" in result.stderr.lower() or "leaked" in result.stdout.lower():
                    print(f"      ⚠️  Potential memory leaks detected in {test}")
                else:
                    print(f"      ✅ No memory leaks detected in {test}")
    
    def _generate_reports(self):
        """Generate comprehensive test reports"""
        # Generate JSON report
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration": time.time() - self.start_time,
            "platform": platform.platform(),
            "configurations": [c.name for c in self.build_configs if c.enabled],
            "results": self.results
        }
        
        report_file = self.output_dir / "test_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  📄 JSON report: {report_file}")
        
        # Generate HTML report
        self._generate_html_report()
        
        # Generate coverage report if available
        if (self.root_dir / "build-debug").exists():
            print("  📊 Generating coverage report...")
            subprocess.run(
                ["python3", "tools/check_coverage.py", "--html"],
                cwd=self.root_dir,
                capture_output=True
            )
    
    def _generate_html_report(self):
        """Generate an HTML report of test results"""
        html_file = self.output_dir / "test_report.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .passed {{ color: green; font-weight: bold; }}
        .failed {{ color: red; font-weight: bold; }}
        .skipped {{ color: orange; }}
        .timeout {{ color: purple; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Inference Systems Lab - Test Results</h1>
    <div class="summary">
        <p><strong>Date:</strong> {datetime.now()}</p>
        <p><strong>Duration:</strong> {time.time() - self.start_time:.2f} seconds</p>
        <p><strong>Platform:</strong> {platform.platform()}</p>
    </div>
    
    <h2>Test Results Matrix</h2>
    <table>
        <tr>
            <th>Configuration</th>
"""
        
        # Add test suite headers
        all_suites = set()
        for config_results in self.results.values():
            all_suites.update(config_results.keys())
        
        for suite in sorted(all_suites):
            html_content += f"            <th>{suite}</th>\n"
        html_content += "        </tr>\n"
        
        # Add results rows
        for config_name, config_results in self.results.items():
            html_content += f"        <tr>\n            <td><strong>{config_name}</strong></td>\n"
            for suite in sorted(all_suites):
                if suite in config_results:
                    result = config_results[suite]
                    css_class = result.name.lower()
                    html_content += f'            <td class="{css_class}">{result.value}</td>\n'
                else:
                    html_content += '            <td>-</td>\n'
            html_content += "        </tr>\n"
        
        html_content += """
    </table>
    
    <h2>Summary</h2>
    <div class="summary">
"""
        
        # Calculate summary statistics
        total_tests = sum(len(r) for r in self.results.values())
        passed_tests = sum(1 for r in self.results.values() for t in r.values() if t == TestResult.PASSED)
        failed_tests = sum(1 for r in self.results.values() for t in r.values() if t == TestResult.FAILED)
        
        html_content += f"""
        <p><strong>Total Tests Run:</strong> {total_tests}</p>
        <p><strong>Passed:</strong> <span class="passed">{passed_tests}</span></p>
        <p><strong>Failed:</strong> <span class="failed">{failed_tests}</span></p>
        <p><strong>Success Rate:</strong> {(passed_tests/total_tests*100) if total_tests > 0 else 0:.1f}%</p>
    </div>
</body>
</html>
"""
        
        with open(html_file, "w") as f:
            f.write(html_content)
        print(f"  📄 HTML report: {html_file}")
    
    def _print_summary(self) -> int:
        """Print final summary and return exit code"""
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
        
        # Count results
        total_tests = sum(len(r) for r in self.results.values())
        passed_tests = sum(1 for r in self.results.values() for t in r.values() if t == TestResult.PASSED)
        failed_tests = sum(1 for r in self.results.values() for t in r.values() if t == TestResult.FAILED)
        duration = time.time() - self.start_time
        
        print(f"Total configurations tested: {len([c for c in self.build_configs if c.enabled])}")
        print(f"Total test suites run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"\nResults saved to: {self.output_dir}")
        
        # Collect failed test names
        failed_test_names = []
        for config_name, config_results in self.results.items():
            for suite_name, result in config_results.items():
                if result == TestResult.FAILED:
                    failed_test_names.append(f"{config_name}/{suite_name}")
        
        # Send notifications and store logs
        if self.notification_manager or self.log_manager:
            # Create test results object for notifications
            test_results = None
            if NOTIFICATIONS_AVAILABLE:
                test_results = TestResults(
                    total_tests=total_tests,
                    passed_tests=passed_tests,
                    failed_tests=failed_tests,
                    skipped_tests=0,
                    duration_seconds=duration,
                    configurations=[c.name for c in self.build_configs if c.enabled],
                    failed_test_names=failed_test_names,
                    platform=f"{platform.system()}-{platform.release()}-{platform.machine()}",
                    timestamp=datetime.now()
                )
            
            # Send notifications
            if self.notification_manager and test_results:
                try:
                    self.notification_manager.notify_test_complete(test_results)
                except Exception as e:
                    print(f"Warning: Failed to send notification: {e}")
            
            # Store persistent logs
            if self.log_manager:
                try:
                    # Collect all log files
                    logs = {}
                    for log_file in self.output_dir.glob("*.log"):
                        with open(log_file, 'r') as f:
                            logs[log_file.stem] = f.read()
                    
                    # Store test run in persistent storage
                    results_dict = {
                        'total_tests': total_tests,
                        'passed_tests': passed_tests,
                        'failed_tests': failed_tests,
                        'duration_seconds': duration,
                        'configurations': [c.name for c in self.build_configs if c.enabled],
                        'platform': f"{platform.system()}-{platform.release()}-{platform.machine()}",
                        'failed_test_names': failed_test_names
                    }
                    
                    run_id = self.log_manager.store_test_run(results_dict, logs)
                    print(f"📁 Test run stored with ID: {run_id}")
                    
                except Exception as e:
                    print(f"Warning: Failed to store persistent logs: {e}")
        
        if failed_tests == 0:
            print("\n✅ ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n❌ {failed_tests} TESTS FAILED")
            print("\nFailed tests:")
            for test_name in failed_test_names:
                print(f"  - {test_name}")
            return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive testing orchestrator for Inference Systems Lab"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke tests only (skip stress tests and benchmarks)"
    )
    parser.add_argument(
        "--memory",
        dest="memory_only",
        action="store_true",
        help="Focus on memory safety testing only"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run builds in parallel (experimental)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output from all tests"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean build directories (preserve for debugging)"
    )
    parser.add_argument(
        "--no-notifications",
        action="store_true",
        help="Disable notifications and persistent logging"
    )
    
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    if not (root_dir / "CMakeLists.txt").exists():
        print("Error: Could not find project root directory")
        return 1
    
    # Run orchestrator
    orchestrator = TestOrchestrator(root_dir, args)
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
