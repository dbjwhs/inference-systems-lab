# Comprehensive Testing Guide

This document provides complete information about the comprehensive testing infrastructure in Inference Systems Laboratory.

## Overview

The project includes a sophisticated testing orchestrator (`tools/run_comprehensive_tests.py`) that provides a single point of execution for all testing activities. This ensures consistent, reproducible, and thorough validation across multiple build configurations and test suites.

## Quick Start

```bash
# Complete testing (recommended before releases)
python3 tools/run_comprehensive_tests.py

# Quick development testing
python3 tools/run_comprehensive_tests.py --quick

# Memory safety focused
python3 tools/run_comprehensive_tests.py --memory

# Keep build directories for debugging
python3 tools/run_comprehensive_tests.py --no-clean
```

## Command Line Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `--quick` | Run essential tests only, skip stress/benchmarks | Rapid iteration, CI smoke tests |
| `--memory` | Focus on memory safety testing | Memory leak investigation |
| `--no-clean` | Preserve build directories | Debugging build issues |
| `--parallel` | Run builds in parallel (experimental) | Faster execution on multi-core |
| `--verbose` | Show detailed output from all tests | Debugging test failures |

## Build Configurations

The orchestrator builds and tests multiple configurations automatically:

### 1. Release Build (`build/`)
- **Purpose**: Performance validation
- **Config**: `-DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON`
- **Features**: Optimized code, benchmarks enabled

### 2. Debug Build (`build-debug/`)
- **Purpose**: Development and coverage
- **Config**: `-DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON`
- **Features**: Debug symbols, coverage instrumentation

### 3. AddressSanitizer Build (`build-sanitizer/`)
- **Purpose**: Memory safety validation
- **Config**: `-DCMAKE_BUILD_TYPE=Debug -DSANITIZER_TYPE=address`
- **Features**: Memory leak detection (Linux only), heap overflow detection
- **Environment**: 
  - Linux: `ASAN_OPTIONS=detect_leaks=1:abort_on_error=0:print_summary=1`
  - macOS: `ASAN_OPTIONS=detect_leaks=0:abort_on_error=0:print_summary=1` (LeakSanitizer not supported)

### 4. ThreadSanitizer Build (`build-tsan/`)
- **Purpose**: Race condition detection
- **Config**: `-DCMAKE_BUILD_TYPE=Debug -DSANITIZER_TYPE=thread`
- **Features**: Data race detection, thread safety validation
- **Note**: Disabled on macOS due to platform issues

### 5. UndefinedBehaviorSanitizer Build (`build-ubsan/`)
- **Purpose**: Undefined behavior detection
- **Config**: `-DCMAKE_BUILD_TYPE=Debug -DSANITIZER_TYPE=address+undefined`
- **Features**: Combined AddressSanitizer + UBSan

## Test Suites

### Core Test Suites

#### 1. Unit Tests
- **Command**: `ctest --output-on-failure -L unit`
- **Timeout**: 300 seconds
- **Coverage**: Individual component testing

#### 2. Integration Tests
- **Command**: `ctest --output-on-failure -L integration`
- **Timeout**: 600 seconds
- **Coverage**: Component interaction testing

#### 3. All CTest
- **Command**: `ctest --output-on-failure --timeout 300`
- **Timeout**: 1800 seconds
- **Coverage**: Complete test suite

### Specialized Test Suites

#### 4. Stress Tests
- **Command**: `./common/concurrency_stress_tests`
- **Timeout**: 900 seconds
- **Purpose**: High-concurrency validation (50-200 threads)
- **Enabled**: Only in full testing mode (not `--quick`)

#### 5. Memory Leak Tests
- **Command**: `./common/concurrency_stress_tests --gtest_filter=*Memory*`
- **Timeout**: 300 seconds
- **Purpose**: Specific memory leak detection
- **Requires**: AddressSanitizer build

#### 6. ML Integration Stress
- **Command**: `./integration/integration_stress_tests`
- **Timeout**: 1200 seconds
- **Purpose**: ML inference system stress testing
- **Enabled**: Only in full testing mode

#### 7. Performance Benchmarks
- **Command**: `./common/result_benchmarks --benchmark_format=json`
- **Timeout**: 600 seconds
- **Purpose**: Performance regression detection
- **Enabled**: Only in optimized builds, not with `--memory` flag

## Memory Safety Testing

### What is AddressSanitizer?

**AddressSanitizer (ASan) is a runtime memory error detector** that instruments C++ code to catch memory safety bugs immediately when they occur. It's essential for projects with complex memory management and concurrent code.

#### Critical Memory Bugs ASan Detects

1. **Heap Buffer Overflow/Underflow**
   ```cpp
   char* buffer = new char[10];
   buffer[15] = 'x';  // ❌ ASan: heap-buffer-overflow
   buffer[-1] = 'y';  // ❌ ASan: heap-buffer-underflow
   ```

2. **Use-After-Free** (Common in concurrent code)
   ```cpp
   char* ptr = new char[100];
   delete[] ptr;
   ptr[0] = 'x';  // ❌ ASan: heap-use-after-free
   ```

3. **Memory Leaks** (with leak detection enabled)
   ```cpp
   char* ptr = new char[100];
   // Missing delete[] ptr  // ❌ ASan: memory leak at exit
   ```

4. **Double-Free**
   ```cpp
   char* ptr = new char[100];
   delete[] ptr;
   delete[] ptr;  // ❌ ASan: double-free
   ```

5. **Stack Buffer Overflow**
   ```cpp
   char buffer[10];
   buffer[15] = 'x';  // ❌ ASan: stack-buffer-overflow
   ```

#### Why ASan is Critical for This Project

- **Concurrent Memory Management**: Our MemoryPool, lock-free queues, and concurrent data structures are prone to race conditions
- **Complex RAII Patterns**: Zero-cost abstractions and custom allocators need validation
- **Early Bug Detection**: Memory bugs often don't crash immediately - ASan catches them when they happen
- **Production Safety**: Prevents memory corruption that could cause security vulnerabilities

#### Real Example from Our Codebase

**Bug Found**: Heap-use-after-free in MemoryPool during concurrent access
```cpp
// Thread A: Reading blocks_ vector
for (auto& block_ptr : blocks_) { /* ... */ }

// Thread B: Simultaneously reallocating vector  
blocks_.push_back(new_block);  // Frees old memory Thread A is using!
```

**AddressSanitizer Output**:
```
==88576==ERROR: AddressSanitizer: heap-use-after-free on address 0x61d0000005d8
READ of size 8 at 0x61d0000005d8 thread T10
    #0 0x0001002864e4 in MemoryPool<unsigned long long>::allocate
    #1 0x000100276ad8 in memory_pool_worker
```

This bug would have been extremely difficult to debug manually but ASan found it immediately.

### AddressSanitizer Integration

The orchestrator automatically configures AddressSanitizer for comprehensive memory safety validation:

```bash
# Environment variables set automatically (platform-dependent)
# Linux:
ASAN_OPTIONS=detect_leaks=1:abort_on_error=0:print_summary=1
# macOS:
ASAN_OPTIONS=detect_leaks=0:abort_on_error=0:print_summary=1
```

**Configuration Details:**
- `detect_leaks=1`: Enable leak detection at program termination (Linux only)
- `detect_leaks=0`: Disable leak detection (macOS - not supported on Apple Silicon)
- `detect_container_overflow=0`: Disable container overflow detection (prevents false positives)
- `abort_on_error=0`: Don't crash immediately, show error and continue testing
- `print_summary=1`: Show detailed leak summary at program exit

**Container Overflow Detection:**
Container overflow detection is disabled (`detect_container_overflow=0`) to prevent false positives that occur in mixed instrumentation scenarios. This is the official workaround recommended by the AddressSanitizer team for projects using GoogleTest with AddressSanitizer. See `docs/ADDRESSSANITIZER_NOTES.md` for detailed explanation.

**Compiler Flags** (set automatically in sanitizer builds):
```bash
-fsanitize=address          # Enable AddressSanitizer instrumentation
-fno-omit-frame-pointer     # Better stack traces in error reports
-O1 -g                      # Some optimization + debug symbols
```

**Performance Impact:**
- **Runtime**: ~2-3x slower execution (worth it for bug detection)
- **Memory**: ~3x more memory usage (needed for shadow memory)
- **Build time**: Slightly slower due to instrumentation

**Key Features:**
- **Leak Detection**: Automatically detects memory leaks at program termination
- **Heap Overflow**: Detects buffer overruns and underruns
- **Use-After-Free**: Detects access to freed memory  
- **Double-Free**: Detects multiple frees of same memory
- **Stack Overflow**: Detects stack buffer overflows

### Memory Analysis Phase

In full testing mode, the orchestrator runs dedicated memory analysis:

1. **Repeat Testing**: Runs tests 3 times to catch intermittent issues
2. **Leak Summary**: Analyzes output for leak patterns
3. **Verbose Reporting**: Uses `verbosity=1` for detailed leak information

### Other Sanitizers

#### ThreadSanitizer (TSan)
**Purpose**: Detects data races and thread safety violations

```cpp
// Example race condition TSan would catch:
int global_counter = 0;

// Thread A:
global_counter++;  // ❌ TSan: data race

// Thread B (simultaneously):
global_counter++;  // ❌ TSan: data race  
```

**When to use**: Testing concurrent algorithms, lock-free data structures
**Note**: Incompatible with AddressSanitizer (run separately)

#### UndefinedBehaviorSanitizer (UBSan)  
**Purpose**: Detects undefined behavior that may work "by accident"

```cpp
// Examples UBSan catches:
int x = INT_MAX;
x++;                    // ❌ UBSan: signed integer overflow

int* p = nullptr;
int& ref = *p;         // ❌ UBSan: null pointer dereference

int arr[10];
return arr[15];        // ❌ UBSan: array bounds violation
```

**When to use**: Combined with ASan for comprehensive safety validation
**Configuration**: `address+undefined` combines both sanitizers

## Report Generation

### Output Directory Structure

```
test-results/YYYYMMDD_HHMMSS/
├── test_report.html          # Human-readable HTML report
├── test_report.json          # Machine-readable JSON report
├── release_unit_tests.log    # Individual test logs
├── asan_memory_leak_tests.log
└── ...
```

### HTML Report Features

- **Test Results Matrix**: Configuration vs. Test Suite grid
- **Summary Statistics**: Pass/fail counts, success rates
- **Execution Details**: Duration, platform information
- **Color Coding**: Visual status indicators

### JSON Report Schema

```json
{
  "timestamp": "2025-08-25T15:28:26",
  "duration": 1234.56,
  "platform": "macOS-14.6.1-arm64-arm-64bit",
  "configurations": ["release", "debug", "asan"],
  "results": {
    "release": {
      "unit_tests": "✅ PASSED",
      "benchmarks": "✅ PASSED"
    }
  }
}
```

## Clean Build Strategy

### Why Clean Builds

The orchestrator **always performs clean builds** by default:

1. **Reproducibility**: Eliminates incremental build artifacts
2. **Dependency Issues**: Catches missing includes/libraries
3. **Configuration Changes**: Ensures CMake changes take effect
4. **Sanitizer Compatibility**: Prevents mixed instrumentation

### Build Directory Management

```bash
# Default behavior (clean builds)
python3 tools/run_comprehensive_tests.py

# Preserve for debugging
python3 tools/run_comprehensive_tests.py --no-clean
```

**Clean Process:**
1. Remove existing build directory completely
2. Create fresh directory
3. Run CMake configuration from scratch
4. Build all targets

## Extending the Testing Infrastructure

### Adding New Test Suites

```python
# In TestOrchestrator._setup_test_suites()
suites.append(TestSuite(
    name="my_new_test",
    command=["./path/to/test", "--options"],
    timeout=300,
    requires_sanitizer=SanitizerType.ADDRESS,  # Optional
    enabled=not self.args.quick,               # Optional
    allow_failure=False                        # Optional
))
```

### Adding New Build Configurations

```python
# In TestOrchestrator._setup_build_configs()
configs.append(BuildConfig(
    name="my_config",
    build_type=BuildType.DEBUG,
    sanitizer=SanitizerType.MEMORY,
    build_dir="build-my-config",
    cmake_args=["-DCMAKE_BUILD_TYPE=Debug", "-DSANITIZER_TYPE=memory"],
    enabled=platform.system() != "Darwin"  # Platform-specific
))
```

### Adding New Sanitizers

1. **Update SanitizerType enum**
2. **Add CMake configuration in `cmake/Sanitizers.cmake`**
3. **Update environment variable setup in orchestrator**
4. **Test on target platforms**

## Performance Considerations

### Execution Times (Typical)

| Mode | Duration | Use Case |
|------|----------|----------|
| `--quick` | 5-10 minutes | Development iteration |
| Full testing | 15-30 minutes | Pre-release validation |
| `--memory` only | 8-15 minutes | Memory issue investigation |

### Resource Usage

- **CPU**: Utilizes all available cores for builds (`make -j$(nproc)`)
- **Memory**: AddressSanitizer increases memory usage ~3x
- **Disk**: Multiple build directories require ~500MB-1GB
- **Temporary Files**: Test outputs saved to `test-results/`

## Troubleshooting

### Common Issues

#### Build Failures
```bash
# Check specific build log
cat test-results/YYYYMMDD_HHMMSS/release_unit_tests.log

# Preserve build directories for investigation
python3 tools/run_comprehensive_tests.py --no-clean
```

#### Memory Issues
```bash
# Focus on memory testing
python3 tools/run_comprehensive_tests.py --memory

# Check AddressSanitizer output
grep -A 10 -B 5 "ERROR: AddressSanitizer" test-results/*/asan_*.log
```

#### Test Timeouts
```bash
# Check test progress
tail -f test-results/YYYYMMDD_HHMMSS/stress_tests.log

# Adjust timeout in TestSuite configuration
```

### Platform-Specific Notes

#### macOS
- ThreadSanitizer disabled due to reliability issues
- Use Homebrew for dependencies: `brew install llvm`
- AddressSanitizer works reliably

#### Linux
- All sanitizers supported
- May need `libc6-dbg` for symbol resolution
- Consider `ulimit` settings for memory-intensive tests

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Comprehensive Testing
  run: |
    python3 tools/run_comprehensive_tests.py --quick
    
- name: Memory Safety Check
  run: |
    python3 tools/run_comprehensive_tests.py --memory
    
- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test-results/
```

### Jenkins Pipeline

```groovy
stage('Comprehensive Tests') {
    steps {
        sh 'python3 tools/run_comprehensive_tests.py'
    }
    post {
        always {
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: true,
                reportDir: 'test-results',
                reportFiles: 'test_report.html',
                reportName: 'Test Report'
            ])
        }
    }
}
```

## Best Practices

### Development Workflow

1. **Regular Testing**: Run `--quick` during development
2. **Pre-Commit**: Run full testing before important commits
3. **Memory Validation**: Use `--memory` when debugging crashes
4. **Report Review**: Check HTML reports for patterns

### Team Workflow

1. **Shared Standards**: All team members use same testing commands
2. **Report Archiving**: Save test reports for release documentation
3. **Failure Triage**: Use JSON reports for automated analysis
4. **Performance Tracking**: Monitor benchmark results over time

### Release Process

```bash
# Pre-release validation
python3 tools/run_comprehensive_tests.py

# Archive results
cp -r test-results/latest release-testing-results/v1.2.3/

# Document any known issues
echo "All tests passed" > release-testing-results/v1.2.3/STATUS.md
```

This comprehensive testing infrastructure ensures the Inference Systems Laboratory maintains the highest standards of code quality, memory safety, and performance throughout development.
