# Developer Tools Suite

A comprehensive collection of development automation tools for the Inference Systems Laboratory project, providing code quality assurance, testing infrastructure, and developer productivity enhancements.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Tools](#core-tools)
  - [Code Quality Tools](#code-quality-tools)
  - [Testing and Coverage](#testing-and-coverage)
  - [Performance Analysis](#performance-analysis)
  - [Development Workflow](#development-workflow)
  - [Project Scaffolding](#project-scaffolding)
- [Tool Reference](#tool-reference)
- [Integration with Development Workflow](#integration-with-development-workflow)
- [Optimization Opportunities](#optimization-opportunities)
- [Future Enhancements](#future-enhancements)

## Overview

The tools directory contains Python scripts and shell utilities that automate common development tasks, enforce code quality standards, and streamline the development workflow. These tools integrate with CMake, Git hooks, and CI/CD pipelines to ensure consistent code quality across the entire project.

### Design Philosophy

- **Automation First**: Minimize manual intervention in repetitive tasks
- **Quality Gates**: Catch issues early in the development cycle
- **Developer Experience**: Provide clear feedback and actionable fixes
- **Performance**: Optimize for large codebases with selective file processing
- **Integration**: Seamless integration with existing developer workflows

## Quick Start

```bash
# Initial setup
cd inference-systems-lab
./tools/setup.sh                                    # Configure and build project

# Install pre-commit hooks
python3 tools/install_hooks.py --install            # Set up quality gates

# Run quality checks
python3 tools/check_format.py --check               # Verify code formatting
python3 tools/check_static_analysis.py --check      # Run static analysis
python3 tools/check_eof_newline.py --check          # Check POSIX compliance

# Run tests with coverage
python3 tools/run_tests.py                          # Execute all tests
python3 tools/check_coverage.py --threshold 80.0    # Verify coverage

# Performance analysis and ML testing
python3 tools/performance_profiler.py               # Analyze performance hotspots
python3 tools/ml_inference_load_test.py             # Test ML inference scenarios

# Create new module
python3 tools/new_module.py my_feature              # Scaffold new component
```

## Core Tools

### Code Quality Tools

#### check_format.py
**Purpose**: Enforces consistent code formatting using clang-format

**Key Features**:
- Automatic discovery of C++ source files
- Dry-run mode for CI/CD verification
- In-place formatting with backup options
- Detailed violation reporting
- Git integration for tracking changes

**Usage Examples**:
```bash
# Check formatting without changes
python3 tools/check_format.py --check

# Fix formatting issues with backup
python3 tools/check_format.py --fix --backup

# Check specific directory
python3 tools/check_format.py --check --filter "common/src/*"

# Fix only staged files (pre-commit friendly)
python3 tools/check_format.py --fix --staged
```

**Arguments**:
- `--check`: Verify formatting without making changes
- `--fix`: Apply formatting fixes
- `--backup`: Create .bak files before fixing
- `--filter PATTERN`: Only process files matching pattern
- `--exclude PATTERNS`: Skip files/directories (comma-separated)
- `--staged`: Only process Git staged files
- `--quiet`: Reduce output verbosity

#### check_static_analysis.py
**Purpose**: Runs clang-tidy static analysis to detect bugs, code smells, and enforce standards

**Key Features**:
- Compilation database integration for accurate analysis
- Severity-based filtering (error, warning, note)
- Fix application with safety checks
- Suppression support for false positives
- Detailed issue categorization
- JSON report generation

**Usage Examples**:
```bash
# Run full static analysis
python3 tools/check_static_analysis.py --check

# Apply safe fixes with backup
python3 tools/check_static_analysis.py --fix --backup

# Auto-fix with git commit (replaces run_clang_tidy.py)
python3 tools/check_static_analysis.py --auto-fix --auto-commit

# Preview auto-fix changes
python3 tools/check_static_analysis.py --auto-fix --dry-run

# Check only errors
python3 tools/check_static_analysis.py --check --severity error

# Generate suppressions file
python3 tools/check_static_analysis.py --generate-suppressions
```

**Arguments**:
- `--check`: Run analysis without fixes
- `--fix`: Apply automated fixes with build validation
- `--auto-fix`: Run in simplified auto-fix mode (like run_clang_tidy.py)
- `--backup`: Create backups before fixing
- `--auto-commit`: Create git commit after fixes (use with --auto-fix)
- `--dry-run`: Preview changes without making them
- `--severity LEVEL`: Filter by severity (error/warning/note)
- `--filter PATTERN`: Only analyze matching files
- `--generate-suppressions`: Create suppressions for current issues
- `--output-json FILE`: Export results as JSON
- `--no-build-validation`: Skip build validation after fixes

#### check_eof_newline.py
**Purpose**: Ensures POSIX compliance by verifying files end with newlines

**Key Features**:
- Automatic text file detection
- Support for multiple file types
- Batch fixing capability
- Integration with pre-commit hooks
- Performance optimized for large repos

**Usage Examples**:
```bash
# Check all text files
python3 tools/check_eof_newline.py --check

# Fix with backups
python3 tools/check_eof_newline.py --fix --backup

# Check Python files only
python3 tools/check_eof_newline.py --check --filter "*.py"

# Process files from list
python3 tools/check_eof_newline.py --fix --filter-from-file staged_files.txt
```

**Arguments**:
- `--check`: Verify EOF newlines
- `--fix`: Add missing newlines
- `--backup`: Create .bak files
- `--filter PATTERN`: Include matching files
- `--filter-from-file FILE`: Read file list
- `--exclude PATTERNS`: Skip patterns
- `--show-details`: Verbose output

### Testing and Coverage

#### run_tests.py
**Purpose**: Comprehensive test runner managing CTest execution across all modules

**Key Features**:
- Module-specific test execution
- Pattern-based test filtering
- Parallel test execution
- JSON result export
- Verbose output modes
- Test discovery and listing

**Usage Examples**:
```bash
# Run all tests
python3 tools/run_tests.py

# Run specific module tests
python3 tools/run_tests.py --module common

# Filter by pattern
python3 tools/run_tests.py --filter Result

# List available tests
python3 tools/run_tests.py --list

# Verbose output with JSON export
python3 tools/run_tests.py --verbose --json-output results.json
```

**Arguments**:
- `--module MODULE`: Run tests for specific module
- `--filter PATTERN`: Run tests matching pattern
- `--list`: List available tests
- `--verbose`: Detailed output
- `--json-output FILE`: Export results as JSON
- `--parallel N`: Number of parallel jobs
- `--timeout SECONDS`: Test timeout

#### check_coverage.py
**Purpose**: Test coverage verification with configurable thresholds

**Key Features**:
- Automatic test discovery and execution
- gcov/llvm-cov support
- HTML report generation
- File and line-level metrics
- Threshold validation
- CI/CD integration

**Usage Examples**:
```bash
# Standard coverage check
python3 tools/check_coverage.py --threshold 80.0

# Generate HTML report
python3 tools/check_coverage.py --html-output coverage.html

# Custom build directory
python3 tools/check_coverage.py --build-dir custom_build

# Exclude test directories
python3 tools/check_coverage.py --exclude-dirs "tests,examples"

# Clean build with coverage
python3 tools/check_coverage.py --clean-build --threshold 90.0
```

**Arguments**:
- `--threshold PERCENT`: Minimum coverage requirement
- `--build-dir PATH`: Build directory location
- `--clean-build`: Clean before building
- `--filter PATTERN`: Test filter pattern
- `--exclude-dirs DIRS`: Exclude directories
- `--html-output FILE`: Generate HTML report
- `--json-output FILE`: Export JSON data
- `--skip-build`: Use existing build
- `--skip-tests`: Use existing coverage data

#### check_documentation.py
**Purpose**: Comprehensive Doxygen documentation generation and validation

**Key Features**:
- Automated Doxygen documentation generation with error handling
- Documentation coverage analysis for undocumented public APIs
- Integration with existing build system and quality gates  
- CI/CD friendly with structured output and exit codes
- Performance optimized for large codebases

**Usage Examples**:
```bash
# Generate documentation
python3 tools/check_documentation.py --generate

# Check coverage only
python3 tools/check_documentation.py --check

# Full workflow with coverage validation
python3 tools/check_documentation.py --generate --check

# Clean generated documentation
python3 tools/check_documentation.py --clean

# Custom coverage threshold
python3 tools/check_documentation.py --check --coverage-threshold 90.0

# Export results as JSON
python3 tools/check_documentation.py --check --json-output coverage.json
```

**Arguments**:
- `--generate`: Generate Doxygen documentation  
- `--check`: Check documentation coverage against threshold
- `--clean`: Clean generated documentation files
- `--build-dir DIR`: Specify build directory (default: build)
- `--coverage-threshold PERCENT`: Set minimum coverage requirement (default: 80.0)
- `--json-output FILE`: Export results as JSON
- `--quiet`: Reduce output verbosity

#### run_benchmarks.py
**Purpose**: Performance regression detection with baseline management

**Key Features**:
- Auto-discovery of benchmark executables
- Baseline storage and comparison
- Statistical regression detection
- Performance trend analysis
- JSON export for CI/CD
- Configurable thresholds

**Usage Examples**:
```bash
# Run benchmarks
python3 tools/run_benchmarks.py

# Save baseline
python3 tools/run_benchmarks.py --save-baseline v1.0.0

# Compare against baseline
python3 tools/run_benchmarks.py --compare-against v1.0.0

# Filter specific benchmarks
python3 tools/run_benchmarks.py --filter "*Result*" --threshold 10.0

# Export results
python3 tools/run_benchmarks.py --output-json results.json
```

**Arguments**:
- `--save-baseline NAME`: Save current results as baseline
- `--compare-against NAME`: Compare with baseline
- `--filter PATTERN`: Run matching benchmarks
- `--threshold PERCENT`: Regression threshold
- `--output-json FILE`: Export results
- `--repetitions N`: Number of runs
- `--timeout SECONDS`: Benchmark timeout

### Performance Analysis

#### performance_profiler.py
**Purpose**: Advanced performance analysis and profiling for C++ codebase

**Key Features**:
- Static code analysis for performance hotspots
- Result<T,E> monadic operations overhead analysis
- Logging system synchronization bottleneck detection  
- Container cache efficiency evaluation
- Schema evolution migration cost assessment
- Google Benchmark result interpretation
- Prioritized optimization recommendations
- Performance scoring (0-100) and ML integration readiness

**Usage Examples**:
```bash
# Full performance analysis
python3 tools/performance_profiler.py

# Custom output location
python3 tools/performance_profiler.py --output perf_analysis.json

# Text report only
python3 tools/performance_profiler.py --format text

# With existing benchmark results
python3 tools/performance_profiler.py --benchmark-results benchmarks.json

# JSON export for CI/CD
python3 tools/performance_profiler.py --format json --output ci_perf.json
```

**Arguments**:
- `--output FILE`: Output file for analysis results (default: performance_analysis.json)
- `--benchmark-results FILE`: Path to existing benchmark results JSON
- `--format FORMAT`: Output format (json/text/both, default: both)

#### ml_inference_load_test.py
**Purpose**: Comprehensive load testing framework for ML inference scenarios

**Key Features**:
- Realistic workload simulation for TensorRT/ONNX integration
- Multiple scenario support: image classification, object detection, NLP transformers
- Concurrent load testing with configurable concurrency levels
- Stress testing with automatic saturation detection
- Mock inference engine for testing without actual ML models
- Performance metrics: throughput, latency percentiles, GPU utilization
- Bottleneck analysis and scaling recommendations
- JSON export with comprehensive performance grading

**Usage Examples**:
```bash
# Run all ML inference scenarios
python3 tools/ml_inference_load_test.py

# Test specific scenario
python3 tools/ml_inference_load_test.py --scenario image_classification

# Custom output location
python3 tools/ml_inference_load_test.py --output ml_results.json

# Object detection stress test
python3 tools/ml_inference_load_test.py --scenario object_detection --output obj_det.json
```

**Arguments**:
- `--output FILE`: Output file for test results (default: ml_load_test_results.json)
- `--scenario SCENARIO`: Specific scenario to test (image_classification/object_detection/nlp_transformers/mixed_workload/all, default: all)

### Development Workflow

#### install_hooks.py
**Purpose**: Manages Git pre-commit hooks for quality enforcement

**Key Features**:
- Automatic hook installation
- Integration with all quality tools
- Selective file checking
- Bypass options for emergencies
- Backup and restoration
- Hook testing capability

**Usage Examples**:
```bash
# Install hooks
python3 tools/install_hooks.py --install

# Check status
python3 tools/install_hooks.py --status

# Test on current changes
python3 tools/install_hooks.py --test

# Remove hooks
python3 tools/install_hooks.py --uninstall

# Force reinstall
python3 tools/install_hooks.py --install --force
```

**Arguments**:
- `--install`: Install pre-commit hooks
- `--uninstall`: Remove hooks
- `--status`: Check installation status
- `--test`: Test hooks on staged files
- `--force`: Overwrite existing hooks
- `--backup`: Backup existing hooks

#### fix_static_analysis_by_file.py
**Purpose**: Systematic file-by-file static analysis fixing

**Key Features**:
- Complexity-based prioritization
- Phase-based fixing strategy
- Progress tracking
- Quick win identification
- Build safety verification

**Usage Examples**:
```bash
# List files by complexity
python3 tools/fix_static_analysis_by_file.py --list-files

# Fix specific file
python3 tools/fix_static_analysis_by_file.py --fix-file common/src/result.hpp

# Fix next easiest file
python3 tools/fix_static_analysis_by_file.py --next-easy

# Show phase 1 files
python3 tools/fix_static_analysis_by_file.py --phase 1

# Fix all in phase
python3 tools/fix_static_analysis_by_file.py --phase 2 --fix-all
```

**Arguments**:
- `--list-files`: Show files by complexity
- `--fix-file FILE`: Fix specific file
- `--next-easy`: Fix next easiest file
- `--phase N`: Show/fix phase N files
- `--fix-all`: Fix all files in phase
- `--dry-run`: Preview changes


### Project Scaffolding

#### new_module.py
**Purpose**: Scaffolds new modules with complete structure

**Key Features**:
- Standard directory creation
- CMake configuration generation
- Test suite scaffolding
- Benchmark setup
- Documentation templates
- Example files

**Usage Examples**:
```bash
# Create basic module
python3 tools/new_module.py neural_symbolic

# Specify author and type
python3 tools/new_module.py probabilistic --type library --author "Jane Doe"

# With description
python3 tools/new_module.py ml_engine --description "Machine learning engine"

# Interface module
python3 tools/new_module.py api --type interface
```

**Arguments**:
- `MODULE_NAME`: Name of new module
- `--type TYPE`: Module type (library/interface/executable)
- `--author AUTHOR`: Module author
- `--description DESC`: Module description
- `--no-tests`: Skip test creation
- `--no-examples`: Skip examples

#### setup.sh
**Purpose**: Initial project setup and build configuration

**Key Features**:
- Dependency checking
- Build type configuration
- Sanitizer integration
- Tool verification
- Platform detection

**Usage Examples**:
```bash
# Standard release build
./tools/setup.sh

# Debug build
./tools/setup.sh --debug

# With AddressSanitizer
./tools/setup.sh --sanitizer address

# Debug with multiple sanitizers
./tools/setup.sh --debug --sanitizer address+undefined

# No tests
./tools/setup.sh --no-tests
```

**Arguments**:
- `--sanitizer TYPE`: Enable sanitizer (address/thread/memory/undefined)
- `--debug`: Debug build
- `--release`: Release build
- `--no-tests`: Disable tests
- `--help`: Show usage

## Integration with Development Workflow

### Typical Development Flow

1. **Initial Setup**
   ```bash
   git clone <repository>
   cd inference-systems-lab
   ./tools/setup.sh
   python3 tools/install_hooks.py --install
   ```

2. **Feature Development**
   ```bash
   # Create new module
   python3 tools/new_module.py my_feature
   
   # Develop code...
   
   # Check quality before commit
   python3 tools/check_format.py --fix
   python3 tools/check_static_analysis.py --check
   python3 tools/run_tests.py --module my_feature
   ```

3. **Pre-Commit Workflow**
   ```bash
   # Stage changes
   git add -A
   
   # Pre-commit hooks automatically run:
   # - Code formatting check
   # - Static analysis
   # - EOF newline check
   # - Build verification
   
   # If issues found, fix and re-stage:
   python3 tools/check_format.py --fix --staged
   git add -A
   git commit -m "Add feature"
   ```

4. **Quality Verification**
   ```bash
   # Full quality check
   python3 tools/run_tests.py
   python3 tools/check_coverage.py --threshold 80.0
   python3 tools/run_benchmarks.py --compare-against main
   ```

### CI/CD Integration

The tools are designed for seamless CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Check Format
  run: python3 tools/check_format.py --check

- name: Static Analysis
  run: python3 tools/check_static_analysis.py --check --severity error

- name: Run Tests
  run: python3 tools/run_tests.py --json-output test-results.json

- name: Check Coverage
  run: python3 tools/check_coverage.py --threshold 80.0 --html-output coverage.html

- name: Performance Check
  run: python3 tools/run_benchmarks.py --compare-against ${{ github.base_ref }}

- name: Performance Analysis
  run: python3 tools/performance_profiler.py --format json --output performance.json

- name: ML Inference Load Test
  run: python3 tools/ml_inference_load_test.py --scenario mixed_workload --output ml_perf.json
```

### Pre-Commit Hook Integration

The pre-commit hook (`pre-commit-hook-template.sh`) integrates multiple tools:

1. **Format Checking**: Ensures consistent code style
2. **Static Analysis**: Catches potential bugs early
3. **EOF Compliance**: Maintains POSIX standards
4. **Build Verification**: Ensures code compiles

To bypass in emergencies:
```bash
git commit --no-verify -m "Emergency fix"
```

## Optimization Opportunities

1. **✅ Consolidated static analysis tools**: Merged `run_clang_tidy.py` functionality into `check_static_analysis.py`
   - Added `--auto-fix` mode for simple automated fixing with git commit support
   - Added `--auto-commit` flag for automatic git commit creation
   - Improved clang-tidy executable detection (macOS/PATH fallback)
   - Maintained all existing comprehensive analysis features

2. **Unified configuration**: Create a central config file for tool settings
3. **Shared utilities**: Extract common functions (file discovery, git operations) into a shared module

## Future Enhancements

### Industry Best Practices (2024-2025)

Based on current enterprise open-source trends, consider implementing:

#### 1. Pre-commit Framework Integration
Migrate to the industry-standard [pre-commit](https://pre-commit.com/) framework:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  
  - repo: https://github.com/pocc/pre-commit-hooks
    rev: v1.3.5
    hooks:
      - id: clang-format
      - id: clang-tidy
```

**Benefits**:
- Self-updating hooks via `pre-commit autoupdate`
- Language-agnostic framework
- 1800+ available hooks
- GitHub Actions integration

#### 2. GitHub Actions Enhancements

##### IssueOps Integration
Use GitHub Issues as an interface for CI/CD operations:

```yaml
on:
  issue_comment:
    types: [created]

jobs:
  deploy:
    if: contains(github.event.comment.body, '/deploy')
    runs-on: ubuntu-latest
    steps:
      - name: Deploy via Issue Command
        run: ./tools/deploy.py --env ${{ github.event.comment.body }}
```

##### Matrix Testing
Expand testing across multiple configurations:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    compiler: [gcc-11, gcc-12, clang-14, clang-15]
    build_type: [Debug, Release]
    sanitizer: [none, address, thread, undefined]
```

#### 3. Security Scanning

##### GitLeaks Integration
Add secret scanning to prevent credential leaks:

```bash
# tools/check_secrets.py
pip install gitleaks
gitleaks detect --source . --verbose
```

##### Dependency Scanning
Implement supply chain security:

```yaml
- uses: actions/dependency-review-action@v3
  with:
    fail-on-severity: moderate
```

#### 4. AI-Powered Development Tools

Following 2024 trends with AI-enabled developer tools:

##### Code Review Automation
```python
# tools/ai_review.py
def review_pull_request(pr_diff):
    """Use LLM for automated code review suggestions"""
    # Integration with GitHub Copilot APIs
    # or OpenAI Codex for review automation
```

##### Intelligent Test Generation
```python
# tools/generate_tests.py
def generate_unit_tests(source_file):
    """AI-powered test case generation"""
    # Analyze code structure
    # Generate comprehensive test cases
    # Ensure edge case coverage
```

#### 5. Performance and Observability

##### Continuous Profiling
```python
# tools/profile_continuous.py
"""
Integration with tools like:
- py-spy for Python profiling
- perf for system profiling
- Tracy for C++ profiling
"""
```

##### Distributed Tracing
```yaml
# OpenTelemetry integration
- name: Setup OTel
  uses: open-telemetry/opentelemetry-collector-action@v0.1.0
```

#### 6. Container and Cloud Native Tools

##### Container Scanning
```bash
# tools/scan_containers.py
trivy image --severity HIGH,CRITICAL my-image:latest
```

##### Kubernetes Integration
```yaml
# tools/k8s_deploy.py
"""
Deploy to Kubernetes with:
- Helm chart generation
- ArgoCD integration
- GitOps workflows
"""
```

#### 7. Documentation Automation

##### API Documentation
```python
# tools/generate_api_docs.py
"""
Auto-generate documentation:
- OpenAPI/Swagger specs
- Doxygen integration
- Architecture diagrams with Mermaid
"""
```

##### Changelog Generation
```bash
# tools/generate_changelog.py
# Conventional commits to changelog
# Semantic versioning automation
```

### Implementation Priorities

1. **Phase 1 (Immediate)**:
   - Migrate to pre-commit framework
   - Add GitLeaks for security
   - Implement dependency scanning

2. **Phase 2 (3-6 months)**:
   - GitHub Actions matrix testing
   - IssueOps for deployment
   - Container scanning

3. **Phase 3 (6-12 months)**:
   - AI-powered code review
   - Continuous profiling
   - Kubernetes integration

### Metrics and KPIs

Track tooling effectiveness with:

- **Code Quality Metrics**:
  - Static analysis issue density
  - Test coverage percentage
  - Code review turnaround time

- **Developer Productivity**:
  - Time to first commit
  - Build/test cycle time
  - Mean time to resolve issues

- **Operational Excellence**:
  - Deployment frequency
  - Change failure rate
  - Mean time to recovery

## Contributing

When adding new tools:

1. Follow the existing Python script structure
2. Include comprehensive docstrings
3. Add argument parsing with clear help text
4. Implement both check and fix modes where applicable
5. Include examples in the docstring
6. Update this README with tool documentation
7. Add integration tests

## License

MIT License - See project root LICENSE file for details.

## Support

For issues or questions about the tools:
1. Check tool-specific `--help` output
2. Review examples in this README
3. Consult the project's CLAUDE.md for context
4. Open an issue in the project repository
