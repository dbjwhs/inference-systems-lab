# CLAUDE.md - AI Assistant Context for Inference Systems Laboratory

This document provides comprehensive context for AI assistants working on the Inference Systems Laboratory project. It contains all essential information needed to understand the project structure, standards, and current state.

## Project Overview

**Inference Systems Laboratory** is a modern C++17+ research and development platform focused on building robust, high-performance inference systems with enterprise-grade tooling. The project emphasizes:

- **Modern C++17+**: Advanced language features, zero-cost abstractions, RAII patterns
- **Enterprise Quality**: Comprehensive testing, automated quality gates, performance monitoring  
- **Developer Experience**: Extensive tooling, pre-commit hooks, automated workflows
- **Performance Focus**: Benchmarking, profiling, cache-friendly data structures

## Current Status & Architecture

### ✅ COMPLETED PHASES (Phases 1-3)

The project has achieved major milestones with enterprise-grade ML infrastructure:

#### **Core Infrastructure (Phase 1)**
- **Error Handling**: Complete `Result<T, E>` implementation with monadic operations (map, and_then, or_else)
- **Logging System**: Thread-safe, structured logging with compile-time filtering and multiple levels
- **Serialization**: Cap'n Proto integration with comprehensive schema evolution and versioning support
- **Build System**: Modular CMake with sanitizers, testing, coverage, and cross-platform support

#### **Advanced ML Infrastructure (Phase 2)**
- **ML Containers**: SIMD-optimized BatchContainer, RealtimeCircularBuffer, FeatureCache with 64-byte alignment
- **Type System**: TypedTensor with compile-time verification, neural network layers, automatic differentiation
- **Memory Management**: Lock-free concurrent data structures, memory pools with O(1) allocation
- **Performance**: Zero-cost abstractions achieving 1.02x overhead ratio with comprehensive benchmarking

#### **ML Tooling Suite (Phase 3 - LATEST)**
- **Model Manager**: Complete version control with semantic versioning, lifecycle management (dev→staging→production)
- **Model Converter**: Automated PyTorch→ONNX→TensorRT pipeline with precision support (FP32/FP16/INT8)
- **Inference Benchmarker**: Performance analysis with latency percentiles (p50/p95/p99), GPU profiling integration
- **Model Validator**: Multi-level validation (basic/standard/strict/exhaustive) with correctness testing

#### **Schema Evolution System** 
Recently completed comprehensive schema versioning framework:
- Semantic versioning (major.minor.patch) with compatibility checking
- Migration framework supporting multiple strategies (direct mapping, transformation, default values, custom logic, lossy)
- `SchemaEvolutionManager` for coordinating migrations between versions
- `VersionValidator` for checking evolution safety and best practices
- `SchemaRegistry` for tracking available schema versions
- `VersionedSerializer` with automatic migration support
- Full backward compatibility preservation with safe evolution paths

#### **Development Tooling Excellence**
- **Quality Assurance**: `clang-format` (Google Style + modern C++), `clang-tidy` (25+ check categories), pre-commit hooks
- **Module Scaffolding**: `tools/new_module.py` generates complete module structure with tests
- **Performance Monitoring**: `tools/run_benchmarks.py` with regression detection and baseline comparison
- **Coverage Analysis**: `tools/check_coverage.py` with configurable thresholds and HTML reports
- **Static Analysis**: Comprehensive modernization with systematic fixing tools (1405→650+ issues resolved, plus 12 easy win files completed)
- **Build Quality**: **ZERO build warnings** achieved - gold standard compilation quality
- **ML Operations**: Complete toolchain with model_manager.py, convert_model.py, benchmark_inference.py, validate_model.py

#### **Quality Standards Achieved**
- **Build Quality**: 100% warning-free compilation across all targets
- **Test Coverage**: All components have comprehensive test suites (152 total tests) with 100% pass rates
- **Static Analysis**: Major modernization completed (98.3% improvement in large headers + 12 easy win files with incremental fixes)
- **Code Standards**: Modern C++17 patterns, RAII compliance, performance benchmarks
- **ML Operations**: Enterprise-grade ML toolchain with 4,000+ lines of production-quality code

### ✅ COMPLETED PHASES (Phases 1-4)

#### **Enterprise Test Coverage Initiative (Phase 4 - LATEST)**
- **Phase 4.1**: Comprehensive engines module testing with unified interface validation, TensorRT mock testing, and inference algorithm coverage
- **Phase 4.2**: Complete integration test implementation replacing all 13 SKIPPED tests with functional ML pipeline testing
- **Phase 4.3**: Comprehensive concurrent stress test suite with multi-threaded validation under extreme load (50-200 threads, >95% success rate)
- **Phase 4.4**: Error injection and recovery testing with fault tolerance validation and graceful degradation
- **Coverage Achievement**: Overall project coverage exceeded 87%+ (from 80.67%), surpassing enterprise 85% target
- **Strategic Impact**: Gold-standard testing infrastructure for production ML inference systems

### 🚧 CURRENT PRIORITIES (Next Phase)

1. **Build System Enhancement**: ENABLE_TENSORRT, ENABLE_ONNX options with ML dependency management
2. **Example Applications**: Real-world ML demonstration servers with monitoring and dashboards
3. **Static Analysis Completion**: Final modernization phases for remaining implementation files

### 📋 PLANNED DEVELOPMENT

- **ONNX Runtime Integration**: Cross-platform model execution with dynamic backend switching
- **TensorRT GPU Integration**: Hardware-accelerated inference with CUDA optimization
- **Inference Engines**: Forward chaining, backward chaining, RETE networks, rule optimization
- **Distributed Systems**: Consensus algorithms (Raft, PBFT), distributed state machines
- **Performance Engineering**: Advanced SIMD optimizations, custom allocators, profiling integration
- **System Integration**: End-to-end distributed inference scenarios

## File Structure & Key Components

```
inference-systems-lab/
├── common/                    # ✅ FOUNDATION COMPLETE
│   ├── src/
│   │   ├── result.hpp        # ✅ Complete Result<T,E> error handling
│   │   ├── logging.hpp       # ✅ Thread-safe structured logging  
│   │   ├── inference_types.hpp # ✅ Cap'n Proto C++ wrappers (modernized)
│   │   ├── schema_evolution.hpp # ✅ Schema versioning system (NEW)
│   │   ├── schema_evolution.cpp # ✅ Schema evolution implementation (NEW)
│   │   └── inference_builders.hpp # ✅ Fluent builder interfaces
│   ├── tests/                # ✅ Comprehensive test suites (100% pass rate)
│   ├── examples/             # ✅ Working demonstrations and learning materials
│   ├── benchmarks/           # ✅ Performance regression tracking
│   ├── schemas/              # ✅ Cap'n Proto schema definitions with versioning
│   └── docs/                 # ✅ API documentation and design principles
├── tools/                     # ✅ COMPLETE AUTOMATION & ML SUITE
│   ├── new_module.py         # ✅ Generate module scaffolding
│   ├── check_format.py       # ✅ Code formatting validation/fixing
│   ├── check_static_analysis.py # ✅ Clang-tidy automation with systematic fixing
│   ├── check_coverage.py     # ✅ Test coverage verification  
│   ├── check_eof_newline.py  # ✅ POSIX compliance validation
│   ├── run_benchmarks.py     # ✅ Performance regression detection
│   ├── install_hooks.py      # ✅ Pre-commit hook management
│   ├── model_manager.py      # ✅ ML model version control and lifecycle (NEW)
│   ├── convert_model.py      # ✅ Automated model conversion pipeline (NEW)
│   ├── benchmark_inference.py # ✅ ML performance analysis framework (NEW)
│   └── validate_model.py     # ✅ Model correctness and accuracy testing (NEW)
├── docs/                     # ✅ COMPREHENSIVE DOCUMENTATION
│   ├── FORMATTING.md         # ✅ Code style and clang-format automation
│   ├── STATIC_ANALYSIS.md    # ✅ Clang-tidy standards and workflow
│   ├── PRE_COMMIT_HOOKS.md   # ✅ Quality gate documentation
│   └── EOF_NEWLINES.md       # ✅ POSIX compliance standards
├── cmake/                    # ✅ MODULAR BUILD SYSTEM
│   ├── CompilerOptions.cmake # ✅ Modern C++17+ configuration
│   ├── Sanitizers.cmake      # ✅ AddressSanitizer, UBSan integration  
│   ├── Testing.cmake         # ✅ GoogleTest framework setup
│   ├── Benchmarking.cmake    # ✅ Google Benchmark integration
│   └── StaticAnalysis.cmake  # ✅ Clang-tidy automation
├── engines/                  # 🚧 EXPANDING - Inference engine implementations
│   ├── src/tensorrt/         # 📋 PLANNED - TensorRT GPU acceleration (Phase 5)
│   ├── src/onnx/             # 📋 PLANNED - ONNX Runtime integration (Phase 5)
│   ├── src/forward_chaining/ # 📋 PLANNED - Rule-based inference engines
│   ├── src/inference_engine.hpp # 📋 PLANNED - Unified inference interface
│   ├── examples/             # 🚧 PHASE 4 - Real-world ML demonstration servers
│   └── benchmarks/           # 📋 PLANNED - Performance comparisons
├── distributed/              # 🚧 PLACEHOLDER - Ready for implementation  
├── performance/              # 🚧 PLACEHOLDER - Ready for implementation
├── integration/              # 🚧 PLACEHOLDER - Ready for implementation
└── experiments/              # 🚧 PLACEHOLDER - Ready for implementation
```

## Coding Standards & Requirements

### Language & Style Requirements
- **C++17 minimum** - Use modern features (structured bindings, std::optional, std::variant, if constexpr)
- **Template naming**: Modern concept-constrained descriptive naming preferred:
  ```cpp
  template<std::copyable ElementType>  // Preferred modern style
  template<typename T>                 // Acceptable for simple cases
  ```
- **Error Handling**: `Result<T, E>` pattern preferred over exceptions
- **RAII**: Strict resource management, prefer stack allocation
- **Zero-cost abstractions**: Performance-critical code with minimal overhead

### Code Quality Standards
- **Testing Required**: Every piece of code needs comprehensive tests (85%+ coverage target, currently 80.67%)
- **Enterprise Test Standards**: Multi-phase systematic approach to coverage excellence
- **Documentation**: All public APIs require Doxygen-style documentation with examples
- **Performance**: Include benchmarks for performance-critical components
- **Memory Safety**: AddressSanitizer, UBSan integration, no memory leaks
- **Build Quality**: Zero warnings, clean static analysis, automated quality gates

### File & Naming Conventions
- **Headers**: `.hpp` extension (not `.h`)
- **Classes**: `PascalCase` 
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Template Parameters**: `PascalCase` (modern descriptive names)
- **Namespaces**: `lower_snake_case`

## Build System & Development Workflow

### Quick Setup Commands
```bash
# Clone and initial setup
git clone <repository-url>
cd inference-systems-lab

# Install pre-commit hooks (recommended)
python3 tools/install_hooks.py --install

# Standard build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Debug build with sanitizers
cmake .. -DCMAKE_BUILD_TYPE=Debug -DSANITIZER_TYPE=address+undefined
make -j$(nproc)

# Verify installation
ctest --output-on-failure
```

### Quality Assurance Workflow
```bash
# Automated quality checks (run by pre-commit hooks)
python3 tools/check_format.py --fix --backup          # Fix formatting
python3 tools/check_static_analysis.py --check        # Static analysis
python3 tools/check_eof_newline.py --fix             # POSIX compliance

# Performance and coverage tracking
python3 tools/run_benchmarks.py --save-baseline name  # Save baseline
python3 tools/run_benchmarks.py --compare-against name # Check regressions
python3 tools/check_coverage.py --threshold 80.0      # Coverage verification
```

### Module Development
```bash
# Create new module with complete scaffolding
python3 tools/new_module.py my_module --author "Name" --description "Description"
# Generates: src/, tests/, examples/, benchmarks/, docs/, CMakeLists.txt
```

## Key Design Patterns & Components

### Result<T, E> Error Handling
```cpp
// Modern error handling without exceptions
auto parse_config() -> Result<Config, ParseError> {
    return Config::from_file("config.json")
        .and_then([](Config cfg) { return cfg.validate(); })
        .map_err([](ParseError err) { return err.add_context("config load"); });
}
```

### Schema Evolution System (NEW)
```cpp
// Semantic versioning with compatibility checking
SchemaVersion v1_0_0(1, 0, 0);
SchemaVersion v1_1_0(1, 1, 0);

// Migration path definition
MigrationPath path(v1_0_0, v1_1_0, 
                   MigrationPath::Strategy::DEFAULT_VALUES,
                   true, "Added optional schema version fields");

// Evolution manager for coordinating migrations
SchemaEvolutionManager manager(v1_1_0);
manager.register_migration_path(path);

// Automatic migration during deserialization
auto migrated_fact = manager.migrate_fact(old_fact, v1_0_0);
```

### Builder Pattern for Complex Objects
```cpp
// Fluent interface for object construction
auto rule = RuleBuilder()
    .with_id(1)
    .with_name("mortality_rule")
    .with_condition("isHuman", {"X"})
    .with_conclusion("isMortal", {"X"})
    .with_priority(10)
    .build();
```

### Performance-Oriented Value Types
```cpp
// Type-safe polymorphic values with zero-cost abstractions
Value val = Value::from_int64(42);
if (auto int_val = val.try_as_int64()) {
    // Safe extraction without exceptions
    process_integer(*int_val);
}
```

## Testing & Performance Philosophy

### Testing Requirements
- **Unit tests** for all public APIs (GoogleTest framework)
- **Integration tests** for component interactions  
- **Performance benchmarks** for critical paths (Google Benchmark)
- **Property-based testing** for algorithmic components
- **Memory safety** validation with sanitizers

### Performance Guidelines  
- **Measure first, optimize second** - All optimizations need benchmarks
- **Cache-friendly design** - Keep hot data together, minimize pointer chasing
- **SIMD considerations** - Design data structures for vectorization
- **Custom allocators** for performance-critical sections

## Recent Major Achievements

### ML Tooling Infrastructure Suite (LATEST - Phase 3 Complete)
Enterprise-grade ML operations toolchain implemented with 152 total tests:
- **Model Manager**: Complete version control with semantic versioning, lifecycle management, rollback capabilities
- **Model Converter**: Automated PyTorch→ONNX→TensorRT pipeline with precision support (FP32/FP16/INT8)
- **Inference Benchmarker**: Performance analysis with latency percentiles (p50/p95/p99), GPU profiling integration
- **Model Validator**: Multi-level validation (basic/standard/strict/exhaustive) with correctness testing
- **Technical Excellence**: 4,000+ lines production-quality code, graceful dependency handling, comprehensive CLI interfaces
- **Integration Ready**: All tools designed to work together as unified ML pipeline with existing systems

### Advanced ML Infrastructure (Phase 2 Complete)
High-performance ML container and type system implementation:
- **SIMD Containers**: BatchContainer, RealtimeCircularBuffer, FeatureCache with 64-byte alignment
- **Type System**: TypedTensor with compile-time verification, neural network layers, automatic differentiation
- **Performance**: Zero-cost abstractions achieving 1.02x overhead ratio with comprehensive benchmarking
- **Thread Safety**: Lock-free concurrent data structures with memory pools for O(1) allocation

### Schema Evolution System
Comprehensive schema versioning and evolution framework:
- Full semantic versioning support with compatibility rules
- Migration system with multiple strategies and validation
- Backward compatibility preservation with safe evolution paths  
- C++ API for schema management (`SchemaEvolutionManager`, `VersionValidator`, `SchemaRegistry`)
- Comprehensive test suite demonstrating all features

### Development Tooling Excellence
Complete automation suite for enterprise-grade development:
- **Pre-commit hooks**: Automatic quality gates preventing low-quality commits
- **Module scaffolding**: Generate complete module structure with single command
- **Performance monitoring**: Regression detection with baseline comparison
- **Coverage tracking**: Automated analysis with configurable thresholds
- **Static analysis**: Systematic fixing tools with build safety integration (1405→650+ issues resolved)

### Static Analysis Easy Wins (LATEST - Phase 4 Progress)
Incremental modernization with build safety focus:
- **12 Files Modernized**: All low-complexity files (≤10 issues) successfully fixed with clang-tidy automation
- **Strategic Targeting**: engines/, common/, integration/ directories with focused scope for quick wins
- **Build Safety**: Backup-restore methodology preserves compilation integrity throughout process
- **Quality Assurance**: All fixes pass comprehensive pre-commit validation (formatting, analysis, build verification)
- **25 Files Remaining**: Medium-high priority files (11-132 issues) ready for systematic incremental approach

## Enterprise Test Coverage Initiative 🎯

### **CRITICAL PRIORITY: Test Coverage Excellence** 
**Status**: 30-40% coverage → Target: 80%+ enterprise-grade standards

The project has excellent infrastructure but **insufficient test coverage for production deployment**. Based on professional assessment:

#### **Current Coverage State** ⚠️
- **Overall Coverage**: ~30-40% (Industry standard: 80%+)
- **Risk Level**: 🔴 **HIGH** - Not suitable for production
- **Critical Gaps**: Engines (~10%), Integration (~5%), Distributed/Performance (0%)

#### **5-Phase Coverage Excellence Plan**

**Phase 1: Coverage Infrastructure** (1 week)
```cmake
# Enable coverage measurement
cmake .. -DENABLE_COVERAGE=ON
make coverage-report
open coverage/index.html
```
- Coverage measurement tools (lcov/gcov integration)
- Automated HTML reports with line-by-line analysis
- CI/CD coverage gates preventing regression

**Phase 2: Baseline Assessment** (1 week)  
- Precise coverage metrics by module/function
- Critical untested path identification
- Coverage improvement roadmap with priorities

**Phase 3: Critical Test Implementation** (2-3 weeks)
- **Engines Module**: 0% → 80% (inference_engine, forward_chaining, model_registry)
- **Integration Tests**: Replace 12 SKIPPED tests with real implementations
- **Error Recovery**: Systematic error injection testing

**Phase 4: Enterprise-Grade Excellence** (2-3 weeks) - CURRENT PRIORITY
- **Phase 4.1**: Engines module comprehensive testing (~15% → 80% coverage)
- **Phase 4.2**: Integration test implementation (replace 12 SKIPPED tests)
- **Phase 4.3**: Concurrent component stress testing under load
- **Phase 4.4**: Error injection and fault tolerance validation
- **Target**: 85%+ minimum line coverage across all production modules

**Phase 5: Continuous Quality Assurance** (ongoing)
- Automated coverage monitoring and dashboards
- Coverage trend analysis and regression alerts
- Enterprise-grade quality gates

#### **Expected Outcomes**
- **100+ new comprehensive tests** targeting untested code
- **Visual coverage reports** with red/yellow/green line indicators
- **Automated quality gates** preventing coverage regression
- **Enterprise deployment readiness** with 80%+ coverage

### **Testing Philosophy Enhancement**
Beyond the existing 80% coverage requirement, implement:
- **Coverage-driven development**: Write tests for all critical paths
- **Mutation testing**: Verify test effectiveness by introducing bugs
- **Property-based testing**: Generative testing for complex scenarios
- **Stress testing**: Concurrent operation validation under load
- **Error injection**: Systematic failure mode testing

## Current Work Context

### Last Completed Task
✅ **Critical Memory Safety Fix & Comprehensive Testing Infrastructure** - Production reliability achievement:
- **Memory Safety Bug Fix**: Fixed critical heap-use-after-free in MemoryPool detected by AddressSanitizer
  - Race condition in concurrent vector access during pool expansion
  - Verified fix eliminates all memory errors under high contention
- **Comprehensive Test Orchestrator**: `tools/run_comprehensive_tests.py` - Single command testing infrastructure  
  - **Clean Builds**: Fresh build directories (Release, Debug, ASan, TSan, UBSan) ensure reproducible results
  - **Complete Test Coverage**: Unit, integration, stress, memory leak, concurrency, benchmarks
  - **Memory Safety Testing**: AddressSanitizer with leak detection (`detect_leaks=1`)
  - **Professional Reports**: HTML/JSON output in `test-results/` directory with detailed statistics
  - **Future-Proof Design**: Easy addition of new sanitizers and test suites
- **Usage Options**: `--quick` (smoke tests), `--memory` (memory focus), `--no-clean` (preserve builds)
- **Strategic Impact**: Single point of execution for all testing activities with systematic validation

### Development Priorities
1. **Next**: Build system ML integration and example applications
   - CMake ENABLE_TENSORRT and ENABLE_ONNX options implementation  
   - Real-world ML demonstration servers (`engines/examples/`)
   - Production monitoring and dashboard integration
2. **Then**: ONNX Runtime cross-platform integration
   - Dynamic backend switching (CPU, GPU, DirectML)
   - Production-ready model serving examples with performance monitoring
3. **Also**: Static analysis final cleanup for remaining implementation files
4. **Future**: Advanced ML inference features and distributed systems integration

## ML Inference Integration Roadmap

### Phase 1: Critical Foundation (COMPLETED ✅)
**Status**: Complete - Core ML infrastructure established
- [x] **Core Data Structures**: Cache-friendly containers, memory pools, concurrent data structures
- [x] **ML Type System**: Advanced tensor types with compile-time verification
- [x] **Error Handling**: Extended `Result<T,E>` for ML-specific error types
- [x] **Development Environment**: Docker, Nix flakes with ML dependencies

### Phase 2: Core Data Structures (COMPLETED ✅)
**Status**: Complete - Advanced ML container and type system implementation
- [x] **Advanced ML Containers**: SIMD-optimized BatchContainer, RealtimeCircularBuffer, FeatureCache
- [x] **Type System**: TypedTensor with compile-time verification, neural network layers, automatic differentiation
- [x] **Performance**: Zero-cost abstractions achieving 1.02x overhead ratio with comprehensive benchmarking
- [x] **Thread Safety**: Lock-free concurrent data structures with memory pools for O(1) allocation

### Phase 3: ML Tooling Infrastructure (COMPLETED ✅)
**Status**: Complete - Enterprise-grade ML operations suite implemented
- [x] **Model Manager**: Complete version control with semantic versioning, lifecycle management
- [x] **Model Converter**: Automated PyTorch→ONNX→TensorRT pipeline with precision support
- [x] **Inference Benchmarker**: Performance analysis with latency percentiles, GPU profiling integration
- [x] **Model Validator**: Multi-level validation with correctness testing and comprehensive reporting

### Phase 4: Enterprise Test Coverage Initiative (COMPLETED ✅)
**Status**: Complete - Enterprise-grade testing infrastructure achieved
- [x] **Phase 4.1**: Comprehensive engines module testing with unified interface validation
- [x] **Phase 4.2**: Complete integration test implementation replacing all SKIPPED tests  
- [x] **Phase 4.3**: Comprehensive concurrent stress test suite with multi-threaded validation
- [x] **Phase 4.4**: Error injection and recovery testing with fault tolerance validation
- [x] **Coverage Achievement**: 87%+ coverage exceeding enterprise 85% target

### Phase 5: ONNX Runtime Cross-Platform (Next Priority)
**Goal**: Universal model format support with multi-backend execution
- [ ] **ONNX Engine**: `engines/src/onnx/onnx_engine.hpp` - Cross-platform model execution
- [ ] **Backend Management**: Dynamic selection between CPU, GPU, and specialized accelerators
- [ ] **Model Versioning**: Schema evolution patterns for ML model lifecycle management
- [ ] **Production Ready**: Enterprise-grade model serving with monitoring and error recovery

### Phase 6: Unified Inference Architecture (Future)
**Goal**: Seamless integration between rule-based and ML inference
```cpp
// Planned unified interface integrating with existing patterns
enum class InferenceBackend : std::uint8_t { 
    RULE_BASED, 
    TENSORRT_GPU, 
    ONNX_RUNTIME,
    HYBRID_NEURAL_SYMBOLIC 
};

auto create_inference_engine(InferenceBackend backend, const ModelConfig& config) 
    -> Result<std::unique_ptr<InferenceEngine>, InferenceError>;

// Integration with existing Result<T,E> and logging patterns
auto run_inference(const InferenceRequest& request) 
    -> Result<InferenceResponse, InferenceError>;
```

### Technical Integration Points
- **Error Handling**: Extend existing `Result<T,E>` patterns for ML-specific error types
- **Logging**: Integrate ML inference metrics with existing structured logging system
- **Schema Evolution**: Apply versioning patterns to ML model lifecycle management  
- **Benchmarking**: Leverage existing performance framework for ML workload comparisons
- **Testing**: Apply existing quality standards (80%+ coverage, comprehensive test suites)
- **Development Workflow**: Use existing automation (pre-commit hooks, static analysis)

## AI Assistant Guidelines

When working on this project:

1. **Quality First**: All code must pass formatting, static analysis, and comprehensive tests
2. **Modern C++17+**: Use advanced language features appropriately (structured bindings, concepts, etc.)
3. **Testing Mandatory**: Every component needs comprehensive test coverage
4. **Performance Aware**: Include benchmarks for performance-critical code
5. **Documentation Required**: All public APIs need Doxygen documentation with examples
6. **Follow Patterns**: Study existing code in `common/` for established patterns
7. **Build Testing**: Verify changes don't break compilation or existing functionality

### Useful Commands for Development
- `python3 tools/run_comprehensive_tests.py` - Complete testing: all configs, all tests (RECOMMENDED)
- `python3 tools/run_comprehensive_tests.py --quick` - Quick smoke tests for development
- `python3 tools/run_comprehensive_tests.py --memory` - Memory safety focused testing
- `make clean && make -j4` - Clean build to verify no compilation issues
- `ctest --output-on-failure` - Run all tests with detailed failure output
- `python3 tools/check_format.py --fix` - Fix formatting issues
- `python3 tools/check_static_analysis.py --check` - Verify static analysis
- `python3 tools/run_benchmarks.py` - Check performance benchmarks

### Pre-Commit Workflow Best Practices
**IMPORTANT**: To avoid commit/hook/fix/commit churn, always run formatting before committing:

```bash
# Recommended workflow before any commit:
python3 tools/check_format.py --fix --staged    # Fix formatting for staged files
git add -A                                       # Stage formatting fixes
git commit -m "Your commit message"              # Commit with clean formatting

# Alternative: Run all quality checks at once
python3 tools/check_format.py --fix
python3 tools/check_static_analysis.py --check --severity warning
git add -A && git commit -m "Your message"

# Emergency bypass (use sparingly)
git commit --no-verify -m "Your message"        # Skip pre-commit hooks
```

**Why this matters**: Pre-commit hooks will block commits with formatting violations, leading to a frustrating cycle of commit → hook failure → fix → commit again. Running the formatter first eliminates this churn and ensures smooth development workflow.

### Context Notes
- **Developer Experience**: Extensive (coding since 1980) - assume advanced knowledge
- **Build System**: Modular CMake with comprehensive tooling integration
- **Dependencies**: Minimal external dependencies, prefer header-only libraries
- **Performance**: Measure-first optimization philosophy with benchmarking infrastructure
- **Quality**: Enterprise-grade standards with automated enforcement

This project represents a modern approach to C++ systems development with comprehensive tooling, testing, and performance monitoring. Every component is built for both educational value and production-quality engineering.
