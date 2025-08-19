# CLAUDE.md - AI Assistant Context for Inference Systems Laboratory

This document provides comprehensive context for AI assistants working on the Inference Systems Laboratory project. It contains all essential information needed to understand the project structure, standards, and current state.

## Project Overview

**Inference Systems Laboratory** is a modern C++17+ research and development platform focused on building robust, high-performance inference systems with enterprise-grade tooling. The project emphasizes:

- **Modern C++17+**: Advanced language features, zero-cost abstractions, RAII patterns
- **Enterprise Quality**: Comprehensive testing, automated quality gates, performance monitoring  
- **Developer Experience**: Extensive tooling, pre-commit hooks, automated workflows
- **Performance Focus**: Benchmarking, profiling, cache-friendly data structures

## Current Status & Architecture

### ✅ COMPLETED FOUNDATION (Phase 1)

The project has established a solid foundation with enterprise-grade infrastructure:

#### **Core Infrastructure**
- **Error Handling**: Complete `Result<T, E>` implementation with monadic operations (map, and_then, or_else)
- **Logging System**: Thread-safe, structured logging with compile-time filtering and multiple levels
- **Serialization**: Cap'n Proto integration with comprehensive schema evolution and versioning support
- **Build System**: Modular CMake with sanitizers, testing, coverage, and cross-platform support

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
- **Static Analysis**: Comprehensive modernization with systematic fixing tools (1405→650+ issues resolved)
- **Build Quality**: **ZERO build warnings** achieved - gold standard compilation quality

#### **Quality Standards Achieved**
- **Build Quality**: 100% warning-free compilation across all targets
- **Test Coverage**: All foundation components have comprehensive test suites with 100% pass rates
- **Static Analysis**: Major modernization completed (98.3% improvement in large headers)
- **Code Standards**: Modern C++17 patterns, RAII compliance, performance benchmarks

### 🚧 CURRENT PRIORITIES

1. **Core Data Structures**: Cache-friendly containers, memory pools, concurrent data structures
2. **Type System**: Common concepts, type traits, strong type aliases  
3. **Forward Chaining Engine**: Basic rule representation, fact database, inference algorithm
4. **Static Analysis Completion**: Final modernization phases for remaining implementation files

### 📋 PLANNED DEVELOPMENT

- **Inference Engines**: Forward chaining, backward chaining, RETE networks, rule optimization
- **Distributed Systems**: Consensus algorithms (Raft, PBFT), distributed state machines
- **Performance Engineering**: SIMD optimizations, custom allocators, profiling integration
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
├── tools/                     # ✅ COMPLETE AUTOMATION SUITE
│   ├── new_module.py         # ✅ Generate module scaffolding
│   ├── check_format.py       # ✅ Code formatting validation/fixing
│   ├── check_static_analysis.py # ✅ Clang-tidy automation with systematic fixing
│   ├── check_coverage.py     # ✅ Test coverage verification  
│   ├── check_eof_newline.py  # ✅ POSIX compliance validation
│   ├── run_benchmarks.py     # ✅ Performance regression detection
│   └── install_hooks.py      # ✅ Pre-commit hook management
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
│   ├── src/tensorrt/         # 📋 PLANNED - TensorRT GPU acceleration (Phase 1)
│   ├── src/onnx/             # 📋 PLANNED - ONNX Runtime integration (Phase 2)
│   ├── src/forward_chaining/ # 📋 PLANNED - Rule-based inference engines
│   ├── src/inference_engine.hpp # 📋 PLANNED - Unified inference interface
│   ├── examples/             # 📋 PLANNED - ML model demos and tutorials
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
- **Testing Required**: Every piece of code needs comprehensive tests (80%+ coverage)
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

### Schema Evolution System (Latest)
Just completed comprehensive schema versioning and evolution framework:
- Full semantic versioning support with compatibility rules
- Migration system with multiple strategies and validation
- Backward compatibility preservation with safe evolution paths  
- C++ API for schema management (`SchemaEvolutionManager`, `VersionValidator`, `SchemaRegistry`)
- Comprehensive test suite demonstrating all features

### Static Analysis Modernization
Systematic modernization across entire codebase:
- **Phase 3 Complete**: Large headers modernized (458→8 issues, 98.3% improvement)
- **Build Quality**: Zero compilation warnings achieved (gold standard)
- **Modern C++17**: Enum base types, trailing return types, special member functions
- **Template modernization**: Concept-constrained descriptive naming patterns

### Development Tooling Excellence
Complete automation suite for enterprise-grade development:
- **Pre-commit hooks**: Automatic quality gates preventing low-quality commits
- **Module scaffolding**: Generate complete module structure with single command
- **Performance monitoring**: Regression detection with baseline comparison
- **Coverage tracking**: Automated analysis with configurable thresholds
- **Static analysis**: Systematic fixing tools with build safety integration

## Current Work Context

### Last Completed Task
✅ **Schema Evolution Implementation** - Comprehensive schema versioning and evolution support:
- Added SchemaVersion, SchemaEvolution, MigrationPath structures to Cap'n Proto schema
- Implemented full C++ API with SchemaEvolutionManager, VersionValidator, SchemaRegistry
- Created comprehensive test suite (schema_evolution_demo.cpp) with 100% pass rate
- Updated all core classes (Fact, Rule) with version tracking support
- Maintained full backward compatibility with existing code
- **Status**: All committed as single comprehensive commit with documentation updates

### Development Priorities
1. **Current**: ML Integration Documentation (Phase 0 - TensorRT/ONNX planning)
2. **Next**: Core data structures implementation (`common/src/containers.hpp`)
3. **Then**: TensorRT GPU inference engine (`engines/src/tensorrt/`)
4. **Also**: ONNX Runtime cross-platform support (`engines/src/onnx/`)
5. **Future**: Neural-symbolic fusion and distributed ML inference

## ML Inference Integration Roadmap

### Phase 0: Documentation and Architecture (CURRENT)
**Status**: In Progress - Documentation updates and dependency planning
- [x] README.md updated with TensorRT/ONNX vision and roadmap
- [ ] CLAUDE.md integration context and development workflow  
- [ ] WORK_TODO.md detailed implementation phases
- [ ] DEVELOPMENT.md ML-specific development workflow
- [ ] Dependencies documentation (TensorRT 8.5+, ONNX Runtime 1.15+)

### Phase 1: TensorRT Foundation (Next Priority - 2-3 months)
**Goal**: GPU-accelerated inference with existing Result<T,E> patterns
- **Core Engine**: `engines/src/tensorrt/tensorrt_engine.hpp` - Basic model loading and execution
- **Error Handling**: Extend `InferenceError` enum for TensorRT-specific errors
- **Examples**: `engines/examples/tensorrt_demo.cpp` - Basic GPU inference demonstration
- **Benchmarks**: Performance comparison between CPU and GPU inference paths
- **Integration**: Seamless integration with existing logging and schema evolution systems

### Phase 2: ONNX Runtime Cross-Platform (3-6 months)  
**Goal**: Universal model format support with multi-backend execution
- **ONNX Engine**: `engines/src/onnx/onnx_engine.hpp` - Cross-platform model execution
- **Backend Management**: Dynamic selection between CPU, GPU, and specialized accelerators
- **Model Versioning**: Schema evolution patterns for ML model lifecycle management
- **Production Ready**: Enterprise-grade model serving with monitoring and error recovery

### Phase 3: Unified Inference Architecture (6-9 months)
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
