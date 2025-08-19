# Development Guide for inference-systems-lab

## Quick Start for New Tasks

**When asking Claude Code for the next coding assignment, this context applies:**

### Core Requirements
- **C++17 or greater** for all C++ code
- **Strongly typed Python** with type hints for utility scripts
- **Comprehensive testing** required for all code, including examples
- **Performance focus** - measure and optimize critical paths
- Developer has extensive experience (coding since 1980) - assume advanced knowledge

## Template naming
- use Modern C++ Template Parameter Naming (2025) using concept-constrained parameters with descriptive names, such as
```cpp
template<std::copyable ElementType>
  instead of the traditional template<typename T>.
```

- This modern pattern emerged from C++20 concepts and represents current best practice because it provides self-documenting interfaces, significantly better compiler error messages, enables aggressive optimizations, and makes code intent crystal clear - major codebases like Google's, Microsoft's, and LLVM have adopted this approach as of 2023-2024, making concept-constrained descriptive naming the gold standard for new C++ code in 2025, though traditional T naming remains acceptable for simple generic utilities.

## Development Environment Setup

### Required Tools
```bash
# C++ Development
cmake >= 3.20
clang++ >= 12 or g++ >= 10  # C++17 support
clang-format                 # Code formatting
clang-tidy                   # Static analysis
valgrind                     # Memory checking
perf                         # Performance profiling

# Python Development (for tooling)
python >= 3.8
mypy                         # Type checking
black                        # Code formatting
pytest                       # Testing framework

# Testing Tools
catch2                       # C++ unit testing
google-benchmark             # C++ benchmarking
lcov                        # Coverage reporting

# ML Integration Tools (Optional)
tensorrt >= 8.5              # NVIDIA GPU inference
onnxruntime >= 1.15          # Cross-platform ML inference
cuda-toolkit >= 11.8        # GPU acceleration support
```

### ML Development Workflow (TensorRT & ONNX Integration)

#### ML Dependencies Setup
```bash
# TensorRT Setup (NVIDIA GPUs)
# Download TensorRT from NVIDIA Developer portal
export TENSORRT_ROOT=/path/to/TensorRT
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH

# ONNX Runtime Setup (Cross-platform)
# Option 1: Package manager
sudo apt install libonnxruntime-dev  # Ubuntu/Debian
brew install onnxruntime             # macOS

# Option 2: Build from source
git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel
```

#### ML Model Development Workflow
```bash
# 1. Model Preparation
# Convert models to optimized formats
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16  # TensorRT optimization
python -c "import onnx; onnx.checker.check_model('model.onnx')"  # ONNX validation

# 2. Integration Development
mkdir engines/examples/models  # Store test models
# Implement inference engine with Result<T,E> patterns
# Add comprehensive unit tests with mock GPU environment

# 3. Performance Validation
./build/engines/tensorrt_benchmarks     # GPU inference performance
./build/engines/onnx_benchmarks         # Cross-platform performance
python3 tools/run_benchmarks.py --compare-ml-backends  # Backend comparison

# 4. Model Versioning
# Use schema evolution patterns for model lifecycle
# Integrate with existing Cap'n Proto versioning system
```

#### ML-Specific Testing Strategy
```cpp
// GPU inference testing with mock environment
class TensorRTEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Mock GPU environment for CI/CD
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required for TensorRT tests";
        }
    }
    
    // Test model loading, inference, error handling
    // Validate GPU memory management and cleanup
    // Performance regression tests with baseline comparison
};

// Cross-platform ONNX testing
class ONNXEngineTest : public ::testing::Test {
    // Test model execution across CPU, GPU, specialized accelerators
    // Validate backend switching and provider selection
    // Memory usage and performance across different platforms
};
```

#### ML Error Handling Patterns
```cpp
// Extend existing Result<T,E> patterns for ML workloads
enum class InferenceError : std::uint8_t {
    MODEL_LOAD_FAILED,
    UNSUPPORTED_MODEL_FORMAT,
    GPU_MEMORY_EXHAUSTED,
    BACKEND_NOT_AVAILABLE,
    INFERENCE_EXECUTION_FAILED,
    INVALID_INPUT_SHAPE,
    MODEL_VERSION_MISMATCH
};

// Integration with existing error handling
auto load_model(const std::string& path, InferenceBackend backend) 
    -> Result<std::unique_ptr<InferenceEngine>, InferenceError>;

auto run_inference(const InferenceRequest& request) 
    -> Result<InferenceResponse, InferenceError>;
```

### Build Instructions
```bash
# Standard build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Debug build with sanitizers
mkdir build-debug && cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DSANITIZE_ADDRESS=ON \
         -DSANITIZE_UNDEFINED=ON
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Run benchmarks
./benchmarks/run_all_benchmarks
```

## Project Architecture Principles

### Memory Management
- Use RAII everywhere
- Prefer stack allocation
- When heap is needed, use smart pointers
- Consider custom allocators for hot paths
- Document ownership clearly

### Concurrency Model
- Lock-free data structures where possible
- Clear documentation of thread safety
- Use `std::atomic` for simple shared state
- Prefer message passing over shared memory

### Error Handling Strategy
```cpp
// Preferred: Result type for expected errors
template<typename T, typename Error>
using Result = std::expected<T, Error>;  // or custom implementation

// Avoid exceptions except for truly exceptional cases
// Never use raw error codes
```

### Testing Philosophy
- **Test-first development encouraged**
- Every bug fix requires a regression test
- Performance tests prevent degradation
- Property-based testing for algorithms

## Code Review Checklist

Before submitting code, verify:

- [ ] All tests pass (`ctest --output-on-failure`)
- [ ] No memory leaks (`valgrind --leak-check=full`)
- [ ] No undefined behavior (run with sanitizers)
- [ ] Performance benchmarks included for critical paths
- [ ] Documentation updated
- [ ] Code formatted (`clang-format -i src/*.cpp`)
- [ ] Static analysis clean (`clang-tidy`)

## Performance Guidelines

### Measurement First
```cpp
// Always benchmark before optimizing
BENCHMARK(ForwardChaining_SimpleRules) {
    KnowledgeBase kb;
    // ... setup ...
    return kb.infer();
}
```

### Cache-Friendly Design
- Keep hot data together
- Minimize pointer chasing
- Use contiguous containers when possible
- Consider data-oriented design for performance-critical sections

### SIMD Considerations
- Design data structures to be SIMD-friendly
- Use aligned allocations for vectorizable data
- Document vectorization opportunities

## Module Development Order

Recommended implementation sequence:

1. **common/** - Foundation utilities
   - Result/Error types
   - Logging infrastructure
   - Serialization framework
   - Basic data structures

2. **engines/** - ML & Rule-based inference (PRIORITY UPDATE)
   - **Phase 1**: TensorRT GPU inference engine (NEW)
   - **Phase 2**: ONNX Runtime cross-platform engine (NEW)
   - **Phase 3**: Unified inference interface (NEW)
   - Forward chaining engine
   - Backward chaining engine
   - RETE network
   - Rule optimization

3. **performance/** - Measurement tools
   - Micro-benchmarking framework
   - Profiling integration
   - Cache analysis tools

4. **distributed/** - Distribution layer
   - Network abstraction
   - Consensus protocols
   - Distributed state machine

5. **integration/** - System integration
   - End-to-end scenarios
   - Real-world applications

## Debugging Tips

### Performance Issues
```bash
# CPU profiling
perf record -g ./your_binary
perf report

# Cache analysis
valgrind --tool=cachegrind ./your_binary

# Lock contention
perf record -e sched:sched_switch -g ./your_binary
```

### Memory Issues
```bash
# Memory leaks
valgrind --leak-check=full --show-leak-kinds=all ./your_binary

# Address sanitizer (compile-time flag)
-fsanitize=address

# Undefined behavior sanitizer
-fsanitize=undefined
```

## Common Patterns

### Factory Pattern for Engines
```cpp
template<typename RuleType>
class EngineFactory {
    static auto create(EngineType type) -> std::unique_ptr<Engine<RuleType>>;
};
```

### Builder Pattern for Complex Objects
```cpp
KnowledgeBase kb = KnowledgeBaseBuilder()
    .with_rules(rules)
    .with_facts(initial_facts)
    .with_inference_strategy(ForwardChaining{})
    .build();
```

## Getting Help

- Check module-specific README files
- Review existing code in `common/` for patterns
- Run tests to understand expected behavior
- Use benchmarks to verify performance assumptions

## Note for Claude Code

When implementing new features:
1. Always include comprehensive tests
2. Use C++17 features appropriately
3. Add benchmarks for performance-critical code
4. Follow the patterns established in existing code
5. Document design decisions and trade-offs
