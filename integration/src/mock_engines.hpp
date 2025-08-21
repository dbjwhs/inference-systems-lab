// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file mock_engines.hpp
 * @brief Mock inference engines for testing without hardware dependencies
 *
 * This header provides sophisticated mock implementations of different inference
 * backends that simulate real behavior without requiring actual hardware (GPU,
 * specialized ML accelerators). These mocks are essential for CI/CD pipelines,
 * development environments without hardware, and comprehensive error injection testing.
 *
 * Key Features:
 * - Realistic behavior simulation with configurable latency and errors
 * - Memory usage simulation for testing resource management
 * - Configurable error injection for testing error handling paths
 * - Performance characteristics matching real backends
 * - GPU memory simulation for TensorRT testing
 * - Cross-platform compatibility simulation for ONNX Runtime
 * - Statistical output generation for model-agnostic testing
 *
 * Design Principles:
 * - Behavioral fidelity: Mocks should behave like real engines
 * - Configurability: Easy to configure for different test scenarios
 * - Error injection: Support systematic error condition testing
 * - Performance modeling: Realistic timing and resource usage
 * - Stateful simulation: Maintain internal state like real engines
 *
 * Architecture:
 * @code
 *   ┌─────────────────┐    inherits     ┌─────────────────┐
 *   │ InferenceEngine │◄────────────────│ MockEngineBase  │
 *   │ (Interface)     │                 │ (Common Mock)   │
 *   └─────────────────┘                 └─────────────────┘
 *                                                │
 *                                                │ provides common
 *                                                ▼
 *                               ┌─────────────────────────────────┐
 *                               │ Configurable Mock Behavior      │
 *                               │ • Latency Simulation            │
 *                               │ • Memory Usage Tracking         │
 *                               │ • Error Injection               │
 *                               │ • Output Generation             │
 *                               └─────────────────────────────────┘
 *                                                │
 *                        ┌───────────────────────┼───────────────────────┐
 *                        ▼                       ▼                       ▼
 *                ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
 *                │ MockTensorRT    │     │ MockONNXRuntime │     │ MockRuleBased   │
 *                │ Engine          │     │ Engine          │     │ Engine          │
 *                │                 │     │                 │     │                 │
 *                │ • GPU Memory    │     │ • Cross Platform│     │ • Symbolic      │
 *                │ • Optimization  │     │ • Multi Backend │     │ • Rule Firing   │
 *                │ • TensorRT APIs │     │ • ONNX Models   │     │ • Logic Engine  │
 *                └─────────────────┘     └─────────────────┘     └─────────────────┘
 * @endcode
 *
 * Usage Example:
 * @code
 * // Create TensorRT mock with realistic GPU simulation
 * auto mock_config = MockEngineConfig{
 *     .simulate_gpu_memory = true,
 *     .base_latency_ms = 5.0f,
 *     .memory_usage_mb = 1024,
 *     .error_injection_rate = 0.01f  // 1% error rate
 * };
 *
 * auto tensorrt_mock = std::make_unique<MockTensorRTEngine>(mock_config);
 *
 * // Configure specific error scenarios
 * tensorrt_mock->inject_error_condition("GPU_MEMORY_EXHAUSTED", 0.05f);
 * tensorrt_mock->set_latency_variation(0.1f);  // 10% latency variation
 *
 * // Use in integration tests
 * framework.inject_mock_backend(InferenceBackend::TENSORRT_GPU,
 *                               [=]() { return std::move(tensorrt_mock); });
 * @endcode
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../../common/src/logging.hpp"
#include "../../common/src/ml_types.hpp"
#include "../../common/src/result.hpp"
#include "../../engines/src/inference_engine.hpp"

namespace inference_lab::integration::mocks {

using namespace common;
using namespace engines;
using namespace common::ml;

//=============================================================================
// Mock Configuration and Behavior Control
//=============================================================================

/**
 * @brief Error injection configuration for systematic testing
 */
struct ErrorInjectionConfig {
    std::unordered_map<std::string, float> error_rates;  ///< Error type -> probability
    bool fail_after_iterations = false;                  ///< Fail after N successful runs
    std::uint32_t failure_iteration = 100;               ///< Iteration to fail on
    bool intermittent_failures = false;                  ///< Random intermittent failures
    float global_error_rate = 0.0f;                      ///< Overall error probability
};

/**
 * @brief Performance simulation configuration
 */
struct PerformanceSimConfig {
    float base_latency_ms = 10.0f;             ///< Base inference latency
    float latency_variation = 0.1f;            ///< Latency variation (0-1)
    std::uint64_t base_memory_usage_mb = 256;  ///< Base memory usage
    float memory_variation = 0.05f;            ///< Memory usage variation
    bool simulate_warmup = true;               ///< Simulate warmup behavior
    std::uint32_t warmup_iterations = 3;       ///< Iterations to warm up
    float warmup_latency_multiplier = 2.0f;    ///< Warmup latency factor
};

/**
 * @brief Comprehensive mock engine configuration
 */
struct MockEngineConfig {
    // Basic behavior
    std::string engine_name = "MockEngine";  ///< Engine identification
    bool simulate_hardware = true;           ///< Simulate hardware constraints
    bool enable_logging = true;              ///< Enable mock operation logging

    // Performance characteristics
    PerformanceSimConfig performance;  ///< Performance simulation settings

    // Error injection
    ErrorInjectionConfig error_injection;  ///< Error injection configuration

    // Resource simulation
    bool simulate_gpu_memory = false;        ///< Simulate GPU memory management
    std::uint64_t max_gpu_memory_mb = 8192;  ///< Simulated GPU memory limit
    bool simulate_model_loading = true;      ///< Simulate model loading time
    float model_loading_time_ms = 500.0f;    ///< Model loading simulation time

    // Output generation
    bool generate_realistic_outputs = true;  ///< Generate realistic vs random outputs
    std::uint32_t random_seed = 42;          ///< Random seed for deterministic testing
    float output_noise_level = 0.01f;        ///< Noise level in generated outputs
};

//=============================================================================
// Base Mock Engine with Common Functionality
//=============================================================================

/**
 * @brief Base class for all mock inference engines
 *
 * Provides common mock functionality including error injection, performance
 * simulation, memory tracking, and realistic behavior modeling. Derived
 * classes implement backend-specific behavior while inheriting core mocking
 * infrastructure.
 */
class MockEngineBase : public InferenceEngine {
  public:
    /**
     * @brief Construct mock engine with configuration
     */
    explicit MockEngineBase(MockEngineConfig config);

    /**
     * @brief Virtual destructor
     */
    ~MockEngineBase() override = default;

    // InferenceEngine interface (common implementation)
    auto get_backend_info() const -> std::string override;
    auto is_ready() const -> bool override;
    auto get_performance_stats() const -> std::string override;

    // Mock-specific configuration interface

    /**
     * @brief Inject specific error condition
     */
    void inject_error_condition(const std::string& error_type, float probability);

    /**
     * @brief Clear all error injections
     */
    void clear_error_injections();

    /**
     * @brief Set latency variation for performance testing
     */
    void set_latency_variation(float variation);

    /**
     * @brief Set memory usage for resource testing
     */
    void set_memory_usage(std::uint64_t memory_mb);

    /**
     * @brief Enable/disable realistic output generation
     */
    void set_realistic_output_generation(bool enable);

    /**
     * @brief Reset performance statistics
     */
    void reset_statistics();

    /**
     * @brief Get detailed mock statistics
     */
    auto get_mock_statistics() const -> std::unordered_map<std::string, std::uint64_t>;

  protected:
    // Configuration
    MockEngineConfig config_;
    mutable std::mutex state_mutex_;

    // Internal state
    std::atomic<bool> is_initialized_{false};
    std::atomic<std::uint64_t> inference_count_{0};
    std::atomic<std::uint64_t> error_count_{0};
    std::atomic<std::uint64_t> total_latency_ms_{0};
    std::atomic<std::uint64_t> memory_allocated_mb_{0};

    // Random number generation
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<float> dist_;

    // Helper methods for derived classes

    /**
     * @brief Simulate model loading delay
     */
    void simulate_model_loading();

    /**
     * @brief Check if error should be injected
     */
    auto should_inject_error() const -> std::optional<InferenceError>;

    /**
     * @brief Simulate inference latency
     */
    void simulate_inference_latency();

    /**
     * @brief Generate realistic output tensors
     */
    auto generate_output_tensors(const std::vector<TensorSpec>& output_specs)
        -> std::vector<TensorOutput>;

    /**
     * @brief Simulate memory allocation/deallocation
     */
    void simulate_memory_operation(std::uint64_t size_mb, bool allocate);

    /**
     * @brief Update performance statistics
     */
    void update_statistics(std::chrono::milliseconds latency, std::uint64_t memory_used);

    /**
     * @brief Generate realistic tensor data
     */
    template <typename T>
    auto generate_realistic_tensor_data(const Shape& shape, const std::string& tensor_name)
        -> std::vector<T>;

  private:
    // Thread-safe error injection check
    auto check_error_injection() const -> std::optional<InferenceError>;

    // Generate output based on input characteristics
    auto generate_contextual_output(const TensorSpec& spec, const std::vector<TensorInput>& inputs)
        -> TensorOutput;
};

//=============================================================================
// TensorRT Mock Implementation
//=============================================================================

/**
 * @brief Mock TensorRT engine for GPU inference simulation
 *
 * Simulates NVIDIA TensorRT behavior including GPU memory management,
 * model optimization, and CUDA-specific operations. Useful for testing
 * GPU-specific code paths without requiring actual GPU hardware.
 */
class MockTensorRTEngine : public MockEngineBase {
  public:
    /**
     * @brief Construct TensorRT mock with GPU-specific configuration
     */
    explicit MockTensorRTEngine(MockEngineConfig config = create_tensorrt_config());

    /**
     * @brief Execute inference with TensorRT-specific behavior
     */
    auto run_inference(const InferenceRequest& request)
        -> Result<InferenceResponse, InferenceError> override;

    /**
     * @brief Get TensorRT-specific backend information
     */
    auto get_backend_info() const -> std::string override;

    // TensorRT-specific mock configuration

    /**
     * @brief Simulate GPU memory exhaustion
     */
    void set_gpu_memory_limit(std::uint64_t limit_mb);

    /**
     * @brief Simulate model optimization time
     */
    void set_optimization_time(std::chrono::milliseconds time);

    /**
     * @brief Enable CUDA error simulation
     */
    void enable_cuda_error_simulation(bool enable);

    /**
     * @brief Set precision mode simulation
     */
    void set_precision_mode(Precision precision);

    /**
     * @brief Create default TensorRT configuration
     */
    static auto create_tensorrt_config() -> MockEngineConfig;

  private:
    // TensorRT-specific state
    std::atomic<std::uint64_t> gpu_memory_used_mb_{0};
    std::uint64_t gpu_memory_limit_mb_{8192};
    Precision current_precision_{Precision::FP32};
    bool model_optimized_{false};
    std::chrono::milliseconds optimization_time_{200};

    // TensorRT-specific simulation
    auto simulate_gpu_memory_check(std::uint64_t required_mb) -> bool;
    auto simulate_model_optimization() -> Result<bool, InferenceError>;
    auto simulate_cuda_operations() -> Result<bool, InferenceError>;
    void simulate_precision_specific_behavior(const InferenceRequest& request);
};

//=============================================================================
// ONNX Runtime Mock Implementation
//=============================================================================

/**
 * @brief Mock ONNX Runtime engine for cross-platform inference simulation
 *
 * Simulates Microsoft ONNX Runtime behavior including model format handling,
 * execution provider selection, and cross-platform compatibility. Supports
 * testing different execution providers without hardware dependencies.
 */
class MockONNXRuntimeEngine : public MockEngineBase {
  public:
    /**
     * @brief Execution providers that ONNX Runtime supports
     */
    enum class ExecutionProvider : std::uint8_t {
        CPU = 0,
        CUDA = 1,
        TENSORRT = 2,
        DIRECTML = 3,
        OPENVINO = 4
    };

    /**
     * @brief Construct ONNX Runtime mock with provider-specific configuration
     */
    explicit MockONNXRuntimeEngine(MockEngineConfig config = create_onnx_config(),
                                   ExecutionProvider provider = ExecutionProvider::CPU);

    /**
     * @brief Execute inference with ONNX Runtime-specific behavior
     */
    auto run_inference(const InferenceRequest& request)
        -> Result<InferenceResponse, InferenceError> override;

    /**
     * @brief Get ONNX Runtime-specific backend information
     */
    auto get_backend_info() const -> std::string override;

    // ONNX Runtime-specific mock configuration

    /**
     * @brief Set execution provider
     */
    void set_execution_provider(ExecutionProvider provider);

    /**
     * @brief Simulate ONNX model compatibility check
     */
    void set_model_compatibility(bool compatible);

    /**
     * @brief Enable dynamic shape simulation
     */
    void enable_dynamic_shapes(bool enable);

    /**
     * @brief Set optimization level
     */
    void set_optimization_level(std::uint32_t level);

    /**
     * @brief Create default ONNX Runtime configuration
     */
    static auto create_onnx_config() -> MockEngineConfig;

  private:
    // ONNX Runtime-specific state
    ExecutionProvider execution_provider_{ExecutionProvider::CPU};
    bool model_compatible_{true};
    bool dynamic_shapes_enabled_{false};
    std::uint32_t optimization_level_{1};

    // Provider-specific simulation
    auto simulate_provider_initialization() -> Result<bool, InferenceError>;
    auto simulate_model_loading() -> Result<bool, InferenceError>;
    auto simulate_dynamic_shape_handling(const InferenceRequest& request)
        -> Result<bool, InferenceError>;
    void apply_provider_specific_optimizations();
    auto get_provider_name(ExecutionProvider provider) const -> std::string;
};

//=============================================================================
// Rule-Based Engine Mock Implementation
//=============================================================================

/**
 * @brief Mock rule-based inference engine for symbolic reasoning simulation
 *
 * Simulates traditional rule-based inference behavior including fact matching,
 * rule firing, and symbolic computation. Useful for testing hybrid neural-symbolic
 * systems without requiring complex rule engine implementation.
 */
class MockRuleBasedEngine : public MockEngineBase {
  public:
    /**
     * @brief Construct rule-based mock with symbolic reasoning configuration
     */
    explicit MockRuleBasedEngine(MockEngineConfig config = create_rule_based_config());

    /**
     * @brief Execute inference with rule-based behavior
     */
    auto run_inference(const InferenceRequest& request)
        -> Result<InferenceResponse, InferenceError> override;

    /**
     * @brief Get rule-based backend information
     */
    auto get_backend_info() const -> std::string override;

    // Rule-based specific mock configuration

    /**
     * @brief Set number of simulated rules
     */
    void set_rule_count(std::uint32_t count);

    /**
     * @brief Set number of simulated facts
     */
    void set_fact_count(std::uint32_t count);

    /**
     * @brief Enable conflict resolution simulation
     */
    void enable_conflict_resolution(bool enable);

    /**
     * @brief Set maximum inference depth
     */
    void set_max_inference_depth(std::uint32_t depth);

    /**
     * @brief Create default rule-based configuration
     */
    static auto create_rule_based_config() -> MockEngineConfig;

  private:
    // Rule-based specific state
    std::uint32_t rule_count_{10};
    std::uint32_t fact_count_{100};
    bool conflict_resolution_enabled_{true};
    std::uint32_t max_inference_depth_{10};

    // Symbolic reasoning simulation
    auto simulate_fact_matching() -> std::uint32_t;
    auto simulate_rule_firing() -> std::uint32_t;
    auto simulate_conflict_resolution() -> bool;
    auto generate_symbolic_output() -> std::vector<std::string>;
};

//=============================================================================
// Factory Functions and Utilities
//=============================================================================

/**
 * @brief Create mock engine for specific backend
 */
auto create_mock_engine(engines::InferenceBackend backend, const MockEngineConfig& config = {})
    -> std::unique_ptr<InferenceEngine>;

/**
 * @brief Create mock engine with error injection
 */
auto create_mock_engine_with_errors(engines::InferenceBackend backend,
                                    const std::vector<std::string>& error_types,
                                    float error_rate = 0.1f) -> std::unique_ptr<InferenceEngine>;

/**
 * @brief Create performance testing mock
 */
auto create_performance_mock(engines::InferenceBackend backend,
                             float latency_ms,
                             std::uint64_t memory_mb) -> std::unique_ptr<InferenceEngine>;

/**
 * @brief Create realistic mock with production-like behavior
 */
auto create_realistic_mock(engines::InferenceBackend backend,
                           const engines::ModelConfig& model_config)
    -> std::unique_ptr<InferenceEngine>;

//=============================================================================
// Mock Test Utilities
//=============================================================================

/**
 * @brief Verify mock engine behavior consistency
 */
auto verify_mock_consistency(InferenceEngine* engine,
                             const InferenceRequest& request,
                             std::uint32_t iterations = 10) -> Result<bool, std::string>;

/**
 * @brief Test mock error injection functionality
 */
auto test_mock_error_injection(MockEngineBase* mock_engine,
                               const std::string& error_type,
                               std::uint32_t test_iterations = 100)
    -> Result<float, std::string>;  // Returns actual error rate

/**
 * @brief Benchmark mock performance characteristics
 */
auto benchmark_mock_performance(InferenceEngine* engine,
                                const InferenceRequest& request,
                                std::uint32_t iterations = 100)
    -> Result<std::unordered_map<std::string, float>, std::string>;

/**
 * @brief Convert ExecutionProvider to string
 */
std::string to_string(MockONNXRuntimeEngine::ExecutionProvider provider);

/**
 * @brief Generate mock configuration for specific test scenario
 */
auto generate_mock_config_for_scenario(const std::string& scenario_name) -> MockEngineConfig;

}  // namespace inference_lab::integration::mocks
