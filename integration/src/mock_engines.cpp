// MIT License
// Copyright (c) 2025 dbjwhs
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * @file mock_engines.cpp
 * @brief Implementation of mock inference engines for testing
 *
 * This file provides the implementation of sophisticated mock engines that
 * simulate real inference backend behavior without requiring actual hardware.
 * The mocks are designed to be realistic while providing controllable behavior
 * for comprehensive testing scenarios.
 */

#include "mock_engines.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <thread>

#include "../../common/src/logging.hpp"

namespace inference_lab::integration::mocks {

using common::Err;
using common::Ok;
using engines::InferenceError;

//=============================================================================
// Utility Functions Implementation
//=============================================================================

std::string to_string(MockONNXRuntimeEngine::ExecutionProvider provider) {
    switch (provider) {
        case MockONNXRuntimeEngine::ExecutionProvider::CPU:
            return "CPU";
        case MockONNXRuntimeEngine::ExecutionProvider::CUDA:
            return "CUDA";
        case MockONNXRuntimeEngine::ExecutionProvider::TENSORRT:
            return "TensorRT";
        case MockONNXRuntimeEngine::ExecutionProvider::DIRECTML:
            return "DirectML";
        case MockONNXRuntimeEngine::ExecutionProvider::OPENVINO:
            return "OpenVINO";
        default:
            return "Unknown";
    }
}

//=============================================================================
// MockEngineBase Implementation
//=============================================================================

MockEngineBase::MockEngineBase(MockEngineConfig config)
    : config_(std::move(config)), rng_(config_.random_seed), dist_(0.0f, 1.0f) {
    if (config_.enable_logging) {
        LOG_INFO_PRINT("Mock engine '{}' initialized", config_.engine_name);
    }

    if (config_.simulate_model_loading) {
        simulate_model_loading();
    }

    is_initialized_.store(true);
}

auto MockEngineBase::get_backend_info() const -> std::string {
    std::stringstream info;
    info << "Mock Engine: " << config_.engine_name
         << ", Hardware Simulation: " << (config_.simulate_hardware ? "Enabled" : "Disabled")
         << ", Inference Count: " << inference_count_.load();
    return info.str();
}

auto MockEngineBase::is_ready() const -> bool {
    return is_initialized_.load();
}

auto MockEngineBase::get_performance_stats() const -> std::string {
    std::stringstream stats;
    auto count = inference_count_.load();
    auto errors = error_count_.load();
    auto total_latency = total_latency_ms_.load();

    stats << "Performance Stats for " << config_.engine_name << ":\n"
          << "  Total Inferences: " << count << "\n"
          << "  Total Errors: " << errors << "\n"
          << "  Success Rate: " << (count > 0 ? (100.0f * (count - errors) / count) : 0.0f)
          << "%\n";

    if (count > 0) {
        stats << "  Average Latency: " << (total_latency / count) << " ms\n";
    }

    return stats.str();
}

void MockEngineBase::inject_error_condition(const std::string& error_type, float probability) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    config_.error_injection.error_rates[error_type] = probability;

    if (config_.enable_logging) {
        LOG_DEBUG_PRINT(
            "Injected error condition '{}' with probability {}", error_type, probability);
    }
}

void MockEngineBase::clear_error_injections() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    config_.error_injection.error_rates.clear();

    if (config_.enable_logging) {
        LOG_DEBUG_PRINT("Cleared all error injections");
    }
}

void MockEngineBase::set_latency_variation(float variation) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    config_.performance.latency_variation = std::clamp(variation, 0.0f, 1.0f);
}

void MockEngineBase::set_memory_usage(std::uint64_t memory_mb) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    config_.performance.base_memory_usage_mb = memory_mb;
}

void MockEngineBase::set_realistic_output_generation(bool enable) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    config_.generate_realistic_outputs = enable;
}

void MockEngineBase::reset_statistics() {
    inference_count_.store(0);
    error_count_.store(0);
    total_latency_ms_.store(0);
    memory_allocated_mb_.store(0);
}

auto MockEngineBase::get_mock_statistics() const -> std::unordered_map<std::string, std::uint64_t> {
    return {{"inference_count", inference_count_.load()},
            {"error_count", error_count_.load()},
            {"total_latency_ms", total_latency_ms_.load()},
            {"memory_allocated_mb", memory_allocated_mb_.load()}};
}

void MockEngineBase::simulate_model_loading() {
    if (config_.model_loading_time_ms > 0) {
        auto delay = std::chrono::milliseconds(static_cast<int>(config_.model_loading_time_ms));
        std::this_thread::sleep_for(delay);
    }
}

auto MockEngineBase::should_inject_error() const -> std::optional<InferenceError> {
    std::lock_guard<std::mutex> lock(state_mutex_);

    // Check global error rate
    if (config_.error_injection.global_error_rate > 0 &&
        dist_(rng_) < config_.error_injection.global_error_rate) {
        return InferenceError::UNKNOWN_ERROR;
    }

    // Check specific error conditions
    for (const auto& [error_type, probability] : config_.error_injection.error_rates) {
        if (dist_(rng_) < probability) {
            if (error_type == "GPU_MEMORY_EXHAUSTED") {
                return InferenceError::GPU_MEMORY_EXHAUSTED;
            } else if (error_type == "MODEL_LOAD_FAILED") {
                return InferenceError::MODEL_LOAD_FAILED;
            } else if (error_type == "INFERENCE_EXECUTION_FAILED") {
                return InferenceError::INFERENCE_EXECUTION_FAILED;
            }
        }
    }

    return std::nullopt;
}

void MockEngineBase::simulate_inference_latency() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    auto base_latency = config_.performance.base_latency_ms;
    auto variation = config_.performance.latency_variation;

    // Add random variation
    float latency_multiplier = 1.0f + (dist_(rng_) - 0.5f) * 2.0f * variation;
    auto actual_latency = base_latency * latency_multiplier;

    if (actual_latency > 0) {
        auto delay = std::chrono::microseconds(static_cast<int>(actual_latency * 1000));
        std::this_thread::sleep_for(delay);
    }
}

auto MockEngineBase::generate_output_tensors(const std::vector<TensorSpec>& output_specs)
    -> std::vector<TensorOutput> {
    std::vector<TensorOutput> outputs;
    outputs.reserve(output_specs.size());

    for (const auto& spec : output_specs) {
        TensorOutput output;
        output.name = spec.name;

        // Generate realistic tensor data based on spec
        if (spec.dtype == DataType::FLOAT32) {
            auto tensor = tensor_factory::random_uniform<float>(spec.shape, -1.0f, 1.0f);
            output.tensor = std::move(tensor);
        } else if (spec.dtype == DataType::INT32) {
            auto tensor = MLTensor<std::int32_t>(spec.shape);
            // Fill with random integers
            for (std::size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = static_cast<std::int32_t>(dist_(rng_) * 100);
            }
            output.tensor = std::move(tensor);
        }

        output.confidence = 0.8f + dist_(rng_) * 0.2f;  // 0.8 to 1.0
        outputs.push_back(std::move(output));
    }

    return outputs;
}

template <typename T>
auto MockEngineBase::generate_realistic_tensor_data(const Shape& shape,
                                                    const std::string& tensor_name)
    -> std::vector<T> {
    std::size_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }

    std::vector<T> data(total_size);

    if (config_.generate_realistic_outputs) {
        // Generate realistic data based on tensor name
        if (tensor_name.find("logits") != std::string::npos ||
            tensor_name.find("output") != std::string::npos) {
            // Logits - normal distribution around 0
            std::normal_distribution<float> normal_dist(0.0f, 1.0f);
            for (auto& value : data) {
                value = static_cast<T>(normal_dist(rng_));
            }
        } else {
            // Default uniform distribution
            for (auto& value : data) {
                value = static_cast<T>(dist_(rng_));
            }
        }
    } else {
        // Simple random data
        for (auto& value : data) {
            value = static_cast<T>(dist_(rng_));
        }
    }

    return data;
}

//=============================================================================
// MockTensorRTEngine Implementation
//=============================================================================

MockTensorRTEngine::MockTensorRTEngine(MockEngineConfig config) : MockEngineBase(config) {
    if (config.simulate_gpu_memory) {
        gpu_memory_limit_mb_ = config.max_gpu_memory_mb;
    }
}

auto MockTensorRTEngine::run_inference(const engines::InferenceRequest& request)
    -> Result<engines::InferenceResponse, engines::InferenceError> {
    auto start_time = std::chrono::steady_clock::now();
    inference_count_.fetch_add(1);

    // Check for error injection
    auto error = should_inject_error();
    if (error.has_value()) {
        error_count_.fetch_add(1);
        return Err(*error);
    }

    // Simulate GPU memory check
    if (config_.simulate_gpu_memory) {
        auto required_memory = request.input_tensors.size() * 100;  // Simplified calculation
        if (!simulate_gpu_memory_check(required_memory)) {
            error_count_.fetch_add(1);
            return Err(InferenceError::GPU_MEMORY_EXHAUSTED);
        }
    }

    // Simulate inference latency
    simulate_inference_latency();

    // Generate response
    engines::InferenceResponse response;

    // Generate mock output tensors
    response.output_tensors.resize(1);
    response.output_tensors[0].resize(1000, 0.5f);  // Mock classification output
    response.output_names = {"output"};

    auto end_time = std::chrono::steady_clock::now();
    response.inference_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Update statistics
    total_latency_ms_.fetch_add(static_cast<uint64_t>(response.inference_time_ms));

    return Ok(std::move(response));
}

auto MockTensorRTEngine::get_backend_info() const -> std::string {
    std::stringstream info;
    info << "MockTensorRTEngine: "
         << "GPU Memory: " << gpu_memory_used_mb_.load() << "/" << gpu_memory_limit_mb_ << " MB, "
         << "Precision: " << (current_precision_ == Precision::FP32 ? "FP32" : "FP16") << ", "
         << MockEngineBase::get_backend_info();
    return info.str();
}

auto MockTensorRTEngine::create_tensorrt_config() -> MockEngineConfig {
    MockEngineConfig config;
    config.engine_name = "MockTensorRTEngine";
    config.simulate_gpu_memory = true;
    config.max_gpu_memory_mb = 8192;
    config.performance.base_latency_ms = 5.0f;
    config.performance.latency_variation = 0.1f;
    config.model_loading_time_ms = 200.0f;
    return config;
}

auto MockTensorRTEngine::simulate_gpu_memory_check(std::uint64_t required_mb) -> bool {
    auto current_usage = gpu_memory_used_mb_.load();
    return (current_usage + required_mb) <= gpu_memory_limit_mb_;
}

//=============================================================================
// MockONNXRuntimeEngine Implementation
//=============================================================================

MockONNXRuntimeEngine::MockONNXRuntimeEngine(MockEngineConfig config, ExecutionProvider provider)
    : MockEngineBase(config), execution_provider_(provider) {}

auto MockONNXRuntimeEngine::run_inference(const engines::InferenceRequest& request)
    -> Result<engines::InferenceResponse, engines::InferenceError> {
    auto start_time = std::chrono::steady_clock::now();
    inference_count_.fetch_add(1);

    // Check for error injection
    auto error = should_inject_error();
    if (error.has_value()) {
        error_count_.fetch_add(1);
        return Err(*error);
    }

    // Simulate provider-specific behavior
    auto provider_result = simulate_provider_initialization();
    if (provider_result.is_err()) {
        error_count_.fetch_add(1);
        return Err(provider_result.unwrap_err());
    }

    // Simulate inference latency (varies by provider)
    simulate_inference_latency();

    // Generate response
    engines::InferenceResponse response;

    // Generate mock output tensors
    response.output_tensors.resize(1);
    response.output_tensors[0].resize(1000, 0.85f);  // Mock classification output
    response.output_names = {"output"};

    auto end_time = std::chrono::steady_clock::now();
    response.inference_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Update statistics
    total_latency_ms_.fetch_add(static_cast<uint64_t>(response.inference_time_ms));

    return Ok(std::move(response));
}

auto MockONNXRuntimeEngine::get_backend_info() const -> std::string {
    std::stringstream info;
    info << "MockONNXRuntimeEngine: "
         << "Provider: " << to_string(execution_provider_) << ", "
         << MockEngineBase::get_backend_info();
    return info.str();
}

auto MockONNXRuntimeEngine::create_onnx_config() -> MockEngineConfig {
    MockEngineConfig config;
    config.engine_name = "MockONNXRuntimeEngine";
    config.simulate_hardware = false;  // CPU-based by default
    config.performance.base_latency_ms = 15.0f;
    config.performance.latency_variation = 0.15f;
    config.model_loading_time_ms = 100.0f;
    return config;
}

auto MockONNXRuntimeEngine::simulate_provider_initialization() -> Result<bool, InferenceError> {
    // Simulate provider-specific initialization
    if (execution_provider_ == ExecutionProvider::CUDA && !config_.simulate_hardware) {
        // Simulate CUDA not available
        return Err(InferenceError::BACKEND_NOT_AVAILABLE);
    }
    return Ok(true);
}

//=============================================================================
// MockRuleBasedEngine Implementation
//=============================================================================

MockRuleBasedEngine::MockRuleBasedEngine(MockEngineConfig config) : MockEngineBase(config) {}

auto MockRuleBasedEngine::run_inference(const engines::InferenceRequest& request)
    -> Result<engines::InferenceResponse, engines::InferenceError> {
    auto start_time = std::chrono::steady_clock::now();
    inference_count_.fetch_add(1);

    // Check for error injection
    auto error = should_inject_error();
    if (error.has_value()) {
        error_count_.fetch_add(1);
        return Err(*error);
    }

    // Simulate rule-based processing
    auto rules_fired = simulate_rule_firing();

    // Very fast processing for rule-based systems
    simulate_inference_latency();

    // Generate symbolic-style response
    engines::InferenceResponse response;

    // Generate mock rule-based output
    response.output_tensors.resize(1);
    response.output_tensors[0].resize(rules_fired, 1.0f);  // Mock derived facts
    response.output_names = {"derived_facts"};

    auto end_time = std::chrono::steady_clock::now();
    response.inference_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Update statistics
    total_latency_ms_.fetch_add(static_cast<uint64_t>(response.inference_time_ms));

    return Ok(std::move(response));
}

auto MockRuleBasedEngine::get_backend_info() const -> std::string {
    std::stringstream info;
    info << "MockRuleBasedEngine: "
         << "Rules: " << rule_count_ << ", "
         << "Facts: " << fact_count_ << ", " << MockEngineBase::get_backend_info();
    return info.str();
}

auto MockRuleBasedEngine::create_rule_based_config() -> MockEngineConfig {
    MockEngineConfig config;
    config.engine_name = "MockRuleBasedEngine";
    config.simulate_hardware = false;
    config.performance.base_latency_ms = 1.0f;
    config.performance.latency_variation = 0.05f;
    config.model_loading_time_ms = 10.0f;
    return config;
}

auto MockRuleBasedEngine::simulate_fact_matching() -> std::uint32_t {
    // Simulate pattern matching process
    return static_cast<std::uint32_t>(dist_(rng_) * fact_count_);
}

auto MockRuleBasedEngine::simulate_rule_firing() -> std::uint32_t {
    // Simulate rule firing process
    return static_cast<std::uint32_t>(dist_(rng_) * rule_count_ * 0.1f);  // 10% of rules fire
}

//=============================================================================
// Factory Functions Implementation
//=============================================================================

auto create_mock_engine(engines::InferenceBackend backend, const MockEngineConfig& config)
    -> std::unique_ptr<InferenceEngine> {
    switch (backend) {
        case engines::InferenceBackend::TENSORRT_GPU:
            return std::make_unique<MockTensorRTEngine>(config);
        case engines::InferenceBackend::ONNX_RUNTIME:
            return std::make_unique<MockONNXRuntimeEngine>(config);
        case engines::InferenceBackend::RULE_BASED:
            return std::make_unique<MockRuleBasedEngine>(config);
        default:
            return nullptr;
    }
}

auto create_mock_engine_with_errors(engines::InferenceBackend backend,
                                    const std::vector<std::string>& error_types,
                                    float error_rate) -> std::unique_ptr<InferenceEngine> {
    MockEngineConfig config;
    for (const auto& error_type : error_types) {
        config.error_injection.error_rates[error_type] = error_rate;
    }

    return create_mock_engine(backend, config);
}

auto create_performance_mock(engines::InferenceBackend backend,
                             float latency_ms,
                             std::uint64_t memory_mb) -> std::unique_ptr<InferenceEngine> {
    MockEngineConfig config;
    config.performance.base_latency_ms = latency_ms;
    config.performance.base_memory_usage_mb = memory_mb;

    return create_mock_engine(backend, config);
}

auto create_realistic_mock(engines::InferenceBackend backend,
                           const engines::ModelConfig& model_config)
    -> std::unique_ptr<InferenceEngine> {
    MockEngineConfig config;
    config.generate_realistic_outputs = true;
    config.simulate_hardware = true;

    // Adjust configuration based on model complexity
    if (model_config.model_path.find("large") != std::string::npos) {
        config.performance.base_latency_ms *= 2.0f;
        config.performance.base_memory_usage_mb *= 2;
    }

    return create_mock_engine(backend, config);
}

auto generate_mock_config_for_scenario(const std::string& scenario_name) -> MockEngineConfig {
    MockEngineConfig config;

    if (scenario_name == "high_performance") {
        config.performance.base_latency_ms = 2.0f;
        config.performance.base_memory_usage_mb = 2048;
        config.simulate_gpu_memory = true;
    } else if (scenario_name == "low_memory") {
        config.performance.base_memory_usage_mb = 64;
        config.simulate_hardware = false;
    } else if (scenario_name == "error_prone") {
        config.error_injection.global_error_rate = 0.1f;
        config.error_injection.intermittent_failures = true;
    } else if (scenario_name == "realistic_production") {
        config.generate_realistic_outputs = true;
        config.performance.simulate_warmup = true;
        config.performance.latency_variation = 0.15f;
    }

    return config;
}

//=============================================================================
// Test Utility Functions Implementation
//=============================================================================

auto verify_mock_consistency(engines::InferenceEngine* engine,
                             const engines::InferenceRequest& request,
                             std::uint32_t iterations) -> Result<bool, std::string> {
    if (!engine || !engine->is_ready()) {
        return Err(std::string("Engine not ready for testing"));
    }

    std::vector<engines::InferenceResponse> responses;
    responses.reserve(iterations);

    // Run multiple inferences to check consistency
    for (std::uint32_t i = 0; i < iterations; ++i) {
        auto result = engine->run_inference(request);
        if (result.is_err()) {
            return Err(std::string("Inference failed on iteration ") + std::to_string(i));
        }
        responses.push_back(result.unwrap());
    }

    // Check basic consistency - all responses should have same structure
    const auto& first_response = responses[0];
    for (std::size_t i = 1; i < responses.size(); ++i) {
        if (responses[i].output_tensors.size() != first_response.output_tensors.size()) {
            return Err(std::string("Inconsistent output tensor count on iteration ") +
                       std::to_string(i));
        }

        if (responses[i].output_names.size() != first_response.output_names.size()) {
            return Err(std::string("Inconsistent output name count on iteration ") +
                       std::to_string(i));
        }

        for (std::size_t j = 0; j < responses[i].output_names.size(); ++j) {
            if (responses[i].output_names[j] != first_response.output_names[j]) {
                return Err(std::string("Inconsistent output name on iteration ") +
                           std::to_string(i));
            }
        }
    }

    return Ok(true);
}

auto test_mock_error_injection(MockEngineBase* mock_engine,
                               const std::string& error_type,
                               std::uint32_t test_iterations) -> Result<float, std::string> {
    if (!mock_engine) {
        return Err(std::string("Null mock engine"));
    }

    // Set error injection rate
    const float expected_error_rate = 0.2f;  // 20%
    mock_engine->inject_error_condition(error_type, expected_error_rate);

    // Create a simple test request
    engines::InferenceRequest test_request;
    test_request.batch_size = 1;
    test_request.input_tensors = {{1.0f, 2.0f, 3.0f}};  // Simple test input
    test_request.input_names = {"input"};

    std::uint32_t error_count = 0;

    // Run test iterations
    for (std::uint32_t i = 0; i < test_iterations; ++i) {
        auto result = mock_engine->run_inference(test_request);
        if (result.is_err()) {
            error_count++;
        }
    }

    float actual_error_rate = static_cast<float>(error_count) / test_iterations;

    // Clear error injection
    mock_engine->clear_error_injections();

    return Ok(actual_error_rate);
}

auto benchmark_mock_performance(engines::InferenceEngine* engine,
                                const engines::InferenceRequest& request,
                                std::uint32_t iterations)
    -> Result<std::unordered_map<std::string, float>, std::string> {
    if (!engine || !engine->is_ready()) {
        return Err(std::string("Engine not ready for benchmarking"));
    }

    std::vector<float> latencies;
    latencies.reserve(iterations);

    auto overall_start = std::chrono::steady_clock::now();

    // Run benchmark iterations
    for (std::uint32_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto result = engine->run_inference(request);
        auto end = std::chrono::steady_clock::now();

        if (result.is_ok()) {
            auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            latencies.push_back(static_cast<float>(latency.count()) / 1000.0f);  // Convert to ms
        }
    }

    auto overall_end = std::chrono::steady_clock::now();
    auto total_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(overall_end - overall_start);

    if (latencies.empty()) {
        return Err(std::string("No successful inferences completed"));
    }

    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());

    float min_latency = latencies.front();
    float max_latency = latencies.back();
    float mean_latency =
        std::accumulate(latencies.begin(), latencies.end(), 0.0f) / latencies.size();
    float throughput = (latencies.size() * 1000.0f) / total_time.count();  // inferences per second

    // Calculate percentiles
    std::size_t p95_idx = static_cast<std::size_t>(0.95f * latencies.size());
    std::size_t p99_idx = static_cast<std::size_t>(0.99f * latencies.size());
    float p95_latency = latencies[std::min(p95_idx, latencies.size() - 1)];
    float p99_latency = latencies[std::min(p99_idx, latencies.size() - 1)];

    return Ok(std::unordered_map<std::string, float>{
        {"min_latency_ms", min_latency},
        {"max_latency_ms", max_latency},
        {"mean_latency_ms", mean_latency},
        {"p95_latency_ms", p95_latency},
        {"p99_latency_ms", p99_latency},
        {"throughput_ips", throughput},
        {"success_rate", static_cast<float>(latencies.size()) / iterations}});
}

}  // namespace inference_lab::integration::mocks
