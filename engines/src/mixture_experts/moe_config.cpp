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

#include "moe_config.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "moe_engine.hpp"  // For MoEError definition

#ifdef __linux__
    #include <sys/sysinfo.h>
#elif defined(__APPLE__)
    #include <mach/mach.h>
    #include <sys/sysctl.h>
#elif defined(_WIN32)
    #include <windows.h>
#endif

namespace engines::mixture_experts {

// MoESystemConfig implementation

auto MoESystemConfig::validate() const -> bool {
    auto result = MoEConfigValidator::validate_configuration(*this);
    return result.is_configuration_valid();
}

auto MoESystemConfig::to_string() const -> std::string {
    std::ostringstream oss;

    oss << "MoE System Configuration (" << system_name << " v" << version << ")\n";
    oss << "=====================================\n";

    oss << "\nCore System:\n";
    oss << "  Debug Logging: " << (enable_debug_logging ? "Enabled" : "Disabled") << "\n";
    oss << "  Performance Monitoring: " << (enable_performance_monitoring ? "Enabled" : "Disabled")
        << "\n";

    oss << "\nExpert Configuration:\n";
    oss << "  Number of Experts: " << num_experts << "\n";
    oss << "  Expert Capacity: " << expert_capacity << "\n";
    oss << "  Parameters per Expert: " << parameters_per_expert << "\n";
    oss << "  Input Dimension: " << input_dimension << "\n";
    oss << "  Output Dimension: " << output_dimension << "\n";

    oss << "\nRouting Configuration:\n";
    oss << "  Hidden Dimension: " << routing_hidden_dimension << "\n";
    oss << "  Learning Rate: " << routing_learning_rate << "\n";
    oss << "  Entropy Regularization: " << entropy_regularization << "\n";
    oss << "  Top-k Experts: " << top_k_experts << "\n";
    oss << "  Gradient Computation: " << (enable_gradient_computation ? "Enabled" : "Disabled")
        << "\n";

    oss << "\nLoad Balancing:\n";
    oss << "  Load Balance Weight: " << load_balance_weight << "\n";
    oss << "  Monitoring Window: " << monitoring_window_ms << "ms\n";
    oss << "  Max Queue Size: " << max_queue_size_per_expert << "\n";
    oss << "  Overload Threshold: " << overload_threshold << "\n";
    oss << "  Adaptive Routing: " << (enable_adaptive_routing ? "Enabled" : "Disabled") << "\n";

    oss << "\nSparse Activation:\n";
    oss << "  Sparsity Threshold: " << sparsity_threshold << "\n";
    oss << "  SIMD Optimization: " << (enable_simd_optimization ? "Enabled" : "Disabled") << "\n";
    oss << "  Vector Alignment: " << vector_alignment << " bytes\n";
    oss << "  Block Size: " << block_size << "\n";
    oss << "  Target Sparsity: " << (target_sparsity_ratio * 100.0f) << "%\n";

    oss << "\nMemory Management:\n";
    oss << "  Memory Pool Size: " << memory_pool_size_mb << "MB\n";
    oss << "  Parameter Compression: " << (enable_parameter_compression ? "Enabled" : "Disabled")
        << "\n";
    oss << "  Compression Ratio: " << compression_ratio << "\n";
    oss << "  Cache Size: " << cache_size_experts << " experts\n";

    oss << "\nPerformance Targets:\n";
    oss << "  Max Concurrent Requests: " << max_concurrent_requests << "\n";
    oss << "  Target P95 Latency: " << target_latency_p95_ms << "ms\n";
    oss << "  Target P50 Latency: " << target_latency_p50_ms << "ms\n";
    oss << "  Throughput Multiplier: " << target_throughput_multiplier << "x\n";
    oss << "  Efficiency Multiplier: " << target_efficiency_multiplier << "x\n";

    return oss.str();
}

auto MoESystemConfig::create_development_config() -> MoESystemConfig {
    MoESystemConfig config;

    config.system_name = "moe_development";
    config.enable_debug_logging = true;
    config.enable_performance_monitoring = true;

    // Smaller scale for development
    config.num_experts = 4;
    config.expert_capacity = 2;
    config.parameters_per_expert = 512;
    config.input_dimension = 128;
    config.output_dimension = 128;

    // Conservative routing parameters
    config.routing_hidden_dimension = 64;
    config.routing_learning_rate = 0.01f;
    config.entropy_regularization = 0.05f;
    config.top_k_experts = 2;

    // Relaxed load balancing
    config.load_balance_weight = 0.2f;
    config.max_queue_size_per_expert = 25;
    config.overload_threshold = 0.9f;

    // Conservative memory usage
    config.memory_pool_size_mb = 100;
    config.enable_parameter_compression = true;
    config.compression_ratio = 0.6f;
    config.cache_size_experts = 2;

    // Relaxed performance targets
    config.max_concurrent_requests = 20;
    config.target_latency_p95_ms = 200.0f;
    config.target_latency_p50_ms = 100.0f;
    config.target_throughput_multiplier = 2.0f;
    config.target_efficiency_multiplier = 10.0f;

    return config;
}

auto MoESystemConfig::create_production_config() -> MoESystemConfig {
    MoESystemConfig config;

    config.system_name = "moe_production";
    config.enable_debug_logging = false;
    config.enable_performance_monitoring = true;

    // Full scale production
    config.num_experts = 16;
    config.expert_capacity = 3;
    config.parameters_per_expert = 2048;
    config.input_dimension = 512;
    config.output_dimension = 512;

    // Optimized routing parameters
    config.routing_hidden_dimension = 256;
    config.routing_learning_rate = 0.001f;
    config.entropy_regularization = 0.1f;
    config.top_k_experts = 3;

    // Aggressive load balancing
    config.load_balance_weight = 0.05f;
    config.max_queue_size_per_expert = 100;
    config.overload_threshold = 0.75f;

    // Ample memory allocation
    config.memory_pool_size_mb = 1000;
    config.enable_parameter_compression = true;
    config.compression_ratio = 0.8f;
    config.cache_size_experts = 8;

    // Production performance targets
    config.max_concurrent_requests = 200;
    config.target_latency_p95_ms = MoEConstants::TARGET_P95_LATENCY_MS;
    config.target_latency_p50_ms = MoEConstants::TARGET_P50_LATENCY_MS;
    config.target_throughput_multiplier = 4.0f;
    config.target_efficiency_multiplier = MoEConstants::TARGET_EFFICIENCY_MIN;

    // Strict quality requirements
    config.minimum_accuracy_threshold = MoEConstants::TARGET_ACCURACY_PRESERVATION;
    config.maximum_utilization_variance = MoEConstants::TARGET_UTILIZATION_VARIANCE_MAX;

    return config;
}

auto MoESystemConfig::create_lightweight_config() -> MoESystemConfig {
    MoESystemConfig config;

    config.system_name = "moe_lightweight";
    config.enable_debug_logging = false;
    config.enable_performance_monitoring = false;

    // Minimal expert configuration
    config.num_experts = 4;
    config.expert_capacity = 1;
    config.parameters_per_expert = 256;
    config.input_dimension = 64;
    config.output_dimension = 64;

    // Minimal routing network
    config.routing_hidden_dimension = 32;
    config.routing_learning_rate = 0.005f;
    config.entropy_regularization = 0.01f;
    config.top_k_experts = 1;
    config.enable_gradient_computation = false;  // Disable to save computation

    // Minimal load balancing
    config.load_balance_weight = 0.05f;
    config.max_queue_size_per_expert = 10;
    config.enable_adaptive_routing = false;

    // Aggressive memory optimization
    config.memory_pool_size_mb = 50;
    config.enable_parameter_compression = true;
    config.compression_ratio = 0.5f;
    config.cache_size_experts = 1;
    config.enable_parameter_streaming = true;

    // High sparsity for efficiency
    config.target_sparsity_ratio = 0.9f;
    config.enable_dynamic_sparsity = true;

    // Minimal concurrent load
    config.max_concurrent_requests = 10;

    return config;
}

auto MoESystemConfig::create_performance_config() -> MoESystemConfig {
    MoESystemConfig config;

    config.system_name = "moe_performance";
    config.enable_debug_logging = false;
    config.enable_performance_monitoring = true;

    // Large scale for maximum throughput
    config.num_experts = 32;
    config.expert_capacity = 4;
    config.parameters_per_expert = 4096;
    config.input_dimension = 1024;
    config.output_dimension = 1024;

    // Aggressive routing optimization
    config.routing_hidden_dimension = 512;
    config.routing_learning_rate = 0.0005f;
    config.entropy_regularization = 0.2f;
    config.top_k_experts = 4;

    // Aggressive load balancing
    config.load_balance_weight = 0.01f;
    config.max_queue_size_per_expert = 200;
    config.overload_threshold = 0.6f;
    config.enable_adaptive_routing = true;

    // Maximum memory allocation
    config.memory_pool_size_mb = 2000;
    config.enable_parameter_compression = false;  // Disable compression for speed
    config.cache_size_experts = 16;

    // Maximum SIMD optimization
    config.enable_simd_optimization = true;
    config.vector_alignment = 64;  // 512-bit AVX-512 alignment
    config.block_size = 128;

    // Aggressive performance targets
    config.max_concurrent_requests = 500;
    config.target_latency_p95_ms = 100.0f;
    config.target_latency_p50_ms = 50.0f;
    config.target_throughput_multiplier = MoEConstants::TARGET_THROUGHPUT_MAX;
    config.target_efficiency_multiplier = MoEConstants::TARGET_EFFICIENCY_MAX;

    return config;
}

// ConfigValidationResult implementation

auto ConfigValidationResult::get_validation_report() const -> std::string {
    std::ostringstream oss;

    oss << "Configuration Validation Report\n";
    oss << "==============================\n";
    oss << "Status: " << (is_configuration_valid() ? "VALID" : "INVALID") << "\n\n";

    if (!errors.empty()) {
        oss << "ERRORS (" << errors.size() << "):\n";
        for (const auto& error : errors) {
            oss << "  - " << error << "\n";
        }
        oss << "\n";
    }

    if (!warnings.empty()) {
        oss << "WARNINGS (" << warnings.size() << "):\n";
        for (const auto& warning : warnings) {
            oss << "  - " << warning << "\n";
        }
        oss << "\n";
    }

    if (!recommendations.empty()) {
        oss << "RECOMMENDATIONS (" << recommendations.size() << "):\n";
        for (const auto& recommendation : recommendations) {
            oss << "  - " << recommendation << "\n";
        }
        oss << "\n";
    }

    return oss.str();
}

// MoEConfigValidator implementation

auto MoEConfigValidator::validate_configuration(const MoESystemConfig& config)
    -> ConfigValidationResult {
    ConfigValidationResult result;

    validate_expert_configuration(config, result);
    validate_routing_configuration(config, result);
    validate_performance_configuration(config, result);
    validate_memory_configuration(config, result);

    if (!result.errors.empty()) {
        result.is_valid = false;
    }

    return result;
}

auto MoEConfigValidator::validate_expert_configuration(const MoESystemConfig& config,
                                                       ConfigValidationResult& result) -> void {
    if (config.num_experts == 0) {
        result.errors.push_back("Number of experts must be greater than 0");
    } else if (config.num_experts > 1000) {
        result.warnings.push_back("Very large number of experts (" +
                                  std::to_string(config.num_experts) + ") may impact performance");
    }

    if (config.expert_capacity == 0) {
        result.errors.push_back("Expert capacity must be greater than 0");
    } else if (config.expert_capacity > config.num_experts) {
        result.errors.push_back("Expert capacity cannot exceed number of experts");
    }

    if (config.parameters_per_expert == 0) {
        result.errors.push_back("Parameters per expert must be greater than 0");
    } else if (config.parameters_per_expert > 100000) {
        result.warnings.push_back(
            "Very large parameter count per expert may require significant memory");
    }

    if (config.input_dimension == 0 || config.output_dimension == 0) {
        result.errors.push_back("Input and output dimensions must be greater than 0");
    }

    // Check dimension alignment for SIMD
    if (config.enable_simd_optimization) {
        if (config.input_dimension % 8 != 0) {
            result.recommendations.push_back(
                "Input dimension should be aligned to 8 for optimal SIMD performance");
        }
        if (config.output_dimension % 8 != 0) {
            result.recommendations.push_back(
                "Output dimension should be aligned to 8 for optimal SIMD performance");
        }
    }
}

auto MoEConfigValidator::validate_routing_configuration(const MoESystemConfig& config,
                                                        ConfigValidationResult& result) -> void {
    if (config.routing_hidden_dimension == 0) {
        result.errors.push_back("Routing hidden dimension must be greater than 0");
    } else if (config.routing_hidden_dimension < config.input_dimension / 4) {
        result.warnings.push_back("Very small routing hidden dimension may limit routing capacity");
    }

    if (config.routing_learning_rate <= 0.0f || config.routing_learning_rate > 1.0f) {
        result.errors.push_back("Routing learning rate must be between 0 and 1");
    } else if (config.routing_learning_rate > 0.1f) {
        result.warnings.push_back("High learning rate may cause routing instability");
    }

    if (config.entropy_regularization < 0.0f || config.entropy_regularization > 1.0f) {
        result.errors.push_back("Entropy regularization must be between 0 and 1");
    }

    if (config.top_k_experts == 0) {
        result.errors.push_back("Top-k experts must be greater than 0");
    } else if (config.top_k_experts > config.num_experts) {
        result.errors.push_back("Top-k experts cannot exceed number of experts");
    } else if (config.top_k_experts != config.expert_capacity) {
        result.warnings.push_back("Top-k experts should typically equal expert capacity");
    }
}

auto MoEConfigValidator::validate_performance_configuration(const MoESystemConfig& config,
                                                            ConfigValidationResult& result)
    -> void {
    if (config.max_concurrent_requests == 0) {
        result.errors.push_back("Maximum concurrent requests must be greater than 0");
    } else if (config.max_concurrent_requests > 10000) {
        result.warnings.push_back("Very high concurrent request limit may exceed system capacity");
    }

    if (config.target_latency_p95_ms <= 0.0f) {
        result.errors.push_back("Target P95 latency must be positive");
    } else if (config.target_latency_p95_ms < config.target_latency_p50_ms) {
        result.errors.push_back("P95 latency must be greater than or equal to P50 latency");
    }

    if (config.target_latency_p50_ms <= 0.0f) {
        result.errors.push_back("Target P50 latency must be positive");
    }

    if (config.target_throughput_multiplier < 1.0f) {
        result.warnings.push_back(
            "Throughput multiplier less than 1.0 indicates performance regression");
    } else if (config.target_throughput_multiplier > MoEConstants::TARGET_THROUGHPUT_MAX) {
        result.warnings.push_back("Very high throughput multiplier may be unrealistic");
    }

    if (config.target_efficiency_multiplier < 1.0f) {
        result.warnings.push_back(
            "Efficiency multiplier less than 1.0 indicates efficiency regression");
    } else if (config.target_efficiency_multiplier > MoEConstants::TARGET_EFFICIENCY_MAX) {
        result.warnings.push_back("Very high efficiency multiplier may be unrealistic");
    }

    // Validate against roadmap targets
    if (config.target_latency_p95_ms > MoEConstants::TARGET_P95_LATENCY_MS * 2.0f) {
        result.warnings.push_back("P95 latency target significantly exceeds roadmap goals");
    }

    if (config.target_efficiency_multiplier < MoEConstants::TARGET_EFFICIENCY_MIN) {
        result.recommendations.push_back(
            "Consider increasing efficiency targets to meet roadmap goals");
    }
}

auto MoEConfigValidator::validate_memory_configuration(const MoESystemConfig& config,
                                                       ConfigValidationResult& result) -> void {
    if (config.memory_pool_size_mb == 0) {
        result.errors.push_back("Memory pool size must be greater than 0");
    }

    if (config.compression_ratio <= 0.0f || config.compression_ratio > 1.0f) {
        result.errors.push_back("Compression ratio must be between 0 and 1");
    } else if (config.compression_ratio < 0.3f) {
        result.warnings.push_back("Very aggressive compression may impact accuracy");
    }

    if (config.cache_size_experts > config.num_experts) {
        result.warnings.push_back("Cache size exceeds number of experts");
    } else if (config.cache_size_experts == 0) {
        result.warnings.push_back("Zero cache size may impact performance");
    }

    if (config.sparsity_threshold <= 0.0f || config.sparsity_threshold > 1.0f) {
        result.errors.push_back("Sparsity threshold must be between 0 and 1");
    }

    if (config.target_sparsity_ratio < 0.0f || config.target_sparsity_ratio > 1.0f) {
        result.errors.push_back("Target sparsity ratio must be between 0 and 1");
    } else if (config.target_sparsity_ratio > 0.95f) {
        result.warnings.push_back("Very high sparsity may impact accuracy");
    }

    // Estimate memory requirements
    std::size_t estimated_memory = estimate_memory_requirements(config);
    if (estimated_memory > config.memory_pool_size_mb * 1024 * 1024) {
        result.warnings.push_back("Estimated memory usage exceeds configured pool size");
    }

    // Check SIMD alignment
    if (config.enable_simd_optimization) {
        if (config.vector_alignment < 16) {
            result.warnings.push_back("Vector alignment should be at least 16 bytes for SIMD");
        } else if (config.vector_alignment > 64) {
            result.warnings.push_back("Very large vector alignment may waste memory");
        }

        if (config.block_size % config.vector_alignment != 0) {
            result.recommendations.push_back(
                "Block size should be aligned to vector alignment boundary");
        }
    }
}

auto MoEConfigValidator::validate_against_system_capabilities(const MoESystemConfig& config)
    -> ConfigValidationResult {
    auto result = validate_configuration(config);

    // Check system memory
    std::size_t estimated_memory = estimate_memory_requirements(config);
    if (!check_system_memory_availability(estimated_memory / (1024 * 1024))) {
        result.warnings.push_back("System may not have sufficient memory for this configuration");
    }

    // Check SIMD capabilities
    if (config.enable_simd_optimization && !check_simd_capability_availability()) {
        result.warnings.push_back("SIMD optimization requested but not available on this system");
        result.recommendations.push_back(
            "Consider disabling SIMD optimization or running on compatible hardware");
    }

    return result;
}

auto MoEConfigValidator::estimate_memory_requirements(const MoESystemConfig& config)
    -> std::size_t {
    std::size_t total_memory = 0;

    // Expert parameters memory
    std::size_t expert_param_memory =
        config.num_experts * config.parameters_per_expert * sizeof(float);
    if (config.enable_parameter_compression) {
        expert_param_memory =
            static_cast<std::size_t>(expert_param_memory * config.compression_ratio);
    }
    total_memory += expert_param_memory;

    // Routing network memory
    std::size_t routing_memory = (config.routing_hidden_dimension * config.input_dimension +
                                  config.num_experts * config.routing_hidden_dimension) *
                                 sizeof(float);
    total_memory += routing_memory;

    // Load balancing structures
    std::size_t load_balance_memory =
        config.num_experts * (sizeof(float) * 10 + sizeof(std::size_t) * 5);
    total_memory += load_balance_memory;

    // Cache memory
    std::size_t cache_memory =
        config.cache_size_experts * config.parameters_per_expert * sizeof(float);
    total_memory += cache_memory;

    // Buffer and alignment overhead (20% estimate)
    total_memory = static_cast<std::size_t>(total_memory * 1.2f);

    return total_memory;
}

auto MoEConfigValidator::check_system_memory_availability(std::size_t required_memory_mb) -> bool {
    std::size_t available_memory_mb = 0;

#ifdef __linux__
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        available_memory_mb = (info.totalram * info.mem_unit) / (1024 * 1024);
    }
#elif defined(__APPLE__)
    int64_t memory_size;
    size_t size = sizeof(memory_size);
    if (sysctlbyname("hw.memsize", &memory_size, &size, nullptr, 0) == 0) {
        available_memory_mb = memory_size / (1024 * 1024);
    }
#elif defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        available_memory_mb = status.ullTotalPhys / (1024 * 1024);
    }
#endif

    if (available_memory_mb == 0) {
        // Cannot determine system memory, assume sufficient
        return true;
    }

    // Require at least 50% of system memory to be available
    return required_memory_mb < (available_memory_mb / 2);
}

auto MoEConfigValidator::check_simd_capability_availability() -> bool {
    // Simplified SIMD detection - in practice would use CPUID or similar
#if defined(__x86_64__) || defined(_M_X64)
    return true;  // Assume modern x86_64 has at least SSE4.2
#elif defined(__aarch64__)
    return true;  // Assume ARM64 has NEON
#else
    return false;  // Conservative assumption for other architectures
#endif
}

auto MoEConfigValidator::estimate_performance_characteristics(const MoESystemConfig& config)
    -> std::pair<float, float> {
    // Simplified performance estimation
    float base_latency = 10.0f;  // Base latency in ms

    // Factor in expert complexity
    float expert_factor = std::log2(static_cast<float>(config.num_experts)) / 4.0f;
    float parameter_factor = std::sqrt(static_cast<float>(config.parameters_per_expert)) / 100.0f;

    float estimated_latency = base_latency + expert_factor + parameter_factor;

    // Apply optimizations
    if (config.enable_simd_optimization) {
        estimated_latency *= 0.7f;  // 30% improvement from SIMD
    }

    if (config.enable_parameter_compression) {
        estimated_latency *= 0.9f;  // 10% improvement from compression
    }

    float sparsity_factor =
        1.0f - (config.target_sparsity_ratio * 0.8f);  // Up to 80% improvement from sparsity
    estimated_latency *= sparsity_factor;

    // Estimate throughput (requests per second)
    float estimated_throughput = 1000.0f / estimated_latency;  // Convert latency to throughput
    estimated_throughput *= config.num_experts * 0.1f;         // Scale with parallelism

    return std::make_pair(estimated_latency, estimated_throughput);
}

auto MoEConfigValidator::recommend_optimizations(const MoESystemConfig& config) -> MoESystemConfig {
    MoESystemConfig optimized = config;

    // Optimize dimensions for SIMD
    if (optimized.enable_simd_optimization) {
        optimized.input_dimension = ((optimized.input_dimension + 7) / 8) * 8;  // Round up to 8
        optimized.output_dimension = ((optimized.output_dimension + 7) / 8) * 8;
        optimized.routing_hidden_dimension = ((optimized.routing_hidden_dimension + 7) / 8) * 8;
    }

    // Optimize cache size
    if (optimized.cache_size_experts <
        std::min(optimized.num_experts / 4, static_cast<std::size_t>(8))) {
        optimized.cache_size_experts =
            std::min(optimized.num_experts / 4, static_cast<std::size_t>(8));
    }

    // Optimize sparsity for target performance
    if (optimized.target_efficiency_multiplier > 15.0f && optimized.target_sparsity_ratio < 0.6f) {
        optimized.target_sparsity_ratio = std::min(0.8f, optimized.target_sparsity_ratio + 0.2f);
    }

    // Optimize memory pool size based on estimated requirements
    std::size_t estimated_memory = estimate_memory_requirements(optimized);
    std::size_t recommended_pool_size = (estimated_memory / (1024 * 1024)) * 2;  // 2x safety factor
    if (recommended_pool_size > optimized.memory_pool_size_mb) {
        optimized.memory_pool_size_mb = recommended_pool_size;
    }

    // Optimize load balancing parameters
    if (optimized.num_experts > 8) {
        optimized.load_balance_weight = std::max(0.05f, optimized.load_balance_weight * 0.8f);
    }

    return optimized;
}

}  // namespace engines::mixture_experts
