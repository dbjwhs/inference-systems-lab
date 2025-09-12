// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#pragma once

#include <array>
#include <chrono>
#include <memory>
#include <vector>

#include "../../common/src/logging.hpp"
#include "../../common/src/ml_types.hpp"
#include "../../common/src/result.hpp"
#include "../inference_engine.hpp"

namespace inference_lab {
namespace engines {
namespace mamba_ssm {

using common::Result;
using common::ml::DataType;
using common::ml::FloatTensor;
using common::ml::Shape;

// Core SSM parameters and configuration
struct MambaSSMConfig {
    // Model dimensions
    size_t d_model = 256;  // Model dimension
    size_t d_state = 16;   // State space dimension N
    size_t d_inner = 512;  // Inner projection dimension (typically 2 * d_model)
    size_t d_conv = 4;     // Convolution kernel size

    // Sequence processing
    size_t max_seq_len = 2048;  // Maximum sequence length
    size_t batch_size = 1;      // Batch size

    // Numerical stability
    double dt_rank = 16.0;        // Rank for Δ parameter matrix
    double dt_scale = 1.0;        // Scale factor for step size
    double dt_init_floor = 1e-4;  // Minimum step size

    // Memory optimization
    bool recompute_in_backward = true;  // Recompute intermediates in backward pass
    bool use_simd_kernels = true;       // Enable SIMD optimizations

    // Activation functions
    enum class ActivationType : uint8_t {
        SILU,  // SiLU/Swish activation
        GELU,  // GELU activation
        RELU   // ReLU activation
    };
    ActivationType activation = ActivationType::SILU;
};

// Selective SSM parameters (input-dependent)
struct SelectiveParameters {
    FloatTensor delta;     // Step size parameters [B, L, D]
    FloatTensor B_matrix;  // Input-to-state matrix [B, L, N]
    FloatTensor C_matrix;  // State-to-output matrix [B, L, N]

    SelectiveParameters(size_t batch, size_t seq_len, size_t d_inner, size_t d_state);
};

// State space model state
struct SSMState {
    FloatTensor hidden_state;  // Hidden state [B, D, N]
    FloatTensor conv_state;    // Convolution state [B, D, conv_width]

    // Performance metrics
    size_t sequence_length = 0;
    std::chrono::microseconds last_update_time{0};

    SSMState(size_t batch, size_t d_inner, size_t d_state, size_t conv_width);

    void reset();
    auto clone() const -> std::unique_ptr<SSMState>;
};

// Error types for Mamba SSM
enum class MambaSSMError : uint8_t {
    INVALID_CONFIGURATION,
    SEQUENCE_TOO_LONG,
    DIMENSION_MISMATCH,
    MEMORY_ALLOCATION_FAILED,
    NUMERICAL_INSTABILITY,
    UNKNOWN_ERROR
};

auto to_string(MambaSSMError error) -> std::string;

// Performance metrics
struct MambaSSMMetrics {
    // Execution metrics
    bool converged = false;
    size_t sequence_length = 0;
    std::chrono::microseconds inference_time_ms{0};
    std::chrono::microseconds selective_scan_time_ms{0};

    // Memory metrics
    size_t memory_usage_bytes = 0;
    size_t peak_memory_bytes = 0;

    // Computational metrics
    size_t total_flops = 0;
    double flops_per_second = 0.0;

    // Selective mechanism metrics
    double average_step_size = 0.0;
    size_t selective_updates = 0;

    // Performance characteristics
    double throughput_tokens_per_sec = 0.0;
    double memory_bandwidth_gb_per_sec = 0.0;
};

// Forward declaration for implementation details
namespace detail {
class SSMKernels;
class SelectiveScanImpl;
}  // namespace detail

// Main Mamba State Space Model engine
class MambaSSMEngine : public InferenceEngine {
  public:
    explicit MambaSSMEngine(const MambaSSMConfig& config = MambaSSMConfig{});
    ~MambaSSMEngine() override;

    // Disable copy and move to align with InferenceEngine base class
    MambaSSMEngine(const MambaSSMEngine&) = delete;
    MambaSSMEngine& operator=(const MambaSSMEngine&) = delete;
    MambaSSMEngine(MambaSSMEngine&&) = delete;
    MambaSSMEngine& operator=(MambaSSMEngine&&) = delete;

    // Core SSM operations
    auto run_mamba_ssm(const FloatTensor& input_sequence) -> Result<FloatTensor, MambaSSMError>;

    auto run_streaming(const FloatTensor& input_token) -> Result<FloatTensor, MambaSSMError>;

    // State management
    void reset_state();
    auto get_state() const -> const SSMState&;
    void set_state(std::unique_ptr<SSMState> state);

    // Configuration management
    void update_config(const MambaSSMConfig& new_config);
    auto get_config() const -> const MambaSSMConfig& { return config_; }

    // Performance monitoring
    auto get_metrics() const -> const MambaSSMMetrics& { return metrics_; }
    void reset_metrics();

    // InferenceEngine interface implementation
    auto run_inference(const InferenceRequest& request)
        -> Result<InferenceResponse, InferenceError> override;
    auto get_backend_info() const -> std::string override;
    auto get_performance_stats() const -> std::string override;
    bool is_ready() const override;

  private:
    // Core implementation methods
    auto selective_scan(const SelectiveParameters& params, const FloatTensor& input)
        -> Result<FloatTensor, MambaSSMError>;

    auto compute_selective_params(const FloatTensor& input)
        -> Result<SelectiveParameters, MambaSSMError>;

    auto discretize_continuous_ssm(const SelectiveParameters& params)
        -> Result<std::pair<FloatTensor, FloatTensor>, MambaSSMError>;

    auto apply_convolution(const FloatTensor& input) -> Result<FloatTensor, MambaSSMError>;

    // SIMD-optimized kernels
    void update_state_simd(const FloatTensor& discrete_A,
                           const FloatTensor& discrete_B,
                           const FloatTensor& input);

    void compute_output_simd(const FloatTensor& C_matrix, FloatTensor& output);

    // Memory management
    auto allocate_working_memory(size_t seq_len) -> Result<void, MambaSSMError>;
    void deallocate_working_memory();

    // Configuration and state
    MambaSSMConfig config_;
    std::unique_ptr<SSMState> state_;
    MambaSSMMetrics metrics_;

    // Learned parameters (would be loaded from trained model)
    FloatTensor A_matrix_;           // State transition matrix [D, N]
    FloatTensor input_projection_;   // Input projection weights
    FloatTensor output_projection_;  // Output projection weights
    FloatTensor dt_projection_;      // Step size projection weights
    FloatTensor B_projection_;       // B matrix projection weights
    FloatTensor C_projection_;       // C matrix projection weights
    FloatTensor conv_weights_;       // Convolution kernel weights

    // Working memory buffers
    std::unique_ptr<uint8_t[]> working_memory_;
    size_t working_memory_size_ = 0;

    // Implementation details
    std::unique_ptr<detail::SSMKernels> kernels_;
    std::unique_ptr<detail::SelectiveScanImpl> scan_impl_;

    // Logging
    static constexpr const char* LOGGER_NAME = "MambaSSM";
};

// Factory function for creating Mamba SSM engines
auto create_mamba_ssm_engine(const MambaSSMConfig& config = MambaSSMConfig{})
    -> Result<std::unique_ptr<MambaSSMEngine>, MambaSSMError>;

// Utility functions for testing and benchmarking
namespace testing {
auto generate_random_sequence(size_t batch_size, size_t seq_len, size_t d_model) -> FloatTensor;

auto create_test_model(size_t d_model = 256, size_t d_state = 16) -> MambaSSMConfig;

auto benchmark_sequence_lengths(const std::vector<size_t>& lengths)
    -> std::vector<std::pair<size_t, MambaSSMMetrics>>;
}  // namespace testing

}  // namespace mamba_ssm
}  // namespace engines
}  // namespace inference_lab
