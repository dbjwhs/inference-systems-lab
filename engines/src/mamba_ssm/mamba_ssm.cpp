// MIT License
// Copyright (c) 2025 Inference Systems Lab

#include "mamba_ssm.hpp"

#include <algorithm>
#include <cmath>
// Platform-specific SIMD includes
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>  // For AVX2 SIMD operations on x86_64
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>  // For NEON SIMD operations on ARM64
#endif
#include <numeric>
#include <random>
#include <sstream>

namespace inference_lab {
namespace engines {
namespace mamba_ssm {

using common::Err;
using common::LogLevel;
using common::Ok;

// Error message conversion
auto to_string(MambaSSMError error) -> std::string {
    switch (error) {
        case MambaSSMError::INVALID_CONFIGURATION:
            return "Invalid Mamba SSM configuration parameters";
        case MambaSSMError::SEQUENCE_TOO_LONG:
            return "Input sequence exceeds maximum allowed length";
        case MambaSSMError::DIMENSION_MISMATCH:
            return "Tensor dimension mismatch in SSM operations";
        case MambaSSMError::MEMORY_ALLOCATION_FAILED:
            return "Failed to allocate memory for SSM computations";
        case MambaSSMError::NUMERICAL_INSTABILITY:
            return "Numerical instability detected in SSM computations";
        case MambaSSMError::UNKNOWN_ERROR:
            return "Unknown Mamba SSM error";
    }
    return "Unknown error";
}

// SelectiveParameters implementation
SelectiveParameters::SelectiveParameters(size_t batch,
                                         size_t seq_len,
                                         size_t d_inner,
                                         size_t d_state)
    : delta(Shape{batch, seq_len, d_inner}),
      B_matrix(Shape{batch, seq_len, d_state}),
      C_matrix(Shape{batch, seq_len, d_state}) {
    // Initialize with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    // Initialize delta with positive values (will be processed through softplus)
    auto delta_data = delta.data();
    for (size_t i = 0; i < delta.size(); ++i) {
        delta_data[i] = std::abs(dist(gen)) + 0.1f;
    }

    // Initialize B and C matrices
    auto B_data = B_matrix.data();
    auto C_data = C_matrix.data();
    for (size_t i = 0; i < B_matrix.size(); ++i) {
        B_data[i] = dist(gen);
    }
    for (size_t i = 0; i < C_matrix.size(); ++i) {
        C_data[i] = dist(gen);
    }
}

// SSMState implementation
SSMState::SSMState(size_t batch, size_t d_inner, size_t d_state, size_t conv_width)
    : hidden_state(Shape{batch, d_inner, d_state}), conv_state(Shape{batch, d_inner, conv_width}) {
    reset();
}

void SSMState::reset() {
    // Zero initialize all states
    std::fill_n(hidden_state.data(), hidden_state.size(), 0.0f);
    std::fill_n(conv_state.data(), conv_state.size(), 0.0f);
    sequence_length = 0;
    last_update_time = std::chrono::microseconds{0};
}

auto SSMState::clone() const -> std::unique_ptr<SSMState> {
    auto cloned = std::make_unique<SSMState>(hidden_state.shape()[0],  // batch
                                             hidden_state.shape()[1],  // d_inner
                                             hidden_state.shape()[2],  // d_state
                                             conv_state.shape()[2]     // conv_width
    );

    // Copy data
    std::copy_n(hidden_state.data(), hidden_state.size(), cloned->hidden_state.data());
    std::copy_n(conv_state.data(), conv_state.size(), cloned->conv_state.data());

    cloned->sequence_length = sequence_length;
    cloned->last_update_time = last_update_time;

    return cloned;
}

// SIMD kernels implementation details
namespace detail {

class SSMKernels {
  public:
    // Cross-platform SIMD parameters
#if defined(__x86_64__) || defined(_M_X64)
    static constexpr size_t SIMD_WIDTH = 8;  // AVX2 processes 8 floats
#elif defined(__aarch64__) || defined(_M_ARM64)
    static constexpr size_t SIMD_WIDTH = 4;  // NEON processes 4 floats
#else
    static constexpr size_t SIMD_WIDTH = 1;  // Scalar fallback
#endif

    // Cross-platform vectorized softplus activation: softplus(x) = log(1 + exp(x))
    static void softplus_simd(const float* input, float* output, size_t size) {
        // Process SIMD-width chunks
        size_t simd_end = (size / SIMD_WIDTH) * SIMD_WIDTH;

        for (size_t i = 0; i < simd_end; i += SIMD_WIDTH) {
#if defined(__x86_64__) || defined(_M_X64)
            __m256 x = _mm256_load_ps(&input[i]);
            __m256 clamped = _mm256_min_ps(x, _mm256_set1_ps(20.0f));
            // Use scalar fallback for exp and log (can be optimized with approximations)
            alignas(32) float temp_in[8], temp_out[8];
            _mm256_store_ps(temp_in, clamped);
            for (int j = 0; j < 8; ++j) {
                temp_out[j] = std::log(1.0f + std::exp(temp_in[j]));
            }
            _mm256_store_ps(&output[i], _mm256_load_ps(temp_out));
#elif defined(__aarch64__) || defined(_M_ARM64)
            float32x4_t x = vld1q_f32(&input[i]);
            float32x4_t clamped = vminq_f32(x, vdupq_n_f32(20.0f));
            // Use scalar fallback for exp and log
            alignas(16) float temp_in[4], temp_out[4];
            vst1q_f32(temp_in, clamped);
            for (int j = 0; j < 4; ++j) {
                temp_out[j] = std::log(1.0f + std::exp(temp_in[j]));
            }
            vst1q_f32(&output[i], vld1q_f32(temp_out));
#else
            // Scalar fallback
            for (size_t j = 0; j < SIMD_WIDTH; ++j) {
                float x = std::min(input[i + j], 20.0f);
                output[i + j] = std::log(1.0f + std::exp(x));
            }
#endif
        }

        // Handle remaining elements
        for (size_t i = simd_end; i < size; ++i) {
            float x = std::min(input[i], 20.0f);
            output[i] = std::log(1.0f + std::exp(x));
        }
    }

    // Cross-platform vectorized matrix exponential for diagonal matrix
    static void matrix_exp_diagonal_simd(const float* diagonal,
                                         float delta,
                                         float* result,
                                         size_t size) {
        // Process SIMD-width chunks
        size_t simd_end = (size / SIMD_WIDTH) * SIMD_WIDTH;

        for (size_t i = 0; i < simd_end; i += SIMD_WIDTH) {
#if defined(__x86_64__) || defined(_M_X64)
            __m256 diag = _mm256_load_ps(&diagonal[i]);
            __m256 scaled = _mm256_mul_ps(diag, _mm256_set1_ps(delta));
            // Use scalar fallback for exp
            alignas(32) float temp_in[8], temp_out[8];
            _mm256_store_ps(temp_in, scaled);
            for (int j = 0; j < 8; ++j) {
                temp_out[j] = std::exp(temp_in[j]);
            }
            _mm256_store_ps(&result[i], _mm256_load_ps(temp_out));
#elif defined(__aarch64__) || defined(_M_ARM64)
            float32x4_t diag = vld1q_f32(&diagonal[i]);
            float32x4_t scaled = vmulq_f32(diag, vdupq_n_f32(delta));
            // Use scalar fallback for exp
            alignas(16) float temp_in[4], temp_out[4];
            vst1q_f32(temp_in, scaled);
            for (int j = 0; j < 4; ++j) {
                temp_out[j] = std::exp(temp_in[j]);
            }
            vst1q_f32(&result[i], vld1q_f32(temp_out));
#else
            // Scalar fallback
            for (size_t j = 0; j < SIMD_WIDTH; ++j) {
                result[i + j] = std::exp(diagonal[i + j] * delta);
            }
#endif
        }

        // Handle remaining elements
        for (size_t i = simd_end; i < size; ++i) {
            result[i] = std::exp(diagonal[i] * delta);
        }
    }

    // Cross-platform vectorized state update: h = A * h + B * x
    static void state_update_simd(const float* A,
                                  const float* h_prev,
                                  const float* B,
                                  float x_val,
                                  float* h_new,
                                  size_t size) {
        // Process SIMD-width chunks
        size_t simd_end = (size / SIMD_WIDTH) * SIMD_WIDTH;

        for (size_t i = 0; i < simd_end; i += SIMD_WIDTH) {
#if defined(__x86_64__) || defined(_M_X64)
            __m256 A_vec = _mm256_load_ps(&A[i]);
            __m256 h_vec = _mm256_load_ps(&h_prev[i]);
            __m256 B_vec = _mm256_load_ps(&B[i]);
            __m256 x_broadcast = _mm256_set1_ps(x_val);

            __m256 A_h = _mm256_mul_ps(A_vec, h_vec);
            __m256 B_x = _mm256_mul_ps(B_vec, x_broadcast);
            __m256 result = _mm256_add_ps(A_h, B_x);

            _mm256_store_ps(&h_new[i], result);
#elif defined(__aarch64__) || defined(_M_ARM64)
            float32x4_t A_vec = vld1q_f32(&A[i]);
            float32x4_t h_vec = vld1q_f32(&h_prev[i]);
            float32x4_t B_vec = vld1q_f32(&B[i]);
            float32x4_t x_broadcast = vdupq_n_f32(x_val);

            float32x4_t A_h = vmulq_f32(A_vec, h_vec);
            float32x4_t B_x = vmulq_f32(B_vec, x_broadcast);
            float32x4_t result = vaddq_f32(A_h, B_x);

            vst1q_f32(&h_new[i], result);
#else
            // Scalar fallback
            for (size_t j = 0; j < SIMD_WIDTH; ++j) {
                h_new[i + j] = A[i + j] * h_prev[i + j] + B[i + j] * x_val;
            }
#endif
        }

        // Handle remaining elements
        for (size_t i = simd_end; i < size; ++i) {
            h_new[i] = A[i] * h_prev[i] + B[i] * x_val;
        }
    }
};

class SelectiveScanImpl {
  public:
    // Core selective scan implementation
    static auto scan(const FloatTensor& discrete_A,
                     const FloatTensor& discrete_B,
                     const FloatTensor& C_matrix,
                     const FloatTensor& input,
                     FloatTensor& hidden_state) -> Result<FloatTensor, MambaSSMError> {
        const auto& A_shape = discrete_A.shape();
        const auto batch = A_shape[0];
        const auto seq_len = A_shape[1];
        const auto d_inner = A_shape[2];
        const auto d_state = A_shape[3];

        // Create output tensor
        FloatTensor output(Shape{batch, seq_len, d_inner});
        auto output_data = output.data();

        const auto* A_data = discrete_A.data();
        const auto* B_data = discrete_B.data();
        const auto* C_data = C_matrix.data();
        const auto* input_data = input.data();
        auto* h_data = hidden_state.data();

        // Sequential scan over time steps
        for (size_t b = 0; b < batch; ++b) {
            for (size_t t = 0; t < seq_len; ++t) {
                for (size_t d = 0; d < d_inner; ++d) {
                    // Compute indices
                    const size_t A_idx = ((b * seq_len + t) * d_inner + d) * d_state;
                    const size_t B_idx = A_idx;
                    const size_t C_idx = (b * seq_len + t) * d_state;
                    const size_t input_idx = (b * seq_len + t) * d_inner + d;
                    const size_t h_idx = (b * d_inner + d) * d_state;

                    // State update using SIMD kernel
                    SSMKernels::state_update_simd(&A_data[A_idx],         // A[b,t,d,:]
                                                  &h_data[h_idx],         // h[b,d,:]
                                                  &B_data[B_idx],         // B[b,t,d,:]
                                                  input_data[input_idx],  // x[b,t,d]
                                                  &h_data[h_idx],         // h_new[b,d,:]
                                                  d_state);

                    // Compute output: y = C * h
                    float output_val = 0.0f;
                    for (size_t n = 0; n < d_state; ++n) {
                        output_val += C_data[C_idx + n] * h_data[h_idx + n];
                    }
                    output_data[(b * seq_len + t) * d_inner + d] = output_val;
                }
            }
        }

        return Ok(std::move(output));
    }
};

}  // namespace detail

// MambaSSMEngine implementation
MambaSSMEngine::MambaSSMEngine(const MambaSSMConfig& config)
    : config_(config),
      state_(std::make_unique<SSMState>(
          config.batch_size, config.d_inner, config.d_state, config.d_conv)),
      kernels_(std::make_unique<detail::SSMKernels>()),
      scan_impl_(std::make_unique<detail::SelectiveScanImpl>()) {
    LOG_INFO_PRINT("Initializing Mamba SSM engine with d_model={}, d_state={}, d_inner={}",
                   config_.d_model,
                   config_.d_state,
                   config_.d_inner);

    // Initialize learned parameters (in practice, these would be loaded from a trained model)
    A_matrix_ = FloatTensor(Shape{config_.d_inner, config_.d_state});
    input_projection_ = FloatTensor(Shape{config_.d_model, config_.d_inner});
    output_projection_ = FloatTensor(Shape{config_.d_inner, config_.d_model});
    dt_projection_ = FloatTensor(Shape{config_.d_inner});
    B_projection_ = FloatTensor(Shape{config_.d_inner, config_.d_state});
    C_projection_ = FloatTensor(Shape{config_.d_inner, config_.d_state});
    conv_weights_ = FloatTensor(Shape{config_.d_inner, config_.d_conv});

    // Initialize A matrix as diagonal with negative real parts for stability
    auto A_data = A_matrix_.data();
    std::fill_n(A_data, A_matrix_.size(), 0.0f);
    for (size_t i = 0; i < std::min(config_.d_inner, config_.d_state); ++i) {
        A_data[i * config_.d_state + i] = -1.0f - static_cast<float>(i) * 0.1f;
    }

    // Initialize other parameters with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    auto init_tensor = [&](FloatTensor& tensor) {
        auto data = tensor.data();
        for (size_t i = 0; i < tensor.size(); ++i) {
            data[i] = dist(gen);
        }
    };

    init_tensor(input_projection_);
    init_tensor(output_projection_);
    init_tensor(dt_projection_);
    init_tensor(B_projection_);
    init_tensor(C_projection_);
    init_tensor(conv_weights_);

    LOG_DEBUG_PRINT("Mamba SSM engine initialized successfully");
}

MambaSSMEngine::~MambaSSMEngine() = default;

auto MambaSSMEngine::run_mamba_ssm(const FloatTensor& input_sequence)
    -> Result<FloatTensor, MambaSSMError> {
    const auto start_time = std::chrono::high_resolution_clock::now();

    LOG_DEBUG_PRINT("Starting Mamba SSM inference with sequence shape: [{}, {}, {}]",
                    input_sequence.shape()[0],
                    input_sequence.shape()[1],
                    input_sequence.shape()[2]);

    // Validate input dimensions
    const auto& input_shape = input_sequence.shape();
    if (input_shape.size() != 3) {
        LOG_ERROR_PRINT("Input tensor must be 3D [batch, seq_len, d_model], got {}D",
                        input_shape.size());
        return Err(MambaSSMError::DIMENSION_MISMATCH);
    }

    const auto batch = input_shape[0];
    const auto seq_len = input_shape[1];
    const auto d_model = input_shape[2];

    if (seq_len > config_.max_seq_len) {
        LOG_ERROR_PRINT("Sequence length {} exceeds maximum {}", seq_len, config_.max_seq_len);
        return Err(MambaSSMError::SEQUENCE_TOO_LONG);
    }

    if (d_model != config_.d_model) {
        LOG_ERROR_PRINT("Model dimension mismatch: expected {}, got {}", config_.d_model, d_model);
        return Err(MambaSSMError::DIMENSION_MISMATCH);
    }

    // 1. Input projection
    LOG_DEBUG_PRINT(
        "Performing input projection from {} to {} dimensions", d_model, config_.d_inner);
    // In practice: x_proj = input @ input_projection
    FloatTensor x_proj(Shape{batch, seq_len, config_.d_inner});
    // Simplified projection (full matrix multiplication would be implemented here)
    std::fill_n(x_proj.data(), x_proj.size(), 0.1f);

    // 2. Convolution (temporal modeling)
    auto conv_result = apply_convolution(x_proj);
    if (!conv_result.is_ok()) {
        return Err(conv_result.unwrap_err());
    }
    auto x_conv = std::move(conv_result).unwrap();

    // 3. Compute selective parameters
    auto params_result = compute_selective_params(x_conv);
    if (!params_result.is_ok()) {
        return Err(params_result.unwrap_err());
    }
    auto selective_params = std::move(params_result).unwrap();

    // 4. Discretize continuous SSM
    auto discrete_result = discretize_continuous_ssm(selective_params);
    if (!discrete_result.is_ok()) {
        return Err(discrete_result.unwrap_err());
    }
    auto discrete_pair = std::move(discrete_result).unwrap();
    auto& discrete_A = discrete_pair.first;
    auto& discrete_B = discrete_pair.second;

    // 5. Selective scan
    LOG_DEBUG_PRINT("Performing selective scan operation");
    auto scan_start = std::chrono::high_resolution_clock::now();

    auto scan_result = detail::SelectiveScanImpl::scan(
        discrete_A, discrete_B, selective_params.C_matrix, x_conv, state_->hidden_state);
    if (!scan_result.is_ok()) {
        return Err(scan_result.unwrap_err());
    }
    auto ssm_output = std::move(scan_result).unwrap();

    auto scan_end = std::chrono::high_resolution_clock::now();
    metrics_.selective_scan_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(scan_end - scan_start);

    // 6. Output projection
    LOG_DEBUG_PRINT("Performing output projection");
    FloatTensor output(Shape{batch, seq_len, config_.d_model});
    // Simplified output projection (full matrix multiplication would be implemented here)
    std::copy_n(ssm_output.data(), std::min(ssm_output.size(), output.size()), output.data());

    const auto end_time = std::chrono::high_resolution_clock::now();
    metrics_.inference_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics_.sequence_length = seq_len;
    metrics_.converged = true;

    // Update performance metrics
    metrics_.total_flops =
        seq_len * config_.d_inner * config_.d_state * 6;  // Approximate FLOP count
    metrics_.flops_per_second = static_cast<double>(metrics_.total_flops) /
                                (metrics_.inference_time_ms.count() / 1000000.0);
    metrics_.throughput_tokens_per_sec =
        static_cast<double>(seq_len) / (metrics_.inference_time_ms.count() / 1000000.0);

    LOG_INFO_PRINT("Mamba SSM inference completed in {}μs with throughput {:.1f} tokens/sec",
                   metrics_.inference_time_ms.count(),
                   metrics_.throughput_tokens_per_sec);

    return Ok(std::move(output));
}

auto MambaSSMEngine::compute_selective_params(const FloatTensor& input)
    -> Result<SelectiveParameters, MambaSSMError> {
    const auto& input_shape = input.shape();
    const auto batch = input_shape[0];
    const auto seq_len = input_shape[1];
    const auto d_inner = input_shape[2];

    SelectiveParameters params(batch, seq_len, d_inner, config_.d_state);

    // Compute delta parameters using softplus activation
    auto delta_data = params.delta.data();
    const auto* input_data = input.data();

    LOG_DEBUG_PRINT(
        "Computing selective parameters for shape [{}, {}, {}]", batch, seq_len, d_inner);

    // Simplified computation (in practice, this would involve learned projections)
    for (size_t i = 0; i < params.delta.size(); ++i) {
        // Apply softplus to ensure positive step sizes
        float x = input_data[i % input.size()] + 1.0f;
        delta_data[i] = std::log(1.0f + std::exp(std::min(x, 20.0f)));  // Clamped softplus
    }

    // Compute average step size for metrics
    metrics_.average_step_size =
        std::accumulate(delta_data, delta_data + params.delta.size(), 0.0) / params.delta.size();

    metrics_.selective_updates = seq_len;

    return Ok(std::move(params));
}

auto MambaSSMEngine::discretize_continuous_ssm(const SelectiveParameters& params)
    -> Result<std::pair<FloatTensor, FloatTensor>, MambaSSMError> {
    const auto& delta_shape = params.delta.shape();
    const auto batch = delta_shape[0];
    const auto seq_len = delta_shape[1];
    const auto d_inner = delta_shape[2];

    // Create discretized tensors
    FloatTensor discrete_A(Shape{batch, seq_len, d_inner, config_.d_state});
    FloatTensor discrete_B(Shape{batch, seq_len, d_inner, config_.d_state});

    const auto* delta_data = params.delta.data();
    const auto* B_data = params.B_matrix.data();
    const auto* A_diag = A_matrix_.data();  // Assuming diagonal A matrix

    auto* discrete_A_data = discrete_A.data();
    auto* discrete_B_data = discrete_B.data();

    LOG_DEBUG_PRINT("Discretizing continuous SSM parameters");

    // Zero-order hold discretization:
    // discrete_A = exp(Δ * A)
    // discrete_B = (Δ * A)^{-1} * (exp(Δ * A) - I) * Δ * B
    // For diagonal A: discrete_A_ii = exp(Δ * A_ii), discrete_B_ii ≈ Δ * B_ii

    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t d = 0; d < d_inner; ++d) {
                const size_t delta_idx = (b * seq_len + t) * d_inner + d;
                const float delta_val = delta_data[delta_idx];

                for (size_t n = 0; n < config_.d_state; ++n) {
                    const size_t A_idx = d * config_.d_state + n;  // Diagonal elements
                    const size_t B_idx = (b * seq_len + t) * config_.d_state + n;
                    const size_t output_idx =
                        ((b * seq_len + t) * d_inner + d) * config_.d_state + n;

                    // Discretize A: exp(Δ * A)
                    if (n < d_inner) {  // Only diagonal elements
                        discrete_A_data[output_idx] = std::exp(delta_val * A_diag[A_idx]);
                    } else {
                        discrete_A_data[output_idx] = 0.0f;
                    }

                    // Discretize B: approximately Δ * B for small Δ
                    discrete_B_data[output_idx] = delta_val * B_data[B_idx];
                }
            }
        }
    }

    return Ok(std::make_pair(std::move(discrete_A), std::move(discrete_B)));
}

auto MambaSSMEngine::apply_convolution(const FloatTensor& input)
    -> Result<FloatTensor, MambaSSMError> {
    LOG_DEBUG_PRINT("Applying 1D convolution with kernel size {}", config_.d_conv);

    // For simplicity, return input unchanged (full convolution would be implemented here)
    // In practice, this would perform causal 1D convolution over the sequence dimension
    FloatTensor output(input.shape());
    std::copy_n(input.data(), input.size(), output.data());

    return Ok(std::move(output));
}

// Interface implementations
auto MambaSSMEngine::run_inference(const InferenceRequest& request)
    -> Result<InferenceResponse, InferenceError> {
    LOG_DEBUG_PRINT("Running inference through unified interface");

    // Create demo input for testing
    const size_t batch = 1;
    const size_t seq_len = 64;
    FloatTensor demo_input(Shape{batch, seq_len, config_.d_model});

    // Fill with test data
    auto data = demo_input.data();
    for (size_t i = 0; i < demo_input.size(); ++i) {
        data[i] = 0.1f * static_cast<float>(i % 10);
    }

    auto result = run_mamba_ssm(demo_input);
    if (!result.is_ok()) {
        return Err(InferenceError::INFERENCE_EXECUTION_FAILED);
    }

    auto output = std::move(result).unwrap();

    InferenceResponse response;

    // Convert FloatTensor to vector<float>
    std::vector<float> output_data(output.data(), output.data() + output.size());
    response.output_tensors.push_back(std::move(output_data));
    response.output_names.push_back("mamba_output");
    response.inference_time_ms = metrics_.inference_time_ms.count() / 1000.0;

    return Ok(std::move(response));
}

auto MambaSSMEngine::get_backend_info() const -> std::string {
    std::ostringstream info;
    info << "Mamba State Space Model Engine\n";
    info << "  Model Dimensions: d_model=" << config_.d_model << ", d_state=" << config_.d_state
         << ", d_inner=" << config_.d_inner << "\n";
    info << "  Max Sequence Length: " << config_.max_seq_len << "\n";
    info << "  SIMD Kernels: " << (config_.use_simd_kernels ? "Enabled" : "Disabled") << "\n";
    info << "  Activation: "
         << (config_.activation == MambaSSMConfig::ActivationType::SILU ? "SiLU" : "Other") << "\n";
    return info.str();
}

auto MambaSSMEngine::get_performance_stats() const -> std::string {
    std::ostringstream stats;
    stats << "Mamba SSM Performance Statistics:\n";
    stats << "  Last Sequence Length: " << metrics_.sequence_length << "\n";
    stats << "  Inference Time: " << metrics_.inference_time_ms.count() << " μs\n";
    stats << "  Selective Scan Time: " << metrics_.selective_scan_time_ms.count() << " μs\n";
    stats << "  Throughput: " << std::fixed << std::setprecision(1)
          << metrics_.throughput_tokens_per_sec << " tokens/sec\n";
    stats << "  FLOP Rate: " << std::fixed << std::setprecision(2)
          << metrics_.flops_per_second / 1e9 << " GFLOPS\n";
    stats << "  Average Step Size: " << std::fixed << std::setprecision(4)
          << metrics_.average_step_size << "\n";
    stats << "  Selective Updates: " << metrics_.selective_updates << "\n";
    return stats.str();
}

bool MambaSSMEngine::is_ready() const {
    return state_ != nullptr && kernels_ != nullptr && scan_impl_ != nullptr;
}

void MambaSSMEngine::reset_state() {
    state_->reset();
    metrics_ = MambaSSMMetrics{};
    LOG_DEBUG_PRINT("Mamba SSM state and metrics reset");
}

auto MambaSSMEngine::get_state() const -> const SSMState& {
    return *state_;
}

void MambaSSMEngine::set_state(std::unique_ptr<SSMState> state) {
    state_ = std::move(state);
}

void MambaSSMEngine::update_config(const MambaSSMConfig& new_config) {
    LOG_INFO_PRINT("Updating Mamba SSM configuration");
    config_ = new_config;

    // Recreate state with new dimensions
    state_ = std::make_unique<SSMState>(
        config_.batch_size, config_.d_inner, config_.d_state, config_.d_conv);
    reset_metrics();
}

void MambaSSMEngine::reset_metrics() {
    metrics_ = MambaSSMMetrics{};
}

// Factory function
auto create_mamba_ssm_engine(const MambaSSMConfig& config)
    -> Result<std::unique_ptr<MambaSSMEngine>, MambaSSMError> {
    LOG_INFO_PRINT("Creating Mamba SSM engine");

    // Validate configuration
    if (config.d_model == 0 || config.d_state == 0 || config.d_inner == 0) {
        LOG_ERROR_PRINT("Invalid configuration: dimensions must be positive");
        return Err(MambaSSMError::INVALID_CONFIGURATION);
    }

    if (config.max_seq_len == 0) {
        LOG_ERROR_PRINT("Invalid configuration: max_seq_len must be positive");
        return Err(MambaSSMError::INVALID_CONFIGURATION);
    }

    try {
        auto engine = std::make_unique<MambaSSMEngine>(config);
        if (!engine->is_ready()) {
            LOG_ERROR_PRINT("Failed to initialize Mamba SSM engine");
            return Err(MambaSSMError::MEMORY_ALLOCATION_FAILED);
        }
        return Ok(std::move(engine));
    } catch (const std::exception& e) {
        LOG_ERROR_PRINT("Exception during Mamba SSM engine creation: {}", e.what());
        return Err(MambaSSMError::MEMORY_ALLOCATION_FAILED);
    }
}

// Testing utilities
namespace testing {
auto generate_random_sequence(size_t batch_size, size_t seq_len, size_t d_model) -> FloatTensor {
    FloatTensor sequence(Shape{batch_size, seq_len, d_model});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    auto data = sequence.data();
    for (size_t i = 0; i < sequence.size(); ++i) {
        data[i] = dist(gen);
    }

    return sequence;
}

auto create_test_model(size_t d_model, size_t d_state) -> MambaSSMConfig {
    MambaSSMConfig config;
    config.d_model = d_model;
    config.d_state = d_state;
    config.d_inner = 2 * d_model;
    config.max_seq_len = 512;
    config.batch_size = 1;
    config.use_simd_kernels = true;
    return config;
}

auto benchmark_sequence_lengths(const std::vector<size_t>& lengths)
    -> std::vector<std::pair<size_t, MambaSSMMetrics>> {
    std::vector<std::pair<size_t, MambaSSMMetrics>> results;

    for (size_t seq_len : lengths) {
        auto config = create_test_model(256, 16);
        config.max_seq_len = std::max(seq_len, config.max_seq_len);

        auto engine_result = create_mamba_ssm_engine(config);
        if (!engine_result.is_ok()) {
            continue;
        }

        auto engine = std::move(engine_result).unwrap();
        auto test_input = generate_random_sequence(1, seq_len, config.d_model);

        auto result = engine->run_mamba_ssm(test_input);
        if (result.is_ok()) {
            results.emplace_back(seq_len, engine->get_metrics());
        }
    }

    return results;
}
}  // namespace testing

}  // namespace mamba_ssm
}  // namespace engines
}  // namespace inference_lab
