#include "sparse_activation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>

#include "moe_engine.hpp"  // For MoEError definition

#ifdef MOE_HAS_X86_SIMD
    #include <cpuid.h>
#endif

namespace engines::mixture_experts {

// Constants for sparse computation
namespace {
constexpr std::size_t DEFAULT_CACHE_LINE_SIZE = 64;
constexpr std::size_t MIN_SIMD_VECTOR_SIZE_AVX2 = 8;
constexpr std::size_t MIN_SIMD_VECTOR_SIZE_NEON = 4;
constexpr std::size_t SIMD_ALIGNMENT_BYTES_AVX2 = 32;
constexpr std::size_t SIMD_ALIGNMENT_BYTES_NEON = 16;
constexpr std::size_t FLOAT_SIZE_BYTES = 4;
constexpr float MICROSECONDS_TO_MILLISECONDS = 1000.0f;
constexpr float DEFAULT_WEIGHT_FACTOR = 1.0f;
constexpr float ZERO_THRESHOLD = 0.0f;
constexpr float UNITY_THRESHOLD = 1.0f;
constexpr float SPARSITY_ESTIMATE_FACTOR = 0.7f;  // 70% sparsity assumption
constexpr std::size_t DEFAULT_EXPERT_PARAM_SIZE = 1024;
constexpr std::size_t DEFAULT_OUTPUT_DIMENSION = 256;
constexpr std::size_t PERF_WINDOW_SIZE = 1000;
constexpr float ADAPTIVE_STATS_SMOOTHING_ALPHA = 0.1f;
constexpr float OVERHEAD_COMPENSATION_FACTOR = 1.1f;
constexpr float ADAPTIVE_THRESHOLD_MULTIPLIER = 0.5f;
constexpr float ADAPTIVE_THRESHOLD_MIN_FACTOR = 0.1f;
constexpr float ADAPTIVE_THRESHOLD_MAX_FACTOR = 10.0f;
constexpr int BENCHMARK_ITERATIONS = 100;
constexpr std::size_t BENCHMARK_MATRIX_ROWS = 256;
}  // anonymous namespace

// Helper function for scalar sparse dot product - shared by all SIMD fallback paths
static inline float scalar_sparse_dot_product(const float* sparse_values,
                                              const std::size_t* sparse_indices,
                                              std::size_t nnz,
                                              const float* dense_vector) noexcept {
    float sum = 0.0f;
    for (std::size_t i = 0; i < nnz; ++i) {
        sum += sparse_values[i] * dense_vector[sparse_indices[i]];
    }
    return sum;
}

// Helper function for scalar sparse multiply-accumulate with bounds checking
static inline float scalar_sparse_dot_product_safe(const float* sparse_values,
                                                   const std::size_t* sparse_indices,
                                                   std::size_t nnz,
                                                   const float* dense_vector,
                                                   std::size_t max_index) noexcept {
    float sum = 0.0f;
    for (std::size_t i = 0; i < nnz; ++i) {
        if (sparse_indices[i] < max_index) {
            sum += sparse_values[i] * dense_vector[sparse_indices[i]];
        }
    }
    return sum;
}

// SparsePattern implementation

auto SparsePattern::from_dense(const std::vector<float>& dense_data, float threshold)
    -> SparsePattern {
    SparsePattern pattern;
    pattern.values_.clear();
    pattern.indices_.clear();

    std::size_t nnz_count = 0;

    for (std::size_t i = 0; i < dense_data.size(); ++i) {
        if (std::abs(dense_data[i]) > threshold) {
            pattern.values_.push_back(dense_data[i]);
            pattern.indices_.push_back(i);
            nnz_count++;
        }
    }

    pattern.sparsity_ratio_ =
        UNITY_THRESHOLD - (static_cast<float>(nnz_count) / static_cast<float>(dense_data.size()));
    return pattern;
}

auto SparsePattern::to_dense(std::size_t size) const -> std::vector<float> {
    std::vector<float> dense(size, 0.0f);

    for (std::size_t i = 0; i < values_.size(); ++i) {
        if (indices_[i] < size) {
            dense[indices_[i]] = values_[i];
        }
    }

    return dense;
}

auto SparsePattern::get_sparsity_ratio() const -> float {
    return sparsity_ratio_;
}

// SparseActivation implementation

SparseActivation::SparseActivation(const SparseConfig& config)
    : config_(config), cache_line_size_(DEFAULT_CACHE_LINE_SIZE) {
    // Static assertions for SIMD alignment requirements
    static_assert(alignof(float) <= SIMD_ALIGNMENT_BYTES_AVX2,
                  "Float alignment must be compatible with AVX2/NEON");
    static_assert(sizeof(float) == FLOAT_SIZE_BYTES, "Float must be 32-bit for SIMD operations");

    detect_simd_capabilities();

    // Pre-allocate aligned buffer for SIMD operations
    aligned_buffer_.resize(config_.block_size, ZERO_THRESHOLD);
}

auto SparseActivation::create(const SparseConfig& config)
    -> inference_lab::common::Result<std::unique_ptr<SparseActivation>, MoEError> {
    if (config.sparsity_threshold <= ZERO_THRESHOLD ||
        config.sparsity_threshold >= UNITY_THRESHOLD) {
        return inference_lab::common::Err(MoEError::SPARSE_ACTIVATION_ERROR);
    }

    if (config.block_size == 0 || config.vector_alignment == 0) {
        return inference_lab::common::Err(MoEError::SPARSE_ACTIVATION_ERROR);
    }

    auto activation = std::unique_ptr<SparseActivation>(new SparseActivation(config));
    return inference_lab::common::Ok(std::move(activation));
}

auto SparseActivation::apply_sparse_activation(const std::vector<float>& input,
                                               const std::vector<float>& expert_weights)
    -> inference_lab::common::Result<SparsePattern, MoEError> {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (input.empty()) {
        return inference_lab::common::Err(MoEError::SPARSE_ACTIVATION_ERROR);
    }

    // Apply expert weights to input (element-wise multiplication)
    std::vector<float> weighted_input;
    weighted_input.reserve(input.size());

    if (expert_weights.empty()) {
        // No expert weighting, use input as-is
        weighted_input = input;
    } else {
        // Apply expert weights with broadcasting if needed
        float weight_factor =
            expert_weights.empty()
                ? DEFAULT_WEIGHT_FACTOR
                : std::accumulate(expert_weights.begin(), expert_weights.end(), ZERO_THRESHOLD) /
                      static_cast<float>(expert_weights.size());

        for (float value : input) {
            weighted_input.push_back(value * weight_factor);
        }
    }

    // Adapt sparsity threshold if dynamic adaptation is enabled
    float threshold = config_.sparsity_threshold;
    if (config_.enable_dynamic_sparsity) {
        threshold = adapt_sparsity_threshold(weighted_input);
    }

    // Create sparse pattern
    SparsePattern pattern = SparsePattern::from_dense(weighted_input, threshold);

    // Update performance statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    float computation_time = static_cast<float>(duration.count()) / MICROSECONDS_TO_MILLISECONDS;

    update_performance_stats(computation_time, pattern.get_sparsity_ratio(), input.size());

    return inference_lab::common::Ok(std::move(pattern));
}

auto SparseActivation::sparse_matrix_vector_multiply(const SparsePattern& sparse_input,
                                                     const std::vector<float>& weight_matrix,
                                                     std::size_t rows,
                                                     std::size_t cols)
    -> inference_lab::common::Result<std::vector<float>, MoEError> {
    if (weight_matrix.size() != rows * cols) {
        return inference_lab::common::Err(MoEError::SPARSE_ACTIVATION_ERROR);
    }

    std::vector<float> output(rows, ZERO_THRESHOLD);

    const auto& values = sparse_input.get_values();
    const auto& indices = sparse_input.get_indices();

    // SIMD-optimized sparse matrix-vector multiplication
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;

        const float* weight_row = weight_matrix.data() + row * cols;

        if (config_.enable_simd_optimization && avx2_available_ &&
            values.size() >= MIN_SIMD_VECTOR_SIZE_AVX2) {
            // Use SIMD optimized version
#ifdef MOE_HAS_X86_SIMD
            sum = sparse_dot_product_avx2(values.data(), indices.data(), values.size(), weight_row);
#else
            // Fallback to scalar version
            sum = scalar_sparse_dot_product_safe(
                values.data(), indices.data(), values.size(), weight_row, cols);
#endif
        } else {
            // Scalar version
            sum = scalar_sparse_dot_product_safe(
                values.data(), indices.data(), values.size(), weight_row, cols);
        }

        output[row] = sum;
    }

    return inference_lab::common::Ok(std::move(output));
}

auto SparseActivation::sparse_elementwise_multiply(const SparsePattern& lhs,
                                                   const SparsePattern& rhs)
    -> inference_lab::common::Result<SparsePattern, MoEError> {
    const auto& lhs_values = lhs.get_values();
    const auto& lhs_indices = lhs.get_indices();
    const auto& rhs_values = rhs.get_values();
    const auto& rhs_indices = rhs.get_indices();

    std::vector<float> result_values;
    std::vector<std::size_t> result_indices;

    // Merge sparse patterns (intersection of non-zero elements)
    std::size_t lhs_idx = 0, rhs_idx = 0;

    while (lhs_idx < lhs_indices.size() && rhs_idx < rhs_indices.size()) {
        if (lhs_indices[lhs_idx] == rhs_indices[rhs_idx]) {
            // Both have non-zero values at this index
            float product = lhs_values[lhs_idx] * rhs_values[rhs_idx];
            if (std::abs(product) > config_.sparsity_threshold) {
                result_values.push_back(product);
                result_indices.push_back(lhs_indices[lhs_idx]);
            }
            lhs_idx++;
            rhs_idx++;
        } else if (lhs_indices[lhs_idx] < rhs_indices[rhs_idx]) {
            lhs_idx++;
        } else {
            rhs_idx++;
        }
    }

    SparsePattern result;
    result.values_ = std::move(result_values);
    result.indices_ = std::move(result_indices);

    // Calculate sparsity ratio (approximate)
    std::size_t max_possible_nnz = std::max(lhs_indices.size(), rhs_indices.size());
    if (max_possible_nnz > 0) {
        result.sparsity_ratio_ = UNITY_THRESHOLD - (static_cast<float>(result.values_.size()) /
                                                    static_cast<float>(max_possible_nnz));
    } else {
        result.sparsity_ratio_ = UNITY_THRESHOLD;
    }

    return inference_lab::common::Ok(std::move(result));
}

auto SparseActivation::detect_simd_capabilities() -> void {
    avx2_available_ = false;
    avx512_available_ = false;
    fma_available_ = false;
    neon_available_ = false;

#ifdef MOE_HAS_X86_SIMD
    // Check CPUID for SIMD support
    unsigned int eax, ebx, ecx, edx;

    // Check for AVX2 support
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        avx2_available_ = (ebx & (1 << 5)) != 0;  // AVX2 bit
    }

    // Check for FMA support
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        fma_available_ = (ecx & (1 << 12)) != 0;  // FMA bit
    }

    // Check for AVX512F support
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        avx512_available_ = (ebx & (1 << 16)) != 0;  // AVX512F bit
    }
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    // ARM NEON is available - use neon_available_ flag specifically
    neon_available_ = true;
#endif
}

#ifdef MOE_HAS_X86_SIMD
auto SparseActivation::sparse_dot_product_avx2(const float* sparse_values,
                                               const std::size_t* sparse_indices,
                                               std::size_t nnz,
                                               const float* dense_vector) -> float {
    // Process 8 elements at a time with AVX2
    __m256 sum_vec = _mm256_setzero_ps();
    std::size_t simd_end = (nnz / MIN_SIMD_VECTOR_SIZE_AVX2) * MIN_SIMD_VECTOR_SIZE_AVX2;

    for (std::size_t i = 0; i < simd_end; i += MIN_SIMD_VECTOR_SIZE_AVX2) {
        // Load sparse values
        __m256 sparse_vals = _mm256_load_ps(&sparse_values[i]);

        // Gather corresponding dense values (manual gather for compatibility)
        float gathered_values[8];
        for (int j = 0; j < 8; ++j) {
            gathered_values[j] = dense_vector[sparse_indices[i + j]];
        }
        __m256 dense_vals = _mm256_load_ps(gathered_values);

        // Multiply and accumulate
        __m256 products = _mm256_mul_ps(sparse_vals, dense_vals);
        sum_vec = _mm256_add_ps(sum_vec, products);
    }

    // Horizontal sum of the vector
    __m128 high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 low = _mm256_extractf128_ps(sum_vec, 0);
    __m128 sum128 = _mm_add_ps(high, low);
    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));

    float result = _mm_cvtss_f32(sum32);

    // Handle remaining elements using scalar helper
    result += scalar_sparse_dot_product(
        sparse_values + simd_end, sparse_indices + simd_end, nnz - simd_end, dense_vector);

    return result;
}
#else
auto SparseActivation::sparse_dot_product_avx2(const float* sparse_values,
                                               const std::size_t* sparse_indices,
                                               std::size_t nnz,
                                               const float* dense_vector) -> float {
    // Fallback scalar implementation
    return scalar_sparse_dot_product(sparse_values, sparse_indices, nnz, dense_vector);
}
#endif

auto SparseActivation::sparse_dot_product_avx512(const float* sparse_values,
                                                 const std::size_t* sparse_indices,
                                                 std::size_t nnz,
                                                 const float* dense_vector) -> float {
    // Simplified implementation - in practice would use AVX512 intrinsics
    return sparse_dot_product_avx2(sparse_values, sparse_indices, nnz, dense_vector);
}

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    #include <arm_neon.h>

auto SparseActivation::sparse_dot_product_neon(const float* sparse_values,
                                               const std::size_t* sparse_indices,
                                               std::size_t nnz,
                                               const float* dense_vector) -> float {
    // Process 4 elements at a time with NEON
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    std::size_t simd_end = (nnz / MIN_SIMD_VECTOR_SIZE_NEON) * MIN_SIMD_VECTOR_SIZE_NEON;

    for (std::size_t i = 0; i < simd_end; i += MIN_SIMD_VECTOR_SIZE_NEON) {
        // Load sparse values
        float32x4_t sparse_vals = vld1q_f32(&sparse_values[i]);

        // Gather corresponding dense values
        float gathered_values[4];
        for (int j = 0; j < 4; ++j) {
            gathered_values[j] = dense_vector[sparse_indices[i + j]];
        }
        float32x4_t dense_vals = vld1q_f32(gathered_values);

        // Multiply and accumulate
        sum_vec = vmlaq_f32(sum_vec, sparse_vals, dense_vals);
    }

    // Horizontal sum of the vector
    float32x2_t sum_pair = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
    float result = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    // Handle remaining elements using scalar helper
    result += scalar_sparse_dot_product(
        sparse_values + simd_end, sparse_indices + simd_end, nnz - simd_end, dense_vector);

    return result;
}
#else
auto SparseActivation::sparse_dot_product_neon(const float* sparse_values,
                                               const std::size_t* sparse_indices,
                                               std::size_t nnz,
                                               const float* dense_vector) -> float {
    // Fallback scalar implementation
    return scalar_sparse_dot_product(sparse_values, sparse_indices, nnz, dense_vector);
}
#endif

auto SparseActivation::adapt_sparsity_threshold(const std::vector<float>& input_distribution)
    -> float {
    if (input_distribution.empty()) {
        return config_.sparsity_threshold;
    }

    // Calculate statistics of input distribution
    float mean = std::accumulate(input_distribution.begin(), input_distribution.end(), 0.0f) /
                 static_cast<float>(input_distribution.size());

    float variance = 0.0f;
    for (float value : input_distribution) {
        float diff = value - mean;
        variance += diff * diff;
    }
    variance /= static_cast<float>(input_distribution.size());

    float std_dev = std::sqrt(variance);

    // Adaptive threshold based on input statistics
    // Target the configured sparsity ratio
    float adaptive_threshold = std::abs(mean) + std_dev * ADAPTIVE_THRESHOLD_MULTIPLIER;

    // Clamp to reasonable bounds
    adaptive_threshold =
        std::max(adaptive_threshold, config_.sparsity_threshold * ADAPTIVE_THRESHOLD_MIN_FACTOR);
    adaptive_threshold =
        std::min(adaptive_threshold, config_.sparsity_threshold * ADAPTIVE_THRESHOLD_MAX_FACTOR);

    return adaptive_threshold;
}

auto SparseActivation::update_performance_stats(float computation_time_ms,
                                                float sparsity_ratio,
                                                std::size_t operations) -> void {
    total_operations_.fetch_add(operations);

    // Update cumulative statistics with exponential moving average
    float current_time = total_computation_time_ms_.load();
    float new_time = current_time + computation_time_ms;
    total_computation_time_ms_.store(new_time);

    float current_sparsity = cumulative_sparsity_ratio_.load();
    float alpha = ADAPTIVE_STATS_SMOOTHING_ALPHA;  // Smoothing factor
    float new_sparsity = alpha * sparsity_ratio + (UNITY_THRESHOLD - alpha) * current_sparsity;
    cumulative_sparsity_ratio_.store(new_sparsity);
}

auto SparseActivation::get_sparse_stats() const -> SparseStats {
    SparseStats stats;

    std::size_t total_ops = total_operations_.load();
    float avg_sparsity = cumulative_sparsity_ratio_.load();

    stats.actual_sparsity_ratio = avg_sparsity;
    stats.total_activations = total_ops;
    stats.sparse_activations = static_cast<std::size_t>(total_ops * avg_sparsity);
    stats.computational_savings = estimate_computational_savings(avg_sparsity, total_ops);
    stats.memory_savings = avg_sparsity;  // Simplified estimate

    // SIMD utilization based on capabilities
    stats.simd_utilization = 0.0f;
    if (config_.enable_simd_optimization) {
        if (avx512_available_) {
            stats.simd_utilization = 0.9f;
        } else if (avx2_available_) {
            stats.simd_utilization = 0.7f;
        } else {
            stats.simd_utilization = 0.1f;  // Minimal SIMD usage
        }
    }

    return stats;
}

auto SparseActivation::estimate_computational_savings(float sparsity_ratio,
                                                      std::size_t vector_size) const -> float {
    // Estimate FLOPS savings from sparse computation
    float dense_flops = static_cast<float>(vector_size);
    float sparse_flops = dense_flops * (UNITY_THRESHOLD - sparsity_ratio);

    // Account for sparse overhead (indexing, etc.)
    float overhead_factor = OVERHEAD_COMPENSATION_FACTOR;
    sparse_flops *= overhead_factor;

    if (dense_flops > 0.0f) {
        return std::max(0.0f, (dense_flops - sparse_flops) / dense_flops);
    } else {
        return 0.0f;
    }
}

auto SparseActivation::validate_simd_capabilities() const
    -> inference_lab::common::Result<std::monostate, MoEError> {
    if (config_.enable_simd_optimization) {
        if (!avx2_available_ && !avx512_available_) {
            // No SIMD support available but requested
            return inference_lab::common::Err(MoEError::SPARSE_ACTIVATION_ERROR);
        }
    }

    return inference_lab::common::Ok(std::monostate{});
}

auto SparseActivation::benchmark_sparse_performance(std::size_t vector_size, float sparsity_ratio)
    -> inference_lab::common::Result<float, MoEError> {
    // Create test vectors
    std::vector<float> test_vector(vector_size);
    std::iota(test_vector.begin(), test_vector.end(), DEFAULT_WEIGHT_FACTOR);

    // Create sparse pattern with target sparsity
    SparsePattern sparse_pattern =
        SparsePattern::from_dense(test_vector,
                                  sparsity_ratio * 10.0f);  // Adjust threshold

    // Create test weight matrix
    std::size_t matrix_rows = BENCHMARK_MATRIX_ROWS;
    std::vector<float> weight_matrix(matrix_rows * vector_size, DEFAULT_WEIGHT_FACTOR);

    // Benchmark sparse matrix-vector multiplication
    auto start_time = std::chrono::high_resolution_clock::now();

    constexpr int num_iterations = BENCHMARK_ITERATIONS;
    for (int i = 0; i < num_iterations; ++i) {
        auto result =
            sparse_matrix_vector_multiply(sparse_pattern, weight_matrix, matrix_rows, vector_size);
        if (!result.is_ok()) {
            return inference_lab::common::Err(std::move(result).unwrap_err());
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    float total_time_ms = static_cast<float>(duration.count()) / 1000.0f;
    float avg_time_ms = total_time_ms / num_iterations;

    // Calculate GFLOPS (simplified)
    std::size_t total_ops = matrix_rows * sparse_pattern.get_nnz() * 2;  // multiply + add
    float gflops = (static_cast<float>(total_ops) / 1e9f) / (avg_time_ms / 1000.0f);

    return inference_lab::common::Ok(gflops);
}

}  // namespace engines::mixture_experts
