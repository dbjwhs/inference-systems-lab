#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <atomic>
#include <variant>

// Platform-specific SIMD includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #include <immintrin.h>  // x86/x64 SIMD intrinsics
    #define MOE_HAS_X86_SIMD 1
#elif defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>   // ARM NEON SIMD intrinsics
    #define MOE_HAS_ARM_NEON 1
#else
    #define MOE_HAS_SIMD 0
#endif

#include "../../../common/src/result.hpp"

namespace engines::mixture_experts {

enum class MoEError : std::uint8_t;

/**
 * @brief Configuration for sparse activation patterns
 */
struct SparseConfig {
    float sparsity_threshold = 0.01f;     // Values below this are considered zero
    bool enable_simd_optimization = true;  // Use SIMD instructions for vectorization
    std::size_t vector_alignment = 32;     // Memory alignment for SIMD (256-bit AVX2)
    std::size_t block_size = 64;           // Processing block size for cache efficiency
    bool enable_dynamic_sparsity = true;   // Adapt sparsity based on input patterns
    float target_sparsity_ratio = 0.7f;    // Target 70% sparsity for efficiency
};

/**
 * @brief Statistics for sparse activation performance
 */
struct SparseStats {
    float actual_sparsity_ratio;           // Percentage of zero/near-zero activations
    std::size_t total_activations;
    std::size_t sparse_activations;
    float computational_savings;           // Estimated FLOPS reduction
    float memory_savings;                  // Memory access reduction
    std::vector<float> block_sparsity_distribution;
    float simd_utilization;                // SIMD instruction utilization rate
};

/**
 * @brief Sparse activation pattern for computational efficiency
 * 
 * Represents sparse data with efficient storage and computation:
 * - Compressed sparse row (CSR) format for memory efficiency
 * - SIMD-optimized operations for vectorized computation
 * - Cache-friendly block processing for memory bandwidth optimization
 * - Dynamic sparsity adaptation based on input patterns
 */
class SparsePattern {
public:
    /**
     * @brief Create sparse pattern from dense vector
     * @param dense_data Input dense vector
     * @param threshold Sparsity threshold for zero detection
     * @return Sparse pattern representation
     */
    static auto from_dense(const std::vector<float>& dense_data, float threshold = 0.01f)
        -> SparsePattern;

    /**
     * @brief Convert back to dense vector
     * @param size Original dense vector size
     * @return Dense vector representation
     */
    auto to_dense(std::size_t size) const -> std::vector<float>;

    /**
     * @brief Get sparsity ratio (fraction of zero elements)
     */
    auto get_sparsity_ratio() const -> float;

    /**
     * @brief Get number of non-zero elements
     */
    auto get_nnz() const -> std::size_t { return values_.size(); }

    /**
     * @brief Get sparse data arrays (CSR format)
     */
    auto get_values() const -> const std::vector<float>& { return values_; }
    auto get_indices() const -> const std::vector<std::size_t>& { return indices_; }

private:
    std::vector<float> values_;        // Non-zero values
    std::vector<std::size_t> indices_; // Indices of non-zero values
    float sparsity_ratio_;
};

/**
 * @brief SIMD-optimized sparse activation system  
 * 
 * Implements high-performance sparse computation patterns:
 * - SIMD vectorization using AVX2/AVX512 instructions
 * - Integration with existing BatchContainer<T> for data alignment
 * - Cache-friendly memory access patterns
 * - Dynamic sparsity adaptation for optimal computational efficiency
 * 
 * Performance targets:
 * - 80%+ theoretical peak FLOPS utilization during sparse operations
 * - 10-100x computational efficiency gains through sparse patterns
 * - <2GB/s peak memory bandwidth compatible with existing infrastructure
 * - Integration with existing SIMD containers without performance regression
 */
class SparseActivation {
public:
    /**
     * @brief Create sparse activation system with specified configuration
     * @param config Sparse activation configuration
     * @return Result containing initialized system or error
     */
    static auto create(const SparseConfig& config) 
        -> inference_lab::common::Result<std::unique_ptr<SparseActivation>, MoEError>;

    /**
     * @brief Apply sparse activation to input vector
     * @param input Dense input vector
     * @param expert_weights Expert selection weights
     * @return Result containing sparse pattern or error
     */
    auto apply_sparse_activation(const std::vector<float>& input,
                                const std::vector<float>& expert_weights)
        -> inference_lab::common::Result<SparsePattern, MoEError>;

    /**
     * @brief Compute sparse matrix-vector multiplication (SIMD optimized)
     * @param sparse_input Sparse input pattern  
     * @param weight_matrix Dense weight matrix (row-major)
     * @param rows Number of rows in weight matrix
     * @param cols Number of columns in weight matrix
     * @return Result containing output vector or error
     */
    auto sparse_matrix_vector_multiply(const SparsePattern& sparse_input,
                                      const std::vector<float>& weight_matrix,
                                      std::size_t rows, std::size_t cols)
        -> inference_lab::common::Result<std::vector<float>, MoEError>;

    /**
     * @brief Compute element-wise sparse operations (SIMD optimized)
     * @param lhs Left-hand sparse pattern
     * @param rhs Right-hand sparse pattern  
     * @return Result containing combined sparse pattern or error
     */
    auto sparse_elementwise_multiply(const SparsePattern& lhs, 
                                    const SparsePattern& rhs)
        -> inference_lab::common::Result<SparsePattern, MoEError>;

    /**
     * @brief Get current sparse activation statistics
     * @return Performance and efficiency metrics
     */
    auto get_sparse_stats() const -> SparseStats;

    /**
     * @brief Validate SIMD optimization availability and performance
     * @return Result indicating SIMD system status
     */
    auto validate_simd_capabilities() const -> inference_lab::common::Result<std::monostate, MoEError>;

    /**
     * @brief Benchmark sparse operations performance
     * @param vector_size Size of test vectors
     * @param sparsity_ratio Target sparsity for benchmarking
     * @return Result containing performance metrics or error
     */
    auto benchmark_sparse_performance(std::size_t vector_size, float sparsity_ratio)
        -> inference_lab::common::Result<float, MoEError>;  // Returns GFLOPS achieved

private:
    explicit SparseActivation(const SparseConfig& config);

    // Configuration
    SparseConfig config_;

    // SIMD capability detection
    bool avx2_available_;
    bool avx512_available_;
    bool fma_available_;

    // Performance monitoring
    mutable std::atomic<std::size_t> total_operations_{0};
    mutable std::atomic<float> total_computation_time_ms_{0.0f};
    mutable std::atomic<float> cumulative_sparsity_ratio_{0.0f};

    // Memory alignment and caching
    std::vector<float> aligned_buffer_;
    std::size_t cache_line_size_;

    // Helper methods for SIMD operations
    auto detect_simd_capabilities() -> void;
    
    auto sparse_dot_product_avx2(const float* sparse_values,
                                 const std::size_t* sparse_indices,
                                 std::size_t nnz,
                                 const float* dense_vector) -> float;
    
    auto sparse_dot_product_avx512(const float* sparse_values,
                                  const std::size_t* sparse_indices,
                                  std::size_t nnz,
                                  const float* dense_vector) -> float;
    
    auto apply_threshold_simd(const float* input, float* output, 
                             std::size_t size, float threshold) -> std::size_t;
    
    auto compress_to_csr(const std::vector<float>& dense_data, float threshold)
        -> std::pair<std::vector<float>, std::vector<std::size_t>>;
    
    auto update_performance_stats(float computation_time_ms, 
                                 float sparsity_ratio,
                                 std::size_t operations) -> void;

    // Cache-friendly block processing
    auto process_sparse_blocks(const SparsePattern& input,
                              const std::vector<float>& weight_matrix,
                              std::size_t rows, std::size_t cols,
                              std::vector<float>& output) -> void;

    // Dynamic sparsity adaptation  
    auto adapt_sparsity_threshold(const std::vector<float>& input_distribution) -> float;
    
    auto estimate_computational_savings(float sparsity_ratio, 
                                       std::size_t vector_size) -> float;
};

/**
 * @brief RAII wrapper for SIMD-aligned memory allocation
 */
template<typename T, std::size_t Alignment = 32>
class AlignedVector {
public:
    explicit AlignedVector(std::size_t size) : size_(size) {
        // Allocate aligned memory compatible with SIMD instructions
        data_ = static_cast<T*>(std::aligned_alloc(Alignment, size * sizeof(T)));
        if (!data_) {
            throw std::bad_alloc();
        }
    }

    ~AlignedVector() {
        if (data_) {
            std::free(data_);
        }
    }

    // Non-copyable, movable
    AlignedVector(const AlignedVector&) = delete;
    AlignedVector& operator=(const AlignedVector&) = delete;
    
    AlignedVector(AlignedVector&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    AlignedVector& operator=(AlignedVector&& other) noexcept {
        if (this != &other) {
            if (data_) std::free(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    auto data() -> T* { return data_; }
    auto data() const -> const T* { return data_; }
    auto size() const -> std::size_t { return size_; }
    
    auto operator[](std::size_t idx) -> T& { return data_[idx]; }
    auto operator[](std::size_t idx) const -> const T& { return data_[idx]; }

private:
    T* data_{nullptr};
    std::size_t size_{0};
};

} // namespace engines::mixture_experts