#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <variant>

#include "../../../common/src/result.hpp"

namespace engines::mixture_experts {

enum class MoEError : std::uint8_t;

/**
 * @brief Configuration for expert parameter management
 */
struct ParameterConfig {
    std::size_t num_experts = 8;
    std::size_t parameters_per_expert = 1024;
    std::size_t memory_pool_size_mb = 500;
    bool enable_parameter_compression = true;
    bool enable_parameter_streaming = false;
    float compression_ratio = 0.8f;
    std::size_t cache_size_experts = 4;  // Number of experts to keep in fast cache
};

/**
 * @brief Expert parameter storage statistics
 */
struct ParameterStats {
    std::size_t total_parameters;
    std::size_t active_parameters;
    std::size_t compressed_parameters;
    float memory_usage_mb;
    float compression_efficiency;
    std::vector<float> expert_memory_usage;
    std::size_t cache_hits;
    std::size_t cache_misses;
};

/**
 * @brief Handle to expert parameters with lazy loading
 */
class ExpertParameterHandle {
public:
    explicit ExpertParameterHandle(std::size_t expert_id, std::size_t parameter_count);
    
    /**
     * @brief Get parameter data (may trigger loading from storage)
     * @return Result containing parameter vector or error
     */
    auto get_parameters() -> inference_lab::common::Result<std::vector<float>&, MoEError>;
    
    /**
     * @brief Check if parameters are currently loaded in memory
     */
    auto is_loaded() const -> bool;
    
    /**
     * @brief Get expert identifier
     */
    auto get_expert_id() const -> std::size_t { return expert_id_; }
    
    /**
     * @brief Get parameter count for this expert
     */
    auto get_parameter_count() const -> std::size_t { return parameter_count_; }

private:
    std::size_t expert_id_;
    std::size_t parameter_count_;
    std::vector<float> parameters_;
    std::atomic<bool> loaded_{false};
    mutable std::mutex parameter_mutex_;
};

/**
 * @brief Memory-efficient expert parameter management system
 * 
 * Implements optimized parameter storage using existing infrastructure:
 * - Memory pool integration with existing MemoryPool<T> for O(1) allocation
 * - Parameter compression using SIMD optimizations
 * - Lazy loading and streaming for large expert networks
 * - Lock-free concurrent access patterns for thread safety
 * 
 * Performance targets:
 * - <500MB memory usage per expert with compression
 * - O(1) parameter allocation and deallocation
 * - <50ms expert loading time for dynamic activation
 * - <10% memory pool integration overhead
 * - Linear memory scaling with number of experts
 */
class ExpertParameters {
public:
    /**
     * @brief Create expert parameter manager with specified configuration
     * @param config Parameter management configuration
     * @return Result containing initialized manager or error
     */
    static auto create(const ParameterConfig& config) 
        -> inference_lab::common::Result<std::unique_ptr<ExpertParameters>, MoEError>;

    /**
     * @brief Get parameter handle for specified expert
     * @param expert_id Expert identifier
     * @return Result containing parameter handle or error
     */
    auto get_expert_handle(std::size_t expert_id) 
        -> inference_lab::common::Result<std::shared_ptr<ExpertParameterHandle>, MoEError>;

    /**
     * @brief Load multiple experts into fast cache
     * @param expert_ids List of expert identifiers to preload
     * @return Result indicating loading success or error
     */
    auto preload_experts(const std::vector<std::size_t>& expert_ids)
        -> inference_lab::common::Result<std::monostate, MoEError>;

    /**
     * @brief Update parameters for specified expert
     * @param expert_id Expert identifier
     * @param new_parameters New parameter values
     * @return Result indicating update success or error
     */
    auto update_expert_parameters(std::size_t expert_id, 
                                 const std::vector<float>& new_parameters)
        -> inference_lab::common::Result<std::monostate, MoEError>;

    /**
     * @brief Get current parameter storage statistics
     * @return Parameter usage and performance metrics
     */
    auto get_parameter_stats() const -> ParameterStats;

    /**
     * @brief Validate parameter storage integrity
     * @return Result indicating storage health status
     */
    auto validate_storage_integrity() const -> inference_lab::common::Result<std::monostate, MoEError>;

    /**
     * @brief Compress parameters for specified experts
     * @param expert_ids List of experts to compress
     * @return Result indicating compression success or error
     */
    auto compress_experts(const std::vector<std::size_t>& expert_ids)
        -> inference_lab::common::Result<std::monostate, MoEError>;

    /**
     * @brief Clear unused experts from cache to free memory
     * @param keep_count Number of most recently used experts to keep
     * @return Result indicating cleanup success or error
     */
    auto cleanup_unused_experts(std::size_t keep_count = 4)
        -> inference_lab::common::Result<std::monostate, MoEError>;

private:
    explicit ExpertParameters(const ParameterConfig& config);

    // Configuration
    ParameterConfig config_;

    // Expert parameter storage
    std::vector<std::shared_ptr<ExpertParameterHandle>> expert_handles_;
    std::unordered_map<std::size_t, std::vector<float>> compressed_parameters_;
    
    // Fast cache for frequently accessed experts
    struct CacheEntry {
        std::size_t expert_id;
        std::shared_ptr<ExpertParameterHandle> handle;
        std::chrono::steady_clock::time_point last_access;
        std::atomic<std::size_t> access_count{0};
    };
    std::vector<CacheEntry> parameter_cache_;
    mutable std::mutex cache_mutex_;

    // Memory pool integration (will use existing MemoryPool<T>)
    std::unique_ptr<void> memory_pool_;  // Will be cast to MemoryPool<float>
    
    // Performance monitoring
    mutable std::atomic<std::size_t> total_cache_hits_{0};
    mutable std::atomic<std::size_t> total_cache_misses_{0};
    mutable std::atomic<float> total_memory_usage_mb_{0.0f};

    // Compression state
    std::unordered_map<std::size_t, float> compression_ratios_;
    mutable std::mutex compression_mutex_;

    // Helper methods
    auto load_expert_from_storage(std::size_t expert_id)
        -> inference_lab::common::Result<std::vector<float>, MoEError>;
    
    auto store_expert_to_storage(std::size_t expert_id, 
                                const std::vector<float>& parameters)
        -> inference_lab::common::Result<std::monostate, MoEError>;
    
    auto compress_parameter_vector(const std::vector<float>& parameters, float ratio)
        -> inference_lab::common::Result<std::vector<float>, MoEError>;
    
    auto decompress_parameter_vector(const std::vector<float>& compressed_params)
        -> inference_lab::common::Result<std::vector<float>, MoEError>;
    
    auto update_cache_entry(std::size_t expert_id, 
                           std::shared_ptr<ExpertParameterHandle> handle) -> void;
    
    auto find_cache_entry(std::size_t expert_id) -> std::shared_ptr<ExpertParameterHandle>;
    
    auto evict_least_recently_used() -> void;

    // Memory management
    auto allocate_parameter_memory(std::size_t size) -> void*;
    auto deallocate_parameter_memory(void* ptr) -> void;
    auto calculate_memory_usage() const -> float;
};

} // namespace engines::mixture_experts