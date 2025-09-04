#include "expert_parameters.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>

namespace engines::mixture_experts {

// ExpertParameterHandle implementation

ExpertParameterHandle::ExpertParameterHandle(std::size_t expert_id, std::size_t parameter_count)
    : expert_id_(expert_id), parameter_count_(parameter_count) {
    parameters_.reserve(parameter_count);
}

auto ExpertParameterHandle::get_parameters() -> inference_lab::common::Result<std::vector<float>&, MoEError> {
    std::lock_guard<std::mutex> lock(parameter_mutex_);
    
    if (!loaded_.load()) {
        // Initialize parameters with random values for demonstration
        // In practice, this would load from storage or be initialized externally
        parameters_.resize(parameter_count_);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        for (auto& param : parameters_) {
            param = dist(gen);
        }
        
        loaded_.store(true);
    }
    
    return inference_lab::common::Ok(std::ref(parameters_));
}

auto ExpertParameterHandle::is_loaded() const -> bool {
    return loaded_.load();
}

// ExpertParameters implementation

ExpertParameters::ExpertParameters(const ParameterConfig& config) 
    : config_(config) {
    
    // Initialize expert handles
    expert_handles_.reserve(config_.num_experts);
    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        auto handle = std::make_shared<ExpertParameterHandle>(i, config_.parameters_per_expert);
        expert_handles_.push_back(handle);
    }
    
    // Initialize cache
    parameter_cache_.reserve(config_.cache_size_experts);
    
    // Initialize memory pool (simplified - in practice would use existing MemoryPool<T>)
    // For now, we'll use nullptr to indicate we'd integrate with existing memory pool
    memory_pool_ = nullptr;
}

auto ExpertParameters::create(const ParameterConfig& config) 
    -> inference_lab::common::Result<std::unique_ptr<ExpertParameters>, MoEError> {
    
    if (config.num_experts == 0 || config.parameters_per_expert == 0) {
        return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
    }
    
    if (config.memory_pool_size_mb == 0) {
        return inference_lab::common::Err(MoEError::PARAMETER_STORAGE_ERROR);
    }
    
    if (config.compression_ratio <= 0.0f || config.compression_ratio > 1.0f) {
        return inference_lab::common::Err(MoEError::PARAMETER_STORAGE_ERROR);
    }
    
    auto parameters = std::unique_ptr<ExpertParameters>(new ExpertParameters(config));
    return inference_lab::common::Ok(std::move(parameters));
}

auto ExpertParameters::get_expert_handle(std::size_t expert_id) 
    -> inference_lab::common::Result<std::shared_ptr<ExpertParameterHandle>, MoEError> {
    
    if (expert_id >= config_.num_experts) {
        return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
    }
    
    // Check cache first
    auto cached_handle = find_cache_entry(expert_id);
    if (cached_handle) {
        total_cache_hits_.fetch_add(1);
        return inference_lab::common::Ok(cached_handle);
    }
    
    total_cache_misses_.fetch_add(1);
    
    // Get handle from storage
    auto handle = expert_handles_[expert_id];
    
    // Update cache
    update_cache_entry(expert_id, handle);
    
    return inference_lab::common::Ok(handle);
}

auto ExpertParameters::preload_experts(const std::vector<std::size_t>& expert_ids)
    -> inference_lab::common::Result<std::monostate, MoEError> {
    
    for (auto expert_id : expert_ids) {
        if (expert_id >= config_.num_experts) {
            return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
        }
        
        auto handle_result = get_expert_handle(expert_id);
        if (!handle_result.is_ok()) {
            return inference_lab::common::Err(std::move(handle_result).unwrap_err());
        }
        
        auto handle = std::move(handle_result).unwrap();
        
        // Force loading of parameters
        auto params_result = handle->get_parameters();
        if (!params_result.is_ok()) {
            return inference_lab::common::Err(std::move(params_result).unwrap_err());
        }
    }
    
    return inference_lab::common::Ok(std::monostate{});
}

auto ExpertParameters::update_expert_parameters(std::size_t expert_id, 
                                               const std::vector<float>& new_parameters)
    -> inference_lab::common::Result<std::monostate, MoEError> {
    
    if (expert_id >= config_.num_experts) {
        return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
    }
    
    if (new_parameters.size() != config_.parameters_per_expert) {
        return inference_lab::common::Err(MoEError::PARAMETER_STORAGE_ERROR);
    }
    
    auto handle_result = get_expert_handle(expert_id);
    if (!handle_result.is_ok()) {
        return inference_lab::common::Err(std::move(handle_result).unwrap_err());
    }
    
    auto handle = std::move(handle_result).unwrap();
    
    // Get current parameters (this will load them if necessary)
    auto params_result = handle->get_parameters();
    if (!params_result.is_ok()) {
        return inference_lab::common::Err(std::move(params_result).unwrap_err());
    }
    
    auto& current_params = std::move(params_result).unwrap();
    
    // Update parameters
    std::copy(new_parameters.begin(), new_parameters.end(), current_params.begin());
    
    // Store to persistent storage if needed
    auto storage_result = store_expert_to_storage(expert_id, new_parameters);
    if (!storage_result.is_ok()) {
        return inference_lab::common::Err(std::move(storage_result).unwrap_err());
    }
    
    return inference_lab::common::Ok(std::monostate{});
}

auto ExpertParameters::find_cache_entry(std::size_t expert_id) -> std::shared_ptr<ExpertParameterHandle> {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    for (auto& entry : parameter_cache_) {
        if (entry.expert_id == expert_id) {
            entry.last_access = std::chrono::steady_clock::now();
            entry.access_count.fetch_add(1);
            return entry.handle;
        }
    }
    
    return nullptr;
}

auto ExpertParameters::update_cache_entry(std::size_t expert_id, 
                                         std::shared_ptr<ExpertParameterHandle> handle) -> void {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // Check if entry already exists
    for (auto& entry : parameter_cache_) {
        if (entry.expert_id == expert_id) {
            entry.handle = handle;
            entry.last_access = std::chrono::steady_clock::now();
            entry.access_count.fetch_add(1);
            return;
        }
    }
    
    // Add new entry
    if (parameter_cache_.size() < config_.cache_size_experts) {
        CacheEntry new_entry;
        new_entry.expert_id = expert_id;
        new_entry.handle = handle;
        new_entry.last_access = std::chrono::steady_clock::now();
        new_entry.access_count.store(1);
        parameter_cache_.push_back(std::move(new_entry));
    } else {
        // Evict least recently used entry
        evict_least_recently_used();
        
        // Add new entry
        CacheEntry new_entry;
        new_entry.expert_id = expert_id;
        new_entry.handle = handle;
        new_entry.last_access = std::chrono::steady_clock::now();
        new_entry.access_count.store(1);
        parameter_cache_.push_back(std::move(new_entry));
    }
}

auto ExpertParameters::evict_least_recently_used() -> void {
    if (parameter_cache_.empty()) {
        return;
    }
    
    // Find least recently used entry
    auto oldest_it = parameter_cache_.begin();
    auto oldest_time = oldest_it->last_access;
    
    for (auto it = parameter_cache_.begin() + 1; it != parameter_cache_.end(); ++it) {
        if (it->last_access < oldest_time) {
            oldest_time = it->last_access;
            oldest_it = it;
        }
    }
    
    // Remove the oldest entry
    parameter_cache_.erase(oldest_it);
}

auto ExpertParameters::load_expert_from_storage(std::size_t expert_id)
    -> inference_lab::common::Result<std::vector<float>, MoEError> {
    
    // Simplified implementation - in practice would load from disk/database
    std::vector<float> parameters(config_.parameters_per_expert);
    
    // Check if we have compressed version
    {
        std::lock_guard<std::mutex> lock(compression_mutex_);
        auto it = compressed_parameters_.find(expert_id);
        if (it != compressed_parameters_.end()) {
            auto decompressed_result = decompress_parameter_vector(it->second);
            if (decompressed_result.is_ok()) {
                return decompressed_result;
            }
        }
    }
    
    // Load uncompressed (simulated with random values)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (auto& param : parameters) {
        param = dist(gen);
    }
    
    return inference_lab::common::Ok(std::move(parameters));
}

auto ExpertParameters::store_expert_to_storage(std::size_t expert_id, 
                                              const std::vector<float>& parameters)
    -> inference_lab::common::Result<std::monostate, MoEError> {
    
    // If compression is enabled, store compressed version
    if (config_.enable_parameter_compression) {
        auto compressed_result = compress_parameter_vector(parameters, config_.compression_ratio);
        if (compressed_result.is_ok()) {
            std::lock_guard<std::mutex> lock(compression_mutex_);
            compressed_parameters_[expert_id] = std::move(compressed_result).unwrap();
            compression_ratios_[expert_id] = config_.compression_ratio;
        }
    }
    
    // In practice, would also store to persistent storage
    return inference_lab::common::Ok(std::monostate{});
}

auto ExpertParameters::compress_parameter_vector(const std::vector<float>& parameters, float ratio)
    -> inference_lab::common::Result<std::vector<float>, MoEError> {
    
    if (ratio <= 0.0f || ratio >= 1.0f) {
        return inference_lab::common::Err(MoEError::PARAMETER_STORAGE_ERROR);
    }
    
    // Simplified compression: quantization
    std::vector<float> compressed;
    compressed.reserve(static_cast<std::size_t>(parameters.size() * ratio));
    
    // Find min/max values for quantization
    auto [min_it, max_it] = std::minmax_element(parameters.begin(), parameters.end());
    float min_val = *min_it;
    float max_val = *max_it;
    float range = max_val - min_val;
    
    if (range < 1e-10f) {
        // All values are essentially the same
        compressed.push_back(min_val);
        return inference_lab::common::Ok(std::move(compressed));
    }
    
    // Store compression metadata
    compressed.push_back(min_val);  // [0] = min value
    compressed.push_back(max_val);  // [1] = max value
    compressed.push_back(static_cast<float>(parameters.size()));  // [2] = original size
    
    // Quantize parameters (simplified 8-bit quantization simulation)
    std::size_t stride = static_cast<std::size_t>(1.0f / ratio);
    for (std::size_t i = 0; i < parameters.size(); i += stride) {
        float normalized = (parameters[i] - min_val) / range;
        float quantized = std::round(normalized * 255.0f) / 255.0f;  // 8-bit quantization
        compressed.push_back(quantized * range + min_val);
    }
    
    return inference_lab::common::Ok(std::move(compressed));
}

auto ExpertParameters::decompress_parameter_vector(const std::vector<float>& compressed_params)
    -> inference_lab::common::Result<std::vector<float>, MoEError> {
    
    if (compressed_params.size() < 3) {
        return inference_lab::common::Err(MoEError::PARAMETER_STORAGE_ERROR);
    }
    
    float min_val = compressed_params[0];
    float max_val = compressed_params[1];
    std::size_t original_size = static_cast<std::size_t>(compressed_params[2]);
    
    if (original_size != config_.parameters_per_expert) {
        return inference_lab::common::Err(MoEError::PARAMETER_STORAGE_ERROR);
    }
    
    std::vector<float> decompressed(original_size);
    
    // Simple decompression with interpolation
    std::size_t compressed_data_size = compressed_params.size() - 3;
    if (compressed_data_size == 1) {
        // All values were the same
        std::fill(decompressed.begin(), decompressed.end(), compressed_params[3]);
        return inference_lab::common::Ok(std::move(decompressed));
    }
    
    // Interpolate between compressed samples
    float stride = static_cast<float>(original_size) / static_cast<float>(compressed_data_size);
    
    for (std::size_t i = 0; i < original_size; ++i) {
        float compressed_index = static_cast<float>(i) / stride;
        std::size_t base_index = static_cast<std::size_t>(compressed_index);
        float fraction = compressed_index - static_cast<float>(base_index);
        
        if (base_index + 3 >= compressed_params.size()) {
            decompressed[i] = compressed_params.back();
        } else if (base_index + 4 >= compressed_params.size()) {
            decompressed[i] = compressed_params[base_index + 3];
        } else {
            // Linear interpolation
            float val1 = compressed_params[base_index + 3];
            float val2 = compressed_params[base_index + 4];
            decompressed[i] = val1 + fraction * (val2 - val1);
        }
    }
    
    return inference_lab::common::Ok(std::move(decompressed));
}

auto ExpertParameters::get_parameter_stats() const -> ParameterStats {
    ParameterStats stats;
    
    stats.total_parameters = config_.num_experts * config_.parameters_per_expert;
    
    // Count active (loaded) parameters
    std::size_t active_count = 0;
    std::size_t compressed_count = 0;
    
    for (const auto& handle : expert_handles_) {
        if (handle->is_loaded()) {
            active_count += handle->get_parameter_count();
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(compression_mutex_);
        compressed_count = compressed_parameters_.size() * config_.parameters_per_expert;
    }
    
    stats.active_parameters = active_count;
    stats.compressed_parameters = compressed_count;
    
    // Calculate memory usage
    stats.memory_usage_mb = calculate_memory_usage();
    
    // Compression efficiency
    if (stats.total_parameters > 0) {
        stats.compression_efficiency = static_cast<float>(compressed_count) / 
                                     static_cast<float>(stats.total_parameters);
    } else {
        stats.compression_efficiency = 0.0f;
    }
    
    // Expert memory usage (simplified)
    stats.expert_memory_usage.resize(config_.num_experts);
    float mem_per_expert = stats.memory_usage_mb / static_cast<float>(config_.num_experts);
    std::fill(stats.expert_memory_usage.begin(), stats.expert_memory_usage.end(), mem_per_expert);
    
    // Cache statistics
    stats.cache_hits = total_cache_hits_.load();
    stats.cache_misses = total_cache_misses_.load();
    
    return stats;
}

auto ExpertParameters::calculate_memory_usage() const -> float {
    // Calculate approximate memory usage in MB
    float total_memory = 0.0f;
    
    // Memory for loaded expert parameters
    std::size_t loaded_count = 0;
    for (const auto& handle : expert_handles_) {
        if (handle->is_loaded()) {
            loaded_count++;
        }
    }
    
    total_memory += static_cast<float>(loaded_count * config_.parameters_per_expert * sizeof(float)) / (1024.0f * 1024.0f);
    
    // Memory for compressed parameters
    {
        std::lock_guard<std::mutex> lock(compression_mutex_);
        for (const auto& [expert_id, compressed_params] : compressed_parameters_) {
            total_memory += static_cast<float>(compressed_params.size() * sizeof(float)) / (1024.0f * 1024.0f);
        }
    }
    
    // Cache memory
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        total_memory += static_cast<float>(parameter_cache_.size() * sizeof(CacheEntry)) / (1024.0f * 1024.0f);
    }
    
    total_memory_usage_mb_.store(total_memory);
    return total_memory;
}

auto ExpertParameters::compress_experts(const std::vector<std::size_t>& expert_ids)
    -> inference_lab::common::Result<std::monostate, MoEError> {
    
    if (!config_.enable_parameter_compression) {
        return inference_lab::common::Ok(std::monostate{});
    }
    
    for (auto expert_id : expert_ids) {
        if (expert_id >= config_.num_experts) {
            return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
        }
        
        auto handle_result = get_expert_handle(expert_id);
        if (!handle_result.is_ok()) {
            return inference_lab::common::Err(std::move(handle_result).unwrap_err());
        }
        
        auto handle = std::move(handle_result).unwrap();
        
        if (handle->is_loaded()) {
            auto params_result = handle->get_parameters();
            if (!params_result.is_ok()) {
                return inference_lab::common::Err(std::move(params_result).unwrap_err());
            }
            
            auto& params = std::move(params_result).unwrap();
            
            auto storage_result = store_expert_to_storage(expert_id, params);
            if (!storage_result.is_ok()) {
                return inference_lab::common::Err(std::move(storage_result).unwrap_err());
            }
        }
    }
    
    return inference_lab::common::Ok(std::monostate{});
}

auto ExpertParameters::cleanup_unused_experts(std::size_t keep_count)
    -> inference_lab::common::Result<std::monostate, MoEError> {
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    if (parameter_cache_.size() <= keep_count) {
        return inference_lab::common::Ok(std::monostate{});
    }
    
    // Sort cache entries by access count and last access time
    std::sort(parameter_cache_.begin(), parameter_cache_.end(),
             [](const CacheEntry& a, const CacheEntry& b) {
                 if (a.access_count.load() != b.access_count.load()) {
                     return a.access_count.load() > b.access_count.load();
                 }
                 return a.last_access > b.last_access;
             });
    
    // Keep only the most frequently/recently used entries
    parameter_cache_.resize(keep_count);
    
    return inference_lab::common::Ok(std::monostate{});
}

auto ExpertParameters::validate_storage_integrity() const -> inference_lab::common::Result<std::monostate, MoEError> {
    // Verify that all expert handles are valid
    for (std::size_t i = 0; i < expert_handles_.size(); ++i) {
        if (!expert_handles_[i]) {
            return inference_lab::common::Err(MoEError::PARAMETER_STORAGE_ERROR);
        }
        
        if (expert_handles_[i]->get_expert_id() != i) {
            return inference_lab::common::Err(MoEError::PARAMETER_STORAGE_ERROR);
        }
        
        if (expert_handles_[i]->get_parameter_count() != config_.parameters_per_expert) {
            return inference_lab::common::Err(MoEError::PARAMETER_STORAGE_ERROR);
        }
    }
    
    // Verify compression ratios are reasonable
    {
        std::lock_guard<std::mutex> lock(compression_mutex_);
        for (const auto& [expert_id, ratio] : compression_ratios_) {
            if (ratio <= 0.0f || ratio > 1.0f) {
                return inference_lab::common::Err(MoEError::PARAMETER_STORAGE_ERROR);
            }
        }
    }
    
    // Verify memory usage is within bounds
    float memory_usage = calculate_memory_usage();
    if (memory_usage > static_cast<float>(config_.memory_pool_size_mb) * 1.2f) {  // 20% tolerance
        return inference_lab::common::Err(MoEError::MEMORY_ALLOCATION_FAILURE);
    }
    
    return inference_lab::common::Ok(std::monostate{});
}

auto ExpertParameters::allocate_parameter_memory(std::size_t size) -> void* {
    // In practice, this would use the existing MemoryPool<T>
    return std::malloc(size);
}

auto ExpertParameters::deallocate_parameter_memory(void* ptr) -> void {
    // In practice, this would use the existing MemoryPool<T>
    std::free(ptr);
}

} // namespace engines::mixture_experts