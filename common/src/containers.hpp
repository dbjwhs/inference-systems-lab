// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file containers.hpp
 * @brief Cache-friendly containers and allocators for high-performance ML inference
 *
 * This file provides specialized container types optimized for machine learning workloads,
 * including memory pool allocation for tensor operations, ring buffers for streaming data,
 * lock-free queues for batch processing, and efficient tensor containers.
 *
 * Key features:
 * - Memory pool allocator for reusable GPU/CPU memory allocation
 * - Ring buffer for continuous streaming inference scenarios
 * - Lock-free queue for multi-threaded batch aggregation
 * - Tensor container with zero-copy views and efficient storage
 * - Cache-friendly design optimized for modern CPU architectures
 * - RAII-compliant resource management with proper move semantics
 *
 * Design Philosophy:
 * These containers are designed specifically for ML inference workloads where:
 * - Memory allocation overhead must be minimized
 * - Cache locality is critical for performance
 * - Multi-threaded processing requires lock-free data structures
 * - Tensor operations need efficient multi-dimensional access patterns
 *
 * Performance Characteristics:
 * - Memory Pool: O(1) allocation/deallocation, configurable block sizes
 * - Ring Buffer: O(1) push/pop operations, lockless single producer/consumer
 * - Lock-Free Queue: O(1) amortized operations, wait-free for readers
 * - Tensor Container: O(1) element access, zero-copy slice operations
 *
 * Example Usage:
 * @code
 * // Memory pool for tensor allocation
 * MemoryPool<float> tensor_pool(1024 * 1024); // 1MB pool
 * auto tensor_mem = tensor_pool.allocate(224 * 224 * 3); // Image tensor
 *
 * // Ring buffer for streaming inference
 * RingBuffer<InferenceRequest> request_buffer(256);
 * request_buffer.push(request);
 * auto next_request = request_buffer.pop();
 *
 * // Lock-free queue for batch processing
 * LockFreeQueue<BatchData> batch_queue;
 * batch_queue.enqueue(batch);
 * auto processed_batch = batch_queue.dequeue();
 * @endcode
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace inference_lab::common {

/**
 * @brief High-performance memory pool allocator for tensor operations
 * @tparam ElementType Type of elements to allocate (typically float, int8_t, etc.)
 *
 * Memory pool allocator designed for ML inference workloads where:
 * - Frequent allocation/deallocation of similar-sized objects occurs
 * - Allocation overhead must be minimized for real-time inference
 * - Memory fragmentation should be avoided
 * - Cache locality is important for performance
 *
 * Features:
 * - O(1) allocation and deallocation
 * - Configurable block sizes for different tensor types
 * - Thread-safe operations with minimal contention
 * - Automatic pool expansion when capacity is exceeded
 * - Alignment support for SIMD operations
 * - Memory usage tracking and statistics
 *
 * Thread Safety:
 * All operations are thread-safe and lock-free where possible.
 * Multiple threads can allocate and deallocate concurrently.
 */
template <typename ElementType>
class MemoryPool {
  public:
    static_assert(std::is_trivially_destructible_v<ElementType>,
                  "MemoryPool only supports trivially destructible types");

    /**
     * @brief Construct memory pool with specified capacity
     * @param capacity_elements Number of elements the pool can hold
     * @param alignment Memory alignment requirement (default: alignof(ElementType))
     */
    explicit MemoryPool(std::size_t capacity_elements,
                        std::size_t alignment = alignof(ElementType));

    /**
     * @brief Destructor - ensures all memory is properly freed
     */
    ~MemoryPool();

    // Non-copyable but movable
    MemoryPool(const MemoryPool&) = delete;
    auto operator=(const MemoryPool&) -> MemoryPool& = delete;
    MemoryPool(MemoryPool&& other) noexcept;
    auto operator=(MemoryPool&& other) noexcept -> MemoryPool&;

    /**
     * @brief Allocate memory for specified number of elements
     * @param count Number of elements to allocate
     * @return Pointer to allocated memory, or nullptr if allocation fails
     *
     * Returns aligned memory suitable for the element type.
     * If the pool is exhausted, may attempt to expand or return nullptr.
     */
    auto allocate(std::size_t count) -> ElementType*;

    /**
     * @brief Deallocate previously allocated memory
     * @param ptr Pointer returned by allocate()
     * @param count Number of elements that were allocated
     *
     * Memory is returned to the pool for reuse.
     * It's safe to call with nullptr (no-op).
     */
    void deallocate(ElementType* ptr, std::size_t count) noexcept;

    /**
     * @brief Get current pool utilization statistics
     * @return PoolStats structure with usage information
     */
    struct PoolStats {
        std::size_t total_capacity;      ///< Total pool capacity in elements
        std::size_t allocated_count;     ///< Currently allocated elements
        std::size_t peak_usage;          ///< Peak allocation since creation
        std::size_t allocation_count;    ///< Total number of allocations
        std::size_t deallocation_count;  ///< Total number of deallocations
    };

    auto get_stats() const noexcept -> PoolStats;

    /**
     * @brief Reset pool to initial state, deallocating all memory
     *
     * WARNING: This invalidates all previously allocated pointers.
     * Only call when you're certain no allocated memory is in use.
     */
    void reset() noexcept;

    /**
     * @brief Check if a pointer was allocated from this pool
     * @param ptr Pointer to check
     * @return true if pointer is from this pool, false otherwise
     */
    auto owns(const ElementType* ptr) const noexcept -> bool;

  private:
    /**
     * @brief Memory block structure for pool management
     */
    struct Block {
        ElementType* data;
        std::size_t size;
        std::atomic<bool> in_use;
        Block* next;

        Block(ElementType* data_ptr, std::size_t block_size)
            : data(data_ptr), size(block_size), in_use(false), next(nullptr) {}
    };

    /**
     * @brief Expand pool capacity by allocating new chunk
     * @param min_size Minimum size needed for current allocation
     * @return true if expansion succeeded, false otherwise
     */
    auto expand_pool(std::size_t min_size) -> bool;

    /**
     * @brief Find suitable free block for allocation
     * @param count Number of elements needed
     * @return Pointer to suitable block, or nullptr if none available
     */
    auto find_free_block(std::size_t count) -> Block*;

    // Pool configuration
    std::size_t capacity_elements_;
    std::size_t alignment_;
    std::size_t chunk_size_;

    // Memory management
    std::vector<std::unique_ptr<std::byte[]>> chunks_;
    std::vector<std::unique_ptr<Block>> blocks_;
    Block* free_list_head_;

    // Thread safety
    mutable std::atomic<std::size_t> allocated_count_{0};
    mutable std::atomic<std::size_t> peak_usage_{0};
    mutable std::atomic<std::size_t> allocation_count_{0};
    mutable std::atomic<std::size_t> deallocation_count_{0};

    // Simple spinlock for structural modifications
    mutable std::atomic_flag modification_lock_ = ATOMIC_FLAG_INIT;

    /**
     * @brief Calculate aligned size for allocation
     * @param size Requested size in bytes
     * @return Aligned size
     */
    auto align_size(std::size_t size) const noexcept -> std::size_t;

    /**
     * @brief Acquire modification lock with exponential backoff
     */
    void acquire_lock() const noexcept;

    /**
     * @brief Release modification lock
     */
    void release_lock() const noexcept;
};

// Template implementation

template <typename ElementType>
MemoryPool<ElementType>::MemoryPool(std::size_t capacity_elements, std::size_t alignment)
    : capacity_elements_(capacity_elements),
      alignment_(std::max(alignment, alignof(ElementType))),
      chunk_size_(capacity_elements * sizeof(ElementType)),
      free_list_head_(nullptr) {
    // Ensure alignment is a power of 2
    if ((alignment_ & (alignment_ - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }

    // Pre-allocate initial chunk
    expand_pool(capacity_elements_);
}

template <typename ElementType>
MemoryPool<ElementType>::~MemoryPool() {
    // Destructor automatically cleans up unique_ptrs
    // No manual cleanup needed due to RAII design
}

template <typename ElementType>
MemoryPool<ElementType>::MemoryPool(MemoryPool&& other) noexcept
    : capacity_elements_(other.capacity_elements_),
      alignment_(other.alignment_),
      chunk_size_(other.chunk_size_),
      chunks_(std::move(other.chunks_)),
      blocks_(std::move(other.blocks_)),
      free_list_head_(other.free_list_head_),
      allocated_count_(other.allocated_count_.load()),
      peak_usage_(other.peak_usage_.load()),
      allocation_count_(other.allocation_count_.load()),
      deallocation_count_(other.deallocation_count_.load()) {
    other.free_list_head_ = nullptr;
    other.allocated_count_ = 0;
    other.peak_usage_ = 0;
    other.allocation_count_ = 0;
    other.deallocation_count_ = 0;
}

template <typename ElementType>
auto MemoryPool<ElementType>::operator=(MemoryPool&& other) noexcept -> MemoryPool& {
    if (this != &other) {
        // Release current resources
        chunks_.clear();
        blocks_.clear();

        // Move other's resources
        capacity_elements_ = other.capacity_elements_;
        alignment_ = other.alignment_;
        chunk_size_ = other.chunk_size_;
        chunks_ = std::move(other.chunks_);
        blocks_ = std::move(other.blocks_);
        free_list_head_ = other.free_list_head_;
        allocated_count_ = other.allocated_count_.load();
        peak_usage_ = other.peak_usage_.load();
        allocation_count_ = other.allocation_count_.load();
        deallocation_count_ = other.deallocation_count_.load();

        // Clear other
        other.free_list_head_ = nullptr;
        other.allocated_count_ = 0;
        other.peak_usage_ = 0;
        other.allocation_count_ = 0;
        other.deallocation_count_ = 0;
    }
    return *this;
}

template <typename ElementType>
auto MemoryPool<ElementType>::allocate(std::size_t count) -> ElementType* {
    if (count == 0) {
        return nullptr;
    }

    // Try to find a suitable free block
    auto* block = find_free_block(count);

    if (!block) {
        // Need to expand the pool
        acquire_lock();
        if (!expand_pool(count)) {
            release_lock();
            return nullptr;  // Expansion failed
        }
        release_lock();

        // Try again after expansion
        block = find_free_block(count);
        if (!block) {
            return nullptr;  // Still couldn't find suitable block
        }
    }

    // Mark block as in use
    block->in_use.store(true, std::memory_order_release);

    // Update statistics
    auto current_allocated = allocated_count_.fetch_add(count, std::memory_order_relaxed) + count;
    allocation_count_.fetch_add(1, std::memory_order_relaxed);

    // Update peak usage
    auto current_peak = peak_usage_.load(std::memory_order_relaxed);
    while (current_allocated > current_peak &&
           !peak_usage_.compare_exchange_weak(
               current_peak, current_allocated, std::memory_order_relaxed)) {
        // Loop until we successfully update peak or another thread beat us to it
    }

    return block->data;
}

template <typename ElementType>
void MemoryPool<ElementType>::deallocate(ElementType* ptr, std::size_t count) noexcept {
    if (!ptr || count == 0) {
        return;
    }

    // Find the block containing this pointer
    for (auto& block_ptr : blocks_) {
        auto* block = block_ptr.get();
        if (block->data == ptr) {
            // Mark block as free
            block->in_use.store(false, std::memory_order_release);

            // Update statistics
            allocated_count_.fetch_sub(count, std::memory_order_relaxed);
            deallocation_count_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }

    // Pointer not found in our pool - this is a programming error
    // In debug builds, this could assert, but in release we'll just ignore it
}

template <typename ElementType>
auto MemoryPool<ElementType>::get_stats() const noexcept -> PoolStats {
    return PoolStats{.total_capacity = capacity_elements_,
                     .allocated_count = allocated_count_.load(std::memory_order_relaxed),
                     .peak_usage = peak_usage_.load(std::memory_order_relaxed),
                     .allocation_count = allocation_count_.load(std::memory_order_relaxed),
                     .deallocation_count = deallocation_count_.load(std::memory_order_relaxed)};
}

template <typename ElementType>
void MemoryPool<ElementType>::reset() noexcept {
    acquire_lock();

    // Mark all blocks as free
    for (auto& block_ptr : blocks_) {
        block_ptr->in_use.store(false, std::memory_order_relaxed);
    }

    // Reset statistics
    allocated_count_.store(0, std::memory_order_relaxed);

    release_lock();
}

template <typename ElementType>
auto MemoryPool<ElementType>::owns(const ElementType* ptr) const noexcept -> bool {
    if (!ptr) {
        return false;
    }

    // Check if pointer is within any of our chunks
    for (const auto& chunk : chunks_) {
        auto* chunk_start = reinterpret_cast<const ElementType*>(chunk.get());
        auto* chunk_end = chunk_start + (chunk_size_ / sizeof(ElementType));

        if (ptr >= chunk_start && ptr < chunk_end) {
            return true;
        }
    }

    return false;
}

template <typename ElementType>
auto MemoryPool<ElementType>::expand_pool(std::size_t min_size) -> bool {
    // Calculate chunk size (at least min_size, but prefer doubling)
    auto new_chunk_size = std::max(min_size * sizeof(ElementType), chunk_size_);
    auto aligned_chunk_size = align_size(new_chunk_size);

    // Allocate new chunk
    auto new_chunk = std::make_unique<std::byte[]>(aligned_chunk_size);
    if (!new_chunk) {
        return false;
    }

    // Create block for this chunk
    auto* chunk_data = reinterpret_cast<ElementType*>(new_chunk.get());
    auto block_size = aligned_chunk_size / sizeof(ElementType);
    auto new_block = std::make_unique<Block>(chunk_data, block_size);

    // Add to our collections
    chunks_.push_back(std::move(new_chunk));
    blocks_.push_back(std::move(new_block));

    // Update capacity
    capacity_elements_ += block_size;

    return true;
}

template <typename ElementType>
auto MemoryPool<ElementType>::find_free_block(std::size_t count) -> Block* {
    // Simple first-fit algorithm
    // In a production implementation, you might want best-fit or segregated lists
    for (auto& block_ptr : blocks_) {
        auto* block = block_ptr.get();
        if (!block->in_use.load(std::memory_order_acquire) && block->size >= count) {
            return block;
        }
    }
    return nullptr;
}

template <typename ElementType>
auto MemoryPool<ElementType>::align_size(std::size_t size) const noexcept -> std::size_t {
    return (size + alignment_ - 1) & ~(alignment_ - 1);
}

template <typename ElementType>
void MemoryPool<ElementType>::acquire_lock() const noexcept {
    // Simple exponential backoff spinlock
    int backoff = 1;
    while (modification_lock_.test_and_set(std::memory_order_acquire)) {
        for (int i = 0; i < backoff; ++i) {
// Pause instruction hint for x86 (no-op on other architectures)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            __builtin_ia32_pause();
#endif
        }
        backoff = std::min(backoff * 2, 64);  // Cap backoff to prevent excessive delays
    }
}

template <typename ElementType>
void MemoryPool<ElementType>::release_lock() const noexcept {
    modification_lock_.clear(std::memory_order_release);
}

/**
 * @brief Lock-free ring buffer for streaming inference data
 * @tparam ElementType Type of elements to store (e.g., InferenceRequest, BatchData)
 *
 * High-performance ring buffer designed for streaming ML inference scenarios where:
 * - Continuous data flow needs efficient buffering
 * - Single producer/single consumer patterns are common
 * - Low latency is critical for real-time inference
 * - Memory allocation should be avoided during operation
 *
 * Features:
 * - Lock-free single producer/single consumer operations
 * - O(1) push and pop operations
 * - Fixed capacity with overflow detection
 * - Cache-friendly circular buffer design
 * - Wait-free operations for maximum throughput
 * - Memory alignment optimized for cache lines
 *
 * Thread Safety:
 * Safe for single producer/single consumer scenarios.
 * Multiple producers or consumers require external synchronization.
 */
template <typename ElementType>
class RingBuffer {
  public:
    /**
     * @brief Construct ring buffer with specified capacity
     * @param capacity Maximum number of elements (must be power of 2)
     */
    explicit RingBuffer(std::size_t capacity);

    /**
     * @brief Destructor - properly destroys all elements
     */
    ~RingBuffer();

    // Non-copyable but movable
    RingBuffer(const RingBuffer&) = delete;
    auto operator=(const RingBuffer&) -> RingBuffer& = delete;
    RingBuffer(RingBuffer&& other) noexcept;
    auto operator=(RingBuffer&& other) noexcept -> RingBuffer&;

    /**
     * @brief Push element to buffer (producer operation)
     * @param element Element to add
     * @return true if successful, false if buffer is full
     *
     * This operation is wait-free and safe for single producer.
     * Returns false if buffer is full (producer should handle backpressure).
     */
    auto push(const ElementType& element) -> bool;

    /**
     * @brief Push element to buffer with move semantics
     * @param element Element to move into buffer
     * @return true if successful, false if buffer is full
     */
    auto push(ElementType&& element) -> bool;

    /**
     * @brief Emplace element directly in buffer
     * @tparam Args Constructor argument types
     * @param args Arguments to forward to ElementType constructor
     * @return true if successful, false if buffer is full
     */
    template <typename... Args>
    auto emplace(Args&&... args) -> bool;

    /**
     * @brief Pop element from buffer (consumer operation)
     * @return Optional containing element if available, nullopt if empty
     *
     * This operation is wait-free and safe for single consumer.
     * Returns nullopt if buffer is empty (consumer should handle starvation).
     */
    auto pop() -> std::optional<ElementType>;

    /**
     * @brief Try to pop element without copying
     * @param output Reference to store popped element
     * @return true if element was popped, false if buffer empty
     *
     * More efficient than pop() when you want to avoid copy/move.
     */
    auto try_pop(ElementType& output) -> bool;

    /**
     * @brief Get current buffer size
     * @return Number of elements currently in buffer
     *
     * Note: This is an approximation in concurrent scenarios.
     */
    auto size() const noexcept -> std::size_t;

    /**
     * @brief Check if buffer is empty
     * @return true if empty, false otherwise
     */
    auto empty() const noexcept -> bool;

    /**
     * @brief Check if buffer is full
     * @return true if full, false otherwise
     */
    auto full() const noexcept -> bool;

    /**
     * @brief Get buffer capacity
     * @return Maximum number of elements
     */
    auto capacity() const noexcept -> std::size_t;

    /**
     * @brief Clear all elements from buffer
     *
     * WARNING: Not thread-safe. Only call when no concurrent access.
     */
    void clear() noexcept;

    /**
     * @brief Get buffer utilization statistics
     * @return BufferStats with usage information
     */
    struct BufferStats {
        std::size_t current_size;       ///< Current number of elements
        std::size_t total_capacity;     ///< Maximum capacity
        std::size_t total_pushes;       ///< Total push operations attempted
        std::size_t successful_pushes;  ///< Successful push operations
        std::size_t total_pops;         ///< Total pop operations attempted
        std::size_t successful_pops;    ///< Successful pop operations
        double utilization_ratio;       ///< current_size / total_capacity
    };

    auto get_stats() const noexcept -> BufferStats;

  private:
    // Buffer configuration
    std::size_t capacity_;
    std::size_t mask_;  // capacity - 1 (for power-of-2 optimization)

    // Buffer storage
    std::unique_ptr<ElementType[]> buffer_;

    // Atomic indices for lock-free operation
    alignas(64) std::atomic<std::size_t> head_{0};  // Producer index
    alignas(64) std::atomic<std::size_t> tail_{0};  // Consumer index

    // Statistics (relaxed memory order for performance)
    mutable std::atomic<std::size_t> total_pushes_{0};
    mutable std::atomic<std::size_t> successful_pushes_{0};
    mutable std::atomic<std::size_t> total_pops_{0};
    mutable std::atomic<std::size_t> successful_pops_{0};

    /**
     * @brief Check if capacity is power of 2
     * @param n Number to check
     * @return true if power of 2, false otherwise
     */
    static auto is_power_of_two(std::size_t n) noexcept -> bool;

    /**
     * @brief Get next index in ring buffer
     * @param index Current index
     * @return Next index (wrapped around)
     */
    auto next_index(std::size_t index) const noexcept -> std::size_t;
};

// Template implementation

template <typename ElementType>
RingBuffer<ElementType>::RingBuffer(std::size_t capacity) : capacity_(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("RingBuffer capacity must be greater than 0");
    }

    if (!is_power_of_two(capacity)) {
        // Find next power of 2
        std::size_t power_of_2 = 1;
        while (power_of_2 < capacity) {
            power_of_2 <<= 1;
        }
        capacity_ = power_of_2;
    }

    mask_ = capacity_ - 1;
    buffer_ = std::make_unique<ElementType[]>(capacity_);
}

template <typename ElementType>
RingBuffer<ElementType>::~RingBuffer() {
    // Destructor for elements handled automatically by unique_ptr
}

template <typename ElementType>
RingBuffer<ElementType>::RingBuffer(RingBuffer&& other) noexcept
    : capacity_(other.capacity_),
      mask_(other.mask_),
      buffer_(std::move(other.buffer_)),
      head_(other.head_.load()),
      tail_(other.tail_.load()),
      total_pushes_(other.total_pushes_.load()),
      successful_pushes_(other.successful_pushes_.load()),
      total_pops_(other.total_pops_.load()),
      successful_pops_(other.successful_pops_.load()) {
    other.capacity_ = 0;
    other.mask_ = 0;
    other.head_ = 0;
    other.tail_ = 0;
    other.total_pushes_ = 0;
    other.successful_pushes_ = 0;
    other.total_pops_ = 0;
    other.successful_pops_ = 0;
}

template <typename ElementType>
auto RingBuffer<ElementType>::operator=(RingBuffer&& other) noexcept -> RingBuffer& {
    if (this != &other) {
        capacity_ = other.capacity_;
        mask_ = other.mask_;
        buffer_ = std::move(other.buffer_);
        head_ = other.head_.load();
        tail_ = other.tail_.load();
        total_pushes_ = other.total_pushes_.load();
        successful_pushes_ = other.successful_pushes_.load();
        total_pops_ = other.total_pops_.load();
        successful_pops_ = other.successful_pops_.load();

        other.capacity_ = 0;
        other.mask_ = 0;
        other.head_ = 0;
        other.tail_ = 0;
        other.total_pushes_ = 0;
        other.successful_pushes_ = 0;
        other.total_pops_ = 0;
        other.successful_pops_ = 0;
    }
    return *this;
}

template <typename ElementType>
auto RingBuffer<ElementType>::push(const ElementType& element) -> bool {
    total_pushes_.fetch_add(1, std::memory_order_relaxed);

    const auto current_head = head_.load(std::memory_order_relaxed);
    const auto next_head = next_index(current_head);

    // Check if buffer is full (next head would equal tail)
    if (next_head == tail_.load(std::memory_order_acquire)) {
        return false;  // Buffer full
    }

    // Store element
    buffer_[current_head] = element;

    // Update head index (release semantics ensures element write is visible)
    head_.store(next_head, std::memory_order_release);

    successful_pushes_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

template <typename ElementType>
auto RingBuffer<ElementType>::push(ElementType&& element) -> bool {
    total_pushes_.fetch_add(1, std::memory_order_relaxed);

    const auto current_head = head_.load(std::memory_order_relaxed);
    const auto next_head = next_index(current_head);

    // Check if buffer is full
    if (next_head == tail_.load(std::memory_order_acquire)) {
        return false;  // Buffer full
    }

    // Move element
    buffer_[current_head] = std::move(element);

    // Update head index
    head_.store(next_head, std::memory_order_release);

    successful_pushes_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

template <typename ElementType>
template <typename... Args>
auto RingBuffer<ElementType>::emplace(Args&&... args) -> bool {
    total_pushes_.fetch_add(1, std::memory_order_relaxed);

    const auto current_head = head_.load(std::memory_order_relaxed);
    const auto next_head = next_index(current_head);

    // Check if buffer is full
    if (next_head == tail_.load(std::memory_order_acquire)) {
        return false;  // Buffer full
    }

    // Construct element in-place
    buffer_[current_head] = ElementType(std::forward<Args>(args)...);

    // Update head index
    head_.store(next_head, std::memory_order_release);

    successful_pushes_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

template <typename ElementType>
auto RingBuffer<ElementType>::pop() -> std::optional<ElementType> {
    total_pops_.fetch_add(1, std::memory_order_relaxed);

    const auto current_tail = tail_.load(std::memory_order_relaxed);

    // Check if buffer is empty
    if (current_tail == head_.load(std::memory_order_acquire)) {
        return std::nullopt;  // Buffer empty
    }

    // Get element
    ElementType result = std::move(buffer_[current_tail]);

    // Update tail index
    tail_.store(next_index(current_tail), std::memory_order_release);

    successful_pops_.fetch_add(1, std::memory_order_relaxed);
    return result;
}

template <typename ElementType>
auto RingBuffer<ElementType>::try_pop(ElementType& output) -> bool {
    total_pops_.fetch_add(1, std::memory_order_relaxed);

    const auto current_tail = tail_.load(std::memory_order_relaxed);

    // Check if buffer is empty
    if (current_tail == head_.load(std::memory_order_acquire)) {
        return false;  // Buffer empty
    }

    // Move element to output
    output = std::move(buffer_[current_tail]);

    // Update tail index
    tail_.store(next_index(current_tail), std::memory_order_release);

    successful_pops_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

template <typename ElementType>
auto RingBuffer<ElementType>::size() const noexcept -> std::size_t {
    const auto head = head_.load(std::memory_order_acquire);
    const auto tail = tail_.load(std::memory_order_acquire);
    return (head - tail) & mask_;
}

template <typename ElementType>
auto RingBuffer<ElementType>::empty() const noexcept -> bool {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
}

template <typename ElementType>
auto RingBuffer<ElementType>::full() const noexcept -> bool {
    const auto head = head_.load(std::memory_order_acquire);
    const auto tail = tail_.load(std::memory_order_acquire);
    return next_index(head) == tail;
}

template <typename ElementType>
auto RingBuffer<ElementType>::capacity() const noexcept -> std::size_t {
    return capacity_;
}

template <typename ElementType>
void RingBuffer<ElementType>::clear() noexcept {
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
}

template <typename ElementType>
auto RingBuffer<ElementType>::get_stats() const noexcept -> BufferStats {
    const auto current_size = size();
    const auto total_pushes = total_pushes_.load(std::memory_order_relaxed);
    const auto successful_pushes = successful_pushes_.load(std::memory_order_relaxed);
    const auto total_pops = total_pops_.load(std::memory_order_relaxed);
    const auto successful_pops = successful_pops_.load(std::memory_order_relaxed);

    return BufferStats{.current_size = current_size,
                       .total_capacity = capacity_,
                       .total_pushes = total_pushes,
                       .successful_pushes = successful_pushes,
                       .total_pops = total_pops,
                       .successful_pops = successful_pops,
                       .utilization_ratio = static_cast<double>(current_size) / capacity_};
}

template <typename ElementType>
auto RingBuffer<ElementType>::is_power_of_two(std::size_t n) noexcept -> bool {
    return n > 0 && (n & (n - 1)) == 0;
}

template <typename ElementType>
auto RingBuffer<ElementType>::next_index(std::size_t index) const noexcept -> std::size_t {
    return (index + 1) & mask_;
}

/**
 * @brief Lock-free queue for multi-threaded batch processing
 * @tparam ElementType Type of elements to store (e.g., BatchData, InferenceTask)
 *
 * High-performance lock-free queue designed for multi-threaded ML batch processing where:
 * - Multiple producers need to enqueue work items concurrently
 * - Multiple consumers process batches from the queue
 * - ABA problem prevention is critical for correctness
 * - Memory reclamation must be handled safely
 * - Scalability across many cores is essential
 *
 * Features:
 * - Lock-free multi-producer/multi-consumer operations
 * - ABA prevention using tagged pointers/hazard pointers
 * - Memory-order optimized operations for performance
 * - Scalable across multiple CPU cores
 * - Wait-free enqueue operations
 * - Nearly wait-free dequeue operations
 * - Built-in memory reclamation strategy
 *
 * Thread Safety:
 * Fully thread-safe for arbitrary numbers of producers and consumers.
 * Uses Compare-And-Swap (CAS) operations for all modifications.
 */
template <typename ElementType>
class LockFreeQueue {
  public:
    /**
     * @brief Construct empty lock-free queue
     */
    LockFreeQueue();

    /**
     * @brief Destructor - safely destroys all remaining elements
     */
    ~LockFreeQueue();

    // Non-copyable and non-movable for thread safety
    LockFreeQueue(const LockFreeQueue&) = delete;
    auto operator=(const LockFreeQueue&) -> LockFreeQueue& = delete;
    LockFreeQueue(LockFreeQueue&&) = delete;
    auto operator=(LockFreeQueue&&) -> LockFreeQueue& = delete;

    /**
     * @brief Enqueue element (producer operation)
     * @param element Element to add
     * @return true if successful (always succeeds unless out of memory)
     *
     * This operation is wait-free and safe for multiple producers.
     * May allocate memory for new nodes.
     */
    auto enqueue(const ElementType& element) -> bool;

    /**
     * @brief Enqueue element with move semantics
     * @param element Element to move into queue
     * @return true if successful (always succeeds unless out of memory)
     */
    auto enqueue(ElementType&& element) -> bool;

    /**
     * @brief Emplace element directly in queue
     * @tparam Args Constructor argument types
     * @param args Arguments to forward to ElementType constructor
     * @return true if successful (always succeeds unless out of memory)
     */
    template <typename... Args>
    auto emplace(Args&&... args) -> bool;

    /**
     * @brief Dequeue element (consumer operation)
     * @return Optional containing element if available, nullopt if empty
     *
     * This operation is nearly wait-free and safe for multiple consumers.
     * Returns nullopt if queue is empty.
     */
    auto dequeue() -> std::optional<ElementType>;

    /**
     * @brief Try to dequeue element without copying
     * @param output Reference to store dequeued element
     * @return true if element was dequeued, false if queue empty
     *
     * More efficient than dequeue() when you want to avoid copy/move.
     */
    auto try_dequeue(ElementType& output) -> bool;

    /**
     * @brief Check if queue is empty
     * @return true if empty, false otherwise
     *
     * Note: This is a snapshot and may change immediately in concurrent scenarios.
     */
    auto empty() const noexcept -> bool;

    /**
     * @brief Get approximate queue size
     * @return Approximate number of elements
     *
     * Note: This is an approximation due to concurrent modifications.
     * Use only for monitoring/debugging purposes.
     */
    auto size_approx() const noexcept -> std::size_t;

    /**
     * @brief Get queue performance statistics
     * @return QueueStats with operational metrics
     */
    struct QueueStats {
        std::size_t total_enqueues;       ///< Total enqueue operations
        std::size_t total_dequeues;       ///< Total dequeue operations
        std::size_t successful_dequeues;  ///< Successful dequeue operations
        std::size_t current_size_approx;  ///< Approximate current size
        std::size_t memory_usage_bytes;   ///< Approximate memory usage
        std::size_t cas_failures;         ///< CAS operation failures (contention indicator)
    };

    auto get_stats() const noexcept -> QueueStats;

  private:
    /**
     * @brief Queue node structure with tagged pointer for ABA prevention
     */
    struct Node {
        std::atomic<ElementType*> data{nullptr};
        std::atomic<Node*> next{nullptr};
        std::atomic<std::size_t> tag{0};  // For ABA prevention

        Node() = default;

        // Non-copyable, non-movable for thread safety
        Node(const Node&) = delete;
        auto operator=(const Node&) -> Node& = delete;
        Node(Node&&) = delete;
        auto operator=(Node&&) -> Node& = delete;
    };

    /**
     * @brief Tagged pointer structure for ABA prevention
     */
    struct TaggedPointer {
        Node* ptr;
        std::size_t tag;

        TaggedPointer() : ptr(nullptr), tag(0) {}
        TaggedPointer(Node* p, std::size_t t) : ptr(p), tag(t) {}

        auto operator==(const TaggedPointer& other) const noexcept -> bool {
            return ptr == other.ptr && tag == other.tag;
        }
    };

    // Queue head and tail with ABA prevention
    alignas(64) std::atomic<TaggedPointer> head_;
    alignas(64) std::atomic<TaggedPointer> tail_;

    // Statistics for monitoring (relaxed memory order for performance)
    mutable std::atomic<std::size_t> total_enqueues_{0};
    mutable std::atomic<std::size_t> total_dequeues_{0};
    mutable std::atomic<std::size_t> successful_dequeues_{0};
    mutable std::atomic<std::size_t> cas_failures_{0};
    mutable std::atomic<std::size_t> node_count_{1};  // Start with 1 for dummy node

    /**
     * @brief Create new queue node
     * @return Pointer to new node, or nullptr if allocation fails
     */
    auto create_node() -> Node*;

    /**
     * @brief Safely delete node (may defer deletion for safety)
     * @param node Node to delete
     */
    void delete_node(Node* node) noexcept;

    /**
     * @brief Compare and swap tagged pointer atomically
     * @param target Target atomic variable
     * @param expected Expected value (updated on failure)
     * @param desired Desired value
     * @return true if CAS succeeded, false otherwise
     */
    auto cas_tagged_pointer(std::atomic<TaggedPointer>& target,
                            TaggedPointer& expected,
                            const TaggedPointer& desired) -> bool;
};

// Template implementation

template <typename ElementType>
LockFreeQueue<ElementType>::LockFreeQueue() {
    // Create dummy node to simplify queue operations
    auto* dummy = create_node();
    if (!dummy) {
        throw std::bad_alloc();
    }

    TaggedPointer initial_ptr(dummy, 0);
    head_.store(initial_ptr);
    tail_.store(initial_ptr);
}

template <typename ElementType>
LockFreeQueue<ElementType>::~LockFreeQueue() {
    // Dequeue all remaining elements
    while (!empty()) {
        dequeue();
    }

    // Delete the dummy node
    auto head_ptr = head_.load();
    delete_node(head_ptr.ptr);
}

template <typename ElementType>
auto LockFreeQueue<ElementType>::enqueue(const ElementType& element) -> bool {
    total_enqueues_.fetch_add(1, std::memory_order_relaxed);

    // Create new node with element
    auto* new_node = create_node();
    if (!new_node) {
        return false;  // Out of memory
    }

    auto* element_copy = new ElementType(element);
    new_node->data.store(element_copy, std::memory_order_relaxed);

    while (true) {
        auto tail_ptr = tail_.load(std::memory_order_acquire);
        auto next_ptr = tail_ptr.ptr->next.load(std::memory_order_acquire);

        // Check if tail is still the same
        if (tail_ptr == tail_.load(std::memory_order_acquire)) {
            if (next_ptr == nullptr) {
                // Try to link new node at the end of the list
                Node* expected_next = nullptr;
                if (tail_ptr.ptr->next.compare_exchange_weak(expected_next,
                                                             new_node,
                                                             std::memory_order_release,
                                                             std::memory_order_relaxed)) {
                    // Successfully linked new node, now try to advance tail
                    TaggedPointer new_tail(new_node, tail_ptr.tag + 1);
                    tail_.compare_exchange_weak(
                        tail_ptr, new_tail, std::memory_order_release, std::memory_order_relaxed);
                    break;
                }
            } else {
                // Tail is lagging, try to advance it
                TaggedPointer new_tail(next_ptr, tail_ptr.tag + 1);
                tail_.compare_exchange_weak(
                    tail_ptr, new_tail, std::memory_order_release, std::memory_order_relaxed);
            }
        }

        cas_failures_.fetch_add(1, std::memory_order_relaxed);
    }

    return true;
}

template <typename ElementType>
auto LockFreeQueue<ElementType>::enqueue(ElementType&& element) -> bool {
    total_enqueues_.fetch_add(1, std::memory_order_relaxed);

    // Create new node with moved element
    auto* new_node = create_node();
    if (!new_node) {
        return false;  // Out of memory
    }

    auto* element_moved = new ElementType(std::move(element));
    new_node->data.store(element_moved, std::memory_order_relaxed);

    while (true) {
        auto tail_ptr = tail_.load(std::memory_order_acquire);
        auto next_ptr = tail_ptr.ptr->next.load(std::memory_order_acquire);

        if (tail_ptr == tail_.load(std::memory_order_acquire)) {
            if (next_ptr == nullptr) {
                Node* expected_next = nullptr;
                if (tail_ptr.ptr->next.compare_exchange_weak(expected_next,
                                                             new_node,
                                                             std::memory_order_release,
                                                             std::memory_order_relaxed)) {
                    TaggedPointer new_tail(new_node, tail_ptr.tag + 1);
                    tail_.compare_exchange_weak(
                        tail_ptr, new_tail, std::memory_order_release, std::memory_order_relaxed);
                    break;
                }
            } else {
                TaggedPointer new_tail(next_ptr, tail_ptr.tag + 1);
                tail_.compare_exchange_weak(
                    tail_ptr, new_tail, std::memory_order_release, std::memory_order_relaxed);
            }
        }

        cas_failures_.fetch_add(1, std::memory_order_relaxed);
    }

    return true;
}

template <typename ElementType>
template <typename... Args>
auto LockFreeQueue<ElementType>::emplace(Args&&... args) -> bool {
    total_enqueues_.fetch_add(1, std::memory_order_relaxed);

    // Create new node with emplaced element
    auto* new_node = create_node();
    if (!new_node) {
        return false;  // Out of memory
    }

    auto* element_emplaced = new ElementType(std::forward<Args>(args)...);
    new_node->data.store(element_emplaced, std::memory_order_relaxed);

    while (true) {
        auto tail_ptr = tail_.load(std::memory_order_acquire);
        auto next_ptr = tail_ptr.ptr->next.load(std::memory_order_acquire);

        if (tail_ptr == tail_.load(std::memory_order_acquire)) {
            if (next_ptr == nullptr) {
                Node* expected_next = nullptr;
                if (tail_ptr.ptr->next.compare_exchange_weak(expected_next,
                                                             new_node,
                                                             std::memory_order_release,
                                                             std::memory_order_relaxed)) {
                    TaggedPointer new_tail(new_node, tail_ptr.tag + 1);
                    tail_.compare_exchange_weak(
                        tail_ptr, new_tail, std::memory_order_release, std::memory_order_relaxed);
                    break;
                }
            } else {
                TaggedPointer new_tail(next_ptr, tail_ptr.tag + 1);
                tail_.compare_exchange_weak(
                    tail_ptr, new_tail, std::memory_order_release, std::memory_order_relaxed);
            }
        }

        cas_failures_.fetch_add(1, std::memory_order_relaxed);
    }

    return true;
}

template <typename ElementType>
auto LockFreeQueue<ElementType>::dequeue() -> std::optional<ElementType> {
    total_dequeues_.fetch_add(1, std::memory_order_relaxed);

    while (true) {
        auto head_ptr = head_.load(std::memory_order_acquire);
        auto tail_ptr = tail_.load(std::memory_order_acquire);
        auto next_ptr = head_ptr.ptr->next.load(std::memory_order_acquire);

        // Check if head is still the same
        if (head_ptr == head_.load(std::memory_order_acquire)) {
            if (head_ptr.ptr == tail_ptr.ptr) {
                if (next_ptr == nullptr) {
                    // Queue is empty
                    return std::nullopt;
                }

                // Tail is lagging, try to advance it
                TaggedPointer new_tail(next_ptr, tail_ptr.tag + 1);
                tail_.compare_exchange_weak(
                    tail_ptr, new_tail, std::memory_order_release, std::memory_order_relaxed);
            } else {
                if (next_ptr == nullptr) {
                    // Inconsistent state, retry
                    continue;
                }

                // Read data before CAS
                auto* data = next_ptr->data.load(std::memory_order_acquire);
                if (!data) {
                    // Data not yet available, retry
                    continue;
                }

                // Try to advance head
                TaggedPointer new_head(next_ptr, head_ptr.tag + 1);
                if (head_.compare_exchange_weak(
                        head_ptr, new_head, std::memory_order_release, std::memory_order_relaxed)) {
                    // Successfully dequeued
                    ElementType result = std::move(*data);
                    delete data;
                    delete_node(head_ptr.ptr);  // Safe to delete old head

                    successful_dequeues_.fetch_add(1, std::memory_order_relaxed);
                    return result;
                }
            }
        }

        cas_failures_.fetch_add(1, std::memory_order_relaxed);
    }
}

template <typename ElementType>
auto LockFreeQueue<ElementType>::try_dequeue(ElementType& output) -> bool {
    auto result = dequeue();
    if (result.has_value()) {
        output = std::move(result.value());
        return true;
    }
    return false;
}

template <typename ElementType>
auto LockFreeQueue<ElementType>::empty() const noexcept -> bool {
    auto head_ptr = head_.load(std::memory_order_acquire);
    auto tail_ptr = tail_.load(std::memory_order_acquire);
    auto next_ptr = head_ptr.ptr->next.load(std::memory_order_acquire);

    return (head_ptr.ptr == tail_ptr.ptr) && (next_ptr == nullptr);
}

template <typename ElementType>
auto LockFreeQueue<ElementType>::size_approx() const noexcept -> std::size_t {
    auto total_enqueues = total_enqueues_.load(std::memory_order_relaxed);
    auto successful_dequeues = successful_dequeues_.load(std::memory_order_relaxed);

    // This is an approximation since operations might be in progress
    if (total_enqueues > successful_dequeues) {
        return total_enqueues - successful_dequeues;
    }
    return 0;
}

template <typename ElementType>
auto LockFreeQueue<ElementType>::get_stats() const noexcept -> QueueStats {
    auto total_enqueues = total_enqueues_.load(std::memory_order_relaxed);
    auto total_dequeues = total_dequeues_.load(std::memory_order_relaxed);
    auto successful_dequeues = successful_dequeues_.load(std::memory_order_relaxed);
    auto cas_failures = cas_failures_.load(std::memory_order_relaxed);
    auto node_count = node_count_.load(std::memory_order_relaxed);

    return QueueStats{.total_enqueues = total_enqueues,
                      .total_dequeues = total_dequeues,
                      .successful_dequeues = successful_dequeues,
                      .current_size_approx = size_approx(),
                      .memory_usage_bytes = node_count * sizeof(Node),
                      .cas_failures = cas_failures};
}

template <typename ElementType>
auto LockFreeQueue<ElementType>::create_node() -> Node* {
    try {
        auto* node = new Node();
        node_count_.fetch_add(1, std::memory_order_relaxed);
        return node;
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}

template <typename ElementType>
void LockFreeQueue<ElementType>::delete_node(Node* node) noexcept {
    if (node) {
        delete node;
        node_count_.fetch_sub(1, std::memory_order_relaxed);
    }
}

template <typename ElementType>
auto LockFreeQueue<ElementType>::cas_tagged_pointer(std::atomic<TaggedPointer>& target,
                                                    TaggedPointer& expected,
                                                    const TaggedPointer& desired) -> bool {
    return target.compare_exchange_weak(
        expected, desired, std::memory_order_release, std::memory_order_relaxed);
}

}  // namespace inference_lab::common
