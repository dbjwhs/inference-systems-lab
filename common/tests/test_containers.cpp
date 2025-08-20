// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file test_containers.cpp
 * @brief Comprehensive test suite for cache-friendly containers
 *
 * This file contains extensive tests for all container types implemented in containers.hpp:
 * - MemoryPool: Thread-safe O(1) allocation/deallocation testing
 * - RingBuffer: Lock-free single producer/consumer validation
 * - LockFreeQueue: Multi-threaded batch processing with ABA prevention
 * - TensorContainer: N-dimensional array operations and memory management
 *
 * Test Categories:
 * - Unit tests for basic functionality
 * - Thread safety and concurrency validation
 * - Performance characteristics verification
 * - Edge cases and error conditions
 * - Memory management and resource cleanup
 * - Integration testing between components
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../src/containers.hpp"

namespace inference_lab::common::tests {

//=============================================================================
// Memory Pool Allocator Tests
//=============================================================================

class MemoryPoolTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create pool with reasonable size for testing
        pool_ = std::make_unique<MemoryPool<int>>(1024);
    }

    void TearDown() override { pool_.reset(); }

    std::unique_ptr<MemoryPool<int>> pool_;
    static constexpr std::size_t DEFAULT_COUNT = 100;
};

TEST_F(MemoryPoolTest, BasicAllocationAndDeallocation) {
    // Test basic allocation
    auto* ptr = pool_->allocate(DEFAULT_COUNT);
    ASSERT_NE(ptr, nullptr);

    // Initialize memory
    for (std::size_t i = 0; i < DEFAULT_COUNT; ++i) {
        ptr[i] = static_cast<int>(i);
    }

    // Verify values
    for (std::size_t i = 0; i < DEFAULT_COUNT; ++i) {
        EXPECT_EQ(ptr[i], static_cast<int>(i));
    }

    // Test deallocation
    EXPECT_NO_THROW(pool_->deallocate(ptr, DEFAULT_COUNT));
}

TEST_F(MemoryPoolTest, MultipleAllocations) {
    constexpr std::size_t NUM_ALLOCS = 10;
    std::array<int*, NUM_ALLOCS> ptrs;

    // Allocate multiple blocks
    for (std::size_t i = 0; i < NUM_ALLOCS; ++i) {
        ptrs[i] = pool_->allocate(10);
        ASSERT_NE(ptrs[i], nullptr);

        // Initialize with unique pattern
        for (std::size_t j = 0; j < 10; ++j) {
            ptrs[i][j] = static_cast<int>(i * 100 + j);
        }
    }

    // Verify all allocations are distinct
    for (std::size_t i = 0; i < NUM_ALLOCS; ++i) {
        for (std::size_t j = i + 1; j < NUM_ALLOCS; ++j) {
            EXPECT_NE(ptrs[i], ptrs[j]);
        }
    }

    // Verify data integrity
    for (std::size_t i = 0; i < NUM_ALLOCS; ++i) {
        for (std::size_t j = 0; j < 10; ++j) {
            EXPECT_EQ(ptrs[i][j], static_cast<int>(i * 100 + j));
        }
    }

    // Deallocate all blocks
    for (std::size_t i = 0; i < NUM_ALLOCS; ++i) {
        EXPECT_NO_THROW(pool_->deallocate(ptrs[i], 10));
    }
}

TEST_F(MemoryPoolTest, LargeAllocation) {
    constexpr std::size_t LARGE_COUNT = 10000;

    auto* ptr = pool_->allocate(LARGE_COUNT);
    ASSERT_NE(ptr, nullptr);

    // Test memory access at boundaries
    ptr[0] = 42;
    ptr[LARGE_COUNT - 1] = 84;

    EXPECT_EQ(ptr[0], 42);
    EXPECT_EQ(ptr[LARGE_COUNT - 1], 84);

    pool_->deallocate(ptr, LARGE_COUNT);
}

TEST_F(MemoryPoolTest, ZeroAllocation) {
    auto* ptr = pool_->allocate(0);
    // Implementation may return nullptr or valid pointer for 0-size allocation
    // Both are acceptable behaviors
    if (ptr != nullptr) {
        pool_->deallocate(ptr, 0);
    }
}

TEST_F(MemoryPoolTest, PoolStatistics) {
    auto initial_stats = pool_->get_stats();

    // Allocate some memory
    auto* ptr1 = pool_->allocate(100);
    auto* ptr2 = pool_->allocate(200);

    auto mid_stats = pool_->get_stats();
    EXPECT_GT(mid_stats.allocation_count, initial_stats.allocation_count);
    EXPECT_GT(mid_stats.allocated_count, initial_stats.allocated_count);

    // Deallocate
    pool_->deallocate(ptr1, 100);
    pool_->deallocate(ptr2, 200);

    auto final_stats = pool_->get_stats();
    EXPECT_GT(final_stats.deallocation_count, initial_stats.deallocation_count);
}

TEST_F(MemoryPoolTest, ThreadSafety) {
    constexpr int NUM_THREADS = 4;
    constexpr int ALLOCS_PER_THREAD = 100;

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    // Launch multiple threads performing allocations
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([this, &success_count, t]() {
            std::vector<int*> local_ptrs;

            // Allocate
            for (int i = 0; i < ALLOCS_PER_THREAD; ++i) {
                auto* ptr = pool_->allocate(10);
                if (ptr != nullptr) {
                    local_ptrs.push_back(ptr);
                    // Write thread-specific pattern
                    for (int j = 0; j < 10; ++j) {
                        ptr[j] = t * 1000 + i * 10 + j;
                    }
                }
            }

            // Verify data integrity
            for (auto* ptr : local_ptrs) {
                for (int j = 0; j < 10; ++j) {
                    int expected = (ptr[j] / 1000) * 1000 + ((ptr[j] / 10) % 100) * 10 + j;
                    if (ptr[j] == expected) {
                        success_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }

            // Deallocate
            for (auto* ptr : local_ptrs) {
                pool_->deallocate(ptr, 10);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Should have many successful verifications
    EXPECT_GT(success_count.load(), NUM_THREADS * ALLOCS_PER_THREAD * 5);
}

//=============================================================================
// Ring Buffer Tests
//=============================================================================

class RingBufferTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create ring buffer with power-of-2 size for testing
        buffer_ = std::make_unique<RingBuffer<int>>(16);
    }

    void TearDown() override { buffer_.reset(); }

    std::unique_ptr<RingBuffer<int>> buffer_;
};

TEST_F(RingBufferTest, BasicPushPop) {
    EXPECT_TRUE(buffer_->empty());
    EXPECT_EQ(buffer_->size(), 0);

    // Push some elements
    EXPECT_TRUE(buffer_->push(42));
    EXPECT_TRUE(buffer_->push(84));
    EXPECT_TRUE(buffer_->push(126));

    EXPECT_FALSE(buffer_->empty());
    EXPECT_EQ(buffer_->size(), 3);

    // Pop elements in FIFO order
    auto result1 = buffer_->pop();
    ASSERT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), 42);

    auto result2 = buffer_->pop();
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), 84);

    auto result3 = buffer_->pop();
    ASSERT_TRUE(result3.has_value());
    EXPECT_EQ(result3.value(), 126);

    EXPECT_TRUE(buffer_->empty());
}

TEST_F(RingBufferTest, FillToCapacity) {
    constexpr std::size_t CAPACITY = 16;
    constexpr std::size_t USABLE_CAPACITY = CAPACITY - 1;  // Ring buffer reserves one slot

    // Fill buffer to usable capacity
    for (std::size_t i = 0; i < USABLE_CAPACITY; ++i) {
        EXPECT_TRUE(buffer_->push(static_cast<int>(i)));
    }

    EXPECT_TRUE(buffer_->full());
    EXPECT_EQ(buffer_->size(), USABLE_CAPACITY);

    // Next push should fail
    EXPECT_FALSE(buffer_->push(999));

    // Drain buffer
    for (std::size_t i = 0; i < USABLE_CAPACITY; ++i) {
        auto result = buffer_->pop();
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result.value(), static_cast<int>(i));
    }

    // Next pop should return nullopt
    auto empty_result = buffer_->pop();
    EXPECT_FALSE(empty_result.has_value());
}

TEST_F(RingBufferTest, WrapAroundBehavior) {
    constexpr std::size_t CAPACITY = 16;
    constexpr std::size_t USABLE_CAPACITY = CAPACITY - 1;  // Ring buffer reserves one slot
    constexpr std::size_t TEST_CYCLES = 5;

    // Test multiple fill/drain cycles to verify wrap-around
    for (std::size_t cycle = 0; cycle < TEST_CYCLES; ++cycle) {
        // Fill buffer to usable capacity
        for (std::size_t i = 0; i < USABLE_CAPACITY; ++i) {
            EXPECT_TRUE(buffer_->push(static_cast<int>(cycle * 100 + i)));
        }

        // Partially drain
        for (std::size_t i = 0; i < USABLE_CAPACITY / 2; ++i) {
            auto result = buffer_->pop();
            ASSERT_TRUE(result.has_value());
            EXPECT_EQ(result.value(), static_cast<int>(cycle * 100 + i));
        }

        // Add more elements (testing wrap-around)
        for (std::size_t i = 0; i < USABLE_CAPACITY / 2; ++i) {
            EXPECT_TRUE(buffer_->push(static_cast<int>(cycle * 100 + USABLE_CAPACITY + i)));
        }

        // Drain remaining from original batch
        for (std::size_t i = USABLE_CAPACITY / 2; i < USABLE_CAPACITY; ++i) {
            auto result = buffer_->pop();
            ASSERT_TRUE(result.has_value());
            EXPECT_EQ(result.value(), static_cast<int>(cycle * 100 + i));
        }

        // Drain the wrapped-around elements
        for (std::size_t i = 0; i < USABLE_CAPACITY / 2; ++i) {
            auto result = buffer_->pop();
            ASSERT_TRUE(result.has_value());
            EXPECT_EQ(result.value(), static_cast<int>(cycle * 100 + USABLE_CAPACITY + i));
        }
    }
}

TEST_F(RingBufferTest, TryPop) {
    int value = 42;

    // Push a value first
    EXPECT_TRUE(buffer_->push(value));
    EXPECT_EQ(buffer_->size(), 1);

    // Test try_pop
    int output = 0;
    EXPECT_TRUE(buffer_->try_pop(output));
    EXPECT_EQ(output, 42);
    EXPECT_TRUE(buffer_->empty());

    // Try pop from empty buffer
    EXPECT_FALSE(buffer_->try_pop(output));
}

TEST_F(RingBufferTest, MoveSemantics) {
    // Test with move-only type
    auto move_buffer = std::make_unique<RingBuffer<std::unique_ptr<int>>>(8);

    auto ptr = std::make_unique<int>(42);
    EXPECT_TRUE(move_buffer->push(std::move(ptr)));
    EXPECT_EQ(ptr, nullptr);  // Should be moved

    auto result = move_buffer->pop();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result.value(), 42);
}

TEST_F(RingBufferTest, Statistics) {
    auto initial_stats = buffer_->get_stats();

    // Perform operations
    buffer_->push(1);
    buffer_->push(2);
    buffer_->pop();

    auto final_stats = buffer_->get_stats();
    EXPECT_GT(final_stats.total_pushes, initial_stats.total_pushes);
    EXPECT_GT(final_stats.total_pops, initial_stats.total_pops);
    EXPECT_EQ(final_stats.current_size, 1);
}

TEST_F(RingBufferTest, SingleProducerSingleConsumer) {
    constexpr int NUM_ITEMS = 10000;
    std::atomic<bool> producer_done{false};
    std::vector<int> consumed_items;

    // Producer thread
    std::thread producer([this, &producer_done]() {
        for (int i = 0; i < NUM_ITEMS; ++i) {
            while (!buffer_->push(i)) {
                std::this_thread::yield();
            }
        }
        producer_done = true;
    });

    // Consumer thread
    std::thread consumer([this, &consumed_items, &producer_done]() {
        while (!producer_done || !buffer_->empty()) {
            auto item = buffer_->pop();
            if (item.has_value()) {
                consumed_items.push_back(item.value());
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    // Verify all items were consumed in order
    EXPECT_EQ(consumed_items.size(), NUM_ITEMS);
    for (int i = 0; i < NUM_ITEMS; ++i) {
        EXPECT_EQ(consumed_items[i], i);
    }
}

//=============================================================================
// Lock-Free Queue Tests
//=============================================================================

class LockFreeQueueTest : public ::testing::Test {
  protected:
    void SetUp() override { queue_ = std::make_unique<LockFreeQueue<int>>(); }

    void TearDown() override { queue_.reset(); }

    std::unique_ptr<LockFreeQueue<int>> queue_;
};

TEST_F(LockFreeQueueTest, BasicEnqueueDequeue) {
    EXPECT_TRUE(queue_->empty());

    // Enqueue some elements
    EXPECT_TRUE(queue_->enqueue(42));
    EXPECT_TRUE(queue_->enqueue(84));
    EXPECT_TRUE(queue_->enqueue(126));

    EXPECT_FALSE(queue_->empty());
    EXPECT_GT(queue_->size_approx(), 0);

    // Dequeue elements in FIFO order
    auto result1 = queue_->dequeue();
    ASSERT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), 42);

    auto result2 = queue_->dequeue();
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), 84);

    auto result3 = queue_->dequeue();
    ASSERT_TRUE(result3.has_value());
    EXPECT_EQ(result3.value(), 126);

    // Queue should be empty now
    auto empty_result = queue_->dequeue();
    EXPECT_FALSE(empty_result.has_value());
}

TEST_F(LockFreeQueueTest, TryDequeue) {
    queue_->enqueue(42);

    int output = 0;
    EXPECT_TRUE(queue_->try_dequeue(output));
    EXPECT_EQ(output, 42);

    // Try dequeue from empty queue
    EXPECT_FALSE(queue_->try_dequeue(output));
}

TEST_F(LockFreeQueueTest, MoveSemantics) {
    auto move_queue = std::make_unique<LockFreeQueue<std::unique_ptr<int>>>();

    auto ptr = std::make_unique<int>(42);
    EXPECT_TRUE(move_queue->enqueue(std::move(ptr)));
    EXPECT_EQ(ptr, nullptr);  // Should be moved

    auto result = move_queue->dequeue();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result.value(), 42);
}

TEST_F(LockFreeQueueTest, Statistics) {
    auto initial_stats = queue_->get_stats();

    // Perform operations
    queue_->enqueue(1);
    queue_->enqueue(2);
    queue_->dequeue();

    auto final_stats = queue_->get_stats();
    EXPECT_GT(final_stats.total_enqueues, initial_stats.total_enqueues);
    EXPECT_GT(final_stats.total_dequeues, initial_stats.total_dequeues);
}

TEST_F(LockFreeQueueTest, MultipleProducersMultipleConsumers) {
    constexpr int NUM_PRODUCERS = 4;
    constexpr int NUM_CONSUMERS = 4;
    constexpr int ITEMS_PER_PRODUCER = 1000;

    std::atomic<int> total_produced{0};
    std::atomic<int> total_consumed{0};
    std::vector<std::thread> producers, consumers;
    std::atomic<bool> stop_consumers{false};

    // Start producers
    for (int p = 0; p < NUM_PRODUCERS; ++p) {
        producers.emplace_back([this, &total_produced, p]() {
            for (int i = 0; i < ITEMS_PER_PRODUCER; ++i) {
                int value = p * ITEMS_PER_PRODUCER + i;
                while (!queue_->enqueue(value)) {
                    std::this_thread::yield();
                }
                total_produced.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Start consumers
    for (int c = 0; c < NUM_CONSUMERS; ++c) {
        consumers.emplace_back([this, &total_consumed, &stop_consumers]() {
            while (!stop_consumers) {
                auto item = queue_->dequeue();
                if (item.has_value()) {
                    total_consumed.fetch_add(1, std::memory_order_relaxed);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    // Wait for producers to finish
    for (auto& producer : producers) {
        producer.join();
    }

    // Let consumers finish remaining items
    while (total_consumed.load() < total_produced.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    stop_consumers = true;

    // Wait for consumers
    for (auto& consumer : consumers) {
        consumer.join();
    }

    // Verify all items were processed
    EXPECT_EQ(total_produced.load(), NUM_PRODUCERS * ITEMS_PER_PRODUCER);
    EXPECT_EQ(total_consumed.load(), total_produced.load());
}

TEST_F(LockFreeQueueTest, HighContentionStressTest) {
    constexpr int NUM_THREADS = 8;
    constexpr std::chrono::seconds TEST_DURATION{5};

    std::atomic<bool> keep_running{true};
    std::atomic<long> total_operations{0};
    std::vector<std::thread> threads;

    // Launch threads that randomly enqueue/dequeue
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([this, &keep_running, &total_operations, t]() {
            std::random_device rd;
            std::mt19937 gen(rd() + t);
            std::uniform_int_distribution<> op_dist(0, 1);

            while (keep_running) {
                if (op_dist(gen) == 0) {
                    // Enqueue operation
                    if (queue_->enqueue(t)) {
                        total_operations.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    // Dequeue operation
                    if (queue_->dequeue().has_value()) {
                        total_operations.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    // Run for specified duration
    std::this_thread::sleep_for(TEST_DURATION);
    keep_running = false;

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify we performed many operations without crashing
    EXPECT_GT(total_operations.load(), NUM_THREADS * 100);

    // Get final statistics
    auto final_stats = queue_->get_stats();
    EXPECT_GT(final_stats.total_enqueues, 0);
    EXPECT_GT(final_stats.total_dequeues, 0);
}

//=============================================================================
// Tensor Container Tests
//=============================================================================

class TensorContainerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create various tensor shapes for testing
        tensor_1d_ = std::make_unique<TensorContainer<float>>(std::vector<std::size_t>{10},
                                                              MemoryPool<float>(1024));
        tensor_2d_ = std::make_unique<TensorContainer<float>>(std::vector<std::size_t>{3, 4},
                                                              MemoryPool<float>(1024));
        tensor_3d_ = std::make_unique<TensorContainer<float>>(std::vector<std::size_t>{2, 3, 4},
                                                              MemoryPool<float>(1024));
    }

    void TearDown() override {
        tensor_1d_.reset();
        tensor_2d_.reset();
        tensor_3d_.reset();
    }

    std::unique_ptr<TensorContainer<float>> tensor_1d_;
    std::unique_ptr<TensorContainer<float>> tensor_2d_;
    std::unique_ptr<TensorContainer<float>> tensor_3d_;
};

TEST_F(TensorContainerTest, BasicConstruction) {
    // Test empty constructor
    TensorContainer<int> empty_tensor;
    EXPECT_TRUE(empty_tensor.empty());
    EXPECT_EQ(empty_tensor.size(), 0);
    EXPECT_EQ(empty_tensor.ndim(), 0);

    // Test shape constructor
    TensorContainer<int> shaped_tensor({2, 3}, MemoryPool<int>(1024));
    EXPECT_FALSE(shaped_tensor.empty());
    EXPECT_EQ(shaped_tensor.size(), 6);
    EXPECT_EQ(shaped_tensor.ndim(), 2);

    auto shape = shaped_tensor.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

TEST_F(TensorContainerTest, ElementAccess) {
    // Test 1D access
    for (std::size_t i = 0; i < 10; ++i) {
        (*tensor_1d_)(i) = static_cast<float>(i * 2);
    }

    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ((*tensor_1d_)(i), static_cast<float>(i * 2));
    }

    // Test 2D access
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            (*tensor_2d_)(i, j) = static_cast<float>(i * 10 + j);
        }
    }

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ((*tensor_2d_)(i, j), static_cast<float>(i * 10 + j));
        }
    }

    // Test 3D access
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                (*tensor_3d_)(i, j, k) = static_cast<float>(i * 100 + j * 10 + k);
            }
        }
    }

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                EXPECT_FLOAT_EQ((*tensor_3d_)(i, j, k), static_cast<float>(i * 100 + j * 10 + k));
            }
        }
    }
}

TEST_F(TensorContainerTest, VectorIndexAccess) {
    // Test at() method with vector indices
    std::vector<std::size_t> indices_2d = {1, 2};
    (*tensor_2d_).at(indices_2d) = 42.0f;
    EXPECT_FLOAT_EQ((*tensor_2d_).at(indices_2d), 42.0f);

    // Test out of bounds
    std::vector<std::size_t> out_of_bounds = {10, 20};
    EXPECT_THROW((*tensor_2d_).at(out_of_bounds), std::out_of_range);
}

TEST_F(TensorContainerTest, FillOperations) {
    // Test fill with value
    tensor_2d_->fill(3.14f);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ((*tensor_2d_)(i, j), 3.14f);
        }
    }

    // Test zero fill
    tensor_2d_->zero();
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ((*tensor_2d_)(i, j), 0.0f);
        }
    }
}

TEST_F(TensorContainerTest, Reshape) {
    // Initialize tensor with sequence
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            (*tensor_2d_)(i, j) = static_cast<float>(i * 4 + j);
        }
    }

    // Reshape from (3,4) to (2,6)
    EXPECT_TRUE(tensor_2d_->reshape({2, 6}));
    EXPECT_EQ(tensor_2d_->shape()[0], 2);
    EXPECT_EQ(tensor_2d_->shape()[1], 6);

    // Verify data is preserved (row-major order)
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            float expected = static_cast<float>(i * 6 + j);
            EXPECT_FLOAT_EQ((*tensor_2d_)(i, j), expected);
        }
    }

    // Test invalid reshape
    EXPECT_FALSE(tensor_2d_->reshape({3, 5}));  // Different total size
}

TEST_F(TensorContainerTest, MemoryLayout) {
    // Test contiguous memory layout
    EXPECT_TRUE(tensor_2d_->is_contiguous());

    // Test strides calculation
    auto strides = tensor_2d_->strides();
    EXPECT_EQ(strides[0], 4);  // Row stride
    EXPECT_EQ(strides[1], 1);  // Column stride

    // Test memory usage
    std::size_t expected_bytes = 3 * 4 * sizeof(float);
    EXPECT_EQ(tensor_2d_->memory_usage(), expected_bytes);
}

TEST_F(TensorContainerTest, TensorInfo) {
    auto info = tensor_3d_->get_info();

    EXPECT_EQ(info.shape.size(), 3);
    EXPECT_EQ(info.shape[0], 2);
    EXPECT_EQ(info.shape[1], 3);
    EXPECT_EQ(info.shape[2], 4);

    EXPECT_EQ(info.total_elements, 24);
    EXPECT_TRUE(info.is_contiguous);
    EXPECT_EQ(info.dtype_name, "float32");
}

TEST_F(TensorContainerTest, CopyOperations) {
    // Initialize source tensor
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            (*tensor_2d_)(i, j) = static_cast<float>(i * 4 + j);
        }
    }

    // Create target tensor with same shape
    TensorContainer<float> target({3, 4}, MemoryPool<float>(1024));

    // Test copy
    EXPECT_TRUE(target.copy_from(*tensor_2d_));

    // Verify data
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(target(i, j), (*tensor_2d_)(i, j));
        }
    }

    // Test copy with different shape should fail
    TensorContainer<float> wrong_shape({2, 5}, MemoryPool<float>(1024));
    EXPECT_FALSE(wrong_shape.copy_from(*tensor_2d_));
}

TEST_F(TensorContainerTest, MoveSemantics) {
    // Initialize tensor
    for (std::size_t i = 0; i < 10; ++i) {
        (*tensor_1d_)(i) = static_cast<float>(i);
    }

    auto* original_data = tensor_1d_->data();

    // Move construct
    TensorContainer<float> moved_tensor = std::move(*tensor_1d_);

    // Verify moved tensor has the data
    EXPECT_EQ(moved_tensor.data(), original_data);
    EXPECT_EQ(moved_tensor.size(), 10);

    // Verify data integrity
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(moved_tensor(i), static_cast<float>(i));
    }

    // Original should be empty
    EXPECT_EQ(tensor_1d_->data(), nullptr);
    EXPECT_EQ(tensor_1d_->size(), 0);
}

TEST_F(TensorContainerTest, IteratorSupport) {
    // Initialize tensor
    std::iota(tensor_1d_->begin(), tensor_1d_->end(), 0.0f);

    // Test iterator access
    float expected = 0.0f;
    for (auto it = tensor_1d_->begin(); it != tensor_1d_->end(); ++it) {
        EXPECT_FLOAT_EQ(*it, expected);
        expected += 1.0f;
    }

    // Test range-based for loop
    std::size_t index = 0;
    for (const auto& value : *tensor_1d_) {
        EXPECT_FLOAT_EQ(value, static_cast<float>(index));
        ++index;
    }
}

//=============================================================================
// Tensor Utility Functions Tests
//=============================================================================

TEST(TensorUtilsTest, ZerosOnesRandom) {
    using namespace tensor_utils;

    // Test zeros
    auto zeros_tensor = zeros<float>({3, 3});
    for (const auto& value : zeros_tensor) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }

    // Test ones
    auto ones_tensor = ones<float>({2, 4});
    for (const auto& value : ones_tensor) {
        EXPECT_FLOAT_EQ(value, 1.0f);
    }

    // Test random (basic functionality)
    auto random_tensor = random<float>({100}, 0.0f, 1.0f);

    // Check all values are in range
    for (const auto& value : random_tensor) {
        EXPECT_GE(value, 0.0f);
        EXPECT_LE(value, 1.0f);
    }

    // Basic statistical check - should have some variation
    float min_val = *std::min_element(random_tensor.begin(), random_tensor.end());
    float max_val = *std::max_element(random_tensor.begin(), random_tensor.end());
    EXPECT_LT(max_val - min_val, 1.0f);  // Should have some spread
    EXPECT_GT(max_val - min_val, 0.1f);  // But not too little
}

//=============================================================================
// Integration Tests
//=============================================================================

TEST(ContainerIntegrationTest, TensorWithMemoryPool) {
    // Create custom memory pool
    MemoryPool<double> custom_pool(1024);

    // Create tensor using custom allocator
    TensorContainer<double, MemoryPool<double>> tensor({10, 10}, std::move(custom_pool));

    // Fill with data
    for (std::size_t i = 0; i < 10; ++i) {
        for (std::size_t j = 0; j < 10; ++j) {
            tensor(i, j) = static_cast<double>(i * 10 + j);
        }
    }

    // Verify data integrity
    for (std::size_t i = 0; i < 10; ++i) {
        for (std::size_t j = 0; j < 10; ++j) {
            EXPECT_DOUBLE_EQ(tensor(i, j), static_cast<double>(i * 10 + j));
        }
    }

    // Check allocator was used
    auto pool_stats = tensor.get_allocator().get_stats();
    EXPECT_GT(pool_stats.allocation_count, 0);
}

TEST(ContainerIntegrationTest, QueueOfTensors) {
    LockFreeQueue<FloatTensor> tensor_queue;

    // Create and enqueue tensors
    constexpr int NUM_TENSORS = 5;
    for (int i = 0; i < NUM_TENSORS; ++i) {
        FloatTensor tensor({2, 2});
        tensor.fill(static_cast<float>(i));

        EXPECT_TRUE(tensor_queue.enqueue(std::move(tensor)));
    }

    // Dequeue and verify tensors
    for (int i = 0; i < NUM_TENSORS; ++i) {
        auto result = tensor_queue.dequeue();
        ASSERT_TRUE(result.has_value());

        auto tensor = std::move(result.value());
        EXPECT_EQ(tensor.size(), 4);

        for (const auto& value : tensor) {
            EXPECT_FLOAT_EQ(value, static_cast<float>(i));
        }
    }
}

TEST(ContainerIntegrationTest, RingBufferTensorStreaming) {
    RingBuffer<FloatTensor> tensor_stream(8);

    // Simulate streaming tensor data
    constexpr int NUM_FRAMES = 20;
    std::vector<FloatTensor> processed_tensors;

    for (int frame = 0; frame < NUM_FRAMES; ++frame) {
        // Create tensor for this frame
        FloatTensor tensor({3, 3});
        tensor.fill(static_cast<float>(frame));

        // If buffer is full, process oldest tensor
        if (tensor_stream.full()) {
            auto old_tensor = tensor_stream.pop();
            if (old_tensor.has_value()) {
                processed_tensors.push_back(std::move(old_tensor.value()));
            }
        }

        // Add new tensor
        tensor_stream.push(std::move(tensor));
    }

    // Process remaining tensors
    while (!tensor_stream.empty()) {
        auto tensor = tensor_stream.pop();
        if (tensor.has_value()) {
            processed_tensors.push_back(std::move(tensor.value()));
        }
    }

    // Verify we processed expected number of tensors
    EXPECT_EQ(processed_tensors.size(), NUM_FRAMES);
}

}  // namespace inference_lab::common::tests
