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
 * @file container_benchmarks.cpp
 * @brief Comprehensive performance benchmarks for cache-friendly containers
 *
 * This file contains detailed performance comparisons between our custom containers
 * and standard library equivalents. Benchmarks measure:
 * - Throughput (operations per second)
 * - Latency (nanoseconds per operation)
 * - Memory usage and cache efficiency
 * - Scalability under concurrent load
 * - Performance degradation patterns
 *
 * Container Comparisons:
 * - MemoryPool vs std::allocator + std::vector
 * - RingBuffer vs std::queue + std::mutex
 * - LockFreeQueue vs std::queue + std::mutex + std::condition_variable
 * - TensorContainer vs std::vector with manual indexing
 *
 * Benchmark Categories:
 * - Single-threaded performance baselines
 * - Multi-threaded scalability (2, 4, 8, 16 threads)
 * - Memory allocation patterns (frequent alloc/dealloc)
 * - Cache performance (sequential vs random access)
 * - Producer-consumer scenarios with varying ratios
 * - Large dataset processing (tensor operations)
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>

#include "../src/containers.hpp"

namespace inference_lab::common::benchmarks {

//=============================================================================
// Memory Pool Benchmarks
//=============================================================================

/**
 * @brief Benchmark custom MemoryPool vs std::allocator
 */
static void BM_MemoryPool_Allocation(benchmark::State& state) {
    const std::size_t num_allocations = static_cast<std::size_t>(state.range(0));
    const std::size_t allocation_size = static_cast<std::size_t>(state.range(1));

    MemoryPool<int> pool(1024 * 1024);  // 1MB pool
    std::vector<int*> ptrs;
    ptrs.reserve(num_allocations);

    for (auto _ : state) {
        // Allocation phase
        for (std::size_t i = 0; i < num_allocations; ++i) {
            auto* ptr = pool.allocate(allocation_size);
            ptrs.push_back(ptr);
            benchmark::DoNotOptimize(ptr);
        }

        // Deallocation phase
        for (auto* ptr : ptrs) {
            pool.deallocate(ptr, allocation_size);
        }
        ptrs.clear();
    }

    state.SetItemsProcessed(state.iterations() * num_allocations);
    state.SetBytesProcessed(state.iterations() * num_allocations * allocation_size * sizeof(int));
}

static void BM_StdAllocator_Allocation(benchmark::State& state) {
    const std::size_t num_allocations = static_cast<std::size_t>(state.range(0));
    const std::size_t allocation_size = static_cast<std::size_t>(state.range(1));

    std::allocator<int> allocator;
    std::vector<int*> ptrs;
    ptrs.reserve(num_allocations);

    for (auto _ : state) {
        // Allocation phase
        for (std::size_t i = 0; i < num_allocations; ++i) {
            auto* ptr = allocator.allocate(allocation_size);
            ptrs.push_back(ptr);
            benchmark::DoNotOptimize(ptr);
        }

        // Deallocation phase
        for (auto* ptr : ptrs) {
            allocator.deallocate(ptr, allocation_size);
        }
        ptrs.clear();
    }

    state.SetItemsProcessed(state.iterations() * num_allocations);
    state.SetBytesProcessed(state.iterations() * num_allocations * allocation_size * sizeof(int));
}

// Register memory pool benchmarks with various allocation patterns
BENCHMARK(BM_MemoryPool_Allocation)->Args({100, 10})->Args({1000, 100})->Args({10000, 1000});
BENCHMARK(BM_StdAllocator_Allocation)->Args({100, 10})->Args({1000, 100})->Args({10000, 1000});

//=============================================================================
// Ring Buffer Benchmarks
//=============================================================================

/**
 * @brief Single producer, single consumer ring buffer performance
 */
static void BM_RingBuffer_SPSC(benchmark::State& state) {
    const std::size_t buffer_size = static_cast<std::size_t>(state.range(0));
    const std::size_t num_items = static_cast<std::size_t>(state.range(1));

    RingBuffer<int> ring_buffer(buffer_size);

    for (auto _ : state) {
        std::size_t produced = 0;
        std::size_t consumed = 0;

        while (consumed < num_items) {
            // Produce items
            while (produced < num_items && !ring_buffer.full()) {
                ring_buffer.push(static_cast<int>(produced));
                ++produced;
            }

            // Consume items
            while (!ring_buffer.empty() && consumed < num_items) {
                auto item = ring_buffer.pop();
                if (item.has_value()) {
                    benchmark::DoNotOptimize(item.value());
                    ++consumed;
                }
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * num_items);
}

static void BM_StdQueue_SPSC(benchmark::State& state) {
    const std::size_t buffer_size = static_cast<std::size_t>(state.range(0));
    const std::size_t num_items = static_cast<std::size_t>(state.range(1));

    std::queue<int> queue;
    std::mutex mutex;

    for (auto _ : state) {
        std::size_t produced = 0;
        std::size_t consumed = 0;

        while (consumed < num_items) {
            // Produce items
            {
                std::lock_guard<std::mutex> lock(mutex);
                while (produced < num_items && queue.size() < buffer_size) {
                    queue.push(static_cast<int>(produced));
                    ++produced;
                }
            }

            // Consume items
            {
                std::lock_guard<std::mutex> lock(mutex);
                while (!queue.empty() && consumed < num_items) {
                    auto item = queue.front();
                    queue.pop();
                    benchmark::DoNotOptimize(item);
                    ++consumed;
                }
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * num_items);
}

BENCHMARK(BM_RingBuffer_SPSC)->Args({64, 10000})->Args({256, 100000})->Args({1024, 1000000});
BENCHMARK(BM_StdQueue_SPSC)->Args({64, 10000})->Args({256, 100000})->Args({1024, 1000000});

//=============================================================================
// Lock-Free Queue Benchmarks
//=============================================================================

/**
 * @brief Multi-producer, multi-consumer queue performance
 */
static void BM_LockFreeQueue_MPMC(benchmark::State& state) {
    const std::size_t num_producers = static_cast<std::size_t>(state.range(0));
    const std::size_t num_consumers = static_cast<std::size_t>(state.range(1));
    const std::size_t items_per_producer = 10000;

    for (auto _ : state) {
        LockFreeQueue<int> queue;
        std::atomic<std::size_t> total_consumed{0};
        std::vector<std::thread> threads;

        // Start producers
        for (std::size_t p = 0; p < num_producers; ++p) {
            threads.emplace_back([&queue, p]() {
                for (std::size_t i = 0; i < items_per_producer; ++i) {
                    while (!queue.enqueue(static_cast<int>(p * items_per_producer + i))) {
                        std::this_thread::yield();
                    }
                }
            });
        }

        // Start consumers
        const std::size_t total_items = num_producers * items_per_producer;
        for (std::size_t c = 0; c < num_consumers; ++c) {
            threads.emplace_back([&queue, &total_consumed, total_items]() {
                while (total_consumed.load() < total_items) {
                    auto item = queue.dequeue();
                    if (item.has_value()) {
                        benchmark::DoNotOptimize(item.value());
                        total_consumed.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        std::this_thread::yield();
                    }
                }
            });
        }

        // Wait for completion
        for (auto& thread : threads) {
            thread.join();
        }
    }

    state.SetItemsProcessed(state.iterations() * num_producers * items_per_producer);
}

static void BM_StdQueue_MPMC(benchmark::State& state) {
    const std::size_t num_producers = static_cast<std::size_t>(state.range(0));
    const std::size_t num_consumers = static_cast<std::size_t>(state.range(1));
    const std::size_t items_per_producer = 10000;

    for (auto _ : state) {
        std::queue<int> queue;
        std::mutex mutex;
        std::condition_variable cv;
        std::atomic<std::size_t> total_consumed{0};
        std::atomic<bool> producers_done{false};
        std::vector<std::thread> threads;

        // Start producers
        for (std::size_t p = 0; p < num_producers; ++p) {
            threads.emplace_back([&queue, &mutex, &cv, p]() {
                for (std::size_t i = 0; i < items_per_producer; ++i) {
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        queue.push(static_cast<int>(p * items_per_producer + i));
                    }
                    cv.notify_one();
                }
            });
        }

        // Start consumers
        const std::size_t total_items = num_producers * items_per_producer;
        for (std::size_t c = 0; c < num_consumers; ++c) {
            threads.emplace_back(
                [&queue, &mutex, &cv, &total_consumed, &producers_done, total_items]() {
                    while (total_consumed.load() < total_items) {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv.wait(lock, [&queue, &producers_done, &total_consumed, total_items]() {
                            return !queue.empty() ||
                                   (producers_done && total_consumed.load() >= total_items);
                        });

                        while (!queue.empty() && total_consumed.load() < total_items) {
                            auto item = queue.front();
                            queue.pop();
                            benchmark::DoNotOptimize(item);
                            total_consumed.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                });
        }

        // Wait for producers to finish
        for (std::size_t i = 0; i < num_producers; ++i) {
            threads[i].join();
        }
        producers_done = true;
        cv.notify_all();

        // Wait for consumers to finish
        for (std::size_t i = num_producers; i < threads.size(); ++i) {
            threads[i].join();
        }
    }

    state.SetItemsProcessed(state.iterations() * num_producers * items_per_producer);
}

BENCHMARK(BM_LockFreeQueue_MPMC)->Args({2, 2})->Args({4, 4})->Args({8, 8});
BENCHMARK(BM_StdQueue_MPMC)->Args({2, 2})->Args({4, 4})->Args({8, 8});

//=============================================================================
// Tensor Container Benchmarks
//=============================================================================

/**
 * @brief Tensor operations vs manual std::vector indexing
 */
static void BM_TensorContainer_MatrixMultiply(benchmark::State& state) {
    const std::size_t matrix_size = static_cast<std::size_t>(state.range(0));

    TensorContainer<float> matrix_a({matrix_size, matrix_size});
    TensorContainer<float> matrix_b({matrix_size, matrix_size});
    TensorContainer<float> result({matrix_size, matrix_size});

    // Initialize matrices
    for (std::size_t i = 0; i < matrix_size; ++i) {
        for (std::size_t j = 0; j < matrix_size; ++j) {
            matrix_a(i, j) = static_cast<float>(i + j);
            matrix_b(i, j) = static_cast<float>(i * j + 1);
        }
    }

    for (auto _ : state) {
        // Simple matrix multiplication
        for (std::size_t i = 0; i < matrix_size; ++i) {
            for (std::size_t j = 0; j < matrix_size; ++j) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < matrix_size; ++k) {
                    sum += matrix_a(i, k) * matrix_b(k, j);
                }
                result(i, j) = sum;
            }
        }
        benchmark::DoNotOptimize(result.data());
    }

    state.SetItemsProcessed(state.iterations() * matrix_size * matrix_size * matrix_size);
    state.SetBytesProcessed(state.iterations() * matrix_size * matrix_size * 3 * sizeof(float));
}

static void BM_StdVector_MatrixMultiply(benchmark::State& state) {
    const std::size_t matrix_size = static_cast<std::size_t>(state.range(0));
    const std::size_t total_elements = matrix_size * matrix_size;

    std::vector<float> matrix_a(total_elements);
    std::vector<float> matrix_b(total_elements);
    std::vector<float> result(total_elements);

    // Initialize matrices (manual indexing)
    for (std::size_t i = 0; i < matrix_size; ++i) {
        for (std::size_t j = 0; j < matrix_size; ++j) {
            matrix_a[i * matrix_size + j] = static_cast<float>(i + j);
            matrix_b[i * matrix_size + j] = static_cast<float>(i * j + 1);
        }
    }

    for (auto _ : state) {
        // Simple matrix multiplication with manual indexing
        for (std::size_t i = 0; i < matrix_size; ++i) {
            for (std::size_t j = 0; j < matrix_size; ++j) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < matrix_size; ++k) {
                    sum += matrix_a[i * matrix_size + k] * matrix_b[k * matrix_size + j];
                }
                result[i * matrix_size + j] = sum;
            }
        }
        benchmark::DoNotOptimize(result.data());
    }

    state.SetItemsProcessed(state.iterations() * matrix_size * matrix_size * matrix_size);
    state.SetBytesProcessed(state.iterations() * matrix_size * matrix_size * 3 * sizeof(float));
}

BENCHMARK(BM_TensorContainer_MatrixMultiply)->Arg(64)->Arg(128)->Arg(256);
BENCHMARK(BM_StdVector_MatrixMultiply)->Arg(64)->Arg(128)->Arg(256);

//=============================================================================
// Memory Access Pattern Benchmarks
//=============================================================================

/**
 * @brief Sequential vs random access patterns
 */
static void BM_TensorContainer_SequentialAccess(benchmark::State& state) {
    const std::size_t tensor_size = static_cast<std::size_t>(state.range(0));
    TensorContainer<float> tensor({tensor_size});

    for (auto _ : state) {
        // Sequential write
        for (std::size_t i = 0; i < tensor_size; ++i) {
            tensor(i) = static_cast<float>(i);
        }

        // Sequential read
        float sum = 0.0f;
        for (std::size_t i = 0; i < tensor_size; ++i) {
            sum += tensor(i);
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(state.iterations() * tensor_size * 2);  // Read + write
    state.SetBytesProcessed(state.iterations() * tensor_size * sizeof(float) * 2);
}

static void BM_TensorContainer_RandomAccess(benchmark::State& state) {
    const std::size_t tensor_size = static_cast<std::size_t>(state.range(0));
    TensorContainer<float> tensor({tensor_size});

    // Generate random indices
    std::vector<std::size_t> indices(tensor_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    for (auto _ : state) {
        // Random write
        for (std::size_t i = 0; i < tensor_size; ++i) {
            tensor(indices[i]) = static_cast<float>(i);
        }

        // Random read
        float sum = 0.0f;
        for (std::size_t i = 0; i < tensor_size; ++i) {
            sum += tensor(indices[i]);
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(state.iterations() * tensor_size * 2);  // Read + write
    state.SetBytesProcessed(state.iterations() * tensor_size * sizeof(float) * 2);
}

BENCHMARK(BM_TensorContainer_SequentialAccess)->Range(1024, 1024 * 1024);
BENCHMARK(BM_TensorContainer_RandomAccess)->Range(1024, 1024 * 1024);

//=============================================================================
// Cache Performance Benchmarks
//=============================================================================

/**
 * @brief Test cache efficiency of different container layouts
 */
static void BM_TensorContainer_CacheEfficiency(benchmark::State& state) {
    const std::size_t rows = 1024;
    const std::size_t cols = 1024;
    TensorContainer<float> tensor({rows, cols});

    // Initialize tensor
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            tensor(i, j) = static_cast<float>(i * cols + j);
        }
    }

    for (auto _ : state) {
        float sum = 0.0f;

        if (state.range(0) == 0) {
            // Row-major access (cache-friendly)
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < cols; ++j) {
                    sum += tensor(i, j);
                }
            }
        } else {
            // Column-major access (cache-unfriendly)
            for (std::size_t j = 0; j < cols; ++j) {
                for (std::size_t i = 0; i < rows; ++i) {
                    sum += tensor(i, j);
                }
            }
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(state.iterations() * rows * cols);
    state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(float));
}

BENCHMARK(BM_TensorContainer_CacheEfficiency)->Arg(0)->Arg(1);  // 0=row-major, 1=col-major

//=============================================================================
// Concurrent Memory Pool Stress Test
//=============================================================================

/**
 * @brief High contention allocation benchmark
 */
static void BM_MemoryPool_Contention(benchmark::State& state) {
    const std::size_t num_threads = static_cast<std::size_t>(state.range(0));
    const std::size_t allocs_per_thread = 1000;

    for (auto _ : state) {
        MemoryPool<int> pool(1024 * 1024);  // 1MB pool
        std::vector<std::thread> threads;
        std::atomic<std::size_t> total_allocs{0};

        for (std::size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&pool, &total_allocs]() {
                std::vector<int*> local_ptrs;
                local_ptrs.reserve(allocs_per_thread);

                for (std::size_t i = 0; i < allocs_per_thread; ++i) {
                    auto* ptr = pool.allocate(10);
                    if (ptr != nullptr) {
                        local_ptrs.push_back(ptr);
                        total_allocs.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                for (auto* ptr : local_ptrs) {
                    pool.deallocate(ptr, 10);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

    state.SetItemsProcessed(state.iterations() * num_threads * allocs_per_thread);
}

BENCHMARK(BM_MemoryPool_Contention)->Range(1, 16);

}  // namespace inference_lab::common::benchmarks

BENCHMARK_MAIN();
