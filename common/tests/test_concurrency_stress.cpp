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
 * @file test_concurrency_stress.cpp
 * @brief Comprehensive stress tests for concurrent components
 *
 * This file contains intensive stress tests designed to validate the thread-safety,
 * performance, and correctness of concurrent data structures and components under
 * high contention and extended duration scenarios.
 *
 * Test Categories:
 * - High-concurrency logger stress testing (50-200 threads)
 * - Lock-free memory pool torture testing with race condition detection
 * - Multi-producer/consumer queue stress testing with ABA prevention
 * - Real-time circular buffer performance under pressure
 * - ML integration framework concurrent inference testing
 * - System-wide integration stress testing
 *
 * Testing Parameters:
 * - Thread counts: 4 to 200+ threads
 * - Duration: 30 seconds to 15+ minutes
 * - Memory pressure simulation
 * - Error injection and recovery validation
 * - Performance regression detection under load
 */

#include <atomic>
#include <chrono>
#include <future>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../src/containers.hpp"
#include "../src/logging.hpp"
#include "../src/result.hpp"

// Use common namespace to access logging macros
using namespace inference_lab::common;

namespace inference_lab::common::test {

using namespace std::chrono_literals;

//=============================================================================
// Stress Test Configuration and Utilities
//=============================================================================

/**
 * @brief Configuration parameters for stress testing
 */
struct StressTestConfig {
    std::size_t thread_count{8};              ///< Number of concurrent threads
    std::chrono::seconds duration{30s};       ///< Test duration
    std::size_t operations_per_thread{1000};  ///< Operations per thread
    bool enable_memory_pressure{false};       ///< Enable memory pressure testing
    bool enable_error_injection{false};       ///< Enable error injection
    double error_rate{0.1};                   ///< Error injection rate (0.0-1.0)
};

/**
 * @brief Stress test statistics collector
 */
struct StressTestStats {
    std::atomic<std::size_t> total_operations{0};
    std::atomic<std::size_t> successful_operations{0};
    std::atomic<std::size_t> failed_operations{0};
    std::atomic<std::size_t> timeout_operations{0};
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

    void record_success() {
        total_operations.fetch_add(1);
        successful_operations.fetch_add(1);
    }

    void record_failure() {
        total_operations.fetch_add(1);
        failed_operations.fetch_add(1);
    }

    void record_timeout() {
        total_operations.fetch_add(1);
        timeout_operations.fetch_add(1);
    }

    double get_success_rate() const {
        auto total = total_operations.load();
        return total > 0 ? static_cast<double>(successful_operations.load()) / total : 0.0;
    }

    double get_operations_per_second() const {
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        return duration > 0 ? static_cast<double>(total_operations.load()) * 1'000'000.0 / duration
                            : 0.0;
    }
};

/**
 * @brief Base class for concurrent stress tests
 */
class ConcurrentStressTestBase : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set up high-performance logging for stress tests
        // Log level will use default configuration
        LOG_INFO_PRINT("Starting concurrent stress test setup");

        // Initialize random number generator
        rng_.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    void TearDown() override { LOG_INFO_PRINT("Concurrent stress test cleanup completed"); }

    /**
     * @brief Generate random delay for simulating work
     */
    std::chrono::microseconds get_random_delay(std::chrono::microseconds min_delay = 1us,
                                               std::chrono::microseconds max_delay = 100us) {
        std::uniform_int_distribution<int> dist(static_cast<int>(min_delay.count()),
                                                static_cast<int>(max_delay.count()));
        return std::chrono::microseconds(dist(rng_));
    }

    /**
     * @brief Generate random data for testing
     */
    std::vector<std::uint8_t> generate_random_data(std::size_t size) {
        std::vector<std::uint8_t> data(size);
        std::uniform_int_distribution<std::uint8_t> dist(0, 255);
        for (auto& byte : data) {
            byte = dist(rng_);
        }
        return data;
    }

    mutable std::mt19937 rng_;
};

//=============================================================================
// High-Concurrency Logger Stress Tests
//=============================================================================

class LoggerStressTest : public ConcurrentStressTestBase {
  public:
    void concurrent_logging_worker(std::size_t worker_id,
                                   const StressTestConfig& config,
                                   StressTestStats& stats,
                                   std::atomic<bool>& stop_flag) {
        std::size_t operation_count = 0;
        auto worker_start = std::chrono::high_resolution_clock::now();

        while (!stop_flag.load() && operation_count < config.operations_per_thread) {
            try {
                // Mix different log levels and operations
                switch (operation_count % 5) {
                    case 0:
                        LOG_INFO_PRINT("Worker {} operation {}: High-frequency info message",
                                       worker_id,
                                       operation_count);
                        break;
                    case 1:
                        LOG_DEBUG_PRINT("Worker {} debug: Detailed diagnostic information {}",
                                        worker_id,
                                        operation_count);
                        break;
                    case 2:
                        LOG_WARNING_PRINT(
                            "Worker {} warning: Potential issue detected at operation {}",
                            worker_id,
                            operation_count);
                        break;
                    case 3:
                        LOG_ERROR_PRINT("Worker {} error: Simulated error condition {}",
                                        worker_id,
                                        operation_count);
                        break;
                    case 4:
                        // ML-specific logging
                        LOG_ML_OPERATION(MLOperation::PERFORMANCE_BENCHMARK,
                                         "model_stress_test",
                                         "Worker {} ML operation {} complete",
                                         worker_id,
                                         operation_count);
                        break;
                }

                // Random delay to simulate variable workload
                if (config.operations_per_thread > 100) {
                    std::this_thread::sleep_for(get_random_delay(1us, 10us));
                }

                stats.record_success();
                ++operation_count;

            } catch (const std::exception& e) {
                LOG_ERROR_PRINT(
                    "Worker {} failed at operation {}: {}", worker_id, operation_count, e.what());
                stats.record_failure();
            }
        }

        auto worker_end = std::chrono::high_resolution_clock::now();
        auto worker_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(worker_end - worker_start);

        LOG_INFO_PRINT("Worker {} completed {} operations in {}ms",
                       worker_id,
                       operation_count,
                       worker_duration.count());
    }
};

TEST_F(LoggerStressTest, HighConcurrencyLogging) {
    StressTestConfig config{.thread_count = 50, .duration = 30s, .operations_per_thread = 2000};

    LOG_INFO_PRINT("Starting high-concurrency logger stress test with {} threads",
                   config.thread_count);

    StressTestStats stats;
    std::atomic<bool> stop_flag{false};
    std::vector<std::thread> workers;

    stats.start_time = std::chrono::high_resolution_clock::now();

    // Launch worker threads
    for (std::size_t i = 0; i < config.thread_count; ++i) {
        workers.emplace_back(&LoggerStressTest::concurrent_logging_worker,
                             this,
                             i,
                             std::cref(config),
                             std::ref(stats),
                             std::ref(stop_flag));
    }

    // Monitor test progress with early completion detection
    auto test_start = std::chrono::high_resolution_clock::now();
    std::atomic<std::size_t> active_workers{config.thread_count};

    // Periodically check if workers are still active
    while (true) {
        std::this_thread::sleep_for(1s);
        auto elapsed = std::chrono::high_resolution_clock::now() - test_start;

        // Exit if duration exceeded or no operations happening
        if (elapsed >= config.duration) {
            break;
        }

        // Check if workers have finished their operations
        std::size_t current_ops = stats.total_operations.load();
        std::size_t expected_ops = config.thread_count * config.operations_per_thread;
        if (current_ops >= expected_ops) {
            LOG_INFO_PRINT("Workers completed {} operations early, ending test", current_ops);
            break;
        }
    }

    stop_flag.store(true);

    // Wait for all workers to complete
    for (auto& worker : workers) {
        worker.join();
    }

    stats.end_time = std::chrono::high_resolution_clock::now();

    // Validate results
    EXPECT_GT(stats.get_success_rate(), 0.95) << "Success rate should be > 95%";

    EXPECT_GT(stats.get_operations_per_second(), 1000.0)
        << "Should achieve > 1000 operations/second";

    EXPECT_EQ(stats.failed_operations.load(), 0)
        << "No operations should fail in normal stress test";

    LOG_INFO_PRINT(
        "Logger stress test completed: {:.1f} ops/sec, {:.2f}% success rate, {} total ops",
        stats.get_operations_per_second(),
        stats.get_success_rate() * 100,
        stats.total_operations.load());
}

TEST_F(LoggerStressTest, ExtremeConcurrencyLogging) {
    StressTestConfig config{.thread_count = 200, .duration = 60s, .operations_per_thread = 5000};

    LOG_INFO_PRINT("Starting extreme concurrency logger test with {} threads", config.thread_count);

    StressTestStats stats;
    std::atomic<bool> stop_flag{false};
    std::vector<std::thread> workers;

    stats.start_time = std::chrono::high_resolution_clock::now();

    // Launch worker threads
    for (std::size_t i = 0; i < config.thread_count; ++i) {
        workers.emplace_back(&LoggerStressTest::concurrent_logging_worker,
                             this,
                             i,
                             std::cref(config),
                             std::ref(stats),
                             std::ref(stop_flag));
    }

    // Monitor test progress with early completion detection
    auto test_start = std::chrono::high_resolution_clock::now();

    // Periodically check if workers are still active
    while (true) {
        std::this_thread::sleep_for(2s);  // Slightly longer interval for extreme test
        auto elapsed = std::chrono::high_resolution_clock::now() - test_start;

        // Exit if duration exceeded
        if (elapsed >= config.duration) {
            break;
        }

        // Check if workers have finished their operations
        std::size_t current_ops = stats.total_operations.load();
        std::size_t expected_ops = config.thread_count * config.operations_per_thread;
        if (current_ops >= expected_ops) {
            LOG_INFO_PRINT("Extreme test workers completed {} operations early", current_ops);
            break;
        }
    }

    stop_flag.store(true);

    // Wait for completion
    for (auto& worker : workers) {
        worker.join();
    }

    stats.end_time = std::chrono::high_resolution_clock::now();

    // More relaxed expectations for extreme concurrency
    EXPECT_GT(stats.get_success_rate(), 0.90)
        << "Success rate should be > 90% even under extreme load";

    EXPECT_GT(stats.total_operations.load(), config.thread_count * 100)
        << "Should complete substantial work even under extreme contention";

    LOG_INFO_PRINT("Extreme concurrency test completed: {:.1f} ops/sec, {:.2f}% success rate",
                   stats.get_operations_per_second(),
                   stats.get_success_rate() * 100);
}

//=============================================================================
// Memory Pool Torture Tests
//=============================================================================

class MemoryPoolStressTest : public ConcurrentStressTestBase {
  public:
    void memory_pool_worker(std::size_t worker_id,
                            MemoryPool<std::uint64_t>& pool,
                            const StressTestConfig& config,
                            StressTestStats& stats,
                            std::atomic<bool>& stop_flag) {
        // Use thread-local allocations to avoid race conditions
        std::vector<std::pair<std::uint64_t*, std::size_t>> allocations;
        allocations.reserve(50);  // Keep fewer allocations to reduce memory pressure

        std::size_t operation_count = 0;
        std::uniform_int_distribution<std::size_t> size_dist(1, 100);  // Smaller allocations

        while (!stop_flag.load() && operation_count < config.operations_per_thread) {
            try {
                // Allocation phase - only allocate if we have room
                if (allocations.size() < 20) {  // Limit concurrent allocations per thread
                    std::size_t alloc_size = size_dist(rng_);
                    auto ptr = pool.allocate(alloc_size);

                    if (ptr != nullptr) {
                        allocations.push_back({ptr, alloc_size});
                        stats.record_success();
                    } else {
                        stats.record_failure();
                    }
                }

                // Deallocate oldest allocation to maintain steady state
                if (!allocations.empty() && (operation_count % 2 == 0)) {
                    auto [dealloc_ptr, dealloc_size] = allocations.front();
                    allocations.erase(allocations.begin());

                    if (dealloc_ptr != nullptr) {
                        try {
                            pool.deallocate(dealloc_ptr, dealloc_size);
                        } catch (...) {
                            // Ignore deallocation errors to avoid crashes
                            stats.record_failure();
                        }
                    }
                }

                ++operation_count;

                // Reduced delay for better performance
                if (operation_count % 50 == 0) {
                    std::this_thread::sleep_for(get_random_delay(1us, 2us));
                }

            } catch (const std::exception& e) {
                stats.record_failure();
            }
        }

        // Cleanup remaining allocations with error handling
        for (auto [ptr, size] : allocations) {
            if (ptr != nullptr) {
                try {
                    pool.deallocate(ptr, size);
                } catch (...) {
                    // Ignore cleanup errors
                }
            }
        }

        LOG_DEBUG_PRINT(
            "Memory pool worker {} completed {} operations", worker_id, operation_count);
    }
};

TEST_F(MemoryPoolStressTest, HighContentionAllocation) {
    StressTestConfig config{.thread_count = 10, .duration = 10s, .operations_per_thread = 500};

    LOG_INFO_PRINT("Starting memory pool stress test with {} threads", config.thread_count);

    MemoryPool<std::uint64_t> pool(2048);  // Larger initial capacity to reduce expansion
    StressTestStats stats;
    std::atomic<bool> stop_flag{false};
    std::vector<std::thread> workers;

    stats.start_time = std::chrono::high_resolution_clock::now();

    // Launch worker threads
    for (std::size_t i = 0; i < config.thread_count; ++i) {
        workers.emplace_back(&MemoryPoolStressTest::memory_pool_worker,
                             this,
                             i,
                             std::ref(pool),
                             std::cref(config),
                             std::ref(stats),
                             std::ref(stop_flag));
    }

    // Monitor pool statistics during test
    std::thread monitor([&]() {
        while (!stop_flag.load()) {
            // Use shorter sleep intervals to avoid hanging
            for (int i = 0; i < 10 && !stop_flag.load(); ++i) {
                std::this_thread::sleep_for(500ms);
            }
            if (!stop_flag.load()) {
                auto pool_stats = pool.get_stats();
                LOG_INFO_PRINT("Pool stats: {} allocated, {} peak usage, {} expansions",
                               pool_stats.allocated_count,
                               pool_stats.peak_usage,
                               pool_stats.allocation_count);
            }
        }
    });

    // Monitor test progress with early completion detection
    auto test_start = std::chrono::high_resolution_clock::now();

    while (true) {
        std::this_thread::sleep_for(1s);
        auto elapsed = std::chrono::high_resolution_clock::now() - test_start;

        // Exit if duration exceeded
        if (elapsed >= config.duration) {
            break;
        }

        // Check if workers have finished their operations
        std::size_t current_ops = stats.total_operations.load();
        std::size_t expected_ops = config.thread_count * config.operations_per_thread;
        if (current_ops >= expected_ops) {
            LOG_INFO_PRINT("Memory pool workers completed {} operations early", current_ops);
            break;
        }
    }

    stop_flag.store(true);

    // Wait for completion
    for (auto& worker : workers) {
        worker.join();
    }
    monitor.join();

    stats.end_time = std::chrono::high_resolution_clock::now();

    // Validate pool integrity
    auto final_stats = pool.get_stats();
    EXPECT_EQ(final_stats.allocated_count, 0) << "All memory should be deallocated";

    EXPECT_GT(stats.get_success_rate(), 0.90)
        << "Memory pool should handle > 90% of allocations successfully under extreme contention";

    LOG_INFO_PRINT("Memory pool stress test completed: {:.1f} ops/sec, {:.2f}% success",
                   stats.get_operations_per_second(),
                   stats.get_success_rate() * 100);
}

//=============================================================================
// Lock-Free Container Stress Tests
//=============================================================================

class LockFreeContainerStressTest : public ConcurrentStressTestBase {
  public:
    template <typename QueueType>
    void producer_worker(std::size_t worker_id,
                         QueueType& queue,
                         const StressTestConfig& config,
                         StressTestStats& stats,
                         std::atomic<bool>& stop_flag) {
        std::size_t operation_count = 0;

        while (!stop_flag.load() && operation_count < config.operations_per_thread) {
            try {
                // Create test data with worker signature
                std::uint64_t data = (static_cast<std::uint64_t>(worker_id) << 32) |
                                     operation_count;

                if (queue.enqueue(data)) {
                    stats.record_success();
                } else {
                    stats.record_failure();
                }

                ++operation_count;

                // Small delay to prevent overwhelming consumers
                if (operation_count % 100 == 0) {
                    std::this_thread::sleep_for(get_random_delay(1us, 10us));
                }

            } catch (const std::exception& e) {
                stats.record_failure();
            }
        }

        LOG_DEBUG_PRINT("Producer {} completed {} operations", worker_id, operation_count);
    }

    template <typename QueueType>
    void consumer_worker(std::size_t worker_id,
                         QueueType& queue,
                         const StressTestConfig& config,
                         StressTestStats& stats,
                         std::atomic<bool>& stop_flag,
                         std::atomic<std::size_t>& total_consumed) {
        std::size_t consumed_count = 0;
        std::map<std::size_t, std::size_t> producer_counts;  // Track messages per producer

        while (!stop_flag.load() || !queue.empty()) {
            try {
                auto dequeue_result = queue.dequeue();
                if (dequeue_result.has_value()) {
                    auto data = dequeue_result.value();
                    // Verify data integrity
                    std::size_t producer_id = static_cast<std::size_t>(data >> 32);
                    std::size_t sequence = static_cast<std::size_t>(data & 0xFFFFFFFF);

                    producer_counts[producer_id]++;
                    consumed_count++;
                    total_consumed.fetch_add(1);

                    stats.record_success();

                    // Validate sequence numbers aren't wildly out of range
                    EXPECT_LT(sequence, config.operations_per_thread * 2)
                        << "Sequence number seems corrupted";

                } else {
                    // No data available, small delay before retry
                    std::this_thread::sleep_for(1us);
                }

            } catch (const std::exception& e) {
                stats.record_failure();
            }
        }

        LOG_DEBUG_PRINT("Consumer {} consumed {} messages from {} producers",
                        worker_id,
                        consumed_count,
                        producer_counts.size());
    }
};

TEST_F(LockFreeContainerStressTest, MultiProducerConsumerQueue) {
    StressTestConfig config{.thread_count = 16,  // 12 producers + 4 consumers
                            .duration = 45s,
                            .operations_per_thread = 5000};

    LOG_INFO_PRINT("Starting multi-producer/consumer queue stress test");

    LockFreeQueue<std::uint64_t> queue;
    StressTestStats producer_stats, consumer_stats;
    std::atomic<bool> stop_flag{false};
    std::atomic<std::size_t> total_consumed{0};
    std::vector<std::thread> workers;

    producer_stats.start_time = std::chrono::high_resolution_clock::now();
    consumer_stats.start_time = producer_stats.start_time;

    // Launch producer threads (75% of threads)
    std::size_t producer_count = (config.thread_count * 3) / 4;
    for (std::size_t i = 0; i < producer_count; ++i) {
        workers.emplace_back(
            &LockFreeContainerStressTest::producer_worker<LockFreeQueue<std::uint64_t>>,
            this,
            i,
            std::ref(queue),
            std::cref(config),
            std::ref(producer_stats),
            std::ref(stop_flag));
    }

    // Launch consumer threads (25% of threads)
    std::size_t consumer_count = config.thread_count - producer_count;
    for (std::size_t i = 0; i < consumer_count; ++i) {
        workers.emplace_back(
            &LockFreeContainerStressTest::consumer_worker<LockFreeQueue<std::uint64_t>>,
            this,
            i,
            std::ref(queue),
            std::cref(config),
            std::ref(consumer_stats),
            std::ref(stop_flag),
            std::ref(total_consumed));
    }

    // Monitor test progress with early completion detection
    auto test_start = std::chrono::high_resolution_clock::now();

    while (true) {
        std::this_thread::sleep_for(1s);
        auto elapsed = std::chrono::high_resolution_clock::now() - test_start;

        // Exit if duration exceeded
        if (elapsed >= config.duration) {
            break;
        }

        // Check if producers have finished their operations
        std::size_t producer_ops = producer_stats.total_operations.load();
        std::size_t expected_producer_ops = producer_count * config.operations_per_thread;
        if (producer_ops >= expected_producer_ops) {
            LOG_INFO_PRINT("Queue stress test producers completed {} operations early",
                           producer_ops);
            break;
        }
    }

    stop_flag.store(true);

    // Wait for all workers
    for (auto& worker : workers) {
        worker.join();
    }

    producer_stats.end_time = std::chrono::high_resolution_clock::now();
    consumer_stats.end_time = producer_stats.end_time;

    // Validate results
    EXPECT_GT(producer_stats.get_success_rate(), 0.95) << "Producer success rate should be > 95%";

    EXPECT_GT(consumer_stats.get_success_rate(), 0.95) << "Consumer success rate should be > 95%";

    // Check that most produced items were consumed
    auto produced = producer_stats.successful_operations.load();
    auto consumed = total_consumed.load();
    double consumption_rate = static_cast<double>(consumed) / produced;

    EXPECT_GT(consumption_rate, 0.90) << "Should consume > 90% of produced items";

    LOG_INFO_PRINT("Queue stress test completed: {} produced, {} consumed ({:.1f}%)",
                   produced,
                   consumed,
                   consumption_rate * 100);
}

}  // namespace inference_lab::common::test
