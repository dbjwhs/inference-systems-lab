#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>

#include "../moe_config.hpp"
#include "../moe_engine.hpp"

namespace engines::mixture_experts {

// Global test engine for benchmarks
static std::unique_ptr<MoEEngine> g_test_engine;
static std::mt19937 g_rng(12345);  // Fixed seed for reproducibility

// Setup function called before benchmarks
static void SetupMoEEngine() {
    if (!g_test_engine) {
        MoEConfig config{};
        config.num_experts = 8;
        config.expert_capacity = 2;
        config.max_concurrent_requests = 100;
        config.memory_pool_size_mb = 500;
        config.enable_sparse_activation = true;

        auto engine_result = MoEEngine::create(config);
        if (engine_result.is_ok()) {
            g_test_engine = std::move(engine_result).unwrap();
        }
    }
}

// Helper function to generate random feature vectors
static std::vector<float> GenerateRandomFeatures(std::size_t dimension, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> features(dimension);
    for (auto& feature : features) {
        feature = dist(rng);
    }
    return features;
}

// Benchmark: End-to-end MoE inference latency
static void BM_MoEInference_EndToEnd(benchmark::State& state) {
    SetupMoEEngine();
    if (!g_test_engine) {
        state.SkipWithError("Failed to create MoE engine");
        return;
    }

    std::size_t feature_dim = static_cast<std::size_t>(state.range(0));
    std::size_t request_id = 0;

    // Warm up
    for (int i = 0; i < 10; ++i) {
        MoEInput input;
        input.features = GenerateRandomFeatures(feature_dim, g_rng);
        input.request_id = request_id++;
        input.priority = 1.0f;
        g_test_engine->run_inference(input);
    }

    for (auto _ : state) {
        MoEInput input;
        input.features = GenerateRandomFeatures(feature_dim, g_rng);
        input.request_id = request_id++;
        input.priority = 1.0f;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = g_test_engine->run_inference(input);
        auto end = std::chrono::high_resolution_clock::now();

        if (result.is_ok()) {
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            state.SetIterationTime(latency.count() / 1e9);  // Convert to seconds
        } else {
            state.SkipWithError("Inference failed");
            break;
        }
    }

    // Calculate and report throughput
    state.counters["requests_per_second"] =
        benchmark::Counter(static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);

    state.counters["feature_dim"] = static_cast<double>(feature_dim);
    state.SetComplexityN(static_cast<int64_t>(feature_dim));
}

BENCHMARK(BM_MoEInference_EndToEnd)
    ->RangeMultiplier(2)
    ->Range(64, 1024)
    ->UseManualTime()
    ->Complexity();

// Benchmark: Expert routing latency
static void BM_ExpertRouting_Latency(benchmark::State& state) {
    // Create router directly for focused benchmarking
    RouterConfig config{};
    config.num_experts = static_cast<std::size_t>(state.range(0));
    config.top_k_experts = 2;
    config.hidden_dimension = 128;
    config.enable_gradient_computation = true;

    auto router_result = ExpertRouter::create(config);
    if (!router_result.is_ok()) {
        state.SkipWithError("Failed to create expert router");
        return;
    }
    auto router = std::move(router_result).unwrap();

    std::size_t feature_dim = 256;

    for (auto _ : state) {
        auto features = GenerateRandomFeatures(feature_dim, g_rng);

        auto start = std::chrono::high_resolution_clock::now();
        auto result = router->select_experts(features);
        auto end = std::chrono::high_resolution_clock::now();

        if (result.is_ok()) {
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            state.SetIterationTime(latency.count() / 1e9);
        } else {
            state.SkipWithError("Expert routing failed");
            break;
        }
    }

    state.counters["num_experts"] = static_cast<double>(config.num_experts);
    state.counters["selections_per_second"] =
        benchmark::Counter(static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);

    state.SetComplexityN(static_cast<int64_t>(config.num_experts));
}

BENCHMARK(BM_ExpertRouting_Latency)
    ->RangeMultiplier(2)
    ->Range(4, 64)
    ->UseManualTime()
    ->Complexity();

// Benchmark: Load balancer performance under high concurrency
static void BM_LoadBalancer_ConcurrentRequests(benchmark::State& state) {
    LoadBalancerConfig config{};
    config.num_experts = 8;
    config.max_queue_size_per_expert = 50;
    config.enable_adaptive_routing = true;

    auto lb_result = LoadBalancer::create(config);
    if (!lb_result.is_ok()) {
        state.SkipWithError("Failed to create load balancer");
        return;
    }
    auto load_balancer = std::move(lb_result).unwrap();

    std::size_t num_threads = static_cast<std::size_t>(state.range(0));
    std::atomic<std::size_t> request_counter{0};

    for (auto _ : state) {
        std::vector<std::thread> threads;
        std::atomic<int> successful_selections{0};

        auto start = std::chrono::high_resolution_clock::now();

        // Launch concurrent threads
        for (std::size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&]() {
                std::vector<std::size_t> candidates = {0, 1, 2, 3};
                std::vector<float> weights = {0.25f, 0.25f, 0.25f, 0.25f};

                auto result = load_balancer->select_optimal_expert(candidates, weights);
                if (result.is_ok()) {
                    successful_selections.fetch_add(1);
                }
            });
        }

        // Wait for all threads
        for (auto& thread : threads) {
            thread.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        state.SetIterationTime(latency.count() / 1e9);

        if (successful_selections.load() < static_cast<int>(num_threads * 0.9f)) {
            state.SkipWithError("Too many load balancer failures");
            break;
        }
    }

    state.counters["concurrent_threads"] = static_cast<double>(num_threads);
    state.counters["selections_per_second"] = benchmark::Counter(
        static_cast<double>(state.iterations() * num_threads), benchmark::Counter::kIsRate);

    state.SetComplexityN(static_cast<int64_t>(num_threads));
}

BENCHMARK(BM_LoadBalancer_ConcurrentRequests)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->UseManualTime()
    ->Complexity();

// Benchmark: Memory usage scaling with number of experts
static void BM_MoE_MemoryScaling(benchmark::State& state) {
    std::size_t num_experts = static_cast<std::size_t>(state.range(0));

    MoEConfig config{};
    config.num_experts = num_experts;
    config.expert_capacity = 2;
    config.max_concurrent_requests = 50;
    config.memory_pool_size_mb = 100;  // Fixed per-expert memory
    config.enable_sparse_activation = true;

    for (auto _ : state) {
        auto engine_result = MoEEngine::create(config);
        if (!engine_result.is_ok()) {
            state.SkipWithError("Failed to create MoE engine");
            break;
        }

        auto engine = std::move(engine_result).unwrap();

        // Measure memory usage
        auto memory_usage = engine->get_memory_usage();
        float total_memory = 0.0f;
        for (auto usage : memory_usage) {
            total_memory += usage;
        }

        state.counters["total_memory_mb"] = static_cast<double>(total_memory);
        state.counters["memory_per_expert_mb"] = static_cast<double>(total_memory / num_experts);

        // Ensure engine is actually functioning
        MoEInput input;
        input.features = GenerateRandomFeatures(256, g_rng);
        input.request_id = 1;
        auto response = engine->run_inference(input);
        if (!response.is_ok()) {
            state.SkipWithError("Engine inference failed");
            break;
        }
    }

    state.counters["num_experts"] = static_cast<double>(num_experts);
    state.SetComplexityN(static_cast<int64_t>(num_experts));
}

BENCHMARK(BM_MoE_MemoryScaling)->RangeMultiplier(2)->Range(2, 32)->Complexity();

// Benchmark: Sparse activation performance
static void BM_SparseActivation_Performance(benchmark::State& state) {
    SparseConfig config{};
    config.enable_simd_optimization = true;

    auto sparse_result = SparseActivation::create(config);
    if (!sparse_result.is_ok()) {
        state.SkipWithError("Failed to create sparse activation");
        return;
    }
    auto sparse_activation = std::move(sparse_result).unwrap();

    std::size_t vector_size = static_cast<std::size_t>(state.range(0));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto _ : state) {
        // Generate input and weights
        std::vector<float> input = GenerateRandomFeatures(vector_size, g_rng);
        std::vector<float> expert_weights(4);  // 4 experts
        for (auto& weight : expert_weights) {
            weight = dist(g_rng);
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto result = sparse_activation->apply_sparse_activation(input, expert_weights);
        auto end = std::chrono::high_resolution_clock::now();

        if (result.is_ok()) {
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            state.SetIterationTime(latency.count() / 1e9);

            // Calculate throughput in elements processed per second
            auto elements_per_second = static_cast<double>(vector_size) / (latency.count() / 1e9);
            state.counters["elements_per_second"] =
                benchmark::Counter(elements_per_second, benchmark::Counter::kIsRate);
        } else {
            state.SkipWithError("Sparse activation failed");
            break;
        }
    }

    state.counters["vector_size"] = static_cast<double>(vector_size);
    state.SetComplexityN(static_cast<int64_t>(vector_size));
}

BENCHMARK(BM_SparseActivation_Performance)
    ->RangeMultiplier(2)
    ->Range(256, 8192)
    ->UseManualTime()
    ->Complexity();

// Benchmark: Expert parameter loading latency
static void BM_ExpertParameters_LoadingLatency(benchmark::State& state) {
    ParameterConfig config{};
    config.num_experts = 16;
    config.memory_pool_size_mb = 500;

    auto params_result = ExpertParameters::create(config);
    if (!params_result.is_ok()) {
        state.SkipWithError("Failed to create expert parameters");
        return;
    }
    auto expert_params = std::move(params_result).unwrap();

    std::uniform_int_distribution<std::size_t> expert_dist(0, config.num_experts - 1);

    for (auto _ : state) {
        std::size_t expert_id = expert_dist(g_rng);

        auto start = std::chrono::high_resolution_clock::now();
        auto handle_result = expert_params->get_expert_handle(expert_id);
        auto end = std::chrono::high_resolution_clock::now();

        if (handle_result.is_ok()) {
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            state.SetIterationTime(latency.count() / 1e9);
        } else {
            state.SkipWithError("Parameter loading failed");
            break;
        }
    }

    state.counters["loads_per_second"] =
        benchmark::Counter(static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_ExpertParameters_LoadingLatency)->UseManualTime()->Iterations(1000);

// Benchmark: Throughput scaling with concurrent requests
static void BM_MoE_ThroughputScaling(benchmark::State& state) {
    SetupMoEEngine();
    if (!g_test_engine) {
        state.SkipWithError("Failed to create MoE engine");
        return;
    }

    std::size_t num_concurrent = static_cast<std::size_t>(state.range(0));

    for (auto _ : state) {
        std::vector<std::thread> threads;
        std::atomic<int> successful_requests{0};
        std::atomic<std::size_t> request_id_counter{0};

        auto start = std::chrono::high_resolution_clock::now();

        // Launch concurrent request threads
        for (std::size_t t = 0; t < num_concurrent; ++t) {
            threads.emplace_back([&]() {
                MoEInput input;
                input.features = GenerateRandomFeatures(256, g_rng);
                input.request_id = request_id_counter.fetch_add(1);
                input.priority = 1.0f;

                auto result = g_test_engine->run_inference(input);
                if (result.is_ok()) {
                    successful_requests.fetch_add(1);
                }
            });
        }

        // Wait for all requests to complete
        for (auto& thread : threads) {
            thread.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        state.SetIterationTime(duration.count() / 1e9);

        if (successful_requests.load() < static_cast<int>(num_concurrent * 0.95f)) {
            state.SkipWithError("Too many request failures");
            break;
        }
    }

    state.counters["concurrent_requests"] = static_cast<double>(num_concurrent);
    state.counters["requests_per_second"] = benchmark::Counter(
        static_cast<double>(state.iterations() * num_concurrent), benchmark::Counter::kIsRate);

    state.SetComplexityN(static_cast<int64_t>(num_concurrent));
}

BENCHMARK(BM_MoE_ThroughputScaling)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->UseManualTime()
    ->Complexity();

// Benchmark: Configuration optimization impact
static void BM_MoE_ConfigurationComparison(benchmark::State& state) {
    // Test different configurations
    std::vector<MoEConfig> configs;

    // Configuration 1: Development (lightweight)
    MoEConfig dev_config{};
    dev_config.num_experts = 4;
    dev_config.expert_capacity = 1;
    dev_config.memory_pool_size_mb = 100;
    dev_config.enable_sparse_activation = false;
    configs.push_back(dev_config);

    // Configuration 2: Production (optimized)
    MoEConfig prod_config{};
    prod_config.num_experts = 8;
    prod_config.expert_capacity = 2;
    prod_config.memory_pool_size_mb = 500;
    prod_config.enable_sparse_activation = true;
    configs.push_back(prod_config);

    // Configuration 3: Performance (high-end)
    MoEConfig perf_config{};
    perf_config.num_experts = 16;
    perf_config.expert_capacity = 4;
    perf_config.memory_pool_size_mb = 1000;
    perf_config.enable_sparse_activation = true;
    configs.push_back(perf_config);

    std::size_t config_index = static_cast<std::size_t>(state.range(0));
    if (config_index >= configs.size()) {
        state.SkipWithError("Invalid configuration index");
        return;
    }

    auto& config = configs[config_index];

    auto engine_result = MoEEngine::create(config);
    if (!engine_result.is_ok()) {
        state.SkipWithError("Failed to create MoE engine");
        return;
    }
    auto engine = std::move(engine_result).unwrap();

    // Warm up
    for (int i = 0; i < 5; ++i) {
        MoEInput input;
        input.features = GenerateRandomFeatures(256, g_rng);
        input.request_id = static_cast<std::size_t>(i);
        engine->run_inference(input);
    }

    std::size_t request_id = 0;
    for (auto _ : state) {
        MoEInput input;
        input.features = GenerateRandomFeatures(256, g_rng);
        input.request_id = request_id++;
        input.priority = 1.0f;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = engine->run_inference(input);
        auto end = std::chrono::high_resolution_clock::now();

        if (result.is_ok()) {
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            state.SetIterationTime(latency.count() / 1e9);

            auto response = std::move(result).unwrap();
            state.counters["routing_latency_ms"] += response.routing_latency_ms;
            state.counters["inference_latency_ms"] += response.inference_latency_ms;
        } else {
            state.SkipWithError("Inference failed");
            break;
        }
    }

    // Calculate averages
    state.counters["routing_latency_ms"] /= static_cast<double>(state.iterations());
    state.counters["inference_latency_ms"] /= static_cast<double>(state.iterations());

    state.counters["config_index"] = static_cast<double>(config_index);
    state.counters["num_experts"] = static_cast<double>(config.num_experts);
    state.counters["expert_capacity"] = static_cast<double>(config.expert_capacity);
    state.counters["requests_per_second"] =
        benchmark::Counter(static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_MoE_ConfigurationComparison)->DenseRange(0, 2)->UseManualTime();

}  // namespace engines::mixture_experts

BENCHMARK_MAIN();
