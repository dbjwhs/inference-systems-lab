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

#include <memory>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "../src/mixture_experts/moe_engine.hpp"

namespace {

using namespace engines::mixture_experts;

/**
 * @brief Global MoE engine for benchmarking
 */
std::unique_ptr<MoEEngine> g_moe_engine;

/**
 * @brief Initialize MoE engine for benchmarking
 */
void InitializeMoEEngine() {
    if (g_moe_engine)
        return;  // Already initialized

    MoEConfig config{.num_experts = 8,
                     .expert_capacity = 2,
                     .load_balancing_weight = 0.1f,
                     .enable_sparse_activation = true,
                     .max_concurrent_requests = 100,
                     .memory_pool_size_mb = 500};

    auto engine_result = MoEEngine::create(config);
    if (engine_result.is_err()) {
        throw std::runtime_error("Failed to create MoE engine for benchmarking");
    }
    g_moe_engine = std::move(engine_result).unwrap();
}

/**
 * @brief Generate random feature vector for benchmarking
 */
std::vector<float> GenerateRandomFeatures(std::size_t size, std::mt19937& gen) {
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::vector<float> features;
    features.reserve(size);

    for (std::size_t i = 0; i < size; ++i) {
        features.push_back(dis(gen));
    }

    return features;
}

/**
 * @brief Generate domain-specific text classification features
 */
std::vector<float> GenerateTextClassificationFeatures(std::size_t domain, std::mt19937& gen) {
    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);

    // Base patterns for different text domains
    std::vector<std::vector<float>> domain_patterns = {
        // News: high formality, factual, time-sensitive
        {0.8f, 0.2f, 0.9f, 0.1f, 0.7f, 0.3f, 0.8f, 0.4f},
        // Reviews: sentiment-heavy, subjective, product-focused
        {0.3f, 0.9f, 0.2f, 0.8f, 0.4f, 0.7f, 0.1f, 0.9f},
        // Technical: precise, structured, documentation-style
        {0.9f, 0.1f, 0.8f, 0.2f, 0.9f, 0.1f, 0.8f, 0.2f},
        // Social: conversational, emoji-rich, informal
        {0.4f, 0.6f, 0.3f, 0.7f, 0.5f, 0.5f, 0.4f, 0.6f},
        // Academic: research-focused, citation-heavy, methodical
        {0.7f, 0.8f, 0.9f, 0.9f, 0.8f, 0.7f, 0.9f, 0.8f}};

    auto base_pattern = domain_patterns[domain % domain_patterns.size()];

    // Add noise to make it more realistic
    for (auto& val : base_pattern) {
        val += noise(gen);
        val = std::max(0.0f, std::min(1.0f, val));  // Clamp to [0,1]
    }

    return base_pattern;
}

/**
 * @brief Generate computer vision task features
 */
std::vector<float> GenerateComputerVisionFeatures(std::size_t task, std::mt19937& gen) {
    std::uniform_real_distribution<float> noise(-0.05f, 0.05f);

    // Base patterns for different vision tasks
    std::vector<std::vector<float>> task_patterns = {
        // Object detection: edge-heavy, corner detection, geometric features
        {0.9f, 0.8f, 0.7f, 0.6f, 0.8f, 0.9f, 0.7f, 0.8f},
        // Scene classification: color distribution, spatial context
        {0.5f, 0.6f, 0.8f, 0.9f, 0.4f, 0.3f, 0.7f, 0.8f},
        // Facial recognition: symmetry, landmark detection, skin texture
        {0.8f, 0.9f, 0.6f, 0.7f, 0.8f, 0.8f, 0.9f, 0.7f}};

    auto base_pattern = task_patterns[task % task_patterns.size()];

    // Add minimal noise for realistic variation
    for (auto& val : base_pattern) {
        val += noise(gen);
        val = std::max(0.0f, std::min(1.0f, val));
    }

    return base_pattern;
}

/**
 * @brief Generate recommendation system context features
 */
std::vector<float> GenerateRecommendationFeatures(std::size_t context, std::mt19937& gen) {
    std::uniform_real_distribution<float> noise(-0.08f, 0.08f);

    // Base patterns for different recommendation contexts
    std::vector<std::vector<float>> context_patterns = {
        // Collaborative filtering: user similarity patterns
        {0.7f, 0.8f, 0.6f, 0.9f, 0.5f, 0.4f, 0.7f, 0.8f},
        // Content-based: item attribute matching
        {0.9f, 0.6f, 0.8f, 0.4f, 0.7f, 0.9f, 0.5f, 0.6f},
        // Demographic: age, location, behavior patterns
        {0.4f, 0.7f, 0.9f, 0.3f, 0.8f, 0.5f, 0.6f, 0.7f},
        // Hybrid: multiple signal combination
        {0.6f, 0.8f, 0.7f, 0.5f, 0.7f, 0.6f, 0.8f, 0.7f}};

    auto base_pattern = context_patterns[context % context_patterns.size()];

    // Add noise for realistic variation
    for (auto& val : base_pattern) {
        val += noise(gen);
        val = std::max(0.0f, std::min(1.0f, val));
    }

    return base_pattern;
}

/**
 * @brief Benchmark text classification MoE inference
 */
static void BM_MoETextClassification(benchmark::State& state) {
    InitializeMoEEngine();

    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::size_t request_id = 0;

    for (auto _ : state) {
        auto domain = request_id % 5;  // 5 different text domains
        auto features = GenerateTextClassificationFeatures(domain, gen);

        MoEInput input{.features = features,
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = request_id++,
                       .priority = 1.0f};

        auto response = g_moe_engine->run_inference(input);

        // Ensure successful inference
        if (response.is_err()) {
            state.SkipWithError("MoE inference failed");
            break;
        }

        // Prevent compiler optimization
        auto result = response.unwrap();
        benchmark::DoNotOptimize(result.outputs.data());
        benchmark::DoNotOptimize(result.outputs.size());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MoETextClassification)->MinTime(1.0);

/**
 * @brief Benchmark computer vision MoE inference
 */
static void BM_MoEComputerVision(benchmark::State& state) {
    InitializeMoEEngine();

    std::mt19937 gen(123);          // Different seed for vision tasks
    std::size_t request_id = 1000;  // Different ID range

    for (auto _ : state) {
        auto task = request_id % 3;  // 3 different vision tasks
        auto features = GenerateComputerVisionFeatures(task, gen);

        MoEInput input{.features = features,
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = request_id++,
                       .priority = 1.0f};

        auto response = g_moe_engine->run_inference(input);

        // Ensure successful inference
        if (response.is_err()) {
            state.SkipWithError("MoE inference failed");
            break;
        }

        // Prevent compiler optimization
        auto result = response.unwrap();
        benchmark::DoNotOptimize(result.outputs.data());
        benchmark::DoNotOptimize(result.outputs.size());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MoEComputerVision)->MinTime(1.0);

/**
 * @brief Benchmark recommendation system MoE inference
 */
static void BM_MoERecommendation(benchmark::State& state) {
    InitializeMoEEngine();

    std::mt19937 gen(456);          // Different seed for recommendation
    std::size_t request_id = 2000;  // Different ID range

    for (auto _ : state) {
        auto context = request_id % 4;  // 4 different recommendation contexts
        auto features = GenerateRecommendationFeatures(context, gen);

        MoEInput input{.features = features,
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = request_id++,
                       .priority = 1.0f};

        auto response = g_moe_engine->run_inference(input);

        // Ensure successful inference
        if (response.is_err()) {
            state.SkipWithError("MoE inference failed");
            break;
        }

        // Prevent compiler optimization
        auto result = response.unwrap();
        benchmark::DoNotOptimize(result.outputs.data());
        benchmark::DoNotOptimize(result.outputs.size());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MoERecommendation)->MinTime(1.0);

/**
 * @brief Benchmark mixed workload (all demo types)
 */
static void BM_MoEMixedWorkload(benchmark::State& state) {
    InitializeMoEEngine();

    std::mt19937 gen(789);          // Different seed for mixed workload
    std::size_t request_id = 3000;  // Different ID range

    for (auto _ : state) {
        std::vector<float> features;

        // Randomly choose workload type
        auto workload_type = request_id % 3;
        switch (workload_type) {
            case 0:
                features = GenerateTextClassificationFeatures(request_id % 5, gen);
                break;
            case 1:
                features = GenerateComputerVisionFeatures(request_id % 3, gen);
                break;
            case 2:
                features = GenerateRecommendationFeatures(request_id % 4, gen);
                break;
        }

        MoEInput input{.features = features,
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = request_id++,
                       .priority = 1.0f};

        auto response = g_moe_engine->run_inference(input);

        // Ensure successful inference
        if (response.is_err()) {
            state.SkipWithError("MoE inference failed");
            break;
        }

        // Prevent compiler optimization
        auto result = response.unwrap();
        benchmark::DoNotOptimize(result.outputs.data());
        benchmark::DoNotOptimize(result.outputs.size());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MoEMixedWorkload)->MinTime(1.0);

/**
 * @brief Benchmark batch processing
 */
static void BM_MoEBatchProcessing(benchmark::State& state) {
    InitializeMoEEngine();

    const auto batch_size = static_cast<std::size_t>(state.range(0));
    std::mt19937 gen(999);
    std::size_t request_id = 4000;

    for (auto _ : state) {
        for (std::size_t i = 0; i < batch_size; ++i) {
            auto features = GenerateRandomFeatures(8, gen);

            MoEInput input{.features = features,
                           .batch_size = 1,  // Individual requests in batch
                           .enable_load_balancing = true,
                           .request_id = request_id++,
                           .priority = 1.0f};

            auto response = g_moe_engine->run_inference(input);

            // Ensure successful inference
            if (response.is_err()) {
                state.SkipWithError("MoE batch inference failed");
                break;
            }

            // Prevent compiler optimization
            auto result = response.unwrap();
            benchmark::DoNotOptimize(result.outputs.data());
            benchmark::DoNotOptimize(result.outputs.size());
        }
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_MoEBatchProcessing)->Arg(1)->Arg(5)->Arg(10)->Arg(20)->Arg(50);

/**
 * @brief Benchmark expert utilization tracking
 */
static void BM_MoEExpertUtilization(benchmark::State& state) {
    InitializeMoEEngine();

    std::mt19937 gen(111);
    std::size_t request_id = 5000;

    for (auto _ : state) {
        // Run inference
        auto features = GenerateRandomFeatures(8, gen);

        MoEInput input{.features = features,
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = request_id++,
                       .priority = 1.0f};

        auto response = g_moe_engine->run_inference(input);

        // Get utilization metrics
        auto utilization = g_moe_engine->get_expert_utilization();

        // Prevent compiler optimization
        benchmark::DoNotOptimize(utilization);

        if (response.is_err()) {
            state.SkipWithError("MoE utilization benchmark failed");
            break;
        }
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MoEExpertUtilization);

/**
 * @brief Benchmark memory usage tracking
 */
static void BM_MoEMemoryUsage(benchmark::State& state) {
    InitializeMoEEngine();

    std::mt19937 gen(222);
    std::size_t request_id = 6000;

    for (auto _ : state) {
        // Run inference
        auto features = GenerateRandomFeatures(8, gen);

        MoEInput input{.features = features,
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = request_id++,
                       .priority = 1.0f};

        auto response = g_moe_engine->run_inference(input);

        // Get memory usage metrics
        auto memory_usage = g_moe_engine->get_memory_usage();

        // Prevent compiler optimization
        benchmark::DoNotOptimize(memory_usage);

        if (response.is_err()) {
            state.SkipWithError("MoE memory usage benchmark failed");
            break;
        }
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MoEMemoryUsage);

/**
 * @brief Benchmark system health validation
 */
static void BM_MoESystemHealth(benchmark::State& state) {
    InitializeMoEEngine();

    for (auto _ : state) {
        auto health_result = g_moe_engine->validate_system_health();

        // Prevent compiler optimization
        benchmark::DoNotOptimize(health_result);

        if (health_result.is_err()) {
            state.SkipWithError("MoE system health validation failed");
            break;
        }
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MoESystemHealth);

}  // anonymous namespace
