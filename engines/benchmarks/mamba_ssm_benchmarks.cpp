// MIT License
// Copyright (c) 2025 Inference Systems Lab

#include <benchmark/benchmark.h>

#include "../src/mamba_ssm/mamba_ssm.hpp"

using namespace inference_lab::engines::mamba_ssm;

namespace inference_lab::engines::mamba_ssm {
namespace benchmark {

class MambaSSMFixture : public ::benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State& state) override {
        // Configure Mamba SSM engine with reasonable defaults
        config_.d_model = 256;
        config_.d_state = 16;
        config_.d_inner = 512;
        config_.d_conv = 4;
        config_.max_seq_len = 2048;
        config_.batch_size = 1;
        config_.use_simd_kernels = true;
        config_.activation = MambaSSMConfig::ActivationType::SILU;

        auto engine_result = create_mamba_ssm_engine(config_);
        if (engine_result.is_ok()) {
            engine_ = std::move(engine_result).unwrap();
        }

        // Create test sequences of different lengths
        small_sequence_ = create_test_sequence(1, 32, config_.d_model);
        medium_sequence_ = create_test_sequence(1, 128, config_.d_model);
        large_sequence_ = create_test_sequence(1, 512, config_.d_model);
    }

    void TearDown(const ::benchmark::State& state) override { engine_.reset(); }

  protected:
    MambaSSMConfig config_;
    std::unique_ptr<MambaSSMEngine> engine_;
    FloatTensor small_sequence_;
    FloatTensor medium_sequence_;
    FloatTensor large_sequence_;

    FloatTensor create_test_sequence(size_t batch, size_t seq_len, size_t d_model) {
        FloatTensor sequence(Shape{batch, seq_len, d_model});

        auto data = sequence.data();
        for (size_t i = 0; i < sequence.size(); ++i) {
            // Create a realistic test pattern with some structure
            float t = static_cast<float>(i % seq_len) / static_cast<float>(seq_len);
            float d = static_cast<float>((i / seq_len) % d_model) / static_cast<float>(d_model);
            data[i] = std::sin(2.0f * 3.14159f * t) * 0.5f + std::cos(4.0f * 3.14159f * d) * 0.3f +
                      0.1f * static_cast<float>(i % 7);
        }

        return sequence;
    }
};

BENCHMARK_F(MambaSSMFixture, SmallSequenceInference)(::benchmark::State& state) {
    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = engine_->run_mamba_ssm(small_sequence_);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto output = std::move(result).unwrap();
            ::benchmark::DoNotOptimize(output);
        }
    }

    state.SetItemsProcessed(state.iterations() * 32);  // 32 tokens per iteration
    auto metrics = engine_->get_metrics();
    state.counters["throughput_tokens_per_sec"] = metrics.throughput_tokens_per_sec;
    state.counters["selective_scan_time_us"] = metrics.selective_scan_time_ms.count();
    state.counters["average_step_size"] = metrics.average_step_size;
}

BENCHMARK_F(MambaSSMFixture, MediumSequenceInference)(::benchmark::State& state) {
    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = engine_->run_mamba_ssm(medium_sequence_);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto output = std::move(result).unwrap();
            ::benchmark::DoNotOptimize(output);
        }
    }

    state.SetItemsProcessed(state.iterations() * 128);  // 128 tokens per iteration
    auto metrics = engine_->get_metrics();
    state.counters["throughput_tokens_per_sec"] = metrics.throughput_tokens_per_sec;
    state.counters["gflops"] = metrics.flops_per_second / 1e9;
    state.counters["selective_updates"] = metrics.selective_updates;
}

BENCHMARK_F(MambaSSMFixture, LargeSequenceInference)(::benchmark::State& state) {
    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = engine_->run_mamba_ssm(large_sequence_);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto output = std::move(result).unwrap();
            ::benchmark::DoNotOptimize(output);
        }
    }

    state.SetItemsProcessed(state.iterations() * 512);  // 512 tokens per iteration
    auto metrics = engine_->get_metrics();
    state.counters["inference_time_ms"] = metrics.inference_time_ms.count() / 1000.0;
    state.counters["memory_usage_mb"] = metrics.memory_usage_bytes / (1024.0 * 1024.0);
}

BENCHMARK_F(MambaSSMFixture, LinearComplexityScaling)(::benchmark::State& state) {
    // Benchmark linear complexity scaling with sequence length
    size_t seq_len = state.range(0);
    auto test_sequence = create_test_sequence(1, seq_len, config_.d_model);

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = engine_->run_mamba_ssm(test_sequence);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto output = std::move(result).unwrap();
            ::benchmark::DoNotOptimize(output);
        }
    }

    state.SetItemsProcessed(state.iterations() * seq_len);
    state.SetComplexityN(seq_len);

    auto metrics = engine_->get_metrics();
    state.counters["tokens_per_sec"] = metrics.throughput_tokens_per_sec;
    state.counters["flops_per_token"] = static_cast<double>(metrics.total_flops) / seq_len;
}
BENCHMARK_REGISTER_F(MambaSSMFixture, LinearComplexityScaling)
    ->Range(16, 1024)               // Test sequence lengths from 16 to 1024
    ->Complexity(::benchmark::oN);  // Expected linear complexity

BENCHMARK_F(MambaSSMFixture, BatchSizeScaling)(::benchmark::State& state) {
    // Test performance with different batch sizes
    size_t batch_size = state.range(0);
    MambaSSMConfig batch_config = config_;
    batch_config.batch_size = batch_size;

    auto batch_engine_result = create_mamba_ssm_engine(batch_config);
    auto batch_engine = std::move(batch_engine_result).unwrap();

    auto batch_sequence = create_test_sequence(batch_size, 128, config_.d_model);

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = batch_engine->run_mamba_ssm(batch_sequence);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto output = std::move(result).unwrap();
            ::benchmark::DoNotOptimize(output);
        }
    }

    state.SetItemsProcessed(state.iterations() * batch_size * 128);
    auto metrics = batch_engine->get_metrics();
    state.counters["batch_throughput"] = metrics.throughput_tokens_per_sec * batch_size;
}
BENCHMARK_REGISTER_F(MambaSSMFixture, BatchSizeScaling)->Range(1, 8);  // Test batch sizes from 1 to
                                                                       // 8

BENCHMARK_F(MambaSSMFixture, ModelDimensionScaling)(::benchmark::State& state) {
    // Test performance scaling with model dimensions
    size_t d_model = state.range(0);
    MambaSSMConfig dim_config = config_;
    dim_config.d_model = d_model;
    dim_config.d_inner = 2 * d_model;                  // Typical ratio
    dim_config.d_state = std::max(8UL, d_model / 16);  // Scale state dimension

    auto dim_engine_result = create_mamba_ssm_engine(dim_config);
    if (!dim_engine_result.is_ok()) {
        state.SkipWithError("Failed to create engine with dimensions");
        return;
    }
    auto dim_engine = std::move(dim_engine_result).unwrap();

    auto dim_sequence = create_test_sequence(1, 64, d_model);

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = dim_engine->run_mamba_ssm(dim_sequence);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto output = std::move(result).unwrap();
            ::benchmark::DoNotOptimize(output);
        }
    }

    state.SetItemsProcessed(state.iterations() * 64);
    auto metrics = dim_engine->get_metrics();
    state.counters["model_dimension"] = d_model;
    state.counters["parameters_approx"] =
        d_model * dim_config.d_inner + dim_config.d_inner * dim_config.d_state;
}
BENCHMARK_REGISTER_F(MambaSSMFixture, ModelDimensionScaling)->Range(64, 512);  // Test model
                                                                               // dimensions from 64
                                                                               // to 512

BENCHMARK_F(MambaSSMFixture, SIMDKernelComparison)(::benchmark::State& state) {
    // Compare performance with and without SIMD kernels
    bool use_simd = state.range(0) != 0;
    MambaSSMConfig simd_config = config_;
    simd_config.use_simd_kernels = use_simd;

    auto simd_engine_result = create_mamba_ssm_engine(simd_config);
    auto simd_engine = std::move(simd_engine_result).unwrap();

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = simd_engine->run_mamba_ssm(medium_sequence_);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto output = std::move(result).unwrap();
            ::benchmark::DoNotOptimize(output);
        }
    }

    state.SetItemsProcessed(state.iterations() * 128);
    state.SetLabel(use_simd ? "SIMD_Enabled" : "SIMD_Disabled");

    auto metrics = simd_engine->get_metrics();
    state.counters["throughput"] = metrics.throughput_tokens_per_sec;
    state.counters["scan_time_us"] = metrics.selective_scan_time_ms.count();
}
BENCHMARK_REGISTER_F(MambaSSMFixture, SIMDKernelComparison)
    ->Arg(0)   // SIMD disabled
    ->Arg(1);  // SIMD enabled

BENCHMARK_F(MambaSSMFixture, ActivationFunctionComparison)(::benchmark::State& state) {
    // Compare different activation functions
    auto activation = static_cast<MambaSSMConfig::ActivationType>(state.range(0));
    MambaSSMConfig act_config = config_;
    act_config.activation = activation;

    auto act_engine_result = create_mamba_ssm_engine(act_config);
    auto act_engine = std::move(act_engine_result).unwrap();

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = act_engine->run_mamba_ssm(medium_sequence_);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto output = std::move(result).unwrap();
            ::benchmark::DoNotOptimize(output);
        }
    }

    state.SetItemsProcessed(state.iterations() * 128);

    const char* activation_names[] = {"SiLU", "GELU", "ReLU"};
    state.SetLabel(activation_names[state.range(0)]);

    auto metrics = act_engine->get_metrics();
    state.counters["inference_time_us"] = metrics.inference_time_ms.count();
}
BENCHMARK_REGISTER_F(MambaSSMFixture, ActivationFunctionComparison)
    ->Arg(0)   // SiLU
    ->Arg(1)   // GELU
    ->Arg(2);  // ReLU

BENCHMARK_F(MambaSSMFixture, StateSpaceDimensionScaling)(::benchmark::State& state) {
    // Test scaling with different state space dimensions
    size_t d_state = state.range(0);
    MambaSSMConfig state_config = config_;
    state_config.d_state = d_state;

    auto state_engine_result = create_mamba_ssm_engine(state_config);
    if (!state_engine_result.is_ok()) {
        state.SkipWithError("Failed to create engine with state dimension");
        return;
    }
    auto state_engine = std::move(state_engine_result).unwrap();

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = state_engine->run_mamba_ssm(medium_sequence_);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto output = std::move(result).unwrap();
            ::benchmark::DoNotOptimize(output);
        }
    }

    state.SetItemsProcessed(state.iterations() * 128);
    state.SetComplexityN(d_state);

    auto metrics = state_engine->get_metrics();
    state.counters["state_dimension"] = d_state;
    state.counters["memory_per_token"] = static_cast<double>(metrics.memory_usage_bytes) / 128;
}
BENCHMARK_REGISTER_F(MambaSSMFixture, StateSpaceDimensionScaling)
    ->Range(8, 64)  // Test state dimensions from 8 to 64
    ->Complexity();

BENCHMARK_F(MambaSSMFixture, EngineCreationCost)(::benchmark::State& state) {
    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto engine_result = create_mamba_ssm_engine(config_);
        ::benchmark::DoNotOptimize(engine_result);

        if (engine_result.is_ok()) {
            auto engine = std::move(engine_result).unwrap();
            ::benchmark::DoNotOptimize(engine);
        }
    }

    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_F(MambaSSMFixture, StateResetPerformance)(::benchmark::State& state) {
    // Benchmark state reset operations
    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        engine_->reset_state();
        ::benchmark::DoNotOptimize(engine_.get());
    }

    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_F(MambaSSMFixture, UnifiedInterfaceOverhead)(::benchmark::State& state) {
    // Compare direct SSM call vs unified interface
    bool use_unified = state.range(0) != 0;

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        if (use_unified) {
            // Use unified interface
            InferenceRequest request;
            auto result = engine_->run_inference(request);
            ::benchmark::DoNotOptimize(result);
        } else {
            // Use direct SSM interface
            auto result = engine_->run_mamba_ssm(medium_sequence_);
            ::benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel(use_unified ? "Unified_Interface" : "Direct_SSM");

    auto metrics = engine_->get_metrics();
    state.counters["inference_time_us"] = metrics.inference_time_ms.count();
}
BENCHMARK_REGISTER_F(MambaSSMFixture, UnifiedInterfaceOverhead)
    ->Arg(0)   // Direct interface
    ->Arg(1);  // Unified interface

// Comparative benchmark against hypothetical Transformer O(n²) complexity
BENCHMARK_F(MambaSSMFixture, ComplexityComparison)(::benchmark::State& state) {
    size_t seq_len = state.range(0);
    auto comparison_sequence = create_test_sequence(1, seq_len, config_.d_model);

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = engine_->run_mamba_ssm(comparison_sequence);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto output = std::move(result).unwrap();
            ::benchmark::DoNotOptimize(output);
        }
    }

    state.SetItemsProcessed(state.iterations() * seq_len);

    auto metrics = engine_->get_metrics();
    // Calculate theoretical O(n²) cost for comparison
    double quadratic_cost = static_cast<double>(seq_len * seq_len);
    double linear_cost = static_cast<double>(seq_len);
    double efficiency_ratio = quadratic_cost / linear_cost;  // How much better than O(n²)

    state.counters["sequence_length"] = seq_len;
    state.counters["actual_time_us"] = metrics.inference_time_ms.count();
    state.counters["efficiency_vs_quadratic"] = efficiency_ratio;
    state.counters["flops_per_token"] = static_cast<double>(metrics.total_flops) / seq_len;
}
BENCHMARK_REGISTER_F(MambaSSMFixture, ComplexityComparison)
    ->Range(32, 2048)  // Test range where O(n²) vs O(n) difference is significant
    ->Complexity(::benchmark::oN);

}  // namespace benchmark
}  // namespace inference_lab::engines::mamba_ssm
