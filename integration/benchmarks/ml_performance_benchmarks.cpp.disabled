// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file ml_performance_benchmarks.cpp
 * @brief Performance benchmarks for ML integration testing framework
 *
 * This file contains comprehensive performance benchmarks for the ML integration
 * testing framework, measuring performance characteristics across different
 * backends, model types, and workload scenarios. The benchmarks are designed
 * to provide baseline measurements and detect performance regressions.
 *
 * Benchmark Categories:
 * - Single backend latency and throughput benchmarks
 * - Cross-backend performance comparison benchmarks
 * - Memory allocation and deallocation benchmarks
 * - Concurrent execution performance benchmarks
 * - Mock engine overhead benchmarks
 * - Framework setup and teardown benchmarks
 *
 * The benchmarks use Google Benchmark for accurate timing measurements
 * and statistical analysis.
 */

#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include "../../common/src/logging.hpp"
#include "../../common/src/ml_types.hpp"
#include "../src/integration_test_utils.hpp"
#include "../src/ml_integration_framework.hpp"
#include "../src/mock_engines.hpp"

using namespace inference_lab::integration;
using namespace inference_lab::integration::utils;
using namespace inference_lab::integration::mocks;
using namespace inference_lab::common;
using namespace inference_lab::common::ml;
using namespace inference_lab::engines;

//=============================================================================
// Benchmark Setup and Utilities
//=============================================================================

/**
 * @brief Global benchmark setup
 */
class BenchmarkSetup {
  public:
    static auto instance() -> BenchmarkSetup& {
        static BenchmarkSetup setup;
        return setup;
    }

    auto get_framework() -> MLIntegrationFramework* { return framework_.get(); }

    auto get_classification_fixture() -> std::shared_ptr<ClassificationTestFixture> {
        return classification_fixture_;
    }

    auto get_test_inputs() -> const std::vector<InferenceRequest>& { return test_inputs_; }

  private:
    BenchmarkSetup() {
        // Set up logging with minimal output for benchmarks
        setup_test_environment("ERROR");

        // Create framework with optimized mock engines
        auto mock_factory = std::make_unique<TestBackendFactory>();
        test_factory_ = mock_factory.get();
        framework_ = std::make_unique<MLIntegrationFramework>(std::move(mock_factory));

        // Set up lightweight mock engines for benchmarking
        setup_benchmark_mocks();

        // Create test fixtures
        classification_fixture_ =
            std::make_shared<ClassificationTestFixture>(ClassificationTestFixture::create()
                                                            .with_model("benchmark_model")
                                                            .with_input_shape({1, 3, 224, 224})
                                                            .with_num_classes(1000));

        // Pre-generate test inputs
        test_inputs_ = classification_fixture_->generate_test_data(100);
    }

    void setup_benchmark_mocks() {
        // TensorRT mock optimized for benchmarking
        auto tensorrt_config = MockTensorRTEngine::create_tensorrt_config();
        tensorrt_config.performance.base_latency_ms = 5.0f;
        tensorrt_config.performance.latency_variation = 0.05f;  // Low variation
        tensorrt_config.simulate_hardware = false;              // Disable heavy simulation
        tensorrt_config.enable_logging = false;                 // Disable logging overhead

        test_factory_->inject_mock_engine(InferenceBackend::TENSORRT_GPU, [tensorrt_config]() {
            return std::make_unique<MockTensorRTEngine>(tensorrt_config);
        });

        // ONNX Runtime mock optimized for benchmarking
        auto onnx_config = MockONNXRuntimeEngine::create_onnx_config();
        onnx_config.performance.base_latency_ms = 15.0f;
        onnx_config.performance.latency_variation = 0.05f;
        onnx_config.simulate_hardware = false;
        onnx_config.enable_logging = false;

        test_factory_->inject_mock_engine(InferenceBackend::ONNX_RUNTIME, [onnx_config]() {
            return std::make_unique<MockONNXRuntimeEngine>(onnx_config);
        });

        // Rule-based mock optimized for benchmarking
        auto rule_config = MockRuleBasedEngine::create_rule_based_config();
        rule_config.performance.base_latency_ms = 1.0f;
        rule_config.performance.latency_variation = 0.02f;
        rule_config.simulate_hardware = false;
        rule_config.enable_logging = false;

        test_factory_->inject_mock_engine(InferenceBackend::RULE_BASED, [rule_config]() {
            return std::make_unique<MockRuleBasedEngine>(rule_config);
        });
    }

    std::unique_ptr<MLIntegrationFramework> framework_;
    TestBackendFactory* test_factory_;
    std::shared_ptr<ClassificationTestFixture> classification_fixture_;
    std::vector<InferenceRequest> test_inputs_;
};

//=============================================================================
// Single Backend Performance Benchmarks
//=============================================================================

/**
 * @brief Benchmark TensorRT mock engine performance
 */
static void BM_TensorRTSingleInference(benchmark::State& state) {
    auto& setup = BenchmarkSetup::instance();
    auto framework = setup.get_framework();
    auto fixture = setup.get_classification_fixture();
    auto test_inputs = setup.get_test_inputs();

    auto model_config = fixture->get_model_config();
    std::vector<InferenceRequest> single_input = {test_inputs[0]};

    for (auto _ : state) {
        auto result = framework->test_single_backend(
            InferenceBackend::TENSORRT_GPU, model_config, single_input);

        if (!result.is_ok()) {
            state.SkipWithError("TensorRT inference failed");
        }
    }

    // Set custom counters
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * sizeof(float) * 1 * 3 * 224 * 224);
}
BENCHMARK(BM_TensorRTSingleInference)->Unit(benchmark::kMillisecond);

/**
 * @brief Benchmark ONNX Runtime mock engine performance
 */
static void BM_ONNXRuntimeSingleInference(benchmark::State& state) {
    auto& setup = BenchmarkSetup::instance();
    auto framework = setup.get_framework();
    auto fixture = setup.get_classification_fixture();
    auto test_inputs = setup.get_test_inputs();

    auto model_config = fixture->get_model_config();
    model_config.backend = InferenceBackend::ONNX_RUNTIME;
    std::vector<InferenceRequest> single_input = {test_inputs[0]};

    for (auto _ : state) {
        auto result = framework->test_single_backend(
            InferenceBackend::ONNX_RUNTIME, model_config, single_input);

        if (!result.is_ok()) {
            state.SkipWithError("ONNX Runtime inference failed");
        }
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ONNXRuntimeSingleInference)->Unit(benchmark::kMillisecond);

/**
 * @brief Benchmark rule-based engine performance
 */
static void BM_RuleBasedSingleInference(benchmark::State& state) {
    auto& setup = BenchmarkSetup::instance();
    auto framework = setup.get_framework();
    auto fixture = setup.get_classification_fixture();
    auto test_inputs = setup.get_test_inputs();

    auto model_config = fixture->get_model_config();
    model_config.backend = InferenceBackend::RULE_BASED;
    std::vector<InferenceRequest> single_input = {test_inputs[0]};

    for (auto _ : state) {
        auto result = framework->test_single_backend(
            InferenceBackend::RULE_BASED, model_config, single_input);

        if (!result.is_ok()) {
            state.SkipWithError("Rule-based inference failed");
        }
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_RuleBasedSingleInference)->Unit(benchmark::kMillisecond);

//=============================================================================
// Batch Processing Benchmarks
//=============================================================================

/**
 * @brief Benchmark batch inference performance
 */
static void BM_BatchInference(benchmark::State& state) {
    auto& setup = BenchmarkSetup::instance();
    auto framework = setup.get_framework();
    auto fixture = setup.get_classification_fixture();
    auto test_inputs = setup.get_test_inputs();

    auto model_config = fixture->get_model_config();
    auto batch_size = state.range(0);

    // Create batch of inputs
    std::vector<InferenceRequest> batch_inputs;
    for (int64_t i = 0; i < batch_size && i < static_cast<int64_t>(test_inputs.size()); ++i) {
        batch_inputs.push_back(std::move(InferenceRequest(test_inputs[i])));
    }

    for (auto _ : state) {
        auto result = framework->test_single_backend(
            InferenceBackend::TENSORRT_GPU, model_config, batch_inputs);

        if (!result.is_ok()) {
            state.SkipWithError("Batch inference failed");
        }
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetBytesProcessed(state.iterations() * batch_size * sizeof(float) * 1 * 3 * 224 * 224);
}
BENCHMARK(BM_BatchInference)->Range(1, 32)->Unit(benchmark::kMillisecond);

//=============================================================================
// Cross-Backend Comparison Benchmarks
//=============================================================================

/**
 * @brief Benchmark cross-backend comparison performance
 */
static void BM_CrossBackendComparison(benchmark::State& state) {
    auto& setup = BenchmarkSetup::instance();
    auto framework = setup.get_framework();
    auto fixture = setup.get_classification_fixture();
    auto test_inputs = setup.get_test_inputs();

    auto model_config = fixture->get_model_config();
    std::vector<InferenceBackend> backends = {InferenceBackend::TENSORRT_GPU,
                                              InferenceBackend::ONNX_RUNTIME};

    // Use smaller input set for comparison
    std::vector<InferenceRequest> comparison_inputs(test_inputs.begin(), test_inputs.begin() + 5);

    for (auto _ : state) {
        auto result = framework->compare_backends(
            backends, model_config, comparison_inputs, ValidationStrategy::STATISTICAL_COMPARISON);

        if (!result.is_ok()) {
            state.SkipWithError("Cross-backend comparison failed");
        }
    }

    state.SetItemsProcessed(state.iterations() * backends.size());
}
BENCHMARK(BM_CrossBackendComparison)->Unit(benchmark::kMillisecond);

//=============================================================================
// Framework Overhead Benchmarks
//=============================================================================

/**
 * @brief Benchmark framework setup overhead
 */
static void BM_FrameworkSetup(benchmark::State& state) {
    for (auto _ : state) {
        auto mock_factory = std::make_unique<TestBackendFactory>();
        auto framework = std::make_unique<MLIntegrationFramework>(std::move(mock_factory));

        // Prevent optimization
        benchmark::DoNotOptimize(framework.get());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_FrameworkSetup)->Unit(benchmark::kMicrosecond);

/**
 * @brief Benchmark test scenario creation
 */
static void BM_TestScenarioCreation(benchmark::State& state) {
    auto fixture =
        std::make_shared<ClassificationTestFixture>(ClassificationTestFixture::create()
                                                        .with_model("benchmark_model")
                                                        .with_input_shape({1, 3, 224, 224})
                                                        .with_num_classes(1000));

    for (auto _ : state) {
        auto scenario = TestScenarioBuilder()
                            .with_name("benchmark_scenario")
                            .with_single_backend(InferenceBackend::TENSORRT_GPU)
                            .with_model_config(fixture->get_model_config())
                            .with_test_fixture(fixture)
                            .with_iterations(1)
                            .build();

        if (!scenario.is_ok()) {
            state.SkipWithError("Scenario creation failed");
        }

        benchmark::DoNotOptimize(scenario.unwrap());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TestScenarioCreation)->Unit(benchmark::kMicrosecond);

//=============================================================================
// Mock Engine Overhead Benchmarks
//=============================================================================

/**
 * @brief Benchmark mock engine creation overhead
 */
static void BM_MockEngineCreation(benchmark::State& state) {
    auto backend = static_cast<InferenceBackend>(state.range(0));
    auto config = MockEngineConfig{};
    config.enable_logging = false;
    config.simulate_hardware = false;

    for (auto _ : state) {
        auto engine = create_mock_engine(backend, config);
        benchmark::DoNotOptimize(engine.get());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MockEngineCreation)
    ->Arg(static_cast<int>(InferenceBackend::TENSORRT_GPU))
    ->Arg(static_cast<int>(InferenceBackend::ONNX_RUNTIME))
    ->Arg(static_cast<int>(InferenceBackend::RULE_BASED))
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Benchmark mock engine inference overhead
 */
static void BM_MockEngineInference(benchmark::State& state) {
    auto backend = static_cast<InferenceBackend>(state.range(0));
    auto config = MockEngineConfig{};
    config.enable_logging = false;
    config.simulate_hardware = false;
    config.performance.base_latency_ms = 0.1f;  // Minimal latency

    auto engine = create_mock_engine(backend, config);
    auto fixture = ClassificationTestFixture::create()
                       .with_model("minimal_model")
                       .with_input_shape({1, 3, 32, 32})  // Small input
                       .with_num_classes(10);

    auto test_inputs = fixture.generate_test_data(1);
    auto request = test_inputs[0];

    for (auto _ : state) {
        auto result = engine->run_inference(request);
        if (!result.is_ok()) {
            state.SkipWithError("Mock inference failed");
        }
        benchmark::DoNotOptimize(result.unwrap());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MockEngineInference)
    ->Arg(static_cast<int>(InferenceBackend::TENSORRT_GPU))
    ->Arg(static_cast<int>(InferenceBackend::ONNX_RUNTIME))
    ->Arg(static_cast<int>(InferenceBackend::RULE_BASED))
    ->Unit(benchmark::kMicrosecond);

//=============================================================================
// Memory Allocation Benchmarks
//=============================================================================

/**
 * @brief Benchmark tensor allocation performance
 */
static void BM_TensorAllocation(benchmark::State& state) {
    auto size = state.range(0);
    Shape shape = {1, 3, static_cast<std::size_t>(size), static_cast<std::size_t>(size)};

    for (auto _ : state) {
        auto tensor = tensor_factory::zeros<float>(shape);
        benchmark::DoNotOptimize(tensor.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * 3 * sizeof(float));
}
BENCHMARK(BM_TensorAllocation)->Range(32, 1024)->Unit(benchmark::kMicrosecond);

/**
 * @brief Benchmark test data generation
 */
static void BM_TestDataGeneration(benchmark::State& state) {
    auto num_samples = state.range(0);
    TestDataGenerator generator;
    Shape input_shape = {1, 3, 224, 224};

    for (auto _ : state) {
        auto data = generator.generate_classification_data(input_shape, 1000, num_samples);
        benchmark::DoNotOptimize(data.size());
    }

    state.SetItemsProcessed(state.iterations() * num_samples);
}
BENCHMARK(BM_TestDataGeneration)->Range(1, 100)->Unit(benchmark::kMillisecond);

//=============================================================================
// Statistical Analysis Benchmarks
//=============================================================================

/**
 * @brief Benchmark statistical analysis performance
 */
static void BM_StatisticalAnalysis(benchmark::State& state) {
    auto data_size = state.range(0);
    std::vector<float> data(data_size);

    // Generate test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& value : data) {
        value = dist(gen);
    }

    for (auto _ : state) {
        auto analysis = StatisticalAnalyzer::analyze_distribution(data);
        benchmark::DoNotOptimize(analysis.mean);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * data_size * sizeof(float));
}
BENCHMARK(BM_StatisticalAnalysis)->Range(100, 10000)->Unit(benchmark::kMicrosecond);

//=============================================================================
// Comprehensive Integration Benchmark
//=============================================================================

/**
 * @brief Comprehensive end-to-end integration benchmark
 */
static void BM_EndToEndIntegration(benchmark::State& state) {
    auto& setup = BenchmarkSetup::instance();
    auto framework = setup.get_framework();
    auto fixture = setup.get_classification_fixture();

    for (auto _ : state) {
        auto scenario = TestScenarioBuilder()
                            .with_name("end_to_end_benchmark")
                            .with_single_backend(InferenceBackend::TENSORRT_GPU)
                            .with_model_config(fixture->get_model_config())
                            .with_test_fixture(fixture)
                            .with_iterations(1)
                            .build();

        if (!scenario.is_ok()) {
            state.SkipWithError("Scenario creation failed");
            continue;
        }

        auto result = framework->run_integration_test(scenario.unwrap());
        if (!result.is_ok()) {
            state.SkipWithError("Integration test failed");
        }

        benchmark::DoNotOptimize(result.unwrap());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_EndToEndIntegration)->Unit(benchmark::kMillisecond);

//=============================================================================
// Benchmark Main Function
//=============================================================================

/**
 * @brief Custom main function for benchmarks with setup
 */
int main(int argc, char** argv) {
    // Initialize benchmark framework
    benchmark::Initialize(&argc, argv);

    // Ensure setup is initialized
    BenchmarkSetup::instance();

    // Run benchmarks if any are registered
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    // Clean up
    cleanup_test_environment();

    return 0;
}
