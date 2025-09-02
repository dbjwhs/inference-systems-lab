// MIT License
// Copyright (c) 2025 Inference Systems Lab

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

#include "../src/ml_config.hpp"
#include "onnx/onnx_engine.hpp"

using namespace inference_lab::engines;
using namespace inference_lab::engines::onnx;
using namespace inference_lab::engines::ml;
using namespace inference_lab::common::ml;

namespace {

struct BenchmarkResult {
    std::string framework;
    std::string provider;
    std::string model_name;
    size_t iterations;

    // Timing statistics
    double mean_us;
    double median_us;
    double min_us;
    double max_us;
    double std_dev_us;
    double p95_us;
    double p99_us;

    // Throughput
    double throughput_qps;

    // Memory (if available)
    size_t memory_mb;
    size_t peak_memory_mb;

    // Success metrics
    size_t successful_runs;
    size_t failed_runs;
    double success_rate;
};

struct ModelSpec {
    std::string name;
    std::string path;
    std::vector<Shape> input_shapes;
    std::vector<DataType> input_types;
};

class MLFrameworkBenchmark {
  private:
    std::vector<ModelSpec> models_;
    std::vector<ExecutionProvider> providers_to_test_;
    size_t default_iterations_;
    bool verbose_;

  public:
    explicit MLFrameworkBenchmark(size_t default_iterations = 1000, bool verbose = false)
        : default_iterations_(default_iterations), verbose_(verbose) {
        // Determine which providers to test based on availability
        auto available_backends = get_available_backends();

        for (const auto& backend : available_backends) {
            switch (backend) {
                case MLBackend::ONNX_RUNTIME:
                    providers_to_test_.push_back(ExecutionProvider::CPU);
                    if (capabilities.gpu_acceleration) {
                        providers_to_test_.push_back(ExecutionProvider::CUDA);
                    }
                    break;
                case MLBackend::TENSORRT_GPU:
                    providers_to_test_.push_back(ExecutionProvider::TENSORRT);
                    break;
                default:
                    break;
            }
        }

        // Remove duplicates
        std::sort(providers_to_test_.begin(), providers_to_test_.end());
        providers_to_test_.erase(std::unique(providers_to_test_.begin(), providers_to_test_.end()),
                                 providers_to_test_.end());
    }

    void add_model(const std::string& name,
                   const std::string& path,
                   const std::vector<Shape>& input_shapes = {},
                   const std::vector<DataType>& input_types = {}) {
        ModelSpec spec;
        spec.name = name;
        spec.path = path;
        spec.input_shapes = input_shapes;
        spec.input_types = input_types;

        // Default input spec for common models if not provided
        if (input_shapes.empty()) {
            spec.input_shapes.push_back({1, 3, 224, 224});  // Common image input
        }
        if (input_types.empty()) {
            spec.input_types.push_back(DataType::FLOAT32);
        }

        models_.push_back(spec);
    }

    std::vector<BenchmarkResult> run_benchmarks() {
        std::vector<BenchmarkResult> results;

        std::cout << "=== ML Framework Benchmark ===\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Default iterations: " << default_iterations_ << "\n";
        std::cout << "  Models: " << models_.size() << "\n";
        std::cout << "  Providers to test: ";
        for (size_t i = 0; i < providers_to_test_.size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << to_string(providers_to_test_[i]);
        }
        std::cout << "\n\n";

        for (const auto& model : models_) {
            std::cout << "Benchmarking model: " << model.name << "\n";
            std::cout << "Path: " << model.path << "\n";

            // Test each provider
            for (const auto& provider : providers_to_test_) {
                auto result = benchmark_model_provider(model, provider);
                if (result.successful_runs > 0) {
                    results.push_back(result);
                    print_result(result);
                } else {
                    std::cout << "  ❌ " << to_string(provider) << ": Failed to run\n";
                }
            }
            std::cout << "\n";
        }

        return results;
    }

    void save_results_csv(const std::vector<BenchmarkResult>& results,
                          const std::string& filename = "ml_benchmark_results.csv") {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << " for writing\n";
            return;
        }

        // CSV header
        file << "Framework,Provider,Model,Iterations,MeanUS,MedianUS,MinUS,MaxUS,StdDevUS,"
             << "P95US,P99US,ThroughputQPS,MemoryMB,PeakMemoryMB,SuccessfulRuns,"
             << "FailedRuns,SuccessRate\n";

        // Data rows
        for (const auto& result : results) {
            file << result.framework << "," << result.provider << "," << result.model_name << ","
                 << result.iterations << "," << std::fixed << std::setprecision(2) << result.mean_us
                 << "," << result.median_us << "," << result.min_us << "," << result.max_us << ","
                 << result.std_dev_us << "," << result.p95_us << "," << result.p99_us << ","
                 << result.throughput_qps << "," << result.memory_mb << "," << result.peak_memory_mb
                 << "," << result.successful_runs << "," << result.failed_runs << ","
                 << result.success_rate << "\n";
        }

        std::cout << "Results saved to: " << filename << "\n";
    }

    void print_summary(const std::vector<BenchmarkResult>& results) {
        if (results.empty()) {
            std::cout << "No benchmark results to summarize.\n";
            return;
        }

        std::cout << "\n=== Benchmark Summary ===\n\n";

        // Group by model
        std::map<std::string, std::vector<BenchmarkResult>> by_model;
        for (const auto& result : results) {
            by_model[result.model_name].push_back(result);
        }

        for (const auto& [model_name, model_results] : by_model) {
            std::cout << "Model: " << model_name << "\n";

            // Find best performing provider
            auto best_it = std::min_element(model_results.begin(),
                                            model_results.end(),
                                            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                                                return a.mean_us < b.mean_us;
                                            });

            if (best_it != model_results.end()) {
                std::cout << "  🏆 Best: " << best_it->provider << " (" << std::fixed
                          << std::setprecision(2) << best_it->mean_us << " µs, "
                          << best_it->throughput_qps << " QPS)\n";
            }

            // Show all results for comparison
            std::cout << "  All results:\n";
            for (const auto& result : model_results) {
                double speedup = best_it != model_results.end() ? result.mean_us / best_it->mean_us
                                                                : 1.0;

                std::cout << "    " << std::setw(12) << result.provider << ": " << std::setw(8)
                          << std::fixed << std::setprecision(2) << result.mean_us << " µs"
                          << " (" << std::setw(7) << result.throughput_qps << " QPS)"
                          << " [" << std::setw(4) << std::setprecision(1) << speedup << "x]";

                if (result.success_rate < 100.0) {
                    std::cout << " ⚠️  " << result.success_rate << "% success";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        // Overall statistics
        if (results.size() > 1) {
            double total_throughput = 0.0;
            double avg_latency = 0.0;

            for (const auto& result : results) {
                total_throughput += result.throughput_qps;
                avg_latency += result.mean_us;
            }

            avg_latency /= results.size();

            std::cout << "Overall Statistics:\n";
            std::cout << "  Total configurations tested: " << results.size() << "\n";
            std::cout << "  Average latency: " << std::fixed << std::setprecision(2) << avg_latency
                      << " µs\n";
            std::cout << "  Combined throughput: " << std::fixed << std::setprecision(1)
                      << total_throughput << " QPS\n";
        }
    }

  private:
    BenchmarkResult benchmark_model_provider(const ModelSpec& model, ExecutionProvider provider) {
        BenchmarkResult result;
        result.framework = "ONNX Runtime";
        result.provider = to_string(provider);
        result.model_name = model.name;
        result.iterations = default_iterations_;
        result.successful_runs = 0;
        result.failed_runs = 0;

        try {
            // Create engine with specified provider
            ONNXRuntimeConfig config;
            config.provider = provider;
            config.optimization_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
            config.enable_profiling = false;

            auto engine_result = create_onnx_engine_from_model(model.path, config);
            if (!engine_result) {
                result.failed_runs = default_iterations_;
                result.success_rate = 0.0;
                return result;
            }

            auto& engine = *engine_result.unwrap();

            // Generate test inputs
            auto inputs = generate_test_inputs(model);

            // Warm up
            for (size_t i = 0; i < 10; ++i) {
                auto warm_result = engine.run_inference(inputs);
                if (!warm_result)
                    break;
            }

            // Collect timing data
            std::vector<double> timings_us;
            timings_us.reserve(default_iterations_);

            for (size_t i = 0; i < default_iterations_; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                auto inference_result = engine.run_inference(inputs);
                auto end = std::chrono::high_resolution_clock::now();

                if (inference_result) {
                    auto duration =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    timings_us.push_back(static_cast<double>(duration.count()));
                    result.successful_runs++;
                } else {
                    result.failed_runs++;
                }
            }

            if (!timings_us.empty()) {
                calculate_statistics(timings_us, result);

                // Get memory information from engine
                const auto& metrics = engine.get_metrics();
                result.memory_mb = metrics.memory_usage_bytes / (1024 * 1024);
                result.peak_memory_mb = metrics.peak_memory_bytes / (1024 * 1024);
            }

        } catch (const std::exception& e) {
            if (verbose_) {
                std::cerr << "Exception in benchmark: " << e.what() << "\n";
            }
            result.failed_runs = default_iterations_;
        }

        result.success_rate =
            static_cast<double>(result.successful_runs) / default_iterations_ * 100.0;

        return result;
    }

    std::vector<FloatTensor> generate_test_inputs(const ModelSpec& model) {
        std::vector<FloatTensor> inputs;

        for (size_t i = 0; i < model.input_shapes.size(); ++i) {
            const auto& shape = model.input_shapes[i];

            // Calculate total size
            size_t total_size = 1;
            for (const auto& dim : shape) {
                total_size *= dim;
            }

            // Generate deterministic test data
            std::vector<float> data(total_size);
            for (size_t j = 0; j < total_size; ++j) {
                data[j] = static_cast<float>(std::sin(j * 0.01) * 0.5 + 0.5);
            }

            inputs.emplace_back(shape, std::move(data));
        }

        return inputs;
    }

    void calculate_statistics(std::vector<double>& timings_us, BenchmarkResult& result) {
        if (timings_us.empty())
            return;

        std::sort(timings_us.begin(), timings_us.end());

        result.min_us = timings_us.front();
        result.max_us = timings_us.back();
        result.median_us = timings_us[timings_us.size() / 2];

        // Mean
        result.mean_us =
            std::accumulate(timings_us.begin(), timings_us.end(), 0.0) / timings_us.size();

        // Standard deviation
        double variance = 0.0;
        for (double timing : timings_us) {
            variance += (timing - result.mean_us) * (timing - result.mean_us);
        }
        result.std_dev_us = std::sqrt(variance / timings_us.size());

        // Percentiles
        result.p95_us = timings_us[static_cast<size_t>(timings_us.size() * 0.95)];
        result.p99_us = timings_us[static_cast<size_t>(timings_us.size() * 0.99)];

        // Throughput (QPS)
        result.throughput_qps = 1000000.0 / result.mean_us;
    }

    void print_result(const BenchmarkResult& result) {
        std::cout << "  ✅ " << result.provider << ":\n";
        std::cout << "    Mean: " << std::fixed << std::setprecision(2) << result.mean_us << " µs ("
                  << result.throughput_qps << " QPS)\n";
        std::cout << "    Range: " << result.min_us << " - " << result.max_us << " µs\n";
        std::cout << "    P95/P99: " << result.p95_us << "/" << result.p99_us << " µs\n";
        std::cout << "    Success rate: " << std::fixed << std::setprecision(1)
                  << result.success_rate << "%\n";

        if (result.memory_mb > 0) {
            std::cout << "    Memory: " << result.memory_mb
                      << " MB (peak: " << result.peak_memory_mb << " MB)\n";
        }
    }
};

}  // anonymous namespace

int main(int argc, char* argv[]) {
    std::cout << "=== ML Framework Benchmark Tool ===\n\n";

    // Check ML framework availability
    const auto& caps = capabilities;
    std::cout << "ML Framework Status: " << caps.to_string() << "\n";

    auto available_backends = get_available_backends();
    std::cout << "Available backends: ";
    for (size_t i = 0; i < available_backends.size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << backend_to_string(available_backends[i]);
    }
    std::cout << "\n\n";

    if (!caps.onnx_runtime_available) {
        std::cout << "❌ ONNX Runtime is not available in this build.\n";
        std::cout << "Cannot run ML framework benchmarks.\n";
        return 1;
    }

    // Parse command line arguments
    bool verbose = false;
    size_t iterations = 1000;
    std::vector<std::string> model_paths;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--iterations" || arg == "-i") {
            if (i + 1 < argc) {
                iterations = std::stoull(argv[++i]);
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options] [model1.onnx] [model2.onnx] ...\n";
            std::cout << "\nOptions:\n";
            std::cout << "  -i, --iterations N    Number of iterations per test (default: 1000)\n";
            std::cout << "  -v, --verbose        Enable verbose output\n";
            std::cout << "  -h, --help           Show this help message\n";
            std::cout << "\nIf no model files are provided, runs in demonstration mode.\n";
            return 0;
        } else {
            model_paths.push_back(arg);
        }
    }

    // Create benchmark suite
    MLFrameworkBenchmark benchmark(iterations, verbose);

    if (model_paths.empty()) {
        std::cout << "No model files provided. Running in demonstration mode...\n\n";

        std::cout << "This benchmark tool can compare:\n";
        std::cout << "  🔹 Different ONNX Runtime execution providers (CPU, CUDA, TensorRT)\n";
        std::cout << "  🔹 Multiple models with various architectures\n";
        std::cout << "  🔹 Performance metrics (latency, throughput, memory usage)\n";
        std::cout << "  🔹 Statistical analysis (mean, median, percentiles, std dev)\n";
        std::cout << "  🔹 CSV export for further analysis\n\n";

        std::cout << "Example usage:\n";
        std::cout << "  " << argv[0] << " model1.onnx model2.onnx --iterations 500\n";
        std::cout << "  " << argv[0] << " resnet50.onnx --verbose\n\n";

        std::cout << "Available execution providers for testing:\n";
        if (capabilities.onnx_runtime_available) {
            std::cout << "  ✅ ONNX Runtime CPU\n";
            if (capabilities.gpu_acceleration) {
                std::cout << "  ✅ ONNX Runtime CUDA (if CUDA available)\n";
                std::cout << "  ✅ TensorRT (if TensorRT available)\n";
            }
        }

        return 0;
    }

    // Add models to benchmark
    for (size_t i = 0; i < model_paths.size(); ++i) {
        std::string model_name = "Model_" + std::to_string(i + 1);
        benchmark.add_model(model_name, model_paths[i]);
    }

    // Run benchmarks
    auto results = benchmark.run_benchmarks();

    if (results.empty()) {
        std::cout << "❌ No successful benchmark runs completed.\n";
        return 1;
    }

    // Print summary
    benchmark.print_summary(results);

    // Save results
    benchmark.save_results_csv(results);

    std::cout << "\n✅ Benchmark completed successfully!\n";
    std::cout << "Results summary:\n";
    std::cout << "  - Configurations tested: " << results.size() << "\n";
    std::cout << "  - Total iterations: " << iterations * results.size() << "\n";
    std::cout << "  - Results exported to CSV for analysis\n";

    return 0;
}
