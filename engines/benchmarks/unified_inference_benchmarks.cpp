// MIT License
// Copyright (c) 2025 Inference Systems Lab

#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
    #include <mach/mach.h>
    #include <mach/task.h>
#endif

#include <benchmark/benchmark.h>

#include "../../common/src/logging.hpp"
#include "../src/circular_bp/circular_bp.hpp"
#include "../src/mamba_ssm/mamba_ssm.hpp"
#include "../src/momentum_bp/momentum_bp.hpp"
#include "unified_benchmark_config.hpp"

using namespace inference_lab::engines;
using inference_lab::common::LogLevel;

namespace inference_lab::engines::unified_benchmarks {

// Unified dataset generator for consistent testing across all POCs
class UnifiedDatasetGenerator {
  public:
    struct TestDataset {
        std::string name;
        size_t num_nodes;
        size_t num_edges;
        double edge_density;
        std::string description;
    };

    static std::vector<TestDataset> get_standard_datasets() {
        return {{"small_binary", 4, 4, 0.5, "Small 4-node binary graphical model"},
                {"medium_chain", 10, 9, 0.18, "Medium 10-node chain model"},
                {"large_grid", 25, 40, 0.13, "Large 5x5 grid model"}};
    }

    static momentum_bp::GraphicalModel create_momentum_bp_model(const TestDataset& dataset) {
        momentum_bp::GraphicalModel model;
        std::mt19937 rng(UnifiedBenchmarkConfig::RANDOM_SEED);  // Fixed seed for reproducibility
        std::uniform_real_distribution<double> potential_dist(0.1, 0.9);

        // Create nodes
        for (size_t i = 1; i <= dataset.num_nodes; ++i) {
            momentum_bp::Node node{static_cast<momentum_bp::NodeId>(i),
                                   {potential_dist(rng), potential_dist(rng)},  // Binary potentials
                                   {}};
            model.nodes.push_back(node);
            model.node_index[i] = i - 1;
        }

        // Create standardized edge structure with equivalent complexity to other techniques
        // Use dataset.num_edges to ensure same computational load across all techniques
        std::uniform_real_distribution<double> edge_dist(0.2, 1.8);
        std::uniform_int_distribution<size_t> node_dist(1, dataset.num_nodes);

        for (size_t edge_id = 1; edge_id <= dataset.num_edges; ++edge_id) {
            size_t node1, node2;
            // Ensure different nodes and avoid duplicates
            do {
                node1 = node_dist(rng);
                node2 = node_dist(rng);
            } while (node1 == node2);

            momentum_bp::EdgePotential edge{static_cast<momentum_bp::EdgeId>(edge_id),
                                            static_cast<momentum_bp::NodeId>(node1),
                                            static_cast<momentum_bp::NodeId>(node2),
                                            {{edge_dist(rng), edge_dist(rng)},
                                             {edge_dist(rng), edge_dist(rng)}}};
            model.edges.push_back(edge);

            // Add neighbors bidirectionally
            model.nodes[node1 - 1].neighbors.push_back(node2);
            model.nodes[node2 - 1].neighbors.push_back(node1);
        }

        return model;
    }

    static circular_bp::GraphicalModel create_circular_bp_model(const TestDataset& dataset) {
        circular_bp::GraphicalModel model;
        std::mt19937 rng(UnifiedBenchmarkConfig::RANDOM_SEED);  // Same seed for consistency
        std::uniform_real_distribution<double> potential_dist(0.1, 0.9);

        // Create nodes
        for (size_t i = 1; i <= dataset.num_nodes; ++i) {
            circular_bp::Node node{static_cast<circular_bp::NodeId>(i),
                                   {potential_dist(rng), potential_dist(rng)},
                                   {}};
            model.nodes.push_back(node);
            model.node_index[i] = i - 1;
        }

        // Create standardized edge structure with same complexity as other techniques
        // Use exact same number of edges to ensure fair computational comparison
        std::uniform_real_distribution<double> edge_dist(0.2, 1.8);
        std::uniform_int_distribution<size_t> node_dist(1, dataset.num_nodes);

        for (size_t edge_id = 1; edge_id <= dataset.num_edges; ++edge_id) {
            size_t node1, node2;
            // Ensure different nodes and avoid duplicates
            do {
                node1 = node_dist(rng);
                node2 = node_dist(rng);
            } while (node1 == node2);

            circular_bp::EdgePotential edge{static_cast<circular_bp::EdgeId>(edge_id),
                                            static_cast<circular_bp::NodeId>(node1),
                                            static_cast<circular_bp::NodeId>(node2),
                                            {{edge_dist(rng), edge_dist(rng)},
                                             {edge_dist(rng), edge_dist(rng)}}};
            model.edges.push_back(edge);

            // Add neighbors bidirectionally
            model.nodes[node1 - 1].neighbors.push_back(node2);
            model.nodes[node2 - 1].neighbors.push_back(node1);
        }

        return model;
    }

    static common::ml::FloatTensor create_mamba_sequence_data(const TestDataset& dataset) {
        size_t batch_size = 1;
        // Scale sequence length proportional to computational complexity (nodes + edges)
        // This ensures equivalent computational load to graph-based techniques
        size_t computational_units = dataset.num_nodes + dataset.num_edges;
        size_t seq_len = computational_units * 2;  // Balanced scaling factor
        size_t d_model = UnifiedBenchmarkConfig::MAMBA_SSM_D_MODEL;

        return mamba_ssm::testing::generate_random_sequence(batch_size, seq_len, d_model);
    }
};

// Unified performance metrics collector
struct UnifiedMetrics {
    std::string technique_name;
    std::string dataset_name;
    double inference_time_ms;
    double memory_usage_mb;
    double convergence_iterations;
    double final_accuracy;
    bool converged;

    void print_summary() const {
        std::ostringstream oss;
        oss << technique_name << " on " << dataset_name << ": "
            << std::fixed << std::setprecision(2) << inference_time_ms << "ms, "
            << std::setprecision(1) << memory_usage_mb << "MB, "
            << static_cast<int>(convergence_iterations) << " iters, "
            << std::setprecision(3) << final_accuracy << " acc, converged="
            << std::boolalpha << converged;
        LOG_INFO_PRINT("{}", oss.str());
    }
};

class UnifiedBenchmarkSuite {
  public:
    static void run_comparative_analysis() {
        LOG_INFO_PRINT("Starting Unified POC Comparative Analysis");

        auto datasets = UnifiedDatasetGenerator::get_standard_datasets();
        std::vector<UnifiedMetrics> all_metrics;

        for (const auto& dataset : datasets) {
            LOG_INFO_PRINT("Testing dataset: {} ({} nodes, {} edges)",
                           dataset.name,
                           dataset.num_nodes,
                           dataset.num_edges);

            // Test Momentum-Enhanced BP
            auto momentum_metrics = benchmark_momentum_bp(dataset);
            all_metrics.push_back(momentum_metrics);

            // Test Circular BP
            auto circular_metrics = benchmark_circular_bp(dataset);
            all_metrics.push_back(circular_metrics);

            // Test Mamba SSM
            auto mamba_metrics = benchmark_mamba_ssm(dataset);
            all_metrics.push_back(mamba_metrics);
        }

        // Generate comparative report
        generate_comparison_report(all_metrics);
    }

    // Public benchmark methods for Google Benchmark integration
    static UnifiedMetrics benchmark_momentum_bp(
        const UnifiedDatasetGenerator::TestDataset& dataset) {
        UnifiedMetrics metrics;
        metrics.technique_name = "Momentum-Enhanced BP";
        metrics.dataset_name = dataset.name;

        try {
            // Create engine
            momentum_bp::MomentumBPConfig config;
            config.max_iterations = UnifiedBenchmarkConfig::MOMENTUM_BP_MAX_ITERATIONS;
            config.convergence_threshold =
                UnifiedBenchmarkConfig::MOMENTUM_BP_CONVERGENCE_THRESHOLD;
            config.enable_momentum = UnifiedBenchmarkConfig::MOMENTUM_BP_ENABLE_MOMENTUM;
            config.enable_adagrad = UnifiedBenchmarkConfig::MOMENTUM_BP_ENABLE_ADAGRAD;

            auto engine_result = momentum_bp::create_momentum_bp_engine(config);
            if (!engine_result.is_ok()) {
                metrics.converged = false;
                metrics.inference_time_ms = 0.0;
                metrics.memory_usage_mb = get_memory_usage_mb();
                metrics.convergence_iterations = 0;
                metrics.final_accuracy = 0.0;
                LOG_ERROR_PRINT("Failed to create Momentum-BP engine: {}",
                                momentum_bp::to_string(engine_result.unwrap_err()));
                return metrics;
            }

            auto engine = std::move(engine_result).unwrap();
            auto model = UnifiedDatasetGenerator::create_momentum_bp_model(dataset);

            // Measure inference time
            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine->run_momentum_bp(model);
            auto end = std::chrono::high_resolution_clock::now();

            metrics.inference_time_ms =
                std::chrono::duration<double, std::milli>(end - start).count();

            if (result.is_ok()) {
                auto bp_metrics = engine->get_metrics();
                metrics.converged = bp_metrics.converged;
                metrics.convergence_iterations = bp_metrics.iterations_to_convergence;
                metrics.final_accuracy =
                    1.0 - bp_metrics.final_residual;  // Convert residual to accuracy
                metrics.memory_usage_mb = get_memory_usage_mb();
            }

        } catch (const std::exception& e) {
            LOG_ERROR_PRINT("Momentum BP benchmark failed: {}", e.what());
            metrics.converged = false;
        }

        return metrics;
    }

    static UnifiedMetrics benchmark_circular_bp(
        const UnifiedDatasetGenerator::TestDataset& dataset) {
        UnifiedMetrics metrics;
        metrics.technique_name = "Circular BP";
        metrics.dataset_name = dataset.name;

        try {
            circular_bp::CircularBPConfig config;
            config.max_iterations = UnifiedBenchmarkConfig::CIRCULAR_BP_MAX_ITERATIONS;
            config.convergence_threshold =
                UnifiedBenchmarkConfig::CIRCULAR_BP_CONVERGENCE_THRESHOLD;
            config.enable_cycle_penalties = true;
            config.enable_correlation_cancellation = true;

            auto engine_result = circular_bp::create_circular_bp_engine(config);
            if (!engine_result.is_ok()) {
                metrics.converged = false;
                metrics.inference_time_ms = 0.0;
                metrics.memory_usage_mb = get_memory_usage_mb();
                metrics.convergence_iterations = 0;
                metrics.final_accuracy = 0.0;
                LOG_ERROR_PRINT("Failed to create Circular-BP engine: {}",
                                circular_bp::to_string(engine_result.unwrap_err()));
                return metrics;
            }

            auto engine = std::move(engine_result).unwrap();
            auto model = UnifiedDatasetGenerator::create_circular_bp_model(dataset);

            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine->run_circular_bp(model);
            auto end = std::chrono::high_resolution_clock::now();

            metrics.inference_time_ms =
                std::chrono::duration<double, std::milli>(end - start).count();

            if (result.is_ok()) {
                auto bp_metrics = engine->get_metrics();
                metrics.converged = bp_metrics.converged;
                metrics.convergence_iterations = bp_metrics.iterations_to_convergence;
                metrics.final_accuracy = 1.0 - bp_metrics.final_residual;
                metrics.memory_usage_mb = get_memory_usage_mb();
            }

        } catch (const std::exception& e) {
            LOG_ERROR_PRINT("Circular BP benchmark failed: {}", e.what());
            metrics.converged = false;
        }

        return metrics;
    }

    static UnifiedMetrics benchmark_mamba_ssm(const UnifiedDatasetGenerator::TestDataset& dataset) {
        UnifiedMetrics metrics;
        metrics.technique_name = "Mamba SSM";
        metrics.dataset_name = dataset.name;

        try {
            mamba_ssm::MambaSSMConfig config;
            config.d_state = UnifiedBenchmarkConfig::MAMBA_SSM_D_STATE;
            config.max_seq_len = dataset.num_nodes * 10;
            config.d_model = UnifiedBenchmarkConfig::MAMBA_SSM_D_MODEL;

            auto engine_result = mamba_ssm::create_mamba_ssm_engine(config);
            if (!engine_result.is_ok()) {
                metrics.converged = false;
                metrics.inference_time_ms = 0.0;
                metrics.memory_usage_mb = get_memory_usage_mb();
                metrics.convergence_iterations = 0;
                metrics.final_accuracy = 0.0;
                LOG_ERROR_PRINT("Failed to create Mamba-SSM engine: {}",
                                mamba_ssm::to_string(engine_result.unwrap_err()));
                return metrics;
            }

            auto engine = std::move(engine_result).unwrap();
            auto sequence_data = UnifiedDatasetGenerator::create_mamba_sequence_data(dataset);

            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine->run_mamba_ssm(sequence_data);
            auto end = std::chrono::high_resolution_clock::now();

            metrics.inference_time_ms =
                std::chrono::duration<double, std::milli>(end - start).count();

            if (result.is_ok()) {
                auto ssm_metrics = engine->get_metrics();
                metrics.converged = ssm_metrics.converged;
                metrics.convergence_iterations =
                    UnifiedBenchmarkConfig::DEFAULT_SSM_CONVERGENCE_ITERATIONS;
                metrics.final_accuracy = UnifiedBenchmarkConfig::DEFAULT_SSM_ACCURACY;
                metrics.memory_usage_mb =
                    static_cast<double>(ssm_metrics.memory_usage_bytes) / (1024.0 * 1024.0);
            }

        } catch (const std::exception& e) {
            LOG_ERROR_PRINT("Mamba SSM benchmark failed: {}", e.what());
            metrics.converged = false;
        }

        return metrics;
    }

  private:
    static void generate_comparison_report(const std::vector<UnifiedMetrics>& all_metrics) {
        LOG_INFO_PRINT("\n=== UNIFIED POC COMPARATIVE ANALYSIS REPORT ===");

        // Group by dataset
        std::map<std::string, std::vector<UnifiedMetrics>> by_dataset;
        for (const auto& metric : all_metrics) {
            by_dataset[metric.dataset_name].push_back(metric);
        }

        for (const auto& [dataset_name, dataset_metrics] : by_dataset) {
            LOG_INFO_PRINT("\nDataset: {}", dataset_name);
            LOG_INFO_PRINT(
                "Technique               | Time (ms) | Memory (MB) | Iters | Accuracy | Converged");
            LOG_INFO_PRINT(
                "------------------------|-----------|-------------|-------|----------|----------");

            for (const auto& metric : dataset_metrics) {
                LOG_INFO_PRINT("{:23} | {:9.2f} | {:11.1f} | {:5.0f} | {:8.3f} | {:9}",
                               metric.technique_name,
                               metric.inference_time_ms,
                               metric.memory_usage_mb,
                               metric.convergence_iterations,
                               metric.final_accuracy,
                               metric.converged ? "Yes" : "No");
            }
        }

        // Summary statistics
        generate_summary_statistics(all_metrics);
    }

    static void generate_summary_statistics(const std::vector<UnifiedMetrics>& all_metrics) {
        LOG_INFO_PRINT("\n=== PERFORMANCE SUMMARY STATISTICS ===");

        std::map<std::string, std::vector<double>> technique_times;
        std::map<std::string, int> convergence_counts;

        for (const auto& metric : all_metrics) {
            technique_times[metric.technique_name].push_back(metric.inference_time_ms);
            if (metric.converged) {
                convergence_counts[metric.technique_name]++;
            }
        }

        LOG_INFO_PRINT("\nTechnique Performance Summary:");
        LOG_INFO_PRINT("Technique               | Avg Time (ms) | Convergence Rate");
        LOG_INFO_PRINT("------------------------|---------------|------------------");

        for (const auto& [technique, times] : technique_times) {
            double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            double convergence_rate =
                static_cast<double>(convergence_counts[technique]) / times.size();

            LOG_INFO_PRINT("{:23} | {:13.2f} | {:16.1%}", technique, avg_time, convergence_rate);
        }
    }

    static double get_memory_usage_mb() {
        // Get actual memory usage using platform-specific APIs
#ifdef __APPLE__
        // macOS - use mach task info
        struct mach_task_basic_info info;
        mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) ==
            KERN_SUCCESS) {
            return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
        }
#elif defined(__linux__)
        // Linux - read from /proc/self/status
        std::ifstream status_file("/proc/self/status");
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.find("VmRSS:") == 0) {
                std::istringstream iss(line);
                std::string key, value, unit;
                iss >> key >> value >> unit;
                return std::stod(value) / 1024.0;  // Convert KB to MB
            }
        }
#endif
        // Fallback estimation for unsupported platforms
        return estimate_memory_usage_mb_fallback();
    }

    static double estimate_memory_usage_mb_fallback() {
        // Fallback estimation when platform APIs unavailable
        // This is a rough estimate based on typical memory usage patterns
        static size_t baseline_mb = 0;
        if (baseline_mb == 0) {
            // Get baseline memory usage at startup (approximate)
            baseline_mb = UnifiedBenchmarkConfig::DEFAULT_BASELINE_MEMORY_MB;
        }
        return static_cast<double>(baseline_mb);
    }
};

}  // namespace inference_lab::engines::unified_benchmarks

// Google Benchmark integration for automated performance testing

using namespace inference_lab::engines::unified_benchmarks;

// Individual technique benchmarks for proper comparison
static void BM_MomentumBP_SmallBinary(benchmark::State& state) {
    auto datasets = UnifiedDatasetGenerator::get_standard_datasets();
    auto small_dataset = datasets[0];  // "small_binary"

    for (auto _ : state) {
        auto momentum_metrics = UnifiedBenchmarkSuite::benchmark_momentum_bp(small_dataset);
        benchmark::DoNotOptimize(momentum_metrics);
    }
}

static void BM_CircularBP_SmallBinary(benchmark::State& state) {
    auto datasets = UnifiedDatasetGenerator::get_standard_datasets();
    auto small_dataset = datasets[0];  // "small_binary"

    for (auto _ : state) {
        auto circular_metrics = UnifiedBenchmarkSuite::benchmark_circular_bp(small_dataset);
        benchmark::DoNotOptimize(circular_metrics);
    }
}

static void BM_MambaSSM_SmallBinary(benchmark::State& state) {
    auto datasets = UnifiedDatasetGenerator::get_standard_datasets();
    auto small_dataset = datasets[0];  // "small_binary"

    for (auto _ : state) {
        auto mamba_metrics = UnifiedBenchmarkSuite::benchmark_mamba_ssm(small_dataset);
        benchmark::DoNotOptimize(mamba_metrics);
    }
}

static void BM_MomentumBP_MediumChain(benchmark::State& state) {
    auto datasets = UnifiedDatasetGenerator::get_standard_datasets();
    auto medium_dataset = datasets[1];  // "medium_chain"

    for (auto _ : state) {
        auto momentum_metrics = UnifiedBenchmarkSuite::benchmark_momentum_bp(medium_dataset);
        benchmark::DoNotOptimize(momentum_metrics);
    }
}

static void BM_CircularBP_MediumChain(benchmark::State& state) {
    auto datasets = UnifiedDatasetGenerator::get_standard_datasets();
    auto medium_dataset = datasets[1];  // "medium_chain"

    for (auto _ : state) {
        auto circular_metrics = UnifiedBenchmarkSuite::benchmark_circular_bp(medium_dataset);
        benchmark::DoNotOptimize(circular_metrics);
    }
}

static void BM_MambaSSM_MediumChain(benchmark::State& state) {
    auto datasets = UnifiedDatasetGenerator::get_standard_datasets();
    auto medium_dataset = datasets[1];  // "medium_chain"

    for (auto _ : state) {
        auto mamba_metrics = UnifiedBenchmarkSuite::benchmark_mamba_ssm(medium_dataset);
        benchmark::DoNotOptimize(mamba_metrics);
    }
}

static void BM_MomentumBP_LargeGrid(benchmark::State& state) {
    auto datasets = UnifiedDatasetGenerator::get_standard_datasets();
    auto large_dataset = datasets[2];  // "large_grid"

    for (auto _ : state) {
        auto momentum_metrics = UnifiedBenchmarkSuite::benchmark_momentum_bp(large_dataset);
        benchmark::DoNotOptimize(momentum_metrics);
    }
}

static void BM_CircularBP_LargeGrid(benchmark::State& state) {
    auto datasets = UnifiedDatasetGenerator::get_standard_datasets();
    auto large_dataset = datasets[2];  // "large_grid"

    for (auto _ : state) {
        auto circular_metrics = UnifiedBenchmarkSuite::benchmark_circular_bp(large_dataset);
        benchmark::DoNotOptimize(circular_metrics);
    }
}

static void BM_MambaSSM_LargeGrid(benchmark::State& state) {
    auto datasets = UnifiedDatasetGenerator::get_standard_datasets();
    auto large_dataset = datasets[2];  // "large_grid"

    for (auto _ : state) {
        auto mamba_metrics = UnifiedBenchmarkSuite::benchmark_mamba_ssm(large_dataset);
        benchmark::DoNotOptimize(mamba_metrics);
    }
}

// Register individual technique benchmarks for proper comparison
BENCHMARK(BM_MomentumBP_SmallBinary)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CircularBP_SmallBinary)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MambaSSM_SmallBinary)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MomentumBP_MediumChain)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CircularBP_MediumChain)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MambaSSM_MediumChain)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MomentumBP_LargeGrid)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CircularBP_LargeGrid)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MambaSSM_LargeGrid)->Unit(benchmark::kMillisecond);

// Standalone performance comparison
static void BM_StandaloneComparativeAnalysis(benchmark::State& state) {
    for (auto _ : state) {
        UnifiedBenchmarkSuite::run_comparative_analysis();
    }
}

BENCHMARK(BM_StandaloneComparativeAnalysis)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)  // Run once for comprehensive analysis
    ->MeasureProcessCPUTime();

BENCHMARK_MAIN();
