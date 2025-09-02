// MIT License
// Copyright (c) 2025 Inference Systems Lab

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../src/ml_config.hpp"
#include "onnx/onnx_engine.hpp"

using namespace inference_lab::engines;
using namespace inference_lab::engines::onnx;
using namespace inference_lab::engines::ml;
using namespace inference_lab::common::ml;

namespace {

// Generate synthetic input data for demo purposes
auto generate_demo_input(const std::vector<TensorInfo>& input_info) -> std::vector<FloatTensor> {
    std::vector<FloatTensor> inputs;

    for (const auto& info : input_info) {
        // Calculate total size
        size_t total_size = 1;
        for (const auto& dim : info.shape) {
            total_size *= dim;
        }

        // Generate random-like data (deterministic for demo)
        std::vector<float> data(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            data[i] = static_cast<float>(std::sin(i * 0.1) * 0.5 + 0.5);
        }

        FloatTensor tensor(info.shape);
        // Copy data into tensor
        auto* tensor_data = tensor.data();
        std::copy(data.begin(), data.end(), tensor_data);
        inputs.push_back(std::move(tensor));
    }

    return inputs;
}

// Performance benchmarking
auto benchmark_inference(ONNXRuntimeEngine& engine,
                         const std::vector<FloatTensor>& inputs,
                         size_t iterations = 100) -> double {
    // Warm up
    for (size_t i = 0; i < 5; ++i) {
        auto result = engine.run_inference(inputs);
        if (!result) {
            std::cerr << "Warmup inference failed\n";
            return -1.0;
        }
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        auto result = engine.run_inference(inputs);
        if (!result) {
            std::cerr << "Benchmark inference failed at iteration " << i << "\n";
            return -1.0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    return static_cast<double>(duration.count()) / iterations;  // microseconds per inference
}

void print_tensor_info(const std::vector<TensorInfo>& tensors, const std::string& type) {
    std::cout << "  " << type << " tensors:\n";
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& info = tensors[i];
        std::cout << "    [" << i << "] " << info.name << ": ";

        // Print shape
        std::cout << "[";
        for (size_t j = 0; j < info.shape.size(); ++j) {
            if (j > 0)
                std::cout << ", ";
            std::cout << info.shape[j];
        }
        std::cout << "] ";

        // Print data type
        switch (info.data_type) {
            case DataType::FLOAT32:
                std::cout << "float32";
                break;
            case DataType::INT32:
                std::cout << "int32";
                break;
            default:
                std::cout << "unknown";
                break;
        }

        if (info.is_dynamic) {
            std::cout << " (dynamic)";
        }
        std::cout << "\n";
    }
}

void print_performance_stats(const ONNXMetrics& metrics) {
    std::cout << "\nPerformance Statistics:\n";
    std::cout << "  Model size: " << (metrics.model_size_bytes / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  Active provider: " << to_string(metrics.active_provider) << "\n";
    std::cout << "  Available providers: ";
    for (size_t i = 0; i < metrics.available_providers.size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << metrics.available_providers[i];
    }
    std::cout << "\n";

    if (metrics.total_inferences > 0) {
        std::cout << "  Average inference time: " << metrics.inference_time_us.count() << " µs\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                  << metrics.inferences_per_second << " inferences/sec\n";
        std::cout << "  Total inferences: " << metrics.total_inferences << "\n";
    }

    if (metrics.memory_usage_bytes > 0) {
        std::cout << "  Memory usage: " << (metrics.memory_usage_bytes / 1024.0 / 1024.0)
                  << " MB\n";
        std::cout << "  Peak memory: " << (metrics.peak_memory_bytes / 1024.0 / 1024.0) << " MB\n";
    }
}

}  // anonymous namespace

int main(int argc, char* argv[]) {
    std::cout << "=== ONNX Runtime Inference Demo ===\n\n";

    // Check ML framework availability
    std::cout << "ML Framework Status:\n";
    const auto& caps = capabilities;
    std::cout << "  " << caps.to_string() << "\n";

    if (!caps.onnx_runtime_available) {
        std::cout << "\n❌ ONNX Runtime is not available in this build.\n";
        std::cout << "To enable ONNX Runtime support:\n";
        std::cout << "  1. Install ONNX Runtime 1.16+\n";
        std::cout << "  2. Set ONNXRUNTIME_ROOT environment variable\n";
        std::cout << "  3. Rebuild with -DENABLE_ONNX_RUNTIME=ON\n";
        return 1;
    }

    std::cout << "✅ ONNX Runtime is available!\n\n";

    // Display available backends
    auto available_backends = get_available_backends();
    std::cout << "Available ML Backends:\n";
    for (const auto& backend : available_backends) {
        std::cout << "  - " << backend_to_string(backend) << "\n";
    }

    auto optimal_backend = detect_optimal_backend();
    std::cout << "Optimal backend: " << backend_to_string(optimal_backend) << "\n\n";

    // Check for command-line model path
    std::string model_path;
    if (argc > 1) {
        model_path = argv[1];
        std::cout << "Using model: " << model_path << "\n\n";
    } else {
        std::cout << "No model file provided.\n";
        std::cout << "Usage: " << argv[0] << " <path_to_onnx_model>\n\n";
        std::cout << "Running in framework detection mode only...\n\n";

        // Still show engine creation capabilities
        std::cout << "Engine Creation Test:\n";
        auto engine_result = create_onnx_engine();
        if (engine_result) {
            std::cout << "✅ ONNX Runtime engine created successfully\n";
            std::cout << "Backend info: " << engine_result.unwrap()->get_backend_info() << "\n";

            // Show performance stats (empty model)
            std::cout << engine_result.unwrap()->get_performance_stats() << "\n";
        } else {
            std::cout << "❌ Failed to create ONNX Runtime engine\n";
        }
        return 0;
    }

    // Create ONNX Runtime engine with configuration
    ONNXRuntimeConfig config;
    config.provider = ExecutionProvider::AUTO;  // Auto-detect best provider
    config.optimization_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
    config.enable_profiling = true;
    config.profile_file_prefix = "onnx_demo_profile";

    std::cout << "Creating ONNX Runtime engine...\n";
    auto engine_result = create_onnx_engine_from_model(model_path, config);

    if (!engine_result) {
        std::cout << "❌ Failed to create ONNX engine and load model\n";
        std::cout << "Error details would be available in a real implementation\n";
        return 1;
    }

    auto& engine = *engine_result.unwrap();
    std::cout << "✅ ONNX Runtime engine created and model loaded successfully\n\n";

    // Get model information
    std::cout << "Model Information:\n";
    auto input_info_result = engine.get_input_info();
    auto output_info_result = engine.get_output_info();

    if (input_info_result && output_info_result) {
        const auto& input_info = input_info_result.unwrap();
        const auto& output_info = output_info_result.unwrap();

        print_tensor_info(input_info, "Input");
        print_tensor_info(output_info, "Output");

        // Generate demo inputs
        std::cout << "\nGenerating synthetic input data...\n";
        auto demo_inputs = generate_demo_input(input_info);

        // Run single inference
        std::cout << "Running inference...\n";
        auto inference_result = engine.run_inference(demo_inputs);

        if (inference_result) {
            const auto& outputs = inference_result.unwrap();
            std::cout << "✅ Inference completed successfully\n";
            std::cout << "Output tensors: " << outputs.size() << "\n";

            // Show output shapes and sample values
            for (size_t i = 0; i < outputs.size() && i < output_info.size(); ++i) {
                const auto& output = outputs[i];
                const auto& info = output_info[i];
                std::cout << "  Output " << i << " (" << info.name << "): shape [";

                for (size_t j = 0; j < output.shape().size(); ++j) {
                    if (j > 0)
                        std::cout << ", ";
                    std::cout << output.shape()[j];
                }
                std::cout << "]\n";

                // Show first few values
                const auto* data = output.data();
                const auto& shape = output.shape();
                size_t total_elements = 1;
                for (auto dim : shape) {
                    total_elements *= dim;
                }
                size_t show_count = std::min(size_t(5), total_elements);
                std::cout << "    First " << show_count << " values: [";
                for (size_t j = 0; j < show_count; ++j) {
                    if (j > 0)
                        std::cout << ", ";
                    std::cout << std::fixed << std::setprecision(4) << data[j];
                }
                if (total_elements > show_count) {
                    std::cout << ", ...";
                }
                std::cout << "]\n";
            }
        } else {
            std::cout << "❌ Inference failed\n";
            return 1;
        }

        // Performance benchmarking
        std::cout << "\nRunning performance benchmark (100 iterations)...\n";
        double avg_time_us = benchmark_inference(engine, demo_inputs, 100);

        if (avg_time_us > 0) {
            std::cout << "✅ Benchmark completed\n";
            std::cout << "Average inference time: " << std::fixed << std::setprecision(2)
                      << avg_time_us << " µs\n";
            std::cout << "Throughput: " << std::fixed << std::setprecision(2)
                      << (1000000.0 / avg_time_us) << " inferences/sec\n";
        }

        // Test different execution providers if available
        auto available_providers = engine.get_available_providers();
        if (available_providers.size() > 1) {
            std::cout << "\nTesting different execution providers:\n";

            for (const auto& provider : available_providers) {
                if (provider == engine.get_current_provider()) {
                    continue;  // Already tested
                }

                std::cout << "  Switching to " << to_string(provider) << "...\n";
                auto switch_result = engine.set_execution_provider(provider);

                if (switch_result.is_ok()) {
                    // Quick benchmark with new provider
                    double provider_time = benchmark_inference(engine, demo_inputs, 10);
                    if (provider_time > 0) {
                        std::cout << "    Performance: " << std::fixed << std::setprecision(2)
                                  << provider_time << " µs/inference\n";
                    } else {
                        std::cout << "    Performance test failed\n";
                    }
                } else {
                    std::cout << "    Failed to switch to provider\n";
                }
            }
        }

    } else {
        std::cout << "❌ Failed to get model information\n";
        return 1;
    }

    // Display performance statistics
    const auto& metrics = engine.get_metrics();
    print_performance_stats(metrics);

    std::cout << "\n✅ ONNX Runtime demo completed successfully!\n";
    std::cout << "\nFor more advanced usage:\n";
    std::cout << "  - Try different ONNX models (classification, detection, etc.)\n";
    std::cout << "  - Experiment with different execution providers\n";
    std::cout << "  - Use named inputs for models with multiple input tensors\n";
    std::cout << "  - Enable profiling for detailed performance analysis\n";

    return 0;
}
