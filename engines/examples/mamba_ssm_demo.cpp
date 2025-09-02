// MIT License
// Copyright (c) 2025 Inference Systems Lab

#include <iomanip>
#include <iostream>
#include <vector>

#include "../src/mamba_ssm/mamba_ssm.hpp"

using namespace inference_lab::engines::mamba_ssm;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(70, '=') << "\n\n";
}

void print_metrics(const std::string& label, const MambaSSMMetrics& metrics) {
    std::cout << label << " Performance Metrics:\n";
    std::cout << "  Sequence Length: " << metrics.sequence_length << "\n";
    std::cout << "  Converged: " << (metrics.converged ? "Yes" : "No") << "\n";
    std::cout << "  Inference Time: " << std::fixed << std::setprecision(3)
              << metrics.inference_time_ms.count() / 1000.0 << " ms\n";
    std::cout << "  Selective Scan Time: " << std::fixed << std::setprecision(3)
              << metrics.selective_scan_time_ms.count() / 1000.0 << " ms\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
              << metrics.throughput_tokens_per_sec << " tokens/sec\n";
    std::cout << "  FLOP Rate: " << std::fixed << std::setprecision(2)
              << metrics.flops_per_second / 1e9 << " GFLOPS\n";
    std::cout << "  Total FLOPs: " << metrics.total_flops << "\n";
    std::cout << "  Average Step Size: " << std::fixed << std::setprecision(4)
              << metrics.average_step_size << "\n";
    std::cout << "  Selective Updates: " << metrics.selective_updates << "\n\n";
}

void print_tensor_sample(const std::string& label,
                         const FloatTensor& tensor,
                         size_t max_elements = 5) {
    std::cout << label << " (shape: [";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
        std::cout << tensor.shape()[i];
        if (i < tensor.shape().size() - 1)
            std::cout << ", ";
    }
    std::cout << "]):\n";

    const auto* data = tensor.data();
    size_t total_elements = tensor.size();
    size_t show_elements = std::min(max_elements, total_elements);

    std::cout << "  First " << show_elements << " elements: [";
    for (size_t i = 0; i < show_elements; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i];
        if (i < show_elements - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";

    if (total_elements > max_elements) {
        std::cout << "  ... (" << (total_elements - max_elements) << " more elements)\n";
        std::cout << "  Last " << std::min(3UL, total_elements - show_elements) << " elements: [";
        for (size_t i = std::max(show_elements, total_elements - 3); i < total_elements; ++i) {
            std::cout << std::fixed << std::setprecision(4) << data[i];
            if (i < total_elements - 1)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

FloatTensor create_structured_sequence(size_t batch, size_t seq_len, size_t d_model) {
    FloatTensor sequence(Shape{batch, seq_len, d_model});

    auto data = sequence.data();
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t d = 0; d < d_model; ++d) {
                size_t idx = (b * seq_len + t) * d_model + d;

                // Create a more interesting test pattern with:
                // - Temporal structure (sine wave over sequence)
                // - Feature structure (cosine wave over dimensions)
                // - Some noise for realism
                float temporal_signal =
                    std::sin(2.0f * 3.14159f * static_cast<float>(t) / static_cast<float>(seq_len));
                float feature_signal =
                    std::cos(2.0f * 3.14159f * static_cast<float>(d) / static_cast<float>(d_model));
                float noise = 0.1f * static_cast<float>((idx * 17) % 100) / 100.0f - 0.05f;
                float batch_offset = 0.1f * static_cast<float>(b);

                data[idx] = 0.5f * temporal_signal + 0.3f * feature_signal + noise + batch_offset;
            }
        }
    }

    return sequence;
}

int main() {
    print_separator("Mamba State Space Model Demonstration");

    std::cout << "This demo showcases Mamba's linear O(n) complexity advantage over\n";
    std::cout << "traditional Transformers' quadratic O(n²) complexity for sequence modeling.\n\n";
    std::cout << "Key Innovations Demonstrated:\n";
    std::cout << "• Selective state space mechanism with O(n) complexity\n";
    std::cout << "• Hardware-efficient SIMD-optimized matrix operations\n";
    std::cout << "• Input-dependent selective updates for token retention\n";
    std::cout << "• Structured state transitions with diagonal matrix optimization\n\n";

    // Test 1: Basic Mamba SSM Functionality
    print_separator("Test 1: Basic Mamba SSM Functionality");

    MambaSSMConfig config;
    config.d_model = 128;       // Model dimension
    config.d_state = 16;        // State space dimension
    config.d_inner = 256;       // Inner projection dimension
    config.d_conv = 4;          // Convolution kernel size
    config.max_seq_len = 1024;  // Maximum sequence length
    config.batch_size = 2;      // Batch size for testing
    config.use_simd_kernels = true;
    config.activation = MambaSSMConfig::ActivationType::SILU;

    std::cout << "Configuration:\n";
    std::cout << "  Model Dimension (d_model): " << config.d_model << "\n";
    std::cout << "  State Dimension (d_state): " << config.d_state << "\n";
    std::cout << "  Inner Dimension (d_inner): " << config.d_inner << "\n";
    std::cout << "  Maximum Sequence Length: " << config.max_seq_len << "\n";
    std::cout << "  SIMD Optimization: " << (config.use_simd_kernels ? "Enabled" : "Disabled")
              << "\n\n";

    auto engine_result = create_mamba_ssm_engine(config);
    if (!engine_result.is_ok()) {
        std::cerr << "Failed to create Mamba SSM engine: " << to_string(engine_result.unwrap_err())
                  << "\n";
        return 1;
    }

    auto engine = std::move(engine_result).unwrap();
    std::cout << "✓ Mamba SSM Engine created successfully\n\n";

    // Create and process test sequence
    const size_t test_seq_len = 64;
    auto test_input = create_structured_sequence(config.batch_size, test_seq_len, config.d_model);

    print_tensor_sample("Input Sequence", test_input, 8);

    auto result = engine->run_mamba_ssm(test_input);
    if (!result.is_ok()) {
        std::cerr << "Inference failed: " << to_string(result.unwrap_err()) << "\n";
        return 1;
    }

    auto output = std::move(result).unwrap();
    print_tensor_sample("Output Sequence", output, 8);
    print_metrics("Basic Inference", engine->get_metrics());

    // Test 2: Linear Complexity Scaling Analysis
    print_separator("Test 2: Linear Complexity Scaling Analysis");

    std::cout << "Testing sequence lengths to demonstrate O(n) scaling vs O(n²) Transformer "
                 "complexity:\n\n";
    std::cout << std::setw(8) << "Seq Len" << std::setw(12) << "Time (ms)" << std::setw(15)
              << "Tokens/sec" << std::setw(12) << "FLOPs/Token" << std::setw(18)
              << "O(n²) Advantage\n";
    std::cout << std::string(65, '-') << "\n";

    std::vector<size_t> sequence_lengths = {32, 64, 128, 256, 512};
    std::vector<double> inference_times;
    std::vector<double> flops_per_token;

    for (size_t seq_len : sequence_lengths) {
        engine->reset_state();

        auto scaling_input = create_structured_sequence(1, seq_len, config.d_model);
        auto scaling_result = engine->run_mamba_ssm(scaling_input);

        if (scaling_result.is_ok()) {
            auto metrics = engine->get_metrics();
            double time_ms = metrics.inference_time_ms.count() / 1000.0;
            double tokens_per_sec = metrics.throughput_tokens_per_sec;
            double flops_per_token_val = static_cast<double>(metrics.total_flops) / seq_len;

            // Calculate theoretical advantage over O(n²) complexity
            double quadratic_advantage = static_cast<double>(seq_len);  // O(n²)/O(n) = O(n)

            std::cout << std::setw(8) << seq_len << std::setw(12) << std::fixed
                      << std::setprecision(2) << time_ms << std::setw(15) << std::fixed
                      << std::setprecision(0) << tokens_per_sec << std::setw(12) << std::fixed
                      << std::setprecision(0) << flops_per_token_val << std::setw(18) << std::fixed
                      << std::setprecision(1) << quadratic_advantage << "x\n";

            inference_times.push_back(time_ms);
            flops_per_token.push_back(flops_per_token_val);
        }
    }

    // Analyze scaling behavior
    std::cout << "\nScaling Analysis:\n";
    if (inference_times.size() >= 2) {
        double time_ratio = inference_times.back() / inference_times.front();
        double length_ratio =
            static_cast<double>(sequence_lengths.back()) / sequence_lengths.front();
        double scaling_factor = time_ratio / length_ratio;

        std::cout << "  Time scaling factor: " << std::fixed << std::setprecision(2)
                  << scaling_factor << " (ideal linear = 1.0)\n";

        if (scaling_factor < 1.5) {
            std::cout << "  ✓ Excellent linear scaling achieved!\n";
        } else if (scaling_factor < 2.0) {
            std::cout << "  ✓ Good linear scaling with some overhead\n";
        } else {
            std::cout << "  ⚠ Scaling appears worse than linear (may include measurement noise)\n";
        }
    }

    // Test 3: Selective Mechanism Analysis
    print_separator("Test 3: Selective Mechanism Analysis");

    std::cout << "Analyzing the selective state space mechanism:\n\n";

    // Test different input patterns to show selectivity
    std::vector<std::pair<std::string, FloatTensor>> test_patterns;

    // Create constant input
    FloatTensor constant_input(Shape{1, 32, config.d_model});
    auto constant_data = constant_input.data();
    std::fill_n(constant_data, constant_input.size(), 0.5f);
    test_patterns.emplace_back("Constant Input", std::move(constant_input));

    // Create other patterns
    test_patterns.emplace_back("Linear Ramp", create_structured_sequence(1, 32, config.d_model));
    test_patterns.emplace_back("Random Noise",
                               testing::generate_random_sequence(1, 32, config.d_model));

    for (const auto& [pattern_name, pattern_input] : test_patterns) {
        engine->reset_state();

        auto pattern_result = engine->run_mamba_ssm(pattern_input);
        if (pattern_result.is_ok()) {
            auto metrics = engine->get_metrics();

            std::cout << pattern_name << ":\n";
            std::cout << "  Average Step Size: " << std::fixed << std::setprecision(4)
                      << metrics.average_step_size << "\n";
            std::cout << "  Selective Updates: " << metrics.selective_updates << "\n";
            std::cout << "  Inference Time: " << std::fixed << std::setprecision(3)
                      << metrics.inference_time_ms.count() / 1000.0 << " ms\n";
            std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
                      << metrics.throughput_tokens_per_sec << " tokens/sec\n\n";
        }
    }

    // Test 4: SIMD Kernel Performance Comparison
    print_separator("Test 4: SIMD Kernel Performance Comparison");

    std::cout << "Comparing SIMD-optimized vs standard kernel performance:\n\n";

    std::vector<bool> simd_settings = {false, true};
    std::vector<std::string> simd_labels = {"Standard Kernels", "SIMD Optimized"};

    for (size_t i = 0; i < simd_settings.size(); ++i) {
        MambaSSMConfig simd_config = config;
        simd_config.use_simd_kernels = simd_settings[i];

        auto simd_engine_result = create_mamba_ssm_engine(simd_config);
        if (simd_engine_result.is_ok()) {
            auto simd_engine = std::move(simd_engine_result).unwrap();

            auto simd_input = create_structured_sequence(1, 128, config.d_model);
            auto simd_result = simd_engine->run_mamba_ssm(simd_input);

            if (simd_result.is_ok()) {
                auto metrics = simd_engine->get_metrics();

                std::cout << simd_labels[i] << ":\n";
                std::cout << "  Inference Time: " << std::fixed << std::setprecision(3)
                          << metrics.inference_time_ms.count() / 1000.0 << " ms\n";
                std::cout << "  Selective Scan Time: " << std::fixed << std::setprecision(3)
                          << metrics.selective_scan_time_ms.count() / 1000.0 << " ms\n";
                std::cout << "  Throughput: " << std::fixed << std::setprecision(0)
                          << metrics.throughput_tokens_per_sec << " tokens/sec\n";
                std::cout << "  FLOP Rate: " << std::fixed << std::setprecision(2)
                          << metrics.flops_per_second / 1e9 << " GFLOPS\n\n";
            }
        }
    }

    // Test 5: Unified Interface Compatibility
    print_separator("Test 5: Unified InferenceEngine Interface");

    std::cout << "Demonstrating compatibility with the unified inference interface:\n\n";

    inference_lab::engines::InferenceRequest request;
    auto interface_result = engine->run_inference(request);

    if (interface_result.is_ok()) {
        auto response = interface_result.unwrap();

        std::cout << "Unified Interface Results:\n";
        std::cout << "  Output Tensors: " << response.output_tensors.size() << "\n";
        std::cout << "  Output Names: " << response.output_names.size() << "\n";
        std::cout << "  Inference Time: " << std::fixed << std::setprecision(3)
                  << response.inference_time_ms << " ms\n\n";

        std::cout << "Backend Information:\n" << engine->get_backend_info() << "\n";
        std::cout << "Performance Statistics:\n" << engine->get_performance_stats() << "\n";
    } else {
        std::cout << "⚠ Unified interface test failed\n\n";
    }

    // Summary
    print_separator("Summary & Key Advantages");

    std::cout << "Mamba State Space Model demonstrates several breakthrough advantages:\n\n";

    std::cout << "🚀 **Linear Complexity**: O(n) scaling vs Transformer's O(n²)\n";
    std::cout << "   • Enables processing of much longer sequences efficiently\n";
    std::cout << "   • Memory requirements grow linearly, not quadratically\n";
    std::cout << "   • Computational cost scales favorably for large sequences\n\n";

    std::cout << "🧠 **Selective Mechanism**: Input-dependent parameter updates\n";
    std::cout << "   • B, C, and Δ parameters adapt based on input content\n";
    std::cout << "   • Selective retention of important information\n";
    std::cout << "   • Dynamic step size adaptation for optimal processing\n\n";

    std::cout << "⚡ **Hardware Efficiency**: SIMD-optimized implementations\n";
    std::cout << "   • Vectorized matrix operations with AVX2/AVX-512 support\n";
    std::cout << "   • Cache-friendly memory layouts with 64-byte alignment\n";
    std::cout << "   • Structured matrices optimize common operations\n\n";

    std::cout << "🔧 **Production Ready**: Enterprise integration patterns\n";
    std::cout << "   • Unified InferenceEngine interface compatibility\n";
    std::cout << "   • Comprehensive error handling with Result<T,E> patterns\n";
    std::cout << "   • Detailed performance metrics and monitoring\n";
    std::cout << "   • Thread-safe operations with modern C++17 design\n\n";

    std::cout << "**Applications particularly suited for Mamba SSM:**\n";
    std::cout << "• Long-form text generation and analysis\n";
    std::cout << "• Time series prediction with long dependencies\n";
    std::cout << "• Sequential decision making in reinforcement learning\n";
    std::cout << "• DNA/protein sequence analysis in bioinformatics\n";
    std::cout << "• Audio processing for speech and music generation\n\n";

    std::cout << "The linear complexity breakthrough makes Mamba SSM a compelling\n";
    std::cout << "alternative to Transformers for applications requiring long-context\n";
    std::cout << "understanding while maintaining competitive accuracy.\n\n";

    return 0;
}
