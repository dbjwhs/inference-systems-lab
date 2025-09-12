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

#include <iomanip>
#include <iostream>

#include "../src/circular_bp/circular_bp.hpp"

using namespace inference_lab::engines::circular_bp;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

void print_marginals(const std::string& label, const std::vector<std::vector<double>>& marginals) {
    std::cout << label << ":\n";
    for (size_t i = 0; i < marginals.size(); ++i) {
        std::cout << "  Node " << (i + 1) << ": [";
        for (size_t j = 0; j < marginals[i].size(); ++j) {
            std::cout << std::fixed << std::setprecision(3) << marginals[i][j];
            if (j < marginals[i].size() - 1)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

void print_metrics(const std::string& label, const CircularBPMetrics& metrics) {
    std::cout << label << " Performance Metrics:\n";
    std::cout << "  Converged: " << (metrics.converged ? "Yes" : "No") << "\n";
    std::cout << "  Iterations: " << metrics.iterations_to_convergence << "\n";
    std::cout << "  Final Residual: " << std::scientific << metrics.final_residual << std::fixed
              << "\n";
    std::cout << "  Inference Time: " << metrics.inference_time_ms.count() << " ms\n";
    std::cout << "  Message Updates: " << metrics.message_updates << "\n";
    std::cout << "  Cycles Detected: " << metrics.cycles_detected << "\n";
    std::cout << "  Correlations Cancelled: " << metrics.correlations_cancelled << "\n";
    std::cout << "  Reverberation Events: " << metrics.reverberation_events << "\n\n";
}

GraphicalModel create_triangle_model() {
    GraphicalModel model;

    // Create 3-node triangle (simplest non-trivial cycle)
    Node node1{1, {0.6, 0.4}, {2, 3}};
    Node node2{2, {0.3, 0.7}, {1, 3}};
    Node node3{3, {0.5, 0.5}, {1, 2}};
    model.nodes = {node1, node2, node3};
    model.node_index[1] = 0;
    model.node_index[2] = 1;
    model.node_index[3] = 2;

    // Create cyclic edges with different potentials
    EdgePotential edge1{1, 1, 2, {{1.2, 0.8}, {0.8, 1.2}}};  // Favors same state
    EdgePotential edge2{2, 2, 3, {{1.1, 0.9}, {0.9, 1.1}}};  // Weakly favors same state
    EdgePotential edge3{3, 3, 1, {{1.3, 0.7}, {0.7, 1.3}}};  // Strongly favors same state
    model.edges = {edge1, edge2, edge3};

    return model;
}

GraphicalModel create_complex_cyclic_model() {
    GraphicalModel model;

    // Create a more complex 5-node model with multiple cycles
    Node node1{1, {0.7, 0.3}, {2, 5}};
    Node node2{2, {0.4, 0.6}, {1, 3, 4}};
    Node node3{3, {0.5, 0.5}, {2, 4}};
    Node node4{4, {0.6, 0.4}, {2, 3, 5}};
    Node node5{5, {0.3, 0.7}, {1, 4}};

    model.nodes = {node1, node2, node3, node4, node5};
    for (size_t i = 0; i < model.nodes.size(); ++i) {
        model.node_index[model.nodes[i].id] = i;
    }

    // Create edges that form multiple overlapping cycles
    EdgePotential edge1{1, 1, 2, {{1.4, 0.6}, {0.6, 1.4}}};
    EdgePotential edge2{2, 2, 3, {{1.2, 0.8}, {0.8, 1.2}}};
    EdgePotential edge3{3, 3, 4, {{1.3, 0.7}, {0.7, 1.3}}};
    EdgePotential edge4{4, 4, 5, {{1.1, 0.9}, {0.9, 1.1}}};
    EdgePotential edge5{5, 5, 1, {{1.5, 0.5}, {0.5, 1.5}}};  // Close main cycle
    EdgePotential edge6{6, 2, 4, {{1.2, 0.8}, {0.8, 1.2}}};  // Create shorter cycle

    model.edges = {edge1, edge2, edge3, edge4, edge5, edge6};

    return model;
}

int main() {
    print_separator("Circular Belief Propagation Demonstration");

    std::cout << "This demo showcases Circular Belief Propagation's ability to handle\n";
    std::cout
        << "cyclic graphical models through cycle detection and correlation cancellation.\n\n";

    // Create different configurations
    CircularBPConfig standard_config;
    standard_config.max_iterations = 100;
    standard_config.convergence_threshold = 1e-6;
    standard_config.correlation_threshold = 0.8;
    standard_config.enable_correlation_cancellation = true;
    standard_config.enable_cycle_penalties = true;
    standard_config.track_message_history = true;

    CircularBPConfig no_enhancements_config = standard_config;
    no_enhancements_config.enable_correlation_cancellation = false;
    no_enhancements_config.enable_cycle_penalties = false;
    no_enhancements_config.track_message_history = false;

    // Test 1: Triangle Model Comparison
    print_separator("Test 1: Triangle Cycle (3 Nodes)");

    auto triangle_model = create_triangle_model();
    std::cout << "Testing on a simple 3-node triangle cycle.\n";
    std::cout << "This is the simplest non-trivial cyclic structure.\n\n";

    // Enhanced Circular BP
    auto enhanced_engine_result = create_circular_bp_engine(standard_config);
    if (enhanced_engine_result.is_ok()) {
        auto enhanced_engine = std::move(enhanced_engine_result).unwrap();

        auto result = enhanced_engine->run_circular_bp(triangle_model);
        if (result.is_ok()) {
            auto marginals = result.unwrap();
            print_marginals("Enhanced Circular-BP Marginals", marginals);
            print_metrics("Enhanced Circular-BP", enhanced_engine->get_metrics());
        }
    }

    // Basic Circular BP (no enhancements)
    auto basic_engine_result = create_circular_bp_engine(no_enhancements_config);
    if (basic_engine_result.is_ok()) {
        auto basic_engine = std::move(basic_engine_result).unwrap();

        auto result = basic_engine->run_circular_bp(triangle_model);
        if (result.is_ok()) {
            auto marginals = result.unwrap();
            print_marginals("Basic Circular-BP Marginals", marginals);
            print_metrics("Basic Circular-BP", basic_engine->get_metrics());
        }
    }

    // Test 2: Complex Multi-Cycle Model
    print_separator("Test 2: Complex Multi-Cycle Model (5 Nodes)");

    auto complex_model = create_complex_cyclic_model();
    std::cout << "Testing on a complex 5-node model with multiple overlapping cycles.\n";
    std::cout
        << "This demonstrates the algorithm's ability to handle complex cyclic structures.\n\n";

    enhanced_engine_result = create_circular_bp_engine(standard_config);
    if (enhanced_engine_result.is_ok()) {
        auto enhanced_engine = std::move(enhanced_engine_result).unwrap();

        auto result = enhanced_engine->run_circular_bp(complex_model);
        if (result.is_ok()) {
            auto marginals = result.unwrap();
            print_marginals("Complex Model - Enhanced Marginals", marginals);
            print_metrics("Complex Model - Enhanced", enhanced_engine->get_metrics());
        }
    }

    // Test 3: Cycle Detection Strategy Comparison
    print_separator("Test 3: Cycle Detection Strategy Comparison");

    std::cout << "Comparing different cycle detection strategies:\n\n";

    std::vector<std::pair<CycleDetectionStrategy, std::string>> strategies = {
        {CycleDetectionStrategy::DEPTH_FIRST_SEARCH, "Depth-First Search"},
        {CycleDetectionStrategy::SPARSE_MATRIX, "Sparse Matrix"},
        {CycleDetectionStrategy::HYBRID_ADAPTIVE, "Hybrid Adaptive"}};

    for (const auto& [strategy, name] : strategies) {
        CircularBPConfig strategy_config = standard_config;
        strategy_config.detection_strategy = strategy;

        auto engine_result = create_circular_bp_engine(strategy_config);
        if (engine_result.is_ok()) {
            auto engine = std::move(engine_result).unwrap();

            auto result = engine->run_circular_bp(complex_model);
            if (result.is_ok()) {
                std::cout << name << " Strategy:\n";
                auto metrics = engine->get_metrics();
                std::cout << "  Cycles Detected: " << metrics.cycles_detected << "\n";
                std::cout << "  Inference Time: " << metrics.inference_time_ms.count() << " ms\n";
                std::cout << "  Correlations Cancelled: " << metrics.correlations_cancelled
                          << "\n\n";
            }
        }
    }

    // Test 4: Unified Interface Demonstration
    print_separator("Test 4: Unified InferenceEngine Interface");

    std::cout << "Demonstrating the unified InferenceEngine interface compatibility:\n\n";

    enhanced_engine_result = create_circular_bp_engine(standard_config);
    if (enhanced_engine_result.is_ok()) {
        auto engine = std::move(enhanced_engine_result).unwrap();

        // Use the unified interface
        inference_lab::engines::InferenceRequest request;
        auto response_result = engine->run_inference(request);

        if (response_result.is_ok()) {
            auto response = response_result.unwrap();

            std::cout << "Unified Interface Results:\n";
            std::cout << "  Output Tensors: " << response.output_tensors.size() << "\n";
            std::cout << "  Output Names: " << response.output_names.size() << "\n";
            std::cout << "  Inference Time: " << response.inference_time_ms << " ms\n\n";

            std::cout << "Backend Info:\n" << engine->get_backend_info() << "\n";
            std::cout << "Performance Stats:\n" << engine->get_performance_stats() << "\n";
        }
    }

    print_separator("Summary");

    std::cout << "Circular Belief Propagation demonstrates several key advantages:\n\n";
    std::cout << "1. **Cycle Detection**: Automatically identifies cycles in graphical models\n";
    std::cout << "2. **Correlation Cancellation**: Mitigates spurious correlations from cycles\n";
    std::cout << "3. **Message History Tracking**: Detects message reverberation patterns\n";
    std::cout << "4. **Flexible Strategies**: Multiple cycle detection algorithms available\n";
    std::cout << "5. **Performance Monitoring**: Comprehensive metrics for analysis\n\n";
    std::cout << "This makes it particularly suitable for:\n";
    std::cout << "- Social network analysis with feedback loops\n";
    std::cout << "- Biological pathway modeling with regulatory cycles\n";
    std::cout << "- Complex constraint satisfaction problems\n";
    std::cout << "- Any graphical model where standard BP fails due to cycles\n\n";

    return 0;
}
