// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file momentum_bp_demo.cpp
 * @brief Demonstration of Momentum-Enhanced Belief Propagation
 *
 * This demo shows the core functionality of momentum-enhanced belief propagation
 * compared to standard BP on a simple graphical model.
 */

#include <chrono>
#include <iomanip>
#include <iostream>

#include "../src/momentum_bp/momentum_bp.hpp"

using namespace inference_lab::engines::momentum_bp;

void print_marginals(const std::vector<std::vector<double>>& marginals) {
    std::cout << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < marginals.size(); ++i) {
        std::cout << "  Node " << (i + 1) << " marginal: [";
        for (std::size_t j = 0; j < marginals[i].size(); ++j) {
            if (j > 0)
                std::cout << ", ";
            std::cout << marginals[i][j];
        }
        std::cout << "]\n";
    }
}

void run_momentum_bp_demo() {
    std::cout << "=== Momentum-Enhanced Belief Propagation Demo ===\n\n";

    // Create configuration with momentum enabled
    MomentumBPConfig momentum_config;
    momentum_config.max_iterations = 50;
    momentum_config.convergence_threshold = 1e-6;
    momentum_config.momentum_factor = 0.9;
    momentum_config.learning_rate = 0.1;
    momentum_config.enable_momentum = true;
    momentum_config.enable_adagrad = true;

    std::cout << "Creating Momentum-BP engine with configuration:\n";
    std::cout << "  Max iterations: " << momentum_config.max_iterations << "\n";
    std::cout << "  Momentum factor: " << momentum_config.momentum_factor << "\n";
    std::cout << "  Learning rate: " << momentum_config.learning_rate << "\n";
    std::cout << "  AdaGrad enabled: " << momentum_config.enable_adagrad << "\n\n";

    auto engine_result = create_momentum_bp_engine(momentum_config);
    if (engine_result.is_err()) {
        std::cerr << "Failed to create Momentum-BP engine: "
                  << to_string(engine_result.unwrap_err()) << std::endl;
        return;
    }

    auto momentum_engine = std::move(engine_result).unwrap();

    // Create a simple 2-node graphical model
    std::cout << "Creating simple 2-node graphical model:\n";
    std::cout << "  Node 1: P(X1=0)=0.6, P(X1=1)=0.4\n";
    std::cout << "  Node 2: P(X2=0)=0.3, P(X2=1)=0.7\n";
    std::cout << "  Edge potential: slightly favors same values\n\n";

    GraphicalModel model;

    Node node1{1, {0.6, 0.4}, {2}};
    Node node2{2, {0.3, 0.7}, {1}};
    model.nodes = {node1, node2};
    model.node_index[1] = 0;
    model.node_index[2] = 1;

    // Edge potential matrix favoring same values
    EdgePotential edge{1, 1, 2, {{1.2, 0.8}, {0.8, 1.2}}};
    model.edges = {edge};

    // Run inference with momentum
    std::cout << "Running Momentum-Enhanced BP inference...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    auto momentum_result = momentum_engine->run_momentum_bp(model);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (momentum_result.is_err()) {
        std::cerr << "Momentum-BP inference failed: " << to_string(momentum_result.unwrap_err())
                  << std::endl;
        return;
    }

    auto momentum_marginals = momentum_result.unwrap();
    auto momentum_metrics = momentum_engine->get_metrics();

    std::cout << "\nMomentum-Enhanced BP Results:\n";
    print_marginals(momentum_marginals);
    std::cout << "\nPerformance:\n";
    std::cout << "  Converged: " << (momentum_metrics.converged ? "Yes" : "No") << "\n";
    std::cout << "  Iterations: " << momentum_metrics.iterations_to_convergence << "\n";
    std::cout << "  Final residual: " << std::scientific << momentum_metrics.final_residual << "\n";
    std::cout << "  Message updates: " << momentum_metrics.message_updates << "\n";
    std::cout << "  Inference time: " << duration.count() << " μs\n\n";

    // Compare with standard BP (momentum disabled)
    std::cout << "Comparing with Standard BP (no momentum/AdaGrad)...\n";

    MomentumBPConfig standard_config = momentum_config;
    standard_config.enable_momentum = false;
    standard_config.enable_adagrad = false;

    auto standard_engine_result = create_momentum_bp_engine(standard_config);
    if (standard_engine_result.is_err()) {
        std::cerr << "Failed to create Standard BP engine\n";
        return;
    }

    auto standard_engine = std::move(standard_engine_result).unwrap();

    start_time = std::chrono::high_resolution_clock::now();
    auto standard_result = standard_engine->run_momentum_bp(model);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (standard_result.is_err()) {
        std::cerr << "Standard BP inference failed: " << to_string(standard_result.unwrap_err())
                  << std::endl;
        return;
    }

    auto standard_marginals = standard_result.unwrap();
    auto standard_metrics = standard_engine->get_metrics();

    std::cout << "\nStandard BP Results:\n";
    print_marginals(standard_marginals);
    std::cout << "\nPerformance:\n";
    std::cout << "  Converged: " << (standard_metrics.converged ? "Yes" : "No") << "\n";
    std::cout << "  Iterations: " << standard_metrics.iterations_to_convergence << "\n";
    std::cout << "  Final residual: " << std::scientific << standard_metrics.final_residual << "\n";
    std::cout << "  Message updates: " << standard_metrics.message_updates << "\n";
    std::cout << "  Inference time: " << duration.count() << " μs\n\n";

    // Test unified InferenceEngine interface
    std::cout << "Testing Unified InferenceEngine Interface...\n";
    inference_lab::engines::InferenceRequest request;  // Empty request uses demo model

    auto unified_result = momentum_engine->run_inference(request);
    if (unified_result.is_ok()) {
        auto response = unified_result.unwrap();
        std::cout << "Unified interface successful:\n";
        std::cout << "  Output tensors: " << response.output_tensors.size() << "\n";
        std::cout << "  Output names: ";
        for (const auto& name : response.output_names) {
            std::cout << name << " ";
        }
        std::cout << "\n  Inference time: " << response.inference_time_ms << " ms\n\n";
    } else {
        std::cout << "Unified interface failed\n\n";
    }

    // Backend information
    std::cout << "Backend Information:\n";
    std::cout << momentum_engine->get_backend_info() << "\n";

    std::cout << "=== Demo completed successfully! ===\n";
}

int main() {
    try {
        run_momentum_bp_demo();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
