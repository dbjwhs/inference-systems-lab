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

/**
 * @file simple_forward_chaining_demo.cpp
 * @brief Simple demonstration of forward chaining inference engine
 *
 * This demonstrates the basic functionality of the forward chaining engine
 * using the classic "Socrates is mortal" logical reasoning example.
 */

#include <iostream>
#include <vector>

#include "../../common/src/inference_types.hpp"
#include "../src/forward_chaining.hpp"

using namespace inference_lab;
using namespace inference_lab::common;
using namespace inference_lab::engines;

void print_facts(const std::vector<Fact>& facts) {
    std::cout << "Facts:\n";
    for (const auto& fact : facts) {
        std::cout << "  " << fact.get_predicate() << "(";
        for (size_t i = 0; i < fact.get_args().size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << fact.get_args()[i].as_text();
        }
        std::cout << ")\n";
    }
}

int main() {
    std::cout << "=== Forward Chaining Inference Engine Demo ===\n\n";

    try {
        // Create forward chaining engine
        auto engine_result =
            create_forward_chaining_engine(ConflictResolutionStrategy::PRIORITY_ORDER,
                                           1000,  // max iterations
                                           true   // enable tracing
            );

        if (engine_result.is_err()) {
            std::cerr << "Failed to create engine: " << to_string(engine_result.unwrap_err())
                      << std::endl;
            return 1;
        }

        auto engine = std::move(engine_result).unwrap();

        std::cout << "Engine created successfully!\n";
        std::cout << "Backend info: " << engine->get_backend_info() << "\n\n";

        // Create some facts
        std::vector<Fact> facts = {Fact(1, "isHuman", {Value::from_text("socrates")}),
                                   Fact(2, "isHuman", {Value::from_text("plato")}),
                                   Fact(3, "isGreek", {Value::from_text("socrates")})};

        // Add facts to the engine
        for (const auto& fact : facts) {
            auto result = engine->add_fact(fact);
            if (result.is_err()) {
                std::cerr << "Failed to add fact\n";
                return 1;
            }
        }

        std::cout << "Initial facts added successfully!\n";
        print_facts(engine->get_all_facts());
        std::cout << "\n";

        // Create mortality rule: isHuman(X) -> isMortal(X)
        std::vector<Rule::Condition> conditions = {{"isHuman", {Value::from_text("X")}, false}};

        std::vector<Rule::Conclusion> conclusions = {{"isMortal", {Value::from_text("X")}, 1.0}};

        Rule mortality_rule(1, "mortality_rule", conditions, conclusions, 10, 1.0);

        auto add_rule_result = engine->add_rule(mortality_rule);
        if (add_rule_result.is_err()) {
            std::cerr << "Failed to add mortality rule\n";
            return 1;
        }

        std::cout << "Mortality rule added: isHuman(X) -> isMortal(X)\n\n";

        // Run inference
        std::cout << "Running forward chaining inference...\n";
        auto inference_result = engine->run_forward_chaining();

        if (inference_result.is_err()) {
            std::cerr << "Inference failed: " << to_string(inference_result.unwrap_err())
                      << std::endl;
            return 1;
        }

        auto derived_facts = inference_result.unwrap();

        std::cout << "\nInference completed successfully!\n";
        std::cout << "Derived " << derived_facts.size() << " new facts:\n";
        print_facts(derived_facts);

        std::cout << "\nAll facts after inference:\n";
        print_facts(engine->get_all_facts());

        std::cout << "\nPerformance metrics:\n";
        std::cout << engine->get_performance_stats() << std::endl;

        std::cout << "\n=== Demo completed successfully! ===\n";

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
