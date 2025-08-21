/**
 * @file socrates_mortality_demo.cpp
 * @brief Classic "Socrates is mortal" forward chaining inference demonstration
 *
 * This example demonstrates the forward chaining inference engine with the classic
 * AI reasoning problem: "Socrates is human, all humans are mortal, therefore Socrates is mortal."
 *
 * The demo shows:
 * - Creating facts and rules
 * - Running forward chaining inference
 * - Examining derived facts and reasoning trace
 * - Performance metrics and debugging information
 */

#include <iostream>
#include <vector>

#include "../../common/src/inference_builders.hpp"
#include "../../common/src/logging.hpp"
#include "../src/forward_chaining.hpp"
#include "../src/inference_engine.hpp"

using namespace inference_lab::engines;
using namespace inference_lab::common;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_facts(const std::vector<Fact>& facts, const std::string& title) {
    std::cout << "\n" << title << ":\n";
    if (facts.empty()) {
        std::cout << "  (none)\n";
        return;
    }

    for (const auto& fact : facts) {
        std::cout << "  " << fact.predicate() << "(";
        for (size_t i = 0; i < fact.args().size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << fact.args()[i].as_text();
        }
        std::cout << ")\n";
    }
}

void print_rules(const std::vector<Rule>& rules, const std::string& title) {
    std::cout << "\n" << title << ":\n";
    if (rules.empty()) {
        std::cout << "  (none)\n";
        return;
    }

    for (const auto& rule : rules) {
        std::cout << "  Rule '" << rule.name() << "' (priority=" << rule.priority() << "):\n";
        std::cout << "    IF ";

        for (size_t i = 0; i < rule.conditions().size(); ++i) {
            if (i > 0)
                std::cout << " AND ";
            const auto& cond = rule.conditions()[i];
            std::cout << cond.predicate << "(";
            for (size_t j = 0; j < cond.args.size(); ++j) {
                if (j > 0)
                    std::cout << ", ";
                std::cout << cond.args[j].as_text();
            }
            std::cout << ")";
        }

        std::cout << "\n    THEN ";
        for (size_t i = 0; i < rule.conclusions().size(); ++i) {
            if (i > 0)
                std::cout << " AND ";
            const auto& concl = rule.conclusions()[i];
            std::cout << concl.predicate << "(";
            for (size_t j = 0; j < concl.args.size(); ++j) {
                if (j > 0)
                    std::cout << ", ";
                std::cout << concl.args[j].as_text();
            }
            std::cout << ")";
        }
        std::cout << "\n";
    }
}

void print_firing_trace(const std::vector<RuleFiring>& trace) {
    std::cout << "\nRule Firing Trace:\n";
    if (trace.empty()) {
        std::cout << "  (no rules fired)\n";
        return;
    }

    for (size_t i = 0; i < trace.size(); ++i) {
        const auto& firing = trace[i];
        std::cout << "  " << (i + 1) << ". Rule '" << firing.rule_name << "' fired:\n";

        // Show variable bindings
        std::cout << "     Bindings: ";
        for (const auto& [var, value] : firing.bindings) {
            std::cout << var << "=" << value.as_text() << " ";
        }
        std::cout << "\n";

        // Show derived facts
        std::cout << "     Derived: ";
        for (size_t j = 0; j < firing.derived_facts.size(); ++j) {
            if (j > 0)
                std::cout << ", ";
            const auto& fact = firing.derived_facts[j];
            std::cout << fact.predicate() << "(";
            for (size_t k = 0; k < fact.args().size(); ++k) {
                if (k > 0)
                    std::cout << ", ";
                std::cout << fact.args()[k].as_text();
            }
            std::cout << ")";
        }
        std::cout << "\n";
    }
}

int main() {
    print_separator("Forward Chaining Inference Engine Demo");
    std::cout << "Demonstrating the classic 'Socrates is mortal' reasoning problem.\n";

    try {
        // Step 1: Create the forward chaining engine
        print_separator("Step 1: Creating Forward Chaining Engine");

        auto engine_result =
            create_forward_chaining_engine(ConflictResolutionStrategy::PRIORITY_ORDER,
                                           1000,  // max iterations
                                           true   // enable tracing for demonstration
            );

        if (engine_result.is_err()) {
            std::cerr << "Error: Failed to create forward chaining engine\n";
            return 1;
        }

        auto engine = std::move(engine_result.unwrap());
        std::cout << "✓ Forward chaining engine created successfully\n";
        std::cout << "  Backend info: " << engine->get_backend_info() << "\n";

        // Step 2: Add initial facts to the knowledge base
        print_separator("Step 2: Adding Initial Facts");

        // Create facts about Greek philosophers
        std::vector<Fact> initial_facts = {
            Fact(1, "isHuman", {Value::from_text("socrates")}),
            Fact(2, "isHuman", {Value::from_text("plato")}),
            Fact(3, "isHuman", {Value::from_text("aristotle")}),
            Fact(4, "livesIn", {Value::from_text("socrates"), Value::from_text("athens")}),
            Fact(5, "livesIn", {Value::from_text("plato"), Value::from_text("athens")}),
            Fact(6, "teaches", {Value::from_text("socrates"), Value::from_text("plato")})};

        for (const auto& fact : initial_facts) {
            auto result = engine->add_fact(fact);
            if (result.is_err()) {
                std::cerr << "Error adding fact: " << to_string(result.err()) << "\n";
                return 1;
            }
        }

        print_facts(engine->get_all_facts(), "Initial Knowledge Base");

        // Step 3: Add inference rules
        print_separator("Step 3: Adding Inference Rules");

        // Rule 1: All humans are mortal
        Rule mortality_rule(1, "mortality_rule");
        mortality_rule.add_condition("isHuman", {Value::from_text("X")});
        mortality_rule.add_conclusion("isMortal", {Value::from_text("X")});
        mortality_rule.set_priority(10);

        // Rule 2: If X teaches Y, then X is wise
        Rule wisdom_rule(2, "wisdom_rule");
        wisdom_rule.add_condition("teaches", {Value::from_text("X"), Value::from_text("Y")});
        wisdom_rule.add_conclusion("isWise", {Value::from_text("X")});
        wisdom_rule.set_priority(8);

        // Rule 3: If X is human and lives in Y, then X is a citizen of Y
        Rule citizenship_rule(3, "citizenship_rule");
        citizenship_rule.add_condition("isHuman", {Value::from_text("X")});
        citizenship_rule.add_condition("livesIn", {Value::from_text("X"), Value::from_text("Y")});
        citizenship_rule.add_conclusion("citizenOf",
                                        {Value::from_text("X"), Value::from_text("Y")});
        citizenship_rule.set_priority(9);

        std::vector<Rule> rules = {mortality_rule, wisdom_rule, citizenship_rule};

        for (const auto& rule : rules) {
            auto result = engine->add_rule(rule);
            if (result.is_err()) {
                std::cerr << "Error adding rule: " << to_string(result.err()) << "\n";
                return 1;
            }
        }

        print_rules(engine->get_all_rules(), "Inference Rules");

        // Step 4: Run forward chaining inference
        print_separator("Step 4: Running Forward Chaining Inference");

        std::cout << "Starting inference...\n";
        auto inference_result = engine->run_forward_chaining();

        if (inference_result.is_err()) {
            std::cerr << "Error during inference: " << to_string(inference_result.err()) << "\n";
            return 1;
        }

        auto derived_facts = inference_result.unwrap();
        std::cout << "✓ Inference completed successfully\n";
        std::cout << "  Derived " << derived_facts.size() << " new facts\n";

        // Step 5: Display results
        print_separator("Step 5: Inference Results");

        print_facts(derived_facts, "Newly Derived Facts");
        print_facts(engine->get_all_facts(), "Complete Knowledge Base (Initial + Derived)");

        // Step 6: Show reasoning trace
        print_separator("Step 6: Reasoning Trace");

        auto trace = engine->get_firing_trace();
        print_firing_trace(trace);

        // Step 7: Performance metrics
        print_separator("Step 7: Performance Metrics");

        auto metrics = engine->get_metrics();
        std::cout << "Performance Summary:\n";
        std::cout << "  Facts processed: " << metrics.facts_processed << "\n";
        std::cout << "  Rules evaluated: " << metrics.rules_evaluated << "\n";
        std::cout << "  Rules fired: " << metrics.rules_fired << "\n";
        std::cout << "  Pattern matches: " << metrics.pattern_matches << "\n";
        std::cout << "  Variable unifications: " << metrics.variable_unifications << "\n";
        std::cout << "  Facts derived: " << metrics.facts_derived << "\n";
        std::cout << "  Total time: " << metrics.total_time_ms.count() << " ms\n";
        std::cout << "  Indexing time: " << metrics.indexing_time_ms.count() << " ms\n";
        std::cout << "  Matching time: " << metrics.matching_time_ms.count() << " ms\n";

        // Step 8: Verify specific conclusions
        print_separator("Step 8: Verifying Key Conclusions");

        // Check if Socrates is mortal
        auto mortal_facts = engine->get_facts_by_predicate("isMortal");
        bool socrates_mortal = false;
        for (const auto& fact : mortal_facts) {
            if (fact.args()[0].as_text() == "socrates") {
                socrates_mortal = true;
                break;
            }
        }

        std::cout << "Key Conclusions:\n";
        std::cout << "  ✓ Socrates is mortal: " << (socrates_mortal ? "YES" : "NO") << "\n";

        // Check wisdom
        auto wise_facts = engine->get_facts_by_predicate("isWise");
        bool socrates_wise = false;
        for (const auto& fact : wise_facts) {
            if (fact.args()[0].as_text() == "socrates") {
                socrates_wise = true;
                break;
            }
        }
        std::cout << "  ✓ Socrates is wise: " << (socrates_wise ? "YES" : "NO") << "\n";

        // Check citizenship
        auto citizen_facts = engine->get_facts_by_predicate("citizenOf");
        int athens_citizens = 0;
        for (const auto& fact : citizen_facts) {
            if (fact.args()[1].as_text() == "athens") {
                athens_citizens++;
            }
        }
        std::cout << "  ✓ Citizens of Athens: " << athens_citizens << "\n";

        print_separator("Demo Completed Successfully");
        std::cout << "The forward chaining engine successfully demonstrated:\n";
        std::cout << "  • Pattern matching with variable binding\n";
        std::cout << "  • Multiple rule firing in correct priority order\n";
        std::cout << "  • Complex reasoning with multi-argument predicates\n";
        std::cout << "  • Performance monitoring and debugging support\n";
        std::cout << "\nClassic conclusion: Since Socrates is human, and all humans are mortal,\n";
        std::cout << "therefore Socrates is mortal. QED.\n";

    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
