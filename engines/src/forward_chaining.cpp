// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file forward_chaining.cpp
 * @brief Implementation of forward chaining inference engine
 */

#include "forward_chaining.hpp"

#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>

using namespace inference_lab::common;  // For LogLevel and other common types

namespace inference_lab::engines {

namespace {
// Anonymous namespace for internal utilities

/**
 * @brief Check if a string represents a variable name
 * Variables start with uppercase letters or underscore
 */
bool is_variable_name(const std::string& name) {
    return !name.empty() && (std::isupper(name[0]) || name[0] == '_');
}

/**
 * @brief Generate current timestamp in milliseconds
 */
std::uint64_t current_timestamp_ms() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

/**
 * @brief Convert value to string for debugging
 */
std::string value_to_debug_string(const common::Value& value) {
    if (value.is_text()) {
        return value.as_text();
    } else if (value.is_int64()) {
        return std::to_string(value.as_int64());
    } else if (value.is_float64()) {
        return std::to_string(value.as_float64());
    } else if (value.is_bool()) {
        return value.as_bool() ? "true" : "false";
    }
    return "<complex_value>";
}
}  // namespace

// Error conversion function
std::string to_string(ForwardChainingError error) {
    switch (error) {
        case ForwardChainingError::INVALID_RULE_FORMAT:
            return "Invalid rule format: malformed conditions or conclusions";
        case ForwardChainingError::VARIABLE_BINDING_FAILED:
            return "Variable binding failed: unable to unify variables across conditions";
        case ForwardChainingError::CYCLE_DETECTED:
            return "Cycle detected: infinite loop prevented in rule firing";
        case ForwardChainingError::FACT_DATABASE_CORRUPT:
            return "Fact database corrupt: inconsistent internal state";
        case ForwardChainingError::RULE_EVALUATION_TIMEOUT:
            return "Rule evaluation timeout: inference exceeded time limit";
        case ForwardChainingError::UNKNOWN_ERROR:
        default:
            return "Unknown forward chaining error";
    }
}

// ForwardChainingEngine implementation

ForwardChainingEngine::ForwardChainingEngine(ConflictResolutionStrategy strategy,
                                             std::uint32_t max_iterations,
                                             bool enable_tracing)
    : conflict_strategy_(strategy),
      max_iterations_(max_iterations),
      tracing_enabled_(enable_tracing) {
    LOG_INFO_PRINT("ForwardChainingEngine created with strategy={}, max_iterations={}, tracing={}",
                   static_cast<int>(strategy),
                   max_iterations,
                   enable_tracing);
}

auto ForwardChainingEngine::run_inference(const InferenceRequest& request)
    -> common::Result<InferenceResponse, InferenceError> {
    LOG_INFO_PRINT("Starting forward chaining inference");

    // For now, we'll use the existing facts and rules in the engine
    // In the future, we can extract facts/rules from the request
    auto result = run_forward_chaining();

    if (result.is_err()) {
        // Convert ForwardChainingError to InferenceError
        LOG_ERROR_PRINT("Forward chaining failed: {}", to_string(result.unwrap_err()));
        return common::Result<InferenceResponse, InferenceError>(
            common::Err<InferenceError>(InferenceError::INFERENCE_EXECUTION_FAILED));
    }

    // Convert derived facts to inference response
    InferenceResponse response;
    response.inference_time_ms = static_cast<double>(metrics_.total_time_ms.count());
    response.memory_used_bytes = all_facts_.size() * sizeof(common::Fact);  // Rough estimate

    // For rule-based inference, we don't have tensor outputs
    // The derived facts would be stored in a future extension:
    // response.derived_facts = result.unwrap();

    LOG_INFO_PRINT("Forward chaining completed successfully: {} facts derived in {}ms",
                   result.unwrap().size(),
                   metrics_.total_time_ms.count());

    return common::Result<InferenceResponse, InferenceError>(
        common::Ok<InferenceResponse>(response));
}

auto ForwardChainingEngine::get_backend_info() const -> std::string {
    std::ostringstream info;
    info << "ForwardChainingEngine{";
    info << "strategy=" << static_cast<int>(conflict_strategy_);
    info << ", max_iterations=" << max_iterations_;
    info << ", tracing=" << (tracing_enabled_ ? "enabled" : "disabled");
    info << ", facts=" << all_facts_.size();
    info << ", rules=" << rules_.size();
    info << "}";
    return info.str();
}

auto ForwardChainingEngine::is_ready() const -> bool {
    // Engine is ready if it has at least one fact or one rule
    return !all_facts_.empty() || !rules_.empty();
}

auto ForwardChainingEngine::get_performance_stats() const -> std::string {
    std::ostringstream stats;
    stats << "ForwardChaining Performance Metrics:\n";
    stats << "  Facts processed: " << metrics_.facts_processed << "\n";
    stats << "  Rules evaluated: " << metrics_.rules_evaluated << "\n";
    stats << "  Rules fired: " << metrics_.rules_fired << "\n";
    stats << "  Pattern matches: " << metrics_.pattern_matches << "\n";
    stats << "  Variable unifications: " << metrics_.variable_unifications << "\n";
    stats << "  Facts derived: " << metrics_.facts_derived << "\n";
    stats << "  Total time: " << metrics_.total_time_ms.count() << "ms\n";
    stats << "  Indexing time: " << metrics_.indexing_time_ms.count() << "ms\n";
    stats << "  Matching time: " << metrics_.matching_time_ms.count() << "ms\n";

    if (!firing_trace_.empty()) {
        stats << "  Rule firings: " << firing_trace_.size() << "\n";
    }

    return stats.str();
}

auto ForwardChainingEngine::add_fact(const common::Fact& fact)
    -> common::Result<std::monostate, ForwardChainingError> {
    if (fact_exists(fact)) {
        LOG_DEBUG_PRINT(
            "Fact already exists: {}({})", fact.get_predicate(), fact.get_args().size());
        return common::Result<std::monostate, ForwardChainingError>(
            common::Ok<std::monostate>(std::monostate{}));
    }

    auto start = std::chrono::high_resolution_clock::now();

    all_facts_.push_back(fact);
    update_fact_index(fact);
    metrics_.facts_processed++;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    metrics_.indexing_time_ms += duration;

    LOG_DEBUG_PRINT("Added fact: {}({})", fact.get_predicate(), fact.get_args().size());

    return common::Result<std::monostate, ForwardChainingError>(
        common::Ok<std::monostate>(std::monostate{}));
}

auto ForwardChainingEngine::add_facts(const std::vector<common::Fact>& facts)
    -> common::Result<std::monostate, ForwardChainingError> {
    for (const auto& fact : facts) {
        auto result = add_fact(fact);
        if (result.is_err()) {
            return result;
        }
    }

    LOG_INFO_PRINT("Added {} facts to knowledge base", facts.size());
    return common::Result<std::monostate, ForwardChainingError>(
        common::Ok<std::monostate>(std::monostate{}));
}

auto ForwardChainingEngine::add_rule(const common::Rule& rule)
    -> common::Result<std::monostate, ForwardChainingError> {
    // Validate rule has conditions and conclusions
    if (rule.get_conditions().empty()) {
        LOG_ERROR_PRINT("Rule {} has no conditions", rule.get_name());
        return common::Result<std::monostate, ForwardChainingError>(
            common::Err<ForwardChainingError>(ForwardChainingError::INVALID_RULE_FORMAT));
    }

    if (rule.get_conclusions().empty()) {
        LOG_ERROR_PRINT("Rule {} has no conclusions", rule.get_name());
        return common::Result<std::monostate, ForwardChainingError>(
            common::Err<ForwardChainingError>(ForwardChainingError::INVALID_RULE_FORMAT));
    }

    rules_.push_back(rule);

    // Sort rules by priority for conflict resolution
    std::sort(rules_.begin(), rules_.end(), [](const common::Rule& a, const common::Rule& b) {
        return a.get_priority() > b.get_priority();  // Higher priority first
    });

    LOG_DEBUG_PRINT("Added rule: {} (priority={})", rule.get_name(), rule.get_priority());

    return common::Result<std::monostate, ForwardChainingError>(
        common::Ok<std::monostate>(std::monostate{}));
}

auto ForwardChainingEngine::add_rules(const std::vector<common::Rule>& rules)
    -> common::Result<std::monostate, ForwardChainingError> {
    for (const auto& rule : rules) {
        auto result = add_rule(rule);
        if (result.is_err()) {
            return result;
        }
    }

    LOG_INFO_PRINT("Added {} rules to rule base", rules.size());
    return common::Result<std::monostate, ForwardChainingError>(
        common::Ok<std::monostate>(std::monostate{}));
}

auto ForwardChainingEngine::run_forward_chaining()
    -> common::Result<std::vector<common::Fact>, ForwardChainingError> {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<common::Fact> derived_facts;
    std::uint32_t iteration = 0;
    bool facts_derived_this_iteration = true;

    LOG_INFO_PRINT(
        "Starting forward chaining: {} initial facts, {} rules", all_facts_.size(), rules_.size());

    // Clear previous firing trace
    if (tracing_enabled_) {
        firing_trace_.clear();
    }

    // Main forward chaining loop
    while (facts_derived_this_iteration && iteration < max_iterations_) {
        facts_derived_this_iteration = false;
        iteration++;

        LOG_DEBUG_PRINT("Forward chaining iteration {}", iteration);

        // Find all applicable rules
        auto applicable_rules = find_applicable_rules();
        metrics_.rules_evaluated += applicable_rules.size();

        if (applicable_rules.empty()) {
            LOG_DEBUG_PRINT("No applicable rules found, terminating");
            break;
        }

        // Apply conflict resolution strategy
        auto ordered_rules = resolve_conflicts(applicable_rules);

        // Fire each applicable rule
        for (const auto& [rule, bindings] : ordered_rules) {
            auto new_facts = fire_rule(rule, bindings);

            if (!new_facts.empty()) {
                facts_derived_this_iteration = true;
                metrics_.rules_fired++;
                metrics_.facts_derived += new_facts.size();

                // Add new facts to knowledge base and result
                for (const auto& fact : new_facts) {
                    auto add_result = add_fact(fact);
                    if (add_result.is_ok()) {
                        derived_facts.push_back(fact);
                    }
                }

                LOG_DEBUG_PRINT(
                    "Rule '{}' fired, derived {} new facts", rule.get_name(), new_facts.size());
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    metrics_.total_time_ms = total_duration;

    if (iteration >= max_iterations_) {
        LOG_WARNING_PRINT("Forward chaining reached maximum iterations ({}), possible cycle",
                          max_iterations_);
        return common::Result<std::vector<common::Fact>, ForwardChainingError>(
            common::Err<ForwardChainingError>(ForwardChainingError::RULE_EVALUATION_TIMEOUT));
    }

    LOG_INFO_PRINT("Forward chaining completed: {} iterations, {} facts derived",
                   iteration,
                   derived_facts.size());

    return common::Result<std::vector<common::Fact>, ForwardChainingError>(
        common::Ok<std::vector<common::Fact>>(derived_facts));
}

void ForwardChainingEngine::clear_facts() {
    all_facts_.clear();
    fact_index_.clear();
    LOG_INFO_PRINT("Cleared all facts from knowledge base");
}

void ForwardChainingEngine::clear_rules() {
    rules_.clear();
    LOG_INFO_PRINT("Cleared all rules from rule base");
}

auto ForwardChainingEngine::get_all_facts() const -> std::vector<common::Fact> {
    return all_facts_;
}

auto ForwardChainingEngine::get_all_rules() const -> std::vector<common::Rule> {
    return rules_;
}

auto ForwardChainingEngine::get_facts_by_predicate(const std::string& predicate) const
    -> std::vector<common::Fact> {
    auto it = fact_index_.find(predicate);
    if (it != fact_index_.end()) {
        return it->second;
    }
    return {};
}

auto ForwardChainingEngine::get_firing_trace() const -> std::vector<RuleFiring> {
    return firing_trace_;
}

auto ForwardChainingEngine::get_metrics() const -> InferenceMetrics {
    return metrics_;
}

void ForwardChainingEngine::reset_metrics() {
    metrics_ = InferenceMetrics{};
    LOG_DEBUG_PRINT("Reset inference metrics");
}

void ForwardChainingEngine::set_conflict_resolution_strategy(ConflictResolutionStrategy strategy) {
    conflict_strategy_ = strategy;
    LOG_INFO_PRINT("Set conflict resolution strategy to {}", static_cast<int>(strategy));
}

void ForwardChainingEngine::set_tracing_enabled(bool enable) {
    tracing_enabled_ = enable;
    if (!enable) {
        firing_trace_.clear();
    }
    LOG_INFO_PRINT("Set rule firing tracing to {}", enable ? "enabled" : "disabled");
}

// Private implementation methods

auto ForwardChainingEngine::find_applicable_rules() const
    -> std::vector<std::pair<common::Rule, VariableBindings>> {
    std::vector<std::pair<common::Rule, VariableBindings>> applicable;

    for (const auto& rule : rules_) {
        auto bindings = can_rule_fire(rule);
        if (bindings.has_value()) {
            applicable.emplace_back(rule, bindings.value());
            LOG_DEBUG_PRINT("Rule '{}' is applicable", rule.get_name());
        }
    }

    return applicable;
}

auto ForwardChainingEngine::can_rule_fire(const common::Rule& rule) const
    -> std::optional<VariableBindings> {
    VariableBindings bindings;

    // All conditions must be satisfied
    for (const auto& condition : rule.get_conditions()) {
        if (!match_condition(condition, bindings)) {
            return std::nullopt;
        }
    }

    metrics_.pattern_matches++;
    return bindings;
}

auto ForwardChainingEngine::match_condition(const common::Rule::Condition& condition,
                                            VariableBindings& bindings) const -> bool {
    // Get facts that match the condition's predicate
    auto candidate_facts = get_facts_by_predicate(condition.predicate_);

    for (const auto& fact : candidate_facts) {
        // Try to unify condition arguments with fact arguments
        if (condition.args_.size() != fact.get_args().size()) {
            continue;  // Arity mismatch
        }

        VariableBindings temp_bindings = bindings;  // Copy current bindings
        bool unification_success = true;

        for (size_t i = 0; i < condition.args_.size(); ++i) {
            if (!unify_values(condition.args_[i], fact.get_args()[i], temp_bindings)) {
                unification_success = false;
                break;
            }
        }

        if (unification_success) {
            bindings = temp_bindings;  // Update bindings
            metrics_.variable_unifications++;
            return true;
        }
    }

    return false;
}

auto ForwardChainingEngine::unify_values(const common::Value& pattern,
                                         const common::Value& instance,
                                         VariableBindings& bindings) const -> bool {
    // If pattern is a variable
    if (is_variable(pattern)) {
        std::string var_name = pattern.as_text();

        auto it = bindings.find(var_name);
        if (it != bindings.end()) {
            // Variable already bound, check consistency
            return unify_values(it->second, instance, bindings);
        } else {
            // Bind variable to instance
            bindings[var_name] = instance;
            return true;
        }
    }

    // If instance is a variable (reverse case)
    if (is_variable(instance)) {
        std::string var_name = instance.as_text();

        auto it = bindings.find(var_name);
        if (it != bindings.end()) {
            return unify_values(pattern, it->second, bindings);
        } else {
            bindings[var_name] = pattern;
            return true;
        }
    }

    // Both are constants - must be equal
    if (pattern.is_text() && instance.is_text()) {
        return pattern.as_text() == instance.as_text();
    } else if (pattern.is_int64() && instance.is_int64()) {
        return pattern.as_int64() == instance.as_int64();
    } else if (pattern.is_float64() && instance.is_float64()) {
        return std::abs(pattern.as_float64() - instance.as_float64()) < 1e-9;
    } else if (pattern.is_bool() && instance.is_bool()) {
        return pattern.as_bool() == instance.as_bool();
    }

    return false;  // Type mismatch or unsupported types
}

auto ForwardChainingEngine::is_variable(const common::Value& value) const -> bool {
    return value.is_text() && is_variable_name(value.as_text());
}

auto ForwardChainingEngine::instantiate_conclusion(const common::Rule::Conclusion& conclusion,
                                                   const VariableBindings& bindings) const
    -> common::Fact {
    std::vector<common::Value> instantiated_args;

    for (const auto& arg : conclusion.args_) {
        if (is_variable(arg)) {
            std::string var_name = arg.as_text();
            auto it = bindings.find(var_name);
            if (it != bindings.end()) {
                instantiated_args.push_back(it->second);
            } else {
                // Unbound variable - use original
                instantiated_args.push_back(arg);
            }
        } else {
            instantiated_args.push_back(arg);
        }
    }

    return common::Fact(generate_fact_id(),
                        conclusion.predicate_,
                        instantiated_args,
                        1.0,  // Default confidence
                        current_timestamp_ms());
}

auto ForwardChainingEngine::fire_rule(const common::Rule& rule, const VariableBindings& bindings)
    -> std::vector<common::Fact> {
    std::vector<common::Fact> new_facts;
    std::vector<std::uint64_t> triggering_fact_ids;

    // Instantiate all conclusions with the variable bindings
    for (const auto& conclusion : rule.get_conclusions()) {
        auto new_fact = instantiate_conclusion(conclusion, bindings);

        // Only add if fact doesn't already exist
        if (!fact_exists(new_fact)) {
            new_facts.push_back(new_fact);
        }
    }

    // Record rule firing if tracing is enabled
    if (tracing_enabled_ && !new_facts.empty()) {
        record_rule_firing(rule, bindings, triggering_fact_ids, new_facts);
    }

    return new_facts;
}

auto ForwardChainingEngine::resolve_conflicts(
    std::vector<std::pair<common::Rule, VariableBindings>>& applicable_rules) const
    -> std::vector<std::pair<common::Rule, VariableBindings>> {
    switch (conflict_strategy_) {
        case ConflictResolutionStrategy::PRIORITY_ORDER:
            // Rules are already sorted by priority in add_rule()
            break;

        case ConflictResolutionStrategy::RECENCY_FIRST:
            // Sort by rule ID (assuming higher ID = more recent)
            std::sort(
                applicable_rules.begin(), applicable_rules.end(), [](const auto& a, const auto& b) {
                    return a.first.get_id() > b.first.get_id();
                });
            break;

        case ConflictResolutionStrategy::SPECIFICITY_FIRST:
            // Sort by number of conditions (more conditions = more specific)
            std::sort(
                applicable_rules.begin(), applicable_rules.end(), [](const auto& a, const auto& b) {
                    return a.first.get_conditions().size() > b.first.get_conditions().size();
                });
            break;

        case ConflictResolutionStrategy::RANDOM_ORDER: {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::shuffle(applicable_rules.begin(), applicable_rules.end(), gen);
        } break;
    }

    return applicable_rules;
}

auto ForwardChainingEngine::fact_exists(const common::Fact& fact) const -> bool {
    auto candidates = get_facts_by_predicate(fact.get_predicate());

    for (const auto& existing : candidates) {
        if (existing.get_predicate() == fact.get_predicate() &&
            existing.get_args().size() == fact.get_args().size()) {
            bool args_match = true;
            for (size_t i = 0; i < existing.get_args().size(); ++i) {
                VariableBindings dummy_bindings;
                if (!unify_values(existing.get_args()[i], fact.get_args()[i], dummy_bindings)) {
                    args_match = false;
                    break;
                }
            }

            if (args_match) {
                return true;
            }
        }
    }

    return false;
}

void ForwardChainingEngine::update_fact_index(const common::Fact& fact) {
    fact_index_[fact.get_predicate()].push_back(fact);
}

void ForwardChainingEngine::rebuild_fact_index() {
    fact_index_.clear();
    for (const auto& fact : all_facts_) {
        update_fact_index(fact);
    }
}

auto ForwardChainingEngine::generate_fact_id() const -> std::uint64_t {
    static std::uint64_t next_id = 1;
    return next_id++;
}

void ForwardChainingEngine::record_rule_firing(const common::Rule& rule,
                                               const VariableBindings& bindings,
                                               const std::vector<std::uint64_t>& triggering_facts,
                                               const std::vector<common::Fact>& derived_facts) {
    if (!tracing_enabled_) {
        return;
    }

    RuleFiring firing;
    firing.rule_id = rule.get_id();
    firing.rule_name = rule.get_name();
    firing.triggering_facts = triggering_facts;
    firing.derived_facts = derived_facts;
    firing.bindings = bindings;
    firing.firing_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch());

    firing_trace_.push_back(firing);
}

// Factory function
auto create_forward_chaining_engine(ConflictResolutionStrategy strategy,
                                    std::uint32_t max_iterations,
                                    bool enable_tracing)
    -> common::Result<std::unique_ptr<ForwardChainingEngine>, ForwardChainingError> {
    try {
        auto engine =
            std::make_unique<ForwardChainingEngine>(strategy, max_iterations, enable_tracing);
        return common::Result<std::unique_ptr<ForwardChainingEngine>, ForwardChainingError>(
            common::Ok<std::unique_ptr<ForwardChainingEngine>>(std::move(engine)));
    } catch (const std::exception& e) {
        LOG_ERROR_PRINT("Failed to create forward chaining engine: {}", e.what());
        return common::Result<std::unique_ptr<ForwardChainingEngine>, ForwardChainingError>(
            common::Err<ForwardChainingError>(ForwardChainingError::UNKNOWN_ERROR));
    }
}

}  // namespace inference_lab::engines
