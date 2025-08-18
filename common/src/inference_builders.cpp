// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file inference_builders.cpp
 * @brief Implementation of fluent builder classes for inference types
 *
 * This file implements the builder pattern classes that provide a fluent interface
 * for constructing Facts, Rules, and Queries. The builders use method chaining to
 * make object construction more readable and less error-prone.
 *
 * Key implementation details:
 * - Thread-safe ID generation using atomic counters
 * - State machines for building complex rules with conditions/conclusions
 * - Automatic type conversion and validation
 * - Proper error handling for incomplete or invalid constructions
 */

#include "inference_builders.hpp"

#include <atomic>
#include <stdexcept>

namespace inference_lab::common {

//=============================================================================
// FactBuilder Implementation
//=============================================================================
// The FactBuilder provides a simple fluent interface for creating facts.
// It accumulates arguments and properties, then constructs the final Fact
// object in the build() method.

FactBuilder::FactBuilder(const std::string& predicate) : predicate_(predicate) {}

FactBuilder& FactBuilder::with_arg(const Value& arg) {
    args_.push_back(arg);
    return *this;
}

FactBuilder& FactBuilder::with_arg(int64_t value) {
    return with_arg(Value::from_int64(value));
}

FactBuilder& FactBuilder::with_arg(double value) {
    return with_arg(Value::from_float64(value));
}

FactBuilder& FactBuilder::with_arg(const std::string& value) {
    return with_arg(Value::from_text(value));
}

FactBuilder& FactBuilder::with_arg(const char* value) {
    return with_arg(Value::from_text(std::string(value)));
}

FactBuilder& FactBuilder::with_arg(bool value) {
    return with_arg(Value::from_bool(value));
}

FactBuilder& FactBuilder::with_id(uint64_t id) {
    id_ = id;
    return *this;
}

FactBuilder& FactBuilder::with_confidence(double confidence) {
    confidence_ = confidence;
    return *this;
}

FactBuilder& FactBuilder::with_timestamp(uint64_t timestamp) {
    timestamp_ = timestamp;
    return *this;
}

FactBuilder& FactBuilder::with_metadata(const std::string& key, const Value& value) {
    metadata_[key] = value;
    return *this;
}

Fact FactBuilder::build() {
    if (id_ == 0) {
        id_ = next_id();
    }

    Fact fact(id_, predicate_, args_, confidence_, timestamp_);
    for (const auto& [key, value] : metadata_) {
        fact.set_metadata(key, value);
    }

    return fact;
}

uint64_t FactBuilder::next_id() {
    // Thread-safe ID generation using atomic counter
    // Starts at 1 since 0 is used to indicate "auto-generate ID"
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1);
}

//=============================================================================
// RuleBuilder Implementation
//=============================================================================
// The RuleBuilder implements a state machine for building complex rules.
// It tracks whether we're currently building a condition or conclusion
// and accumulates arguments for the current item being built.

RuleBuilder::RuleBuilder(const std::string& name) : name_(name) {}

RuleBuilder& RuleBuilder::when_condition(const std::string& predicate,
                                         const std::vector<Value>& args,
                                         bool negated) {
    finish_current_condition();
    finish_current_conclusion();

    Rule::Condition condition;
    condition.predicate_ = predicate;
    condition.args_ = args;
    condition.negated_ = negated;
    conditions_.push_back(condition);

    return *this;
}

RuleBuilder& RuleBuilder::when(const std::string& predicate) {
    finish_current_condition();
    finish_current_conclusion();

    current_predicate_ = predicate;
    current_args_.clear();
    current_negated_ = false;
    building_state_ = BuildingState::CONDITION;

    return *this;
}

RuleBuilder& RuleBuilder::when_not(const std::string& predicate) {
    finish_current_condition();
    finish_current_conclusion();

    current_predicate_ = predicate;
    current_args_.clear();
    current_negated_ = true;
    building_state_ = BuildingState::CONDITION;

    return *this;
}

RuleBuilder& RuleBuilder::with_arg(const Value& arg) {
    if (building_state_ == BuildingState::NONE) {
        throw std::runtime_error("Must call when() or then() before adding arguments");
    }
    current_args_.push_back(arg);
    return *this;
}

RuleBuilder& RuleBuilder::with_arg(int64_t value) {
    return with_arg(Value::from_int64(value));
}

RuleBuilder& RuleBuilder::with_arg(double value) {
    return with_arg(Value::from_float64(value));
}

RuleBuilder& RuleBuilder::with_arg(const std::string& value) {
    return with_arg(Value::from_text(value));
}

RuleBuilder& RuleBuilder::with_arg(const char* value) {
    return with_arg(Value::from_text(std::string(value)));
}

RuleBuilder& RuleBuilder::with_arg(bool value) {
    return with_arg(Value::from_bool(value));
}

RuleBuilder& RuleBuilder::with_variable(const std::string& var_name) {
    // Ensure variable name starts with uppercase
    std::string name = var_name;
    if (!name.empty() && std::islower(name[0])) {
        name[0] = std::toupper(name[0]);
    }
    return with_arg(Value::from_text(name));
}

RuleBuilder& RuleBuilder::then_conclusion(const std::string& predicate,
                                          const std::vector<Value>& args,
                                          double confidence) {
    finish_current_condition();
    finish_current_conclusion();

    Rule::Conclusion conclusion;
    conclusion.predicate_ = predicate;
    conclusion.args_ = args;
    conclusion.confidence_ = confidence;
    conclusions_.push_back(conclusion);

    return *this;
}

RuleBuilder& RuleBuilder::then(const std::string& predicate) {
    finish_current_condition();
    finish_current_conclusion();

    current_predicate_ = predicate;
    current_args_.clear();
    current_confidence_ = 1.0;
    building_state_ = BuildingState::CONCLUSION;

    return *this;
}

RuleBuilder& RuleBuilder::with_id(uint64_t id) {
    id_ = id;
    return *this;
}

RuleBuilder& RuleBuilder::with_priority(int32_t priority) {
    priority_ = priority;
    return *this;
}

RuleBuilder& RuleBuilder::with_confidence(double confidence) {
    if (building_state_ == BuildingState::CONCLUSION) {
        current_confidence_ = confidence;
    } else {
        confidence_ = confidence;
    }
    return *this;
}

Rule RuleBuilder::build() {
    finish_current_condition();
    finish_current_conclusion();

    if (conditions_.empty()) {
        throw std::runtime_error("Rule must have at least one condition");
    }
    if (conclusions_.empty()) {
        throw std::runtime_error("Rule must have at least one conclusion");
    }

    if (id_ == 0) {
        id_ = next_id();
    }

    return Rule(id_, name_, conditions_, conclusions_, priority_, confidence_);
}

void RuleBuilder::finish_current_condition() {
    // Complete the current condition being built and add it to the conditions list
    // This is called automatically when starting a new condition or conclusion
    if (building_state_ == BuildingState::CONDITION && !current_predicate_.empty()) {
        Rule::Condition condition;
        condition.predicate_ = current_predicate_;
        condition.args_ = current_args_;
        condition.negated_ = current_negated_;
        conditions_.push_back(condition);

        // Clear state for next condition/conclusion
        current_predicate_.clear();
        current_args_.clear();
        building_state_ = BuildingState::NONE;
    }
}

void RuleBuilder::finish_current_conclusion() {
    // Complete the current conclusion being built and add it to the conclusions list
    // This is called automatically when starting a new condition or conclusion
    if (building_state_ == BuildingState::CONCLUSION && !current_predicate_.empty()) {
        Rule::Conclusion conclusion;
        conclusion.predicate_ = current_predicate_;
        conclusion.args_ = current_args_;
        conclusion.confidence_ = current_confidence_;
        conclusions_.push_back(conclusion);

        // Clear state for next condition/conclusion
        current_predicate_.clear();
        current_args_.clear();
        current_confidence_ = 1.0;
        building_state_ = BuildingState::NONE;
    }
}

uint64_t RuleBuilder::next_id() {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1);
}

//=============================================================================
// QueryBuilder Implementation
//=============================================================================
// The QueryBuilder provides fluent interface for creating queries.
// It maintains the goal pattern being built and query parameters.

QueryBuilder::QueryBuilder(Query::Type type) : type_(type) {}

QueryBuilder& QueryBuilder::goal(const std::string& predicate,
                                 const std::vector<Value>& args,
                                 bool negated) {
    goal_predicate_ = predicate;
    goal_args_ = args;
    goal_negated_ = negated;
    goal_set_ = true;
    return *this;
}

QueryBuilder& QueryBuilder::goal(const std::string& predicate) {
    goal_predicate_ = predicate;
    goal_args_.clear();
    goal_negated_ = false;
    goal_set_ = true;
    return *this;
}

QueryBuilder& QueryBuilder::with_arg(const Value& arg) {
    if (!goal_set_) {
        throw std::runtime_error("Must call goal() before adding arguments");
    }
    goal_args_.push_back(arg);
    return *this;
}

QueryBuilder& QueryBuilder::with_arg(int64_t value) {
    return with_arg(Value::from_int64(value));
}

QueryBuilder& QueryBuilder::with_arg(double value) {
    return with_arg(Value::from_float64(value));
}

QueryBuilder& QueryBuilder::with_arg(const std::string& value) {
    return with_arg(Value::from_text(value));
}

QueryBuilder& QueryBuilder::with_arg(const char* value) {
    return with_arg(Value::from_text(std::string(value)));
}

QueryBuilder& QueryBuilder::with_arg(bool value) {
    return with_arg(Value::from_bool(value));
}

QueryBuilder& QueryBuilder::with_variable(const std::string& var_name) {
    // Ensure variable name starts with uppercase
    std::string name = var_name;
    if (!name.empty() && std::islower(name[0])) {
        name[0] = std::toupper(name[0]);
    }
    return with_arg(Value::from_text(name));
}

QueryBuilder& QueryBuilder::with_id(uint64_t id) {
    id_ = id;
    return *this;
}

QueryBuilder& QueryBuilder::max_results(uint32_t max) {
    max_results_ = max;
    return *this;
}

QueryBuilder& QueryBuilder::timeout(uint32_t timeoutMs) {
    timeout_ms_ = timeoutMs;
    return *this;
}

QueryBuilder& QueryBuilder::with_metadata(const std::string& key, const Value& value) {
    metadata_[key] = value;
    return *this;
}

Query QueryBuilder::build() {
    if (!goal_set_ || goal_predicate_.empty()) {
        throw std::runtime_error("Query must have a goal");
    }

    if (id_ == 0) {
        id_ = next_id();
    }

    Rule::Condition goal;
    goal.predicate_ = goal_predicate_;
    goal.args_ = goal_args_;
    goal.negated_ = goal_negated_;

    return Query(id_, type_, goal, max_results_, timeout_ms_);
}

QueryBuilder QueryBuilder::find_all() {
    return QueryBuilder(Query::Type::FIND_ALL);
}

QueryBuilder QueryBuilder::prove() {
    return QueryBuilder(Query::Type::PROVE);
}

QueryBuilder QueryBuilder::find_first(uint32_t limit) {
    return QueryBuilder(Query::Type::FIND_FIRST).max_results(limit);
}

QueryBuilder QueryBuilder::explain() {
    return QueryBuilder(Query::Type::EXPLAIN);
}

uint64_t QueryBuilder::next_id() {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1);
}

}  // namespace inference_lab::common
