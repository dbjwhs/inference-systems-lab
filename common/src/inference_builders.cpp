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

FactBuilder& FactBuilder::withArg(const Value& arg) {
    args_.push_back(arg);
    return *this;
}

FactBuilder& FactBuilder::withArg(int64_t value) {
    return withArg(Value::fromInt64(value));
}

FactBuilder& FactBuilder::withArg(double value) {
    return withArg(Value::fromFloat64(value));
}

FactBuilder& FactBuilder::withArg(const std::string& value) {
    return withArg(Value::fromText(value));
}

FactBuilder& FactBuilder::withArg(const char* value) {
    return withArg(Value::fromText(std::string(value)));
}

FactBuilder& FactBuilder::withArg(bool value) {
    return withArg(Value::fromBool(value));
}

FactBuilder& FactBuilder::withId(uint64_t id) {
    id_ = id;
    return *this;
}

FactBuilder& FactBuilder::withConfidence(double confidence) {
    confidence_ = confidence;
    return *this;
}

FactBuilder& FactBuilder::withTimestamp(uint64_t timestamp) {
    timestamp_ = timestamp;
    return *this;
}

FactBuilder& FactBuilder::withMetadata(const std::string& key, const Value& value) {
    metadata_[key] = value;
    return *this;
}

Fact FactBuilder::build() {
    if (id_ == 0) {
        id_ = nextId();
    }
    
    Fact fact(id_, predicate_, args_, confidence_, timestamp_);
    for (const auto& [key, value] : metadata_) {
        fact.setMetadata(key, value);
    }
    
    return fact;
}

uint64_t FactBuilder::nextId() {
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

RuleBuilder& RuleBuilder::whenCondition(const std::string& predicate, const std::vector<Value>& args, bool negated) {
    finishCurrentCondition();
    finishCurrentConclusion();
    
    Rule::Condition condition;
    condition.predicate = predicate;
    condition.args = args;
    condition.negated = negated;
    conditions_.push_back(condition);
    
    return *this;
}

RuleBuilder& RuleBuilder::when(const std::string& predicate) {
    finishCurrentCondition();
    finishCurrentConclusion();
    
    currentPredicate_ = predicate;
    currentArgs_.clear();
    currentNegated_ = false;
    buildingState_ = BuildingState::Condition;
    
    return *this;
}

RuleBuilder& RuleBuilder::whenNot(const std::string& predicate) {
    finishCurrentCondition();
    finishCurrentConclusion();
    
    currentPredicate_ = predicate;
    currentArgs_.clear();
    currentNegated_ = true;
    buildingState_ = BuildingState::Condition;
    
    return *this;
}

RuleBuilder& RuleBuilder::withArg(const Value& arg) {
    if (buildingState_ == BuildingState::None) {
        throw std::runtime_error("Must call when() or then() before adding arguments");
    }
    currentArgs_.push_back(arg);
    return *this;
}

RuleBuilder& RuleBuilder::withArg(int64_t value) {
    return withArg(Value::fromInt64(value));
}

RuleBuilder& RuleBuilder::withArg(double value) {
    return withArg(Value::fromFloat64(value));
}

RuleBuilder& RuleBuilder::withArg(const std::string& value) {
    return withArg(Value::fromText(value));
}

RuleBuilder& RuleBuilder::withArg(const char* value) {
    return withArg(Value::fromText(std::string(value)));
}

RuleBuilder& RuleBuilder::withArg(bool value) {
    return withArg(Value::fromBool(value));
}

RuleBuilder& RuleBuilder::withVariable(const std::string& varName) {
    // Ensure variable name starts with uppercase
    std::string name = varName;
    if (!name.empty() && std::islower(name[0])) {
        name[0] = std::toupper(name[0]);
    }
    return withArg(Value::fromText(name));
}

RuleBuilder& RuleBuilder::thenConclusion(const std::string& predicate, const std::vector<Value>& args, double confidence) {
    finishCurrentCondition();
    finishCurrentConclusion();
    
    Rule::Conclusion conclusion;
    conclusion.predicate = predicate;
    conclusion.args = args;
    conclusion.confidence = confidence;
    conclusions_.push_back(conclusion);
    
    return *this;
}

RuleBuilder& RuleBuilder::then(const std::string& predicate) {
    finishCurrentCondition();
    finishCurrentConclusion();
    
    currentPredicate_ = predicate;
    currentArgs_.clear();
    currentConfidence_ = 1.0;
    buildingState_ = BuildingState::Conclusion;
    
    return *this;
}

RuleBuilder& RuleBuilder::withId(uint64_t id) {
    id_ = id;
    return *this;
}

RuleBuilder& RuleBuilder::withPriority(int32_t priority) {
    priority_ = priority;
    return *this;
}

RuleBuilder& RuleBuilder::withConfidence(double confidence) {
    if (buildingState_ == BuildingState::Conclusion) {
        currentConfidence_ = confidence;
    } else {
        confidence_ = confidence;
    }
    return *this;
}

Rule RuleBuilder::build() {
    finishCurrentCondition();
    finishCurrentConclusion();
    
    if (conditions_.empty()) {
        throw std::runtime_error("Rule must have at least one condition");
    }
    if (conclusions_.empty()) {
        throw std::runtime_error("Rule must have at least one conclusion");
    }
    
    if (id_ == 0) {
        id_ = nextId();
    }
    
    return Rule(id_, name_, conditions_, conclusions_, priority_, confidence_);
}

void RuleBuilder::finishCurrentCondition() {
    // Complete the current condition being built and add it to the conditions list
    // This is called automatically when starting a new condition or conclusion
    if (buildingState_ == BuildingState::Condition && !currentPredicate_.empty()) {
        Rule::Condition condition;
        condition.predicate = currentPredicate_;
        condition.args = currentArgs_;
        condition.negated = currentNegated_;
        conditions_.push_back(condition);
        
        // Clear state for next condition/conclusion
        currentPredicate_.clear();
        currentArgs_.clear();
        buildingState_ = BuildingState::None;
    }
}

void RuleBuilder::finishCurrentConclusion() {
    // Complete the current conclusion being built and add it to the conclusions list
    // This is called automatically when starting a new condition or conclusion
    if (buildingState_ == BuildingState::Conclusion && !currentPredicate_.empty()) {
        Rule::Conclusion conclusion;
        conclusion.predicate = currentPredicate_;
        conclusion.args = currentArgs_;
        conclusion.confidence = currentConfidence_;
        conclusions_.push_back(conclusion);
        
        // Clear state for next condition/conclusion
        currentPredicate_.clear();
        currentArgs_.clear();
        currentConfidence_ = 1.0;
        buildingState_ = BuildingState::None;
    }
}

uint64_t RuleBuilder::nextId() {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1);
}

//=============================================================================
// QueryBuilder Implementation
//=============================================================================
// The QueryBuilder provides fluent interface for creating queries.
// It maintains the goal pattern being built and query parameters.

QueryBuilder::QueryBuilder(Query::Type type) : type_(type) {}

QueryBuilder& QueryBuilder::goal(const std::string& predicate, const std::vector<Value>& args, bool negated) {
    goalPredicate_ = predicate;
    goalArgs_ = args;
    goalNegated_ = negated;
    goalSet_ = true;
    return *this;
}

QueryBuilder& QueryBuilder::goal(const std::string& predicate) {
    goalPredicate_ = predicate;
    goalArgs_.clear();
    goalNegated_ = false;
    goalSet_ = true;
    return *this;
}

QueryBuilder& QueryBuilder::withArg(const Value& arg) {
    if (!goalSet_) {
        throw std::runtime_error("Must call goal() before adding arguments");
    }
    goalArgs_.push_back(arg);
    return *this;
}

QueryBuilder& QueryBuilder::withArg(int64_t value) {
    return withArg(Value::fromInt64(value));
}

QueryBuilder& QueryBuilder::withArg(double value) {
    return withArg(Value::fromFloat64(value));
}

QueryBuilder& QueryBuilder::withArg(const std::string& value) {
    return withArg(Value::fromText(value));
}

QueryBuilder& QueryBuilder::withArg(const char* value) {
    return withArg(Value::fromText(std::string(value)));
}

QueryBuilder& QueryBuilder::withArg(bool value) {
    return withArg(Value::fromBool(value));
}

QueryBuilder& QueryBuilder::withVariable(const std::string& varName) {
    // Ensure variable name starts with uppercase
    std::string name = varName;
    if (!name.empty() && std::islower(name[0])) {
        name[0] = std::toupper(name[0]);
    }
    return withArg(Value::fromText(name));
}

QueryBuilder& QueryBuilder::withId(uint64_t id) {
    id_ = id;
    return *this;
}

QueryBuilder& QueryBuilder::maxResults(uint32_t max) {
    maxResults_ = max;
    return *this;
}

QueryBuilder& QueryBuilder::timeout(uint32_t timeoutMs) {
    timeoutMs_ = timeoutMs;
    return *this;
}

QueryBuilder& QueryBuilder::withMetadata(const std::string& key, const Value& value) {
    metadata_[key] = value;
    return *this;
}

Query QueryBuilder::build() {
    if (!goalSet_ || goalPredicate_.empty()) {
        throw std::runtime_error("Query must have a goal");
    }
    
    if (id_ == 0) {
        id_ = nextId();
    }
    
    Rule::Condition goal;
    goal.predicate = goalPredicate_;
    goal.args = goalArgs_;
    goal.negated = goalNegated_;
    
    return Query(id_, type_, goal, maxResults_, timeoutMs_);
}

QueryBuilder QueryBuilder::findAll() {
    return QueryBuilder(Query::Type::FindAll);
}

QueryBuilder QueryBuilder::prove() {
    return QueryBuilder(Query::Type::Prove);
}

QueryBuilder QueryBuilder::findFirst(uint32_t limit) {
    return QueryBuilder(Query::Type::FindFirst).maxResults(limit);
}

QueryBuilder QueryBuilder::explain() {
    return QueryBuilder(Query::Type::Explain);
}

uint64_t QueryBuilder::nextId() {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1);
}

} // namespace inference_lab::common