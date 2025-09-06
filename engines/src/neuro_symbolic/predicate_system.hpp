// MIT License
// Copyright (c) 2025 dbjwhs

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <string>

#include "logic_types.hpp"
#include "symbolic_logic.hpp"
#include "../../common/src/result.hpp"

namespace inference_lab::engines::neuro_symbolic {

using inference_lab::common::Result;
using inference_lab::common::Ok;
using inference_lab::common::Err;

/**
 * @brief Advanced predicate system for first-order logic
 * 
 * This module extends the basic symbolic logic with sophisticated predicate
 * operations, including predicate definitions, type checking, and advanced
 * reasoning capabilities.
 */

/**
 * @brief Predicate signature defining name, arity, and argument types
 */
struct PredicateSignature {
    std::string name;
    std::size_t arity;
    std::vector<std::string> argument_types;  // Optional type information
    
    PredicateSignature(std::string pred_name, std::size_t pred_arity)
        : name(std::move(pred_name)), arity(pred_arity) {}
    
    PredicateSignature(std::string pred_name, std::vector<std::string> types)
        : name(std::move(pred_name)), arity(types.size()), argument_types(std::move(types)) {}
    
    [[nodiscard]] auto to_string() const -> std::string;
    [[nodiscard]] auto equals(const PredicateSignature& other) const -> bool;
};

/**
 * @brief Rule in first-order logic: Head :- Body1, Body2, ..., BodyN
 * 
 * Represents logical rules of the form:
 * mortal(X) :- human(X)
 * grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
 */
class LogicRule {
public:
    LogicRule(std::unique_ptr<LogicFormula> head, std::vector<std::unique_ptr<LogicFormula>> body);

    // Non-copyable but moveable
    LogicRule(const LogicRule&) = delete;
    auto operator=(const LogicRule&) -> LogicRule& = delete;
    LogicRule(LogicRule&&) = default;
    auto operator=(LogicRule&&) -> LogicRule& = default;

    [[nodiscard]] auto get_head() const -> const LogicFormula& { return *head_; }
    [[nodiscard]] auto get_body() const -> const std::vector<std::unique_ptr<LogicFormula>>& { return body_; }
    [[nodiscard]] auto get_body_size() const -> std::size_t { return body_.size(); }
    [[nodiscard]] auto is_fact() const -> bool { return body_.empty(); }
    
    [[nodiscard]] auto to_string() const -> std::string;
    [[nodiscard]] auto clone() const -> std::unique_ptr<LogicRule>;
    
    // Get all variables in the rule
    [[nodiscard]] auto collect_variables() const -> std::unordered_set<SymbolId>;
    
    // Apply substitution to the entire rule
    [[nodiscard]] auto apply_substitution(const Substitution& subst) const -> std::unique_ptr<LogicRule>;

private:
    std::unique_ptr<LogicFormula> head_;
    std::vector<std::unique_ptr<LogicFormula>> body_;
};

/**
 * @brief Registry for predicate signatures and type checking
 */
class PredicateRegistry {
public:
    PredicateRegistry() = default;

    /**
     * @brief Register a predicate signature
     */
    void register_predicate(const PredicateSignature& signature);
    
    /**
     * @brief Check if a predicate is registered
     */
    [[nodiscard]] auto is_registered(const std::string& name, std::size_t arity) const -> bool;
    
    /**
     * @brief Get predicate signature
     */
    [[nodiscard]] auto get_signature(const std::string& name, std::size_t arity) const 
        -> Result<PredicateSignature, LogicError>;
    
    /**
     * @brief Validate a predicate against its signature
     */
    [[nodiscard]] auto validate_predicate(const Predicate& predicate) const 
        -> Result<bool, LogicError>;
    
    /**
     * @brief Get all registered predicates
     */
    [[nodiscard]] auto get_all_signatures() const -> const std::vector<PredicateSignature>& {
        return signatures_;
    }
    
    /**
     * @brief Clear all registered predicates
     */
    void clear() { signatures_.clear(); }

private:
    std::vector<PredicateSignature> signatures_;
    
    auto find_signature(const std::string& name, std::size_t arity) const 
        -> std::vector<PredicateSignature>::const_iterator;
};

/**
 * @brief Advanced knowledge base with rules and sophisticated querying
 */
class RuleBasedKnowledgeBase {
public:
    RuleBasedKnowledgeBase() = default;

    // Non-copyable but moveable
    RuleBasedKnowledgeBase(const RuleBasedKnowledgeBase&) = delete;
    auto operator=(const RuleBasedKnowledgeBase&) -> RuleBasedKnowledgeBase& = delete;
    RuleBasedKnowledgeBase(RuleBasedKnowledgeBase&&) = default;
    auto operator=(RuleBasedKnowledgeBase&&) -> RuleBasedKnowledgeBase& = default;

    /**
     * @brief Add a fact to the knowledge base
     */
    void add_fact(std::unique_ptr<LogicFormula> fact);
    
    /**
     * @brief Add a rule to the knowledge base
     */
    void add_rule(std::unique_ptr<LogicRule> rule);
    
    /**
     * @brief Query for facts matching a pattern
     */
    [[nodiscard]] auto query_facts(const LogicFormula& query) const 
        -> std::vector<std::pair<const LogicFormula*, Substitution>>;
    
    /**
     * @brief Get all rules with head matching a pattern
     */
    [[nodiscard]] auto get_matching_rules(const LogicFormula& head_pattern) const 
        -> std::vector<std::pair<const LogicRule*, Substitution>>;
    
    /**
     * @brief Get all facts
     */
    [[nodiscard]] auto get_facts() const -> const std::vector<std::unique_ptr<LogicFormula>>& {
        return facts_;
    }
    
    /**
     * @brief Get all rules
     */
    [[nodiscard]] auto get_rules() const -> const std::vector<std::unique_ptr<LogicRule>>& {
        return rules_;
    }
    
    /**
     * @brief Get total number of facts and rules
     */
    [[nodiscard]] auto size() const -> std::size_t { return facts_.size() + rules_.size(); }
    
    /**
     * @brief Check if knowledge base is empty
     */
    [[nodiscard]] auto empty() const -> bool { return facts_.empty() && rules_.empty(); }
    
    /**
     * @brief Clear all facts and rules
     */
    void clear() { facts_.clear(); rules_.clear(); }

private:
    std::vector<std::unique_ptr<LogicFormula>> facts_;
    std::vector<std::unique_ptr<LogicRule>> rules_;
};

/**
 * @brief Advanced logical reasoner with SLD resolution
 * 
 * Implements SLD (Selective Linear Definite) resolution for logic programming,
 * which is the foundation of Prolog-style reasoning.
 */
class SLDReasoner {
public:
    explicit SLDReasoner(std::shared_ptr<RuleBasedKnowledgeBase> kb);

    /**
     * @brief Configuration for reasoning process
     */
    struct Config {
        std::size_t max_depth;        // Maximum recursion depth
        std::size_t max_solutions;    // Maximum number of solutions to find
        bool use_occurs_check;        // Enable occurs check in unification
        bool trace_execution;         // Enable execution tracing
        
        // Constructor with default values
        Config() : max_depth(100), max_solutions(10), use_occurs_check(true), trace_execution(false) {}
        
        Config(std::size_t depth, std::size_t solutions, bool occurs_check, bool trace)
            : max_depth(depth), max_solutions(solutions), use_occurs_check(occurs_check), trace_execution(trace) {}
    };

    /**
     * @brief Query result with solution and trace information
     */
    struct QueryResult {
        std::vector<Substitution> solutions;
        std::vector<std::string> trace;
        bool timeout = false;
        std::size_t depth_reached = 0;
    };

    /**
     * @brief Answer a query using SLD resolution
     */
    [[nodiscard]] auto query(const LogicFormula& goal, const Config& config = Config()) const -> QueryResult;
    
    /**
     * @brief Answer a query with multiple goals
     */
    [[nodiscard]] auto query_goals(const std::vector<std::unique_ptr<LogicFormula>>& goals, 
                                  const Config& config = Config()) const -> QueryResult;
    
    /**
     * @brief Check if a goal is provable (has at least one solution)
     */
    [[nodiscard]] auto is_provable(const LogicFormula& goal, const Config& config = Config()) const -> bool;

private:
    std::shared_ptr<RuleBasedKnowledgeBase> knowledge_base_;
    
    // Internal SLD resolution implementation
    struct SLDState {
        std::vector<std::unique_ptr<LogicFormula>> goals;
        Substitution substitution;
        std::size_t depth;
        std::vector<std::string> trace;
    };
    
    void sld_resolve(const SLDState& state, 
                    const Config& config,
                    QueryResult& result) const;
    
    auto resolve_goal(const LogicFormula& goal, 
                     const LogicRule& rule,
                     const Substitution& current_subst) const -> std::vector<SLDState>;
    
    auto apply_substitution_to_goals(const std::vector<std::unique_ptr<LogicFormula>>& goals,
                                    const Substitution& subst) const 
                                    -> std::vector<std::unique_ptr<LogicFormula>>;
};

/**
 * @brief Utility functions for predicate manipulation
 */
namespace predicates {

/**
 * @brief Create a simple predicate from name and argument strings
 */
auto make_predicate(const std::string& name, const std::vector<std::string>& args) 
    -> std::unique_ptr<Predicate>;

/**
 * @brief Create a variable term
 */
auto make_variable(const std::string& name) -> std::unique_ptr<Variable>;

/**
 * @brief Create a constant term
 */
auto make_constant(const std::string& name) -> std::unique_ptr<Constant>;

/**
 * @brief Create a compound term
 */
auto make_compound(const std::string& functor, const std::vector<std::string>& args) 
    -> std::unique_ptr<CompoundTerm>;

/**
 * @brief Create an atomic formula from a predicate
 */
auto make_atomic_formula(std::unique_ptr<Predicate> predicate) -> std::unique_ptr<LogicFormula>;

/**
 * @brief Create an implication formula: antecedent → consequent
 */
auto make_implication(std::unique_ptr<LogicFormula> antecedent, std::unique_ptr<LogicFormula> consequent)
    -> std::unique_ptr<LogicFormula>;

/**
 * @brief Create a conjunction formula: formula1 ∧ formula2
 */
auto make_conjunction(std::vector<std::unique_ptr<LogicFormula>> formulas)
    -> std::unique_ptr<LogicFormula>;

/**
 * @brief Create a universal quantification: ∀var formula
 */
auto make_universal(std::unique_ptr<Variable> var, std::unique_ptr<LogicFormula> formula)
    -> std::unique_ptr<LogicFormula>;

/**
 * @brief Parse a simple predicate string like "likes(john, mary)"
 */
auto parse_predicate(const std::string& predicate_str) -> Result<std::unique_ptr<Predicate>, LogicError>;

/**
 * @brief Parse a simple rule string like "mortal(X) :- human(X)"
 */
auto parse_rule(const std::string& rule_str) -> Result<std::unique_ptr<LogicRule>, LogicError>;

} // namespace predicates

} // namespace inference_lab::engines::neuro_symbolic
