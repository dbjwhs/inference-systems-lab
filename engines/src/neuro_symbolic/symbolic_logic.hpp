// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#pragma once

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "../../common/src/result.hpp"
#include "logic_types.hpp"

namespace inference_lab::engines::neuro_symbolic {

using inference_lab::common::Err;
using inference_lab::common::Ok;
using inference_lab::common::Result;

/**
 * @brief Core symbolic logic operations and reasoning
 *
 * This module provides the fundamental operations for symbolic logic programming,
 * including unification, substitution, and basic inference rules.
 */

/**
 * @brief Logical formula representation
 *
 * Represents compound logical formulas with operators like AND, OR, NOT, IMPLIES, etc.
 * Supports both atomic predicates and complex nested formulas.
 */
class LogicFormula {
  public:
    /**
     * @brief Formula types for different logical constructs
     */
    enum class Type : std::uint8_t {
        ATOMIC = 0,     // Simple predicate: likes(john, mary)
        COMPOUND = 1,   // Compound formula with operator: P ∧ Q
        QUANTIFIED = 2  // Quantified formula: ∀X likes(X, mary)
    };

    // Atomic formula constructor
    explicit LogicFormula(std::unique_ptr<Predicate> predicate);

    // Compound formula constructor
    LogicFormula(LogicOperator op, std::vector<std::unique_ptr<LogicFormula>> operands);

    // Quantified formula constructor
    LogicFormula(LogicOperator quantifier,
                 std::unique_ptr<Variable> var,
                 std::unique_ptr<LogicFormula> body);

    // Non-copyable but moveable
    LogicFormula(const LogicFormula&) = delete;
    auto operator=(const LogicFormula&) -> LogicFormula& = delete;
    LogicFormula(LogicFormula&&) = default;
    auto operator=(LogicFormula&&) -> LogicFormula& = default;

    [[nodiscard]] auto get_type() const -> Type { return type_; }
    [[nodiscard]] auto to_string() const -> std::string;
    [[nodiscard]] auto clone() const -> std::unique_ptr<LogicFormula>;

    // Accessors for different formula types
    [[nodiscard]] auto get_predicate() const -> const Predicate* {
        return type_ == Type::ATOMIC ? predicate_.get() : nullptr;
    }

    [[nodiscard]] auto get_operator() const -> LogicOperator { return operator_; }
    [[nodiscard]] auto get_operands() const -> const std::vector<std::unique_ptr<LogicFormula>>& {
        return operands_;
    }

    [[nodiscard]] auto get_quantified_variable() const -> const Variable* {
        return type_ == Type::QUANTIFIED ? quantified_var_.get() : nullptr;
    }

    [[nodiscard]] auto get_body() const -> const LogicFormula* {
        return type_ == Type::QUANTIFIED && !operands_.empty() ? operands_[0].get() : nullptr;
    }

    // Collect all variables in the formula
    [[nodiscard]] auto collect_variables() const -> std::unordered_set<SymbolId>;

    // Collect free variables (not bound by quantifiers)
    [[nodiscard]] auto collect_free_variables() const -> std::unordered_set<SymbolId>;

    // Apply substitution to the formula
    [[nodiscard]] auto apply_substitution(const Substitution& subst) const
        -> std::unique_ptr<LogicFormula>;

  private:
    Type type_;
    LogicOperator operator_;
    std::unique_ptr<Predicate> predicate_;                 // For atomic formulas
    std::vector<std::unique_ptr<LogicFormula>> operands_;  // For compound formulas
    std::unique_ptr<Variable> quantified_var_;             // For quantified formulas

    // Helper methods for variable capture detection
    [[nodiscard]] auto collect_free_variables_impl(std::unordered_set<SymbolId>& bound_vars) const
        -> std::unordered_set<SymbolId>;
    [[nodiscard]] auto would_capture_variables(const Substitution& subst) const -> bool;
};

/**
 * @brief Unification algorithm for terms
 *
 * Implements the unification algorithm that finds substitutions to make terms identical.
 * This is the core operation for logical reasoning and pattern matching.
 */
class Unifier {
  public:
    /**
     * @brief Unify two terms
     * @param term1 First term to unify
     * @param term2 Second term to unify
     * @return UnificationResult with substitutions if successful
     */
    static auto unify(const Term& term1, const Term& term2) -> UnificationResult;

    /**
     * @brief Unify two predicates
     * @param pred1 First predicate to unify
     * @param pred2 Second predicate to unify
     * @return UnificationResult with substitutions if successful
     */
    static auto unify_predicates(const Predicate& pred1, const Predicate& pred2)
        -> UnificationResult;

    /**
     * @brief Apply substitution to a term
     * @param term Term to apply substitution to
     * @param subst Substitution mapping
     * @return New term with substitution applied
     */
    static auto apply_substitution(const Term& term, const Substitution& subst)
        -> std::unique_ptr<Term>;

    /**
     * @brief Compose two substitutions
     * @param subst1 First substitution
     * @param subst2 Second substitution
     * @return Composed substitution
     */
    static auto compose_substitutions(const Substitution& subst1, const Substitution& subst2)
        -> Substitution;

  private:
    // Helper methods for unification algorithm
    static auto unify_terms_recursive(const Term& term1, const Term& term2, Substitution& subst)
        -> bool;
    static auto occurs_check(SymbolId var_id, const Term& term) -> bool;
    static auto apply_substitution_to_substitution(const Substitution& target,
                                                   const Substitution& subst) -> Substitution;
};

/**
 * @brief Basic inference rules for symbolic logic
 *
 * Implements fundamental inference rules like modus ponens, modus tollens,
 * universal instantiation, etc.
 */
class InferenceRules {
  public:
    /**
     * @brief Modus Ponens: From P and P→Q, infer Q
     * @param premise The premise P
     * @param implication The implication P→Q
     * @return Result containing the conclusion Q if successful
     */
    static auto modus_ponens(const LogicFormula& premise, const LogicFormula& implication)
        -> Result<std::unique_ptr<LogicFormula>, LogicError>;

    /**
     * @brief Modus Tollens: From ¬Q and P→Q, infer ¬P
     * @param negated_conclusion The negated conclusion ¬Q
     * @param implication The implication P→Q
     * @return Result containing the negated premise ¬P if successful
     */
    static auto modus_tollens(const LogicFormula& negated_conclusion,
                              const LogicFormula& implication)
        -> Result<std::unique_ptr<LogicFormula>, LogicError>;

    /**
     * @brief Universal Instantiation: From ∀X P(X), infer P(c) for constant c
     * @param universal_formula The universally quantified formula ∀X P(X)
     * @param constant The constant to instantiate with
     * @return Result containing the instantiated formula P(c)
     */
    static auto universal_instantiation(const LogicFormula& universal_formula, const Term& constant)
        -> Result<std::unique_ptr<LogicFormula>, LogicError>;

    /**
     * @brief Existential Generalization: From P(c), infer ∃X P(X)
     * @param formula The formula P(c) with constant c
     * @param constant The constant to generalize
     * @param variable The variable to introduce
     * @return Result containing the existentially quantified formula ∃X P(X)
     */
    static auto existential_generalization(const LogicFormula& formula,
                                           const Term& constant,
                                           const Variable& variable)
        -> Result<std::unique_ptr<LogicFormula>, LogicError>;

  private:
    // Helper methods for inference rules
    static auto is_implication(const LogicFormula& formula) -> bool;
    static auto extract_implication_parts(const LogicFormula& implication)
        -> std::pair<const LogicFormula*, const LogicFormula*>;
    static auto formulas_match(const LogicFormula& formula1, const LogicFormula& formula2) -> bool;
};

/**
 * @brief Knowledge base for storing logical facts and rules
 *
 * Maintains a collection of logical formulas representing known facts and rules.
 * Supports querying and basic inference operations.
 */
class KnowledgeBase {
  public:
    KnowledgeBase() = default;

    // Non-copyable but moveable
    KnowledgeBase(const KnowledgeBase&) = delete;
    auto operator=(const KnowledgeBase&) -> KnowledgeBase& = delete;
    KnowledgeBase(KnowledgeBase&&) = default;
    auto operator=(KnowledgeBase&&) -> KnowledgeBase& = default;

    /**
     * @brief Add a fact or rule to the knowledge base
     * @param formula The logical formula to add
     */
    void add_formula(std::unique_ptr<LogicFormula> formula);

    /**
     * @brief Query the knowledge base for matching facts
     * @param query The query formula to match against
     * @return Vector of matching formulas with their substitutions
     */
    [[nodiscard]] auto query(const LogicFormula& query) const
        -> std::vector<std::pair<const LogicFormula*, UnificationResult>>;

    /**
     * @brief Get all formulas in the knowledge base
     */
    [[nodiscard]] auto get_formulas() const -> const std::vector<std::unique_ptr<LogicFormula>>& {
        return formulas_;
    }

    /**
     * @brief Get number of formulas in the knowledge base
     */
    [[nodiscard]] auto size() const -> std::size_t { return formulas_.size(); }

    /**
     * @brief Check if knowledge base is empty
     */
    [[nodiscard]] auto empty() const -> bool { return formulas_.empty(); }

    /**
     * @brief Clear all formulas from the knowledge base
     */
    void clear() { formulas_.clear(); }

  private:
    std::vector<std::unique_ptr<LogicFormula>> formulas_;
};

/**
 * @brief Simple logical reasoner
 *
 * Provides basic reasoning capabilities over a knowledge base using
 * fundamental inference rules.
 */
class LogicReasoner {
  public:
    explicit LogicReasoner(std::shared_ptr<KnowledgeBase> kb);

    /**
     * @brief Answer a query using the knowledge base
     * @param query The query to answer
     * @return Vector of answer substitutions
     */
    [[nodiscard]] auto answer_query(const LogicFormula& query) const -> std::vector<Substitution>;

    /**
     * @brief Perform one step of forward chaining
     * @return Number of new facts derived
     */
    auto forward_chain_step() -> std::size_t;

    /**
     * @brief Perform forward chaining until fixpoint
     * @param max_iterations Maximum number of iterations
     * @return Total number of new facts derived
     */
    auto forward_chain(std::size_t max_iterations = 100) -> std::size_t;

  private:
    std::shared_ptr<KnowledgeBase> knowledge_base_;
};

}  // namespace inference_lab::engines::neuro_symbolic
