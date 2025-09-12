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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "../../common/src/result.hpp"

namespace inference_lab::engines::neuro_symbolic {

using inference_lab::common::Err;
using inference_lab::common::Ok;
using inference_lab::common::Result;

/**
 * @brief Core type definitions for symbolic logic programming
 *
 * This module defines the fundamental types and structures needed for
 * symbolic reasoning, including terms, variables, constants, and logic operations.
 */

// Forward declarations
class Term;
class Variable;
class Constant;
class Predicate;
class LogicFormula;

/**
 * @brief Unique identifier for variables and constants
 */
using SymbolId = std::uint64_t;

/**
 * @brief Error types for symbolic logic operations
 */
enum class LogicError : std::uint8_t {
    INVALID_TERM = 0,
    UNIFICATION_FAILED = 1,
    VARIABLE_NOT_BOUND = 2,
    INVALID_ARITY = 3,
    INVALID_FORMULA = 4,
    PARSING_ERROR = 5,
    TYPE_MISMATCH = 6,
    CIRCULAR_DEPENDENCY = 7
};

/**
 * @brief Convert logic error to string
 */
auto to_string(LogicError error) -> std::string;

/**
 * @brief Basic term types in first-order logic
 */
enum class TermType : std::uint8_t { VARIABLE = 0, CONSTANT = 1, FUNCTION = 2, COMPOUND = 3 };

/**
 * @brief Truth values for symbolic logic
 *
 * Supports three-valued logic: TRUE, FALSE, UNKNOWN
 * This will later be extended to fuzzy logic with continuous values
 */
enum class TruthValue : std::uint8_t { FALSE_VAL = 0, UNKNOWN = 1, TRUE_VAL = 2 };

/**
 * @brief Convert truth value to string
 */
auto to_string(TruthValue truth) -> std::string;

/**
 * @brief Logical operators for compound formulas
 */
enum class LogicOperator : std::uint8_t {
    NOT = 0,      // ¬
    AND = 1,      // ∧
    OR = 2,       // ∨
    IMPLIES = 3,  // →
    IFF = 4,      // ↔
    FORALL = 5,   // ∀
    EXISTS = 6    // ∃
};

/**
 * @brief Convert logic operator to string
 */
auto to_string(LogicOperator op) -> std::string;

/**
 * @brief Base class for all terms in first-order logic
 *
 * Terms can be variables (x, y, z), constants (john, 5, "hello"),
 * or compound terms (f(x, y), g(a, b, c))
 */
class Term {
  public:
    explicit Term(TermType type, std::string name = "");
    virtual ~Term() = default;

    // Non-copyable but moveable
    Term(const Term&) = delete;
    auto operator=(const Term&) -> Term& = delete;
    Term(Term&&) = default;
    auto operator=(Term&&) -> Term& = default;

    [[nodiscard]] auto get_type() const -> TermType { return type_; }
    [[nodiscard]] auto get_name() const -> const std::string& { return name_; }
    [[nodiscard]] auto get_id() const -> SymbolId { return id_; }

    // Virtual interface for term operations
    [[nodiscard]] virtual auto to_string() const -> std::string;
    [[nodiscard]] virtual auto clone() const -> std::unique_ptr<Term> = 0;
    [[nodiscard]] virtual auto equals(const Term& other) const -> bool;
    [[nodiscard]] virtual auto get_arity() const -> std::size_t { return 0; }

    // Collect all variables in this term
    [[nodiscard]] virtual auto collect_variables() const -> std::unordered_set<SymbolId>;

  private:
    TermType type_;
    std::string name_;
    SymbolId id_;
    static std::atomic<SymbolId> next_id_;
};

/**
 * @brief Variable term (e.g., X, Y, person, etc.)
 *
 * Variables can be bound to other terms during unification
 */
class Variable : public Term {
  public:
    explicit Variable(std::string name);

    [[nodiscard]] auto clone() const -> std::unique_ptr<Term> override;
    [[nodiscard]] auto to_string() const -> std::string override;
    [[nodiscard]] auto collect_variables() const -> std::unordered_set<SymbolId> override;

    // Variable binding
    [[nodiscard]] auto bind(std::unique_ptr<Term> term) -> bool;  // Returns false if cycle detected
    void unbind();
    [[nodiscard]] auto is_bound() const -> bool { return bound_term_ != nullptr; }
    [[nodiscard]] auto get_binding() const -> const Term* { return bound_term_.get(); }

  private:
    std::unique_ptr<Term> bound_term_;

    // Helper method for cycle detection
    [[nodiscard]] auto contains_variable(const Term& term, SymbolId var_id) const -> bool;
};

/**
 * @brief Constant term (e.g., john, 42, "hello")
 */
class Constant : public Term {
  public:
    explicit Constant(std::string name);

    [[nodiscard]] auto clone() const -> std::unique_ptr<Term> override;
    [[nodiscard]] auto to_string() const -> std::string override;
    [[nodiscard]] auto collect_variables() const -> std::unordered_set<SymbolId> override;
};

/**
 * @brief Compound term with function symbol and arguments (e.g., f(x,y), likes(john, mary))
 */
class CompoundTerm : public Term {
  public:
    CompoundTerm(std::string functor, std::vector<std::unique_ptr<Term>> args);

    [[nodiscard]] auto clone() const -> std::unique_ptr<Term> override;
    [[nodiscard]] auto to_string() const -> std::string override;
    [[nodiscard]] auto get_arity() const -> std::size_t override { return arguments_.size(); }
    [[nodiscard]] auto collect_variables() const -> std::unordered_set<SymbolId> override;

    [[nodiscard]] auto get_functor() const -> const std::string& { return get_name(); }
    [[nodiscard]] auto get_arguments() const -> const std::vector<std::unique_ptr<Term>>& {
        return arguments_;
    }
    [[nodiscard]] auto get_argument(std::size_t index) const -> const Term* {
        return index < arguments_.size() ? arguments_[index].get() : nullptr;
    }

  private:
    std::vector<std::unique_ptr<Term>> arguments_;
};

/**
 * @brief Predicate with name and arguments (e.g., likes(john, mary), mortal(X))
 */
class Predicate {
  public:
    Predicate(std::string name, std::vector<std::unique_ptr<Term>> args);

    // Non-copyable but moveable
    Predicate(const Predicate&) = delete;
    auto operator=(const Predicate&) -> Predicate& = delete;
    Predicate(Predicate&&) = default;
    auto operator=(Predicate&&) -> Predicate& = default;

    [[nodiscard]] auto get_name() const -> const std::string& { return name_; }
    [[nodiscard]] auto get_arity() const -> std::size_t { return arguments_.size(); }
    [[nodiscard]] auto get_arguments() const -> const std::vector<std::unique_ptr<Term>>& {
        return arguments_;
    }
    [[nodiscard]] auto get_argument(std::size_t index) const -> const Term* {
        return index < arguments_.size() ? arguments_[index].get() : nullptr;
    }

    [[nodiscard]] auto to_string() const -> std::string;
    [[nodiscard]] auto clone() const -> std::unique_ptr<Predicate>;
    [[nodiscard]] auto collect_variables() const -> std::unordered_set<SymbolId>;

  private:
    std::string name_;
    std::vector<std::unique_ptr<Term>> arguments_;
};

/**
 * @brief Variable substitution for unification
 *
 * Maps variable IDs to their substituted terms
 */
using Substitution = std::unordered_map<SymbolId, std::shared_ptr<Term>>;

/**
 * @brief Unification result containing substitutions
 */
struct UnificationResult {
    Substitution substitution;
    bool success;

    UnificationResult() : success(false) {}
    explicit UnificationResult(Substitution subst)
        : substitution(std::move(subst)), success(true) {}
    UnificationResult(bool result, Substitution subst)
        : substitution(std::move(subst)), success(result) {}
};

}  // namespace inference_lab::engines::neuro_symbolic
