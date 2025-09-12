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

#include "logic_types.hpp"

#include <atomic>
#include <cassert>
#include <sstream>

namespace inference_lab::engines::neuro_symbolic {

//=============================================================================
// Static ID generation for terms
//=============================================================================

std::atomic<SymbolId> Term::next_id_{1};

//=============================================================================
// Error handling
//=============================================================================

auto to_string(LogicError error) -> std::string {
    switch (error) {
        case LogicError::INVALID_TERM:
            return "Invalid term structure or type";
        case LogicError::UNIFICATION_FAILED:
            return "Unable to unify terms - incompatible structures";
        case LogicError::VARIABLE_NOT_BOUND:
            return "Attempted to use unbound variable";
        case LogicError::INVALID_ARITY:
            return "Function/predicate called with wrong number of arguments";
        case LogicError::INVALID_FORMULA:
            return "Malformed logical formula";
        case LogicError::PARSING_ERROR:
            return "Failed to parse logical expression";
        case LogicError::TYPE_MISMATCH:
            return "Type mismatch in logical operation";
        case LogicError::CIRCULAR_DEPENDENCY:
            return "Circular dependency detected in logical structure";
        default:
            return "Unknown logic error";
    }
}

auto to_string(TruthValue truth) -> std::string {
    switch (truth) {
        case TruthValue::FALSE_VAL:
            return "FALSE";
        case TruthValue::UNKNOWN:
            return "UNKNOWN";
        case TruthValue::TRUE_VAL:
            return "TRUE";
        default:
            return "INVALID_TRUTH_VALUE";
    }
}

auto to_string(LogicOperator op) -> std::string {
    switch (op) {
        case LogicOperator::NOT:
            return "¬";
        case LogicOperator::AND:
            return "∧";
        case LogicOperator::OR:
            return "∨";
        case LogicOperator::IMPLIES:
            return "→";
        case LogicOperator::IFF:
            return "↔";
        case LogicOperator::FORALL:
            return "∀";
        case LogicOperator::EXISTS:
            return "∃";
        default:
            return "INVALID_OPERATOR";
    }
}

//=============================================================================
// Term base class
//=============================================================================

Term::Term(TermType type, std::string name)
    : type_(type), name_(std::move(name)), id_(next_id_.fetch_add(1, std::memory_order_relaxed)) {}

auto Term::to_string() const -> std::string {
    return name_.empty() ? std::to_string(id_) : name_;
}

auto Term::equals(const Term& other) const -> bool {
    return type_ == other.type_ && name_ == other.name_;
}

auto Term::collect_variables() const -> std::unordered_set<SymbolId> {
    // Base implementation returns empty set (for constants)
    return {};
}

//=============================================================================
// Variable implementation
//=============================================================================

Variable::Variable(std::string name) : Term(TermType::VARIABLE, std::move(name)) {}

auto Variable::clone() const -> std::unique_ptr<Term> {
    auto var = std::make_unique<Variable>(get_name());
    if (is_bound()) {
        // Note: In cloning, we assume no cycles exist in the original
        // If binding fails during cloning, there's a logic error in the original
        bool binding_successful = var->bind(bound_term_->clone());
        (void)binding_successful;  // Suppress unused variable warning
        assert(binding_successful &&
               "Cycle detected during variable cloning - original had invalid state");
    }
    return var;
}

auto Variable::to_string() const -> std::string {
    if (is_bound()) {
        auto name_str = get_name().empty() ? std::to_string(get_id()) : get_name();
        return name_str + "=" + bound_term_->to_string();
    }
    // Use base class behavior for empty names
    return Term::to_string();
}

auto Variable::collect_variables() const -> std::unordered_set<SymbolId> {
    return {get_id()};
}

auto Variable::bind(std::unique_ptr<Term> term) -> bool {
    // Cycle detection: check if the term we're binding to contains this variable
    if (contains_variable(*term, get_id())) {
        return false;  // Cycle detected, binding rejected
    }

    bound_term_ = std::move(term);
    return true;  // Binding successful
}

auto Variable::contains_variable(const Term& term, SymbolId var_id) const -> bool {
    // Check if this term is the variable we're looking for
    if (term.get_id() == var_id) {
        return true;
    }

    // Recursively check compound terms
    if (term.get_type() == TermType::COMPOUND) {
        const auto* compound = dynamic_cast<const CompoundTerm*>(&term);
        if (compound) {
            for (const auto& arg : compound->get_arguments()) {
                if (contains_variable(*arg, var_id)) {
                    return true;
                }
            }
        }
    }

    // Check bound variables (follow the binding chain)
    if (term.get_type() == TermType::VARIABLE) {
        const auto* var = dynamic_cast<const Variable*>(&term);
        if (var && var->is_bound()) {
            return contains_variable(*var->get_binding(), var_id);
        }
    }

    return false;
}

void Variable::unbind() {
    bound_term_.reset();
}

//=============================================================================
// Constant implementation
//=============================================================================

Constant::Constant(std::string name) : Term(TermType::CONSTANT, std::move(name)) {}

auto Constant::clone() const -> std::unique_ptr<Term> {
    return std::make_unique<Constant>(get_name());
}

auto Constant::to_string() const -> std::string {
    // Use base class behavior for empty names
    return Term::to_string();
}

auto Constant::collect_variables() const -> std::unordered_set<SymbolId> {
    return {};  // Constants contain no variables
}

//=============================================================================
// CompoundTerm implementation
//=============================================================================

CompoundTerm::CompoundTerm(std::string functor, std::vector<std::unique_ptr<Term>> args)
    : Term(TermType::COMPOUND, std::move(functor)), arguments_(std::move(args)) {}

auto CompoundTerm::clone() const -> std::unique_ptr<Term> {
    std::vector<std::unique_ptr<Term>> cloned_args;
    cloned_args.reserve(arguments_.size());

    for (const auto& arg : arguments_) {
        cloned_args.push_back(arg->clone());
    }

    return std::make_unique<CompoundTerm>(get_name(), std::move(cloned_args));
}

auto CompoundTerm::to_string() const -> std::string {
    if (arguments_.empty()) {
        return get_name();
    }

    std::ostringstream oss;
    oss << get_name() << "(";

    for (std::size_t i = 0; i < arguments_.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << arguments_[i]->to_string();
    }

    oss << ")";
    return oss.str();
}

auto CompoundTerm::collect_variables() const -> std::unordered_set<SymbolId> {
    std::unordered_set<SymbolId> variables;
    for (const auto& arg : arguments_) {
        auto arg_vars = arg->collect_variables();
        variables.insert(arg_vars.begin(), arg_vars.end());
    }
    return variables;
}

//=============================================================================
// Predicate implementation
//=============================================================================

Predicate::Predicate(std::string name, std::vector<std::unique_ptr<Term>> args)
    : name_(std::move(name)), arguments_(std::move(args)) {}

auto Predicate::to_string() const -> std::string {
    if (arguments_.empty()) {
        return name_;
    }

    std::ostringstream oss;
    oss << name_ << "(";

    for (std::size_t i = 0; i < arguments_.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << arguments_[i]->to_string();
    }

    oss << ")";
    return oss.str();
}

auto Predicate::clone() const -> std::unique_ptr<Predicate> {
    std::vector<std::unique_ptr<Term>> cloned_args;
    cloned_args.reserve(arguments_.size());

    for (const auto& arg : arguments_) {
        cloned_args.push_back(arg->clone());
    }

    return std::make_unique<Predicate>(name_, std::move(cloned_args));
}

auto Predicate::collect_variables() const -> std::unordered_set<SymbolId> {
    std::unordered_set<SymbolId> variables;
    for (const auto& arg : arguments_) {
        auto arg_vars = arg->collect_variables();
        variables.insert(arg_vars.begin(), arg_vars.end());
    }
    return variables;
}

}  // namespace inference_lab::engines::neuro_symbolic
