// MIT License
// Copyright (c) 2025 dbjwhs

#include "symbolic_logic.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace inference_lab::engines::neuro_symbolic {

//=============================================================================
// LogicFormula implementation
//=============================================================================

LogicFormula::LogicFormula(std::unique_ptr<Predicate> predicate)
    : type_(Type::ATOMIC), operator_(LogicOperator::AND), predicate_(std::move(predicate)) {}

LogicFormula::LogicFormula(LogicOperator op, std::vector<std::unique_ptr<LogicFormula>> operands)
    : type_(Type::COMPOUND), operator_(op), operands_(std::move(operands)) {}

LogicFormula::LogicFormula(LogicOperator quantifier,
                           std::unique_ptr<Variable> var,
                           std::unique_ptr<LogicFormula> body)
    : type_(Type::QUANTIFIED), operator_(quantifier), quantified_var_(std::move(var)) {
    operands_.push_back(std::move(body));
}

auto LogicFormula::to_string() const -> std::string {
    switch (type_) {
        case Type::ATOMIC:
            return predicate_->to_string();

        case Type::COMPOUND: {
            if (operator_ == LogicOperator::NOT && operands_.size() == 1) {
                return "¬" + operands_[0]->to_string();
            }

            if (operands_.size() == 2) {
                return "(" + operands_[0]->to_string() + " " +
                       neuro_symbolic::to_string(operator_) + " " + operands_[1]->to_string() + ")";
            }

            // Multiple operands
            std::ostringstream oss;
            oss << "(";
            for (std::size_t i = 0; i < operands_.size(); ++i) {
                if (i > 0)
                    oss << " " << neuro_symbolic::to_string(operator_) << " ";
                oss << operands_[i]->to_string();
            }
            oss << ")";
            return oss.str();
        }

        case Type::QUANTIFIED:
            return neuro_symbolic::to_string(operator_) + quantified_var_->to_string() + " (" +
                   operands_[0]->to_string() + ")";
    }

    return "INVALID_FORMULA";
}

auto LogicFormula::clone() const -> std::unique_ptr<LogicFormula> {
    switch (type_) {
        case Type::ATOMIC:
            return std::make_unique<LogicFormula>(predicate_->clone());

        case Type::COMPOUND: {
            std::vector<std::unique_ptr<LogicFormula>> cloned_operands;
            cloned_operands.reserve(operands_.size());
            for (const auto& operand : operands_) {
                cloned_operands.push_back(operand->clone());
            }
            return std::make_unique<LogicFormula>(operator_, std::move(cloned_operands));
        }

        case Type::QUANTIFIED: {
            auto cloned_term = quantified_var_->clone();
            auto* var_ptr = dynamic_cast<Variable*>(cloned_term.get());
            if (!var_ptr) {
                // This should never happen if the type system is correct
                return nullptr;
            }

            return std::make_unique<LogicFormula>(
                operator_,
                std::unique_ptr<Variable>(static_cast<Variable*>(cloned_term.release())),
                operands_[0]->clone());
        }
    }

    return nullptr;
}

auto LogicFormula::collect_variables() const -> std::unordered_set<SymbolId> {
    std::unordered_set<SymbolId> variables;

    switch (type_) {
        case Type::ATOMIC: {
            for (const auto& arg : predicate_->get_arguments()) {
                if (arg->get_type() == TermType::VARIABLE) {
                    variables.insert(arg->get_id());
                }
            }
            break;
        }

        case Type::COMPOUND: {
            for (const auto& operand : operands_) {
                auto operand_vars = operand->collect_variables();
                variables.insert(operand_vars.begin(), operand_vars.end());
            }
            break;
        }

        case Type::QUANTIFIED: {
            variables.insert(quantified_var_->get_id());
            if (!operands_.empty()) {
                auto body_vars = operands_[0]->collect_variables();
                variables.insert(body_vars.begin(), body_vars.end());
            }
            break;
        }
    }

    return variables;
}

auto LogicFormula::apply_substitution(const Substitution& subst) const
    -> std::unique_ptr<LogicFormula> {
    switch (type_) {
        case Type::ATOMIC: {
            // Apply substitution to predicate arguments
            std::vector<std::unique_ptr<Term>> new_args;
            new_args.reserve(predicate_->get_arguments().size());

            for (const auto& arg : predicate_->get_arguments()) {
                new_args.push_back(Unifier::apply_substitution(*arg, subst));
            }

            auto new_predicate =
                std::make_unique<Predicate>(predicate_->get_name(), std::move(new_args));
            return std::make_unique<LogicFormula>(std::move(new_predicate));
        }

        case Type::COMPOUND: {
            std::vector<std::unique_ptr<LogicFormula>> new_operands;
            new_operands.reserve(operands_.size());

            for (const auto& operand : operands_) {
                new_operands.push_back(operand->apply_substitution(subst));
            }

            return std::make_unique<LogicFormula>(operator_, std::move(new_operands));
        }

        case Type::QUANTIFIED: {
            // For quantified formulas, we need to be careful about variable capture
            // Use comprehensive capture detection
            if (would_capture_variables(subst)) {
                return clone();  // Avoid variable capture
            }

            // Also check if we're directly substituting the quantified variable
            auto quantified_id = quantified_var_->get_id();
            if (subst.find(quantified_id) != subst.end()) {
                return clone();  // Avoid substituting bound variable
            }

            auto cloned_term = quantified_var_->clone();
            auto* var_ptr = dynamic_cast<Variable*>(cloned_term.get());
            if (!var_ptr) {
                // Type safety violation - return original formula
                return clone();
            }

            cloned_term.release();  // Transfer ownership to var_ptr
            return std::make_unique<LogicFormula>(operator_,
                                                  std::unique_ptr<Variable>(var_ptr),
                                                  operands_[0]->apply_substitution(subst));
        }
    }

    return clone();
}

auto LogicFormula::collect_free_variables() const -> std::unordered_set<SymbolId> {
    std::unordered_set<SymbolId> bound_vars;
    return collect_free_variables_impl(bound_vars);
}

auto LogicFormula::collect_free_variables_impl(std::unordered_set<SymbolId>& bound_vars) const
    -> std::unordered_set<SymbolId> {
    switch (type_) {
        case Type::ATOMIC: {
            // Collect all variables in the predicate, excluding bound ones
            std::unordered_set<SymbolId> free_vars;
            auto all_vars = predicate_->collect_variables();
            for (auto var_id : all_vars) {
                if (bound_vars.find(var_id) == bound_vars.end()) {
                    free_vars.insert(var_id);
                }
            }
            return free_vars;
        }

        case Type::COMPOUND: {
            std::unordered_set<SymbolId> free_vars;
            for (const auto& operand : operands_) {
                auto operand_free_vars = operand->collect_free_variables_impl(bound_vars);
                free_vars.insert(operand_free_vars.begin(), operand_free_vars.end());
            }
            return free_vars;
        }

        case Type::QUANTIFIED: {
            // Add quantified variable to bound variables
            bound_vars.insert(quantified_var_->get_id());

            // Collect free variables from body (excluding newly bound variable)
            std::unordered_set<SymbolId> free_vars;
            if (!operands_.empty()) {
                free_vars = operands_[0]->collect_free_variables_impl(bound_vars);
            }

            // Remove the quantified variable from bound set for sibling formulas
            bound_vars.erase(quantified_var_->get_id());

            return free_vars;
        }
    }

    return {};
}

auto LogicFormula::would_capture_variables(const Substitution& subst) const -> bool {
    if (type_ != Type::QUANTIFIED) {
        return false;  // Only quantified formulas can capture variables
    }

    // Check if any substitution term contains free variables that would be captured
    // by our quantified variable
    auto quantified_id = quantified_var_->get_id();

    for (const auto& [var_id, term] : subst) {
        // If we're substituting for the quantified variable itself, that's fine
        if (var_id == quantified_id) {
            continue;
        }

        // Check if the substitution term contains the quantified variable
        auto term_vars = term->collect_variables();
        if (term_vars.find(quantified_id) != term_vars.end()) {
            return true;  // Variable capture detected
        }
    }

    return false;
}

//=============================================================================
// Unifier implementation
//=============================================================================

auto Unifier::unify(const Term& term1, const Term& term2) -> UnificationResult {
    Substitution subst;
    if (unify_terms_recursive(term1, term2, subst)) {
        return UnificationResult(std::move(subst));
    }
    return UnificationResult();
}

auto Unifier::unify_predicates(const Predicate& pred1, const Predicate& pred2)
    -> UnificationResult {
    // Predicates must have same name and arity
    if (pred1.get_name() != pred2.get_name() || pred1.get_arity() != pred2.get_arity()) {
        return UnificationResult();
    }

    Substitution subst;
    const auto& args1 = pred1.get_arguments();
    const auto& args2 = pred2.get_arguments();

    for (std::size_t i = 0; i < args1.size(); ++i) {
        if (!unify_terms_recursive(*args1[i], *args2[i], subst)) {
            return UnificationResult();
        }
    }

    return UnificationResult(std::move(subst));
}

auto Unifier::apply_substitution(const Term& term, const Substitution& subst)
    -> std::unique_ptr<Term> {
    if (term.get_type() == TermType::VARIABLE) {
        auto it = subst.find(term.get_id());
        if (it != subst.end()) {
            return it->second->clone();
        }
    } else if (term.get_type() == TermType::COMPOUND) {
        const auto* compound = static_cast<const CompoundTerm*>(&term);
        std::vector<std::unique_ptr<Term>> new_args;
        new_args.reserve(compound->get_arguments().size());

        for (const auto& arg : compound->get_arguments()) {
            new_args.push_back(apply_substitution(*arg, subst));
        }

        return std::make_unique<CompoundTerm>(compound->get_functor(), std::move(new_args));
    }

    return term.clone();
}

auto Unifier::compose_substitutions(const Substitution& subst1, const Substitution& subst2)
    -> Substitution {
    Substitution result = subst1;

    // Apply subst2 to the range of subst1
    auto temp_result = apply_substitution_to_substitution(result, subst2);
    result = std::move(temp_result);

    // Add mappings from subst2 that are not in subst1
    for (const auto& [var_id, term] : subst2) {
        if (result.find(var_id) == result.end()) {
            result[var_id] = term;
        }
    }

    return result;
}

auto Unifier::unify_terms_recursive(const Term& term1, const Term& term2, Substitution& subst)
    -> bool {
    // Same term
    if (&term1 == &term2) {
        return true;
    }

    // Variable unification
    if (term1.get_type() == TermType::VARIABLE) {
        auto var_id = term1.get_id();
        if (occurs_check(var_id, term2)) {
            return false;  // Occurs check failed
        }
        subst[var_id] = std::shared_ptr<Term>(term2.clone().release());
        return true;
    }

    if (term2.get_type() == TermType::VARIABLE) {
        auto var_id = term2.get_id();
        if (occurs_check(var_id, term1)) {
            return false;  // Occurs check failed
        }
        subst[var_id] = std::shared_ptr<Term>(term1.clone().release());
        return true;
    }

    // Constant unification
    if (term1.get_type() == TermType::CONSTANT && term2.get_type() == TermType::CONSTANT) {
        return term1.get_name() == term2.get_name();
    }

    // Compound term unification
    if (term1.get_type() == TermType::COMPOUND && term2.get_type() == TermType::COMPOUND) {
        const auto* compound1 = static_cast<const CompoundTerm*>(&term1);
        const auto* compound2 = static_cast<const CompoundTerm*>(&term2);

        if (compound1->get_functor() != compound2->get_functor() ||
            compound1->get_arity() != compound2->get_arity()) {
            return false;
        }

        const auto& args1 = compound1->get_arguments();
        const auto& args2 = compound2->get_arguments();

        for (std::size_t i = 0; i < args1.size(); ++i) {
            if (!unify_terms_recursive(*args1[i], *args2[i], subst)) {
                return false;
            }
        }

        return true;
    }

    return false;  // Different types that can't unify
}

auto Unifier::occurs_check(SymbolId var_id, const Term& term) -> bool {
    if (term.get_type() == TermType::VARIABLE) {
        return term.get_id() == var_id;
    } else if (term.get_type() == TermType::COMPOUND) {
        const auto* compound = static_cast<const CompoundTerm*>(&term);
        for (const auto& arg : compound->get_arguments()) {
            if (occurs_check(var_id, *arg)) {
                return true;
            }
        }
    }
    return false;
}

auto Unifier::apply_substitution_to_substitution(const Substitution& target,
                                                 const Substitution& subst) -> Substitution {
    Substitution result;
    for (const auto& [var_id, term] : target) {
        result[var_id] = apply_substitution(*term, subst);
    }
    return result;
}

//=============================================================================
// InferenceRules implementation
//=============================================================================

auto InferenceRules::modus_ponens(const LogicFormula& premise, const LogicFormula& implication)
    -> Result<std::unique_ptr<LogicFormula>, LogicError> {
    if (!is_implication(implication)) {
        return Err(LogicError::INVALID_FORMULA);
    }

    auto [antecedent, consequent] = extract_implication_parts(implication);
    if (!antecedent || !consequent) {
        return Err(LogicError::INVALID_FORMULA);
    }

    if (formulas_match(premise, *antecedent)) {
        return Ok(consequent->clone());
    }

    return Err(LogicError::UNIFICATION_FAILED);
}

auto InferenceRules::modus_tollens(const LogicFormula& negated_conclusion,
                                   const LogicFormula& implication)
    -> Result<std::unique_ptr<LogicFormula>, LogicError> {
    if (!is_implication(implication)) {
        return Err(LogicError::INVALID_FORMULA);
    }

    auto [antecedent, consequent] = extract_implication_parts(implication);
    if (!antecedent || !consequent) {
        return Err(LogicError::INVALID_FORMULA);
    }

    // Check if negated_conclusion is ¬consequent
    if (negated_conclusion.get_type() == LogicFormula::Type::COMPOUND &&
        negated_conclusion.get_operator() == LogicOperator::NOT &&
        negated_conclusion.get_operands().size() == 1 &&
        formulas_match(*negated_conclusion.get_operands()[0], *consequent)) {
        // Return ¬antecedent
        std::vector<std::unique_ptr<LogicFormula>> neg_operands;
        neg_operands.push_back(antecedent->clone());
        return Ok(std::make_unique<LogicFormula>(LogicOperator::NOT, std::move(neg_operands)));
    }

    return Err(LogicError::UNIFICATION_FAILED);
}

auto InferenceRules::universal_instantiation(const LogicFormula& universal_formula,
                                             const Term& constant)
    -> Result<std::unique_ptr<LogicFormula>, LogicError> {
    if (universal_formula.get_type() != LogicFormula::Type::QUANTIFIED ||
        universal_formula.get_operator() != LogicOperator::FORALL) {
        return Err(LogicError::INVALID_FORMULA);
    }

    const auto* quantified_var = universal_formula.get_quantified_variable();
    const auto* body = universal_formula.get_body();

    if (!quantified_var || !body) {
        return Err(LogicError::INVALID_FORMULA);
    }

    // Create substitution: variable -> constant
    Substitution subst;
    subst[quantified_var->get_id()] = constant.clone();

    return Ok(body->apply_substitution(subst));
}

auto InferenceRules::existential_generalization(const LogicFormula& formula,
                                                const Term& constant,
                                                const Variable& variable)
    -> Result<std::unique_ptr<LogicFormula>, LogicError> {
    // Create substitution: constant -> variable
    Substitution subst;
    // This is a reverse substitution for generalization
    // We need to find occurrences of the constant and replace with the variable

    // Safely clone and cast the variable
    auto cloned_term = variable.clone();
    auto* var_ptr = dynamic_cast<Variable*>(cloned_term.get());
    if (!var_ptr) {
        return Err(LogicError::TYPE_MISMATCH);
    }

    // For now, return a simple existential quantification
    return Ok(std::make_unique<LogicFormula>(
        LogicOperator::EXISTS,
        std::unique_ptr<Variable>(static_cast<Variable*>(cloned_term.release())),
        formula.clone()));
}

auto InferenceRules::is_implication(const LogicFormula& formula) -> bool {
    return formula.get_type() == LogicFormula::Type::COMPOUND &&
           formula.get_operator() == LogicOperator::IMPLIES && formula.get_operands().size() == 2;
}

auto InferenceRules::extract_implication_parts(const LogicFormula& implication)
    -> std::pair<const LogicFormula*, const LogicFormula*> {
    if (!is_implication(implication)) {
        return {nullptr, nullptr};
    }

    const auto& operands = implication.get_operands();
    return {operands[0].get(), operands[1].get()};
}

auto InferenceRules::formulas_match(const LogicFormula& formula1, const LogicFormula& formula2)
    -> bool {
    // Simple structural matching - could be enhanced with unification
    return formula1.to_string() == formula2.to_string();
}

//=============================================================================
// KnowledgeBase implementation
//=============================================================================

void KnowledgeBase::add_formula(std::unique_ptr<LogicFormula> formula) {
    formulas_.push_back(std::move(formula));
}

auto KnowledgeBase::query(const LogicFormula& query) const
    -> std::vector<std::pair<const LogicFormula*, UnificationResult>> {
    std::vector<std::pair<const LogicFormula*, UnificationResult>> results;

    for (const auto& formula : formulas_) {
        if (formula->get_type() == LogicFormula::Type::ATOMIC &&
            query.get_type() == LogicFormula::Type::ATOMIC) {
            const auto* formula_pred = formula->get_predicate();
            const auto* query_pred = query.get_predicate();

            if (formula_pred && query_pred) {
                auto unification_result = Unifier::unify_predicates(*formula_pred, *query_pred);
                if (unification_result.success) {
                    results.emplace_back(formula.get(), std::move(unification_result));
                }
            }
        }
    }

    return results;
}

//=============================================================================
// LogicReasoner implementation
//=============================================================================

LogicReasoner::LogicReasoner(std::shared_ptr<KnowledgeBase> kb) : knowledge_base_(std::move(kb)) {}

auto LogicReasoner::answer_query(const LogicFormula& query) const -> std::vector<Substitution> {
    std::vector<Substitution> answers;

    auto matches = knowledge_base_->query(query);
    for (const auto& [formula, unification_result] : matches) {
        answers.push_back(std::move(unification_result.substitution));
    }

    return answers;
}

auto LogicReasoner::forward_chain_step() -> std::size_t {
    // Simple forward chaining - could be greatly enhanced
    std::size_t new_facts = 0;
    auto initial_size = knowledge_base_->size();

    // Look for implications and try to apply modus ponens
    const auto& formulas = knowledge_base_->get_formulas();
    for (std::size_t i = 0; i < formulas.size(); ++i) {
        for (std::size_t j = 0; j < formulas.size(); ++j) {
            if (i != j) {
                auto result = InferenceRules::modus_ponens(*formulas[i], *formulas[j]);
                if (result.is_ok()) {
                    knowledge_base_->add_formula(std::move(result).unwrap());
                    new_facts++;
                }
            }
        }
    }

    return new_facts;
}

auto LogicReasoner::forward_chain(std::size_t max_iterations) -> std::size_t {
    std::size_t total_new_facts = 0;

    for (std::size_t i = 0; i < max_iterations; ++i) {
        auto new_facts = forward_chain_step();
        if (new_facts == 0) {
            break;  // Fixpoint reached
        }
        total_new_facts += new_facts;
    }

    return total_new_facts;
}

}  // namespace inference_lab::engines::neuro_symbolic
