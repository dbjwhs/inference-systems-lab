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

#include "predicate_system.hpp"

#include <algorithm>
#include <functional>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace inference_lab::engines::neuro_symbolic {

//=============================================================================
// PredicateSignature implementation
//=============================================================================

auto PredicateSignature::to_string() const -> std::string {
    std::ostringstream oss;
    oss << name << "/" << arity;
    if (!argument_types.empty()) {
        oss << " : (";
        for (std::size_t i = 0; i < argument_types.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << argument_types[i];
        }
        oss << ")";
    }
    return oss.str();
}

auto PredicateSignature::equals(const PredicateSignature& other) const -> bool {
    return name == other.name && arity == other.arity;
}

//=============================================================================
// LogicRule implementation
//=============================================================================

LogicRule::LogicRule(std::unique_ptr<LogicFormula> head,
                     std::vector<std::unique_ptr<LogicFormula>> body)
    : head_(std::move(head)), body_(std::move(body)) {}

auto LogicRule::to_string() const -> std::string {
    std::ostringstream oss;
    oss << head_->to_string();

    if (!body_.empty()) {
        oss << " :- ";
        for (std::size_t i = 0; i < body_.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << body_[i]->to_string();
        }
    }

    return oss.str();
}

auto LogicRule::clone() const -> std::unique_ptr<LogicRule> {
    std::vector<std::unique_ptr<LogicFormula>> cloned_body;
    cloned_body.reserve(body_.size());

    for (const auto& formula : body_) {
        cloned_body.push_back(formula->clone());
    }

    return std::make_unique<LogicRule>(head_->clone(), std::move(cloned_body));
}

auto LogicRule::collect_variables() const -> std::unordered_set<SymbolId> {
    // Use name-based deduplication for rule variables
    std::unordered_map<std::string, SymbolId> unique_vars_by_name;

    std::function<void(const LogicFormula&)> collect_vars_by_name;
    collect_vars_by_name = [&](const LogicFormula& formula) {
        switch (formula.get_type()) {
            case LogicFormula::Type::ATOMIC: {
                if (const auto* pred = formula.get_predicate()) {
                    for (const auto& arg : pred->get_arguments()) {
                        if (arg->get_type() == TermType::VARIABLE) {
                            const std::string& var_name = arg->get_name();
                            if (unique_vars_by_name.find(var_name) == unique_vars_by_name.end()) {
                                unique_vars_by_name[var_name] = arg->get_id();
                            }
                        }
                    }
                }
                break;
            }
            case LogicFormula::Type::COMPOUND: {
                for (const auto& operand : formula.get_operands()) {
                    collect_vars_by_name(*operand);
                }
                break;
            }
            case LogicFormula::Type::QUANTIFIED: {
                const std::string& var_name = formula.get_quantified_variable()->get_name();
                if (unique_vars_by_name.find(var_name) == unique_vars_by_name.end()) {
                    unique_vars_by_name[var_name] = formula.get_quantified_variable()->get_id();
                }
                if (!formula.get_operands().empty()) {
                    collect_vars_by_name(*formula.get_operands()[0]);
                }
                break;
            }
        }
    };

    // Collect from head
    collect_vars_by_name(*head_);

    // Collect from body
    for (const auto& formula : body_) {
        collect_vars_by_name(*formula);
    }

    // Convert to set of IDs (keeping one ID per unique name)
    std::unordered_set<SymbolId> variables;
    for (const auto& [name, id] : unique_vars_by_name) {
        variables.insert(id);
    }

    return variables;
}

auto LogicRule::apply_substitution(const Substitution& subst) const -> std::unique_ptr<LogicRule> {
    std::vector<std::unique_ptr<LogicFormula>> new_body;
    new_body.reserve(body_.size());

    for (const auto& formula : body_) {
        new_body.push_back(formula->apply_substitution(subst));
    }

    return std::make_unique<LogicRule>(head_->apply_substitution(subst), std::move(new_body));
}

//=============================================================================
// PredicateRegistry implementation
//=============================================================================

void PredicateRegistry::register_predicate(const PredicateSignature& signature) {
    // Check if already registered
    if (!is_registered(signature.name, signature.arity)) {
        signatures_.push_back(signature);
    }
}

auto PredicateRegistry::is_registered(const std::string& name, std::size_t arity) const -> bool {
    return find_signature(name, arity) != signatures_.end();
}

auto PredicateRegistry::get_signature(const std::string& name, std::size_t arity) const
    -> Result<PredicateSignature, LogicError> {
    auto it = find_signature(name, arity);
    if (it != signatures_.end()) {
        return Ok(*it);
    }
    return Err(LogicError::INVALID_TERM);
}

auto PredicateRegistry::validate_predicate(const Predicate& predicate) const
    -> Result<bool, LogicError> {
    if (!is_registered(predicate.get_name(), predicate.get_arity())) {
        return Err(LogicError::INVALID_TERM);
    }

    // For now, just check arity - type checking would go here
    return Ok(true);
}

auto PredicateRegistry::find_signature(const std::string& name, std::size_t arity) const
    -> std::vector<PredicateSignature>::const_iterator {
    return std::find_if(
        signatures_.begin(), signatures_.end(), [&name, arity](const PredicateSignature& sig) {
            return sig.name == name && sig.arity == arity;
        });
}

//=============================================================================
// RuleBasedKnowledgeBase implementation
//=============================================================================

void RuleBasedKnowledgeBase::add_fact(std::unique_ptr<LogicFormula> fact) {
    facts_.push_back(std::move(fact));
}

void RuleBasedKnowledgeBase::add_rule(std::unique_ptr<LogicRule> rule) {
    rules_.push_back(std::move(rule));
}

auto RuleBasedKnowledgeBase::query_facts(const LogicFormula& query) const
    -> std::vector<std::pair<const LogicFormula*, Substitution>> {
    std::vector<std::pair<const LogicFormula*, Substitution>> results;

    for (const auto& fact : facts_) {
        if (fact->get_type() == LogicFormula::Type::ATOMIC &&
            query.get_type() == LogicFormula::Type::ATOMIC) {
            const auto* fact_pred = fact->get_predicate();
            const auto* query_pred = query.get_predicate();

            if (fact_pred && query_pred) {
                auto unification_result = Unifier::unify_predicates(*fact_pred, *query_pred);
                if (unification_result.success) {
                    results.emplace_back(fact.get(), std::move(unification_result.substitution));
                }
            }
        }
    }

    return results;
}

auto RuleBasedKnowledgeBase::get_matching_rules(const LogicFormula& head_pattern) const
    -> std::vector<std::pair<const LogicRule*, Substitution>> {
    std::vector<std::pair<const LogicRule*, Substitution>> results;

    for (const auto& rule : rules_) {
        const auto& rule_head = rule->get_head();

        if (rule_head.get_type() == LogicFormula::Type::ATOMIC &&
            head_pattern.get_type() == LogicFormula::Type::ATOMIC) {
            const auto* rule_pred = rule_head.get_predicate();
            const auto* pattern_pred = head_pattern.get_predicate();

            if (rule_pred && pattern_pred) {
                auto unification_result = Unifier::unify_predicates(*rule_pred, *pattern_pred);
                if (unification_result.success) {
                    results.emplace_back(rule.get(), std::move(unification_result.substitution));
                }
            }
        }
    }

    return results;
}

//=============================================================================
// SLDReasoner implementation
//=============================================================================

SLDReasoner::SLDReasoner(std::shared_ptr<RuleBasedKnowledgeBase> kb)
    : knowledge_base_(std::move(kb)) {}

auto SLDReasoner::query(const LogicFormula& goal, const Config& config) const -> QueryResult {
    std::vector<std::unique_ptr<LogicFormula>> goals;
    goals.push_back(goal.clone());
    return query_goals(goals, config);
}

auto SLDReasoner::query_goals(const std::vector<std::unique_ptr<LogicFormula>>& goals,
                              const Config& config) const -> QueryResult {
    QueryResult result;

    SLDState initial_state;
    for (const auto& goal : goals) {
        initial_state.goals.push_back(goal->clone());
    }
    initial_state.depth = 0;

    sld_resolve(initial_state, config, result);

    return result;
}

auto SLDReasoner::is_provable(const LogicFormula& goal, const Config& config) const -> bool {
    auto limited_config = config;
    limited_config.max_solutions = 1;

    auto result = query(goal, limited_config);
    return !result.solutions.empty();
}

void SLDReasoner::sld_resolve(const SLDState& state,
                              const Config& config,
                              QueryResult& result) const {
    // Check termination conditions
    if (state.depth > config.max_depth) {
        result.timeout = true;
        return;
    }

    if (result.solutions.size() >= config.max_solutions) {
        return;
    }

    result.depth_reached = std::max(result.depth_reached, state.depth);

    // If no goals left, we have a solution
    if (state.goals.empty()) {
        result.solutions.push_back(state.substitution);
        if (config.trace_execution) {
            result.trace.insert(result.trace.end(), state.trace.begin(), state.trace.end());
            result.trace.push_back("Solution found: " + std::to_string(result.solutions.size()));
        }
        return;
    }

    // Select first goal to resolve
    const auto& current_goal = *state.goals[0];

    if (config.trace_execution) {
        auto trace_entry =
            "Depth " + std::to_string(state.depth) + ": Resolving " + current_goal.to_string();
        const_cast<SLDState&>(state).trace.push_back(trace_entry);
    }

    // Try to match against facts
    auto fact_matches = knowledge_base_->query_facts(current_goal);
    for (const auto& [fact, fact_subst] : fact_matches) {
        SLDState new_state;
        new_state.depth = state.depth + 1;
        new_state.trace = state.trace;

        // Compose substitutions
        new_state.substitution = Unifier::compose_substitutions(state.substitution, fact_subst);

        // Remove resolved goal and apply substitution to remaining goals
        for (std::size_t i = 1; i < state.goals.size(); ++i) {
            new_state.goals.push_back(state.goals[i]->apply_substitution(fact_subst));
        }

        if (config.trace_execution) {
            new_state.trace.push_back("Matched fact: " + fact->to_string());
        }

        sld_resolve(new_state, config, result);
    }

    // Try to match against rules
    auto rule_matches = knowledge_base_->get_matching_rules(current_goal);
    for (const auto& [rule, rule_subst] : rule_matches) {
        SLDState new_state;
        new_state.depth = state.depth + 1;
        new_state.trace = state.trace;

        // Compose substitutions
        new_state.substitution = Unifier::compose_substitutions(state.substitution, rule_subst);

        // Replace current goal with rule body and apply substitution to all goals
        const auto& rule_body = rule->get_body();
        for (const auto& body_goal : rule_body) {
            new_state.goals.push_back(body_goal->apply_substitution(rule_subst));
        }

        for (std::size_t i = 1; i < state.goals.size(); ++i) {
            new_state.goals.push_back(state.goals[i]->apply_substitution(rule_subst));
        }

        if (config.trace_execution) {
            new_state.trace.push_back("Applied rule: " + rule->to_string());
        }

        sld_resolve(new_state, config, result);
    }
}

auto SLDReasoner::apply_substitution_to_goals(
    const std::vector<std::unique_ptr<LogicFormula>>& goals, const Substitution& subst) const
    -> std::vector<std::unique_ptr<LogicFormula>> {
    std::vector<std::unique_ptr<LogicFormula>> new_goals;
    new_goals.reserve(goals.size());

    for (const auto& goal : goals) {
        new_goals.push_back(goal->apply_substitution(subst));
    }

    return new_goals;
}

//=============================================================================
// Utility functions in predicates namespace
//=============================================================================

namespace predicates {

auto make_predicate(const std::string& name, const std::vector<std::string>& args)
    -> std::unique_ptr<Predicate> {
    std::vector<std::unique_ptr<Term>> terms;
    terms.reserve(args.size());

    for (const auto& arg : args) {
        // Simple heuristic: uppercase names are variables, lowercase are constants
        if (!arg.empty() && std::isupper(arg[0])) {
            terms.push_back(std::make_unique<Variable>(arg));
        } else {
            terms.push_back(std::make_unique<Constant>(arg));
        }
    }

    return std::make_unique<Predicate>(name, std::move(terms));
}

auto make_variable(const std::string& name) -> std::unique_ptr<Variable> {
    return std::make_unique<Variable>(name);
}

auto make_constant(const std::string& name) -> std::unique_ptr<Constant> {
    return std::make_unique<Constant>(name);
}

auto make_compound(const std::string& functor, const std::vector<std::string>& args)
    -> std::unique_ptr<CompoundTerm> {
    std::vector<std::unique_ptr<Term>> terms;
    terms.reserve(args.size());

    for (const auto& arg : args) {
        if (!arg.empty() && std::isupper(arg[0])) {
            terms.push_back(std::make_unique<Variable>(arg));
        } else {
            terms.push_back(std::make_unique<Constant>(arg));
        }
    }

    return std::make_unique<CompoundTerm>(functor, std::move(terms));
}

auto make_atomic_formula(std::unique_ptr<Predicate> predicate) -> std::unique_ptr<LogicFormula> {
    return std::make_unique<LogicFormula>(std::move(predicate));
}

auto make_implication(std::unique_ptr<LogicFormula> antecedent,
                      std::unique_ptr<LogicFormula> consequent) -> std::unique_ptr<LogicFormula> {
    std::vector<std::unique_ptr<LogicFormula>> operands;
    operands.push_back(std::move(antecedent));
    operands.push_back(std::move(consequent));

    return std::make_unique<LogicFormula>(LogicOperator::IMPLIES, std::move(operands));
}

auto make_conjunction(std::vector<std::unique_ptr<LogicFormula>> formulas)
    -> std::unique_ptr<LogicFormula> {
    if (formulas.size() == 1) {
        return std::move(formulas[0]);
    }

    return std::make_unique<LogicFormula>(LogicOperator::AND, std::move(formulas));
}

auto make_universal(std::unique_ptr<Variable> var, std::unique_ptr<LogicFormula> formula)
    -> std::unique_ptr<LogicFormula> {
    return std::make_unique<LogicFormula>(
        LogicOperator::FORALL, std::move(var), std::move(formula));
}

auto parse_predicate(const std::string& predicate_str)
    -> Result<std::unique_ptr<Predicate>, LogicError> {
    // Simple regex-based parser for predicates like "likes(john, mary)"
    std::regex predicate_regex(R"((\w+)\(([^)]*)\))");
    std::smatch matches;

    if (!std::regex_match(predicate_str, matches, predicate_regex)) {
        // Try parsing as a simple predicate without arguments
        std::regex simple_predicate_regex(R"(^\w+$)");
        if (std::regex_match(predicate_str, simple_predicate_regex)) {
            return Ok(
                std::make_unique<Predicate>(predicate_str, std::vector<std::unique_ptr<Term>>{}));
        }
        return Err(LogicError::PARSING_ERROR);
    }

    std::string name = matches[1].str();
    std::string args_str = matches[2].str();

    std::vector<std::unique_ptr<Term>> arguments;

    if (!args_str.empty()) {
        // Split arguments by comma
        std::istringstream args_stream(args_str);
        std::string arg;

        while (std::getline(args_stream, arg, ',')) {
            // Trim whitespace
            arg.erase(0, arg.find_first_not_of(" \t"));
            arg.erase(arg.find_last_not_of(" \t") + 1);

            if (!arg.empty()) {
                // Variables start with uppercase, constants with lowercase
                if (std::isupper(arg[0])) {
                    arguments.push_back(std::make_unique<Variable>(arg));
                } else {
                    arguments.push_back(std::make_unique<Constant>(arg));
                }
            }
        }
    }

    return Ok(std::make_unique<Predicate>(name, std::move(arguments)));
}

auto parse_rule(const std::string& rule_str) -> Result<std::unique_ptr<LogicRule>, LogicError> {
    // Parse rules like "mortal(X) :- human(X)" or facts like "human(socrates)"

    auto rule_separator_pos = rule_str.find(":-");

    if (rule_separator_pos == std::string::npos) {
        // This is a fact, not a rule
        auto head_result = parse_predicate(rule_str);
        if (head_result.is_err()) {
            return Err(head_result.unwrap_err());
        }

        auto head_formula = make_atomic_formula(std::move(head_result).unwrap());
        return Ok(std::make_unique<LogicRule>(std::move(head_formula),
                                              std::vector<std::unique_ptr<LogicFormula>>{}));
    }

    // Parse head and body
    std::string head_str = rule_str.substr(0, rule_separator_pos);
    std::string body_str = rule_str.substr(rule_separator_pos + 2);

    // Trim whitespace
    head_str.erase(0, head_str.find_first_not_of(" \t"));
    head_str.erase(head_str.find_last_not_of(" \t") + 1);
    body_str.erase(0, body_str.find_first_not_of(" \t"));
    body_str.erase(body_str.find_last_not_of(" \t") + 1);

    // Parse head
    auto head_result = parse_predicate(head_str);
    if (head_result.is_err()) {
        return Err(head_result.unwrap_err());
    }

    auto head_formula = make_atomic_formula(std::move(head_result).unwrap());

    // Parse body (comma-separated predicates)
    std::vector<std::unique_ptr<LogicFormula>> body_formulas;
    std::istringstream body_stream(body_str);
    std::string body_predicate;

    while (std::getline(body_stream, body_predicate, ',')) {
        // Trim whitespace
        body_predicate.erase(0, body_predicate.find_first_not_of(" \t"));
        body_predicate.erase(body_predicate.find_last_not_of(" \t") + 1);

        if (!body_predicate.empty()) {
            auto pred_result = parse_predicate(body_predicate);
            if (pred_result.is_err()) {
                return Err(pred_result.unwrap_err());
            }

            body_formulas.push_back(make_atomic_formula(std::move(pred_result).unwrap()));
        }
    }

    return Ok(std::make_unique<LogicRule>(std::move(head_formula), std::move(body_formulas)));
}

}  // namespace predicates

}  // namespace inference_lab::engines::neuro_symbolic
