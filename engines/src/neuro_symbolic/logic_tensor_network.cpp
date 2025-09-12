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

#include "logic_tensor_network.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>

namespace inference_lab::engines::neuro_symbolic {

// ================================================================================================
// CONSTRUCTION AND CONFIGURATION
// ================================================================================================

auto LogicTensorNetwork::create(const LTNConfig& config)
    -> common::Result<std::unique_ptr<LogicTensorNetwork>, LTNError> {
    // Validate configuration
    if (config.embedding_dim == 0) {
        return common::Err<LTNError>(LTNError::INVALID_CONFIGURATION);
    }
    if (config.learning_rate <= 0.0f || config.learning_rate > 1.0f) {
        return common::Err<LTNError>(LTNError::INVALID_CONFIGURATION);
    }

    // Use make_unique with private constructor access via friendship/pimpl
    auto ltn = std::unique_ptr<LogicTensorNetwork>(new LogicTensorNetwork(config));
    return common::Ok<std::unique_ptr<LogicTensorNetwork>>(std::move(ltn));
}

LogicTensorNetwork::LogicTensorNetwork(LTNConfig config) : config_(std::move(config)) {
    // Initialize optimizer state if needed
    if (config_.optimizer_type == "adam") {
        // Adam optimizer requires momentum and velocity terms - will be initialized per parameter
    }
}

// ================================================================================================
// SYMBOL AND PREDICATE MANAGEMENT
// ================================================================================================

auto LogicTensorNetwork::add_individual(const std::string& name)
    -> common::Result<std::size_t, LTNError> {
    if (individuals_.find(name) != individuals_.end()) {
        // Individual already exists, return existing ID
        return common::Ok<std::size_t>(individuals_[name].id);
    }

    try {
        Individual individual(name, next_individual_id_++, config_.embedding_dim);
        std::size_t id = individual.id;
        individuals_[name] = std::move(individual);

        return common::Ok<std::size_t>(id);
    } catch (const std::exception&) {
        return common::Err<LTNError>(LTNError::MEMORY_ALLOCATION_ERROR);
    }
}

auto LogicTensorNetwork::add_predicate(const std::string& name, std::size_t arity)
    -> common::Result<std::size_t, LTNError> {
    if (predicates_.find(name) != predicates_.end()) {
        // Predicate already exists, return existing ID
        return common::Ok<std::size_t>(predicates_[name].id);
    }

    if (arity == 0) {
        return common::Err<LTNError>(LTNError::INVALID_CONFIGURATION);
    }

    try {
        LTNPredicate predicate(name, arity, next_predicate_id_++, config_.embedding_dim);
        std::size_t id = predicate.id;
        predicates_[name] = std::move(predicate);

        return common::Ok<std::size_t>(id);
    } catch (const std::exception&) {
        return common::Err<LTNError>(LTNError::MEMORY_ALLOCATION_ERROR);
    }
}

auto LogicTensorNetwork::get_individual(const std::string& name)
    -> common::Result<Individual*, LTNError> {
    auto it = individuals_.find(name);
    if (it == individuals_.end()) {
        return common::Err<LTNError>(LTNError::UNDEFINED_INDIVIDUAL);
    }

    return common::Ok<Individual*>(&it->second);
}

auto LogicTensorNetwork::get_individual(const std::string& name) const
    -> common::Result<const Individual*, LTNError> {
    auto it = individuals_.find(name);
    if (it == individuals_.end()) {
        return common::Err<LTNError>(LTNError::UNDEFINED_INDIVIDUAL);
    }

    return common::Ok<const Individual*>(&it->second);
}

auto LogicTensorNetwork::get_predicate(const std::string& name)
    -> common::Result<LTNPredicate*, LTNError> {
    auto it = predicates_.find(name);
    if (it == predicates_.end()) {
        return common::Err<LTNError>(LTNError::UNDEFINED_PREDICATE);
    }

    return common::Ok<LTNPredicate*>(&it->second);
}

auto LogicTensorNetwork::get_predicate(const std::string& name) const
    -> common::Result<const LTNPredicate*, LTNError> {
    auto it = predicates_.find(name);
    if (it == predicates_.end()) {
        return common::Err<LTNError>(LTNError::UNDEFINED_PREDICATE);
    }

    return common::Ok<const LTNPredicate*>(&it->second);
}

// ================================================================================================
// PREDICATE EVALUATION
// ================================================================================================

auto LogicTensorNetwork::evaluate_predicate(const std::string& predicate_name,
                                            const std::string& individual_name)
    -> common::Result<FuzzyValue, LTNError> {
    auto predicate_result = get_predicate(predicate_name);
    if (predicate_result.is_err()) {
        return common::Err<LTNError>(predicate_result.unwrap_err());
    }
    const auto* predicate = predicate_result.unwrap();

    if (predicate->arity != 1) {
        return common::Err<LTNError>(LTNError::DIMENSION_MISMATCH);
    }

    auto individual_result = get_individual(individual_name);
    if (individual_result.is_err()) {
        return common::Err<LTNError>(individual_result.unwrap_err());
    }
    const auto* individual = individual_result.unwrap();

    // Evaluate neural predicate: sigmoid(W * embedding + b)
    std::vector<std::vector<float>> embeddings = {individual->embedding};
    FuzzyValue result = evaluate_neural_predicate(*predicate, embeddings);

    return common::Ok<FuzzyValue>(result);
}

auto LogicTensorNetwork::evaluate_relation(const std::string& predicate_name,
                                           const std::string& individual1,
                                           const std::string& individual2)
    -> common::Result<FuzzyValue, LTNError> {
    auto predicate_result = get_predicate(predicate_name);
    if (predicate_result.is_err()) {
        return common::Err<LTNError>(predicate_result.unwrap_err());
    }
    const auto* predicate = predicate_result.unwrap();

    if (predicate->arity != 2) {
        return common::Err<LTNError>(LTNError::DIMENSION_MISMATCH);
    }

    auto ind1_result = get_individual(individual1);
    auto ind2_result = get_individual(individual2);
    if (ind1_result.is_err()) {
        return common::Err<LTNError>(ind1_result.unwrap_err());
    }
    if (ind2_result.is_err()) {
        return common::Err<LTNError>(ind2_result.unwrap_err());
    }

    const auto* ind1 = ind1_result.unwrap();
    const auto* ind2 = ind2_result.unwrap();

    // Evaluate binary relation: concatenate embeddings and apply neural network
    std::vector<std::vector<float>> embeddings = {ind1->embedding, ind2->embedding};
    FuzzyValue result = evaluate_neural_predicate(*predicate, embeddings);

    return common::Ok<FuzzyValue>(result);
}

auto LogicTensorNetwork::batch_evaluate_predicate(const std::string& predicate_name,
                                                  const std::vector<std::string>& individual_names)
    -> common::Result<std::vector<FuzzyValue>, LTNError> {
    std::vector<FuzzyValue> results;
    results.reserve(individual_names.size());

    for (const auto& name : individual_names) {
        auto result = evaluate_predicate(predicate_name, name);
        if (result.is_err()) {
            return common::Err<LTNError>(result.unwrap_err());
        }
        results.push_back(result.unwrap());
    }

    return common::Ok<std::vector<FuzzyValue>>(std::move(results));
}

// ================================================================================================
// NEURAL PREDICATE EVALUATION
// ================================================================================================

auto LogicTensorNetwork::evaluate_neural_predicate(
    const LTNPredicate& predicate, const std::vector<std::vector<float>>& embeddings)
    -> FuzzyValue {
    // Concatenate all input embeddings
    std::vector<float> input;
    for (const auto& embedding : embeddings) {
        input.insert(input.end(), embedding.begin(), embedding.end());
    }

    // Compute linear combination: W * x + b
    float linear_output = predicate.bias[0];
    for (std::size_t i = 0; i < input.size(); ++i) {
        linear_output += predicate.weights[i] * input[i];
    }

    // Apply sigmoid activation to get fuzzy truth value
    FuzzyValue result = sigmoid_membership(linear_output, 1.0f, 0.0f);
    return clamp_fuzzy_value(result);
}

// ================================================================================================
// FORMULA EVALUATION IMPLEMENTATIONS
// ================================================================================================

auto LogicTensorNetwork::AtomicFormula::evaluate(LogicTensorNetwork& ltn)
    -> common::Result<FuzzyValue, LTNError> {
    if (arguments_.size() == 1) {
        return ltn.evaluate_predicate(predicate_name_, arguments_[0]);
    } else if (arguments_.size() == 2) {
        return ltn.evaluate_relation(predicate_name_, arguments_[0], arguments_[1]);
    } else {
        // For higher arity, would need more sophisticated handling
        return common::Err<LTNError>(LTNError::DIMENSION_MISMATCH);
    }
}

auto LogicTensorNetwork::AtomicFormula::to_string() const -> std::string {
    std::ostringstream oss;
    oss << predicate_name_ << "(";
    for (std::size_t i = 0; i < arguments_.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << arguments_[i];
    }
    oss << ")";
    return oss.str();
}

auto LogicTensorNetwork::AtomicFormula::free_variables() const -> std::vector<std::string> {
    std::vector<std::string> vars;
    for (const auto& arg : arguments_) {
        // Simple heuristic: variables start with lowercase, constants with uppercase
        if (!arg.empty() && std::islower(arg[0])) {
            vars.push_back(arg);
        }
    }
    return vars;
}

auto LogicTensorNetwork::ConjunctionFormula::evaluate(LogicTensorNetwork& ltn)
    -> common::Result<FuzzyValue, LTNError> {
    auto left_result = left_->evaluate(ltn);
    auto right_result = right_->evaluate(ltn);

    if (left_result.is_err()) {
        return left_result;
    }
    if (right_result.is_err()) {
        return right_result;
    }

    FuzzyValue result = fuzzy_and(left_result.unwrap(), right_result.unwrap());
    return common::Ok<FuzzyValue>(result);
}

auto LogicTensorNetwork::ConjunctionFormula::to_string() const -> std::string {
    return "(" + left_->to_string() + " ∧ " + right_->to_string() + ")";
}

auto LogicTensorNetwork::ConjunctionFormula::free_variables() const -> std::vector<std::string> {
    auto left_vars = left_->free_variables();
    auto right_vars = right_->free_variables();

    // Combine and deduplicate
    std::vector<std::string> all_vars = left_vars;
    for (const auto& var : right_vars) {
        if (std::find(all_vars.begin(), all_vars.end(), var) == all_vars.end()) {
            all_vars.push_back(var);
        }
    }
    return all_vars;
}

auto LogicTensorNetwork::DisjunctionFormula::evaluate(LogicTensorNetwork& ltn)
    -> common::Result<FuzzyValue, LTNError> {
    auto left_result = left_->evaluate(ltn);
    auto right_result = right_->evaluate(ltn);

    if (left_result.is_err()) {
        return left_result;
    }
    if (right_result.is_err()) {
        return right_result;
    }

    FuzzyValue result = fuzzy_or(left_result.unwrap(), right_result.unwrap());
    return common::Ok<FuzzyValue>(result);
}

auto LogicTensorNetwork::DisjunctionFormula::to_string() const -> std::string {
    return "(" + left_->to_string() + " ∨ " + right_->to_string() + ")";
}

auto LogicTensorNetwork::DisjunctionFormula::free_variables() const -> std::vector<std::string> {
    auto left_vars = left_->free_variables();
    auto right_vars = right_->free_variables();

    std::vector<std::string> all_vars = left_vars;
    for (const auto& var : right_vars) {
        if (std::find(all_vars.begin(), all_vars.end(), var) == all_vars.end()) {
            all_vars.push_back(var);
        }
    }
    return all_vars;
}

auto LogicTensorNetwork::ImplicationFormula::evaluate(LogicTensorNetwork& ltn)
    -> common::Result<FuzzyValue, LTNError> {
    auto antecedent_result = antecedent_->evaluate(ltn);
    auto consequent_result = consequent_->evaluate(ltn);

    if (antecedent_result.is_err()) {
        return antecedent_result;
    }
    if (consequent_result.is_err()) {
        return consequent_result;
    }

    FuzzyValue result = fuzzy_implies(antecedent_result.unwrap(), consequent_result.unwrap());
    return common::Ok<FuzzyValue>(result);
}

auto LogicTensorNetwork::ImplicationFormula::to_string() const -> std::string {
    return "(" + antecedent_->to_string() + " → " + consequent_->to_string() + ")";
}

auto LogicTensorNetwork::ImplicationFormula::free_variables() const -> std::vector<std::string> {
    auto ant_vars = antecedent_->free_variables();
    auto cons_vars = consequent_->free_variables();

    std::vector<std::string> all_vars = ant_vars;
    for (const auto& var : cons_vars) {
        if (std::find(all_vars.begin(), all_vars.end(), var) == all_vars.end()) {
            all_vars.push_back(var);
        }
    }
    return all_vars;
}

auto LogicTensorNetwork::NegationFormula::evaluate(LogicTensorNetwork& ltn)
    -> common::Result<FuzzyValue, LTNError> {
    auto operand_result = operand_->evaluate(ltn);
    if (operand_result.is_err()) {
        return operand_result;
    }

    FuzzyValue result = fuzzy_not(operand_result.unwrap());
    return common::Ok<FuzzyValue>(result);
}

auto LogicTensorNetwork::NegationFormula::to_string() const -> std::string {
    return "¬" + operand_->to_string();
}

auto LogicTensorNetwork::NegationFormula::free_variables() const -> std::vector<std::string> {
    return operand_->free_variables();
}

// ================================================================================================
// FORMULA CONSTRUCTION HELPERS
// ================================================================================================

auto LogicTensorNetwork::atomic(const std::string& predicate_name,
                                const std::vector<std::string>& arguments)
    -> std::unique_ptr<Formula> {
    return std::make_unique<AtomicFormula>(predicate_name, arguments);
}

auto LogicTensorNetwork::conjunction(std::unique_ptr<Formula> left, std::unique_ptr<Formula> right)
    -> std::unique_ptr<Formula> {
    return std::make_unique<ConjunctionFormula>(std::move(left), std::move(right));
}

auto LogicTensorNetwork::disjunction(std::unique_ptr<Formula> left, std::unique_ptr<Formula> right)
    -> std::unique_ptr<Formula> {
    return std::make_unique<DisjunctionFormula>(std::move(left), std::move(right));
}

auto LogicTensorNetwork::implication(std::unique_ptr<Formula> antecedent,
                                     std::unique_ptr<Formula> consequent)
    -> std::unique_ptr<Formula> {
    return std::make_unique<ImplicationFormula>(std::move(antecedent), std::move(consequent));
}

auto LogicTensorNetwork::negation(std::unique_ptr<Formula> operand) -> std::unique_ptr<Formula> {
    return std::make_unique<NegationFormula>(std::move(operand));
}

// ================================================================================================
// TRAINING AND OPTIMIZATION
// ================================================================================================

auto LogicTensorNetwork::add_formula(const std::string& name,
                                     std::unique_ptr<Formula> formula,
                                     float weight) -> common::Result<std::monostate, LTNError> {
    if (weight <= 0.0f) {
        return common::Err<LTNError>(LTNError::INVALID_CONFIGURATION);
    }

    formulas_[name] = std::move(formula);
    formula_weights_[name] = weight;

    return common::Ok<std::monostate>(std::monostate{});
}

auto LogicTensorNetwork::train(const std::vector<Example>& examples, std::size_t epochs)
    -> common::Result<float, LTNError> {
    float final_loss = 0.0f;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;

        // Supervised loss from examples
        for (const auto& example : examples) {
            auto prediction_result =
                evaluate_predicate(example.predicate_name,
                                   example.arguments[0]);  // Simplified for unary
            if (prediction_result.is_err()) {
                return common::Err<LTNError>(prediction_result.unwrap_err());
            }

            float prediction = prediction_result.unwrap();
            float error = prediction - example.target_truth;
            epoch_loss += example.weight * error * error;  // MSE
        }

        // Knowledge base constraint loss
        for (const auto& [name, formula] : formulas_) {
            auto satisfaction_result = formula->evaluate(*this);
            if (satisfaction_result.is_err()) {
                return common::Err<LTNError>(satisfaction_result.unwrap_err());
            }

            float satisfaction = satisfaction_result.unwrap();
            float constraint_loss = formula_weights_[name] * (1.0f - satisfaction);
            epoch_loss += constraint_loss;
        }

        current_loss_ = epoch_loss;
        final_loss = epoch_loss;

        // Compute and apply gradients (simplified implementation)
        auto grad_result = compute_gradients(epoch_loss);
        if (grad_result.is_err()) {
            return common::Err<LTNError>(grad_result.unwrap_err());
        }

        auto update_result = update_parameters();
        if (update_result.is_err()) {
            return common::Err<LTNError>(update_result.unwrap_err());
        }

        training_step_++;

        // Check for convergence
        if (epoch_loss < config_.convergence_threshold) {
            break;
        }
    }

    return common::Ok<float>(final_loss);
}

auto LogicTensorNetwork::compute_gradients(float loss) -> common::Result<std::monostate, LTNError> {
    // Simplified gradient computation - in practice would use automatic differentiation
    // This is a placeholder that demonstrates the structure

    parameter_gradients_.clear();

    // For each predicate, compute gradients w.r.t. weights and biases
    for (auto& [name, predicate] : predicates_) {
        if (predicate.is_trainable) {
            std::vector<float> weight_grads(predicate.weights.size(), 0.0f);
            std::vector<float> bias_grads(predicate.bias.size(), 0.0f);

            // Placeholder: finite differences for gradient estimation
            const float eps = 1e-5f;

            // This is conceptually what would happen - actual implementation
            // would use automatic differentiation framework
            parameter_gradients_[name + "_weights"] = std::move(weight_grads);
            parameter_gradients_[name + "_bias"] = std::move(bias_grads);
        }
    }

    // For each individual, compute embedding gradients
    for (auto& [name, individual] : individuals_) {
        if (individual.is_trainable) {
            std::vector<float> embed_grads(individual.embedding.size(), 0.0f);
            parameter_gradients_[name + "_embedding"] = std::move(embed_grads);
        }
    }

    return common::Ok<std::monostate>(std::monostate{});
}

auto LogicTensorNetwork::update_parameters() -> common::Result<std::monostate, LTNError> {
    // Apply gradients using chosen optimizer

    if (config_.optimizer_type == "sgd") {
        // Simple SGD updates
        for (auto& [name, predicate] : predicates_) {
            if (predicate.is_trainable) {
                auto weight_grad_it = parameter_gradients_.find(name + "_weights");
                auto bias_grad_it = parameter_gradients_.find(name + "_bias");

                if (weight_grad_it != parameter_gradients_.end()) {
                    const auto& grads = weight_grad_it->second;
                    for (std::size_t i = 0; i < predicate.weights.size(); ++i) {
                        predicate.weights[i] -= config_.learning_rate * grads[i];
                    }
                }

                if (bias_grad_it != parameter_gradients_.end()) {
                    const auto& grads = bias_grad_it->second;
                    for (std::size_t i = 0; i < predicate.bias.size(); ++i) {
                        predicate.bias[i] -= config_.learning_rate * grads[i];
                    }
                }
            }
        }

        for (auto& [name, individual] : individuals_) {
            if (individual.is_trainable) {
                auto embed_grad_it = parameter_gradients_.find(name + "_embedding");

                if (embed_grad_it != parameter_gradients_.end()) {
                    const auto& grads = embed_grad_it->second;
                    for (std::size_t i = 0; i < individual.embedding.size(); ++i) {
                        individual.embedding[i] -= config_.learning_rate * grads[i];
                    }
                }
            }
        }
    }
    // Other optimizers (Adam, RMSprop) would be implemented similarly

    return common::Ok<std::monostate>(std::monostate{});
}

// ================================================================================================
// INFERENCE AND QUERYING
// ================================================================================================

auto LogicTensorNetwork::query(Formula& formula) -> common::Result<FuzzyValue, LTNError> {
    return formula.evaluate(*this);
}

auto LogicTensorNetwork::find_individuals(const std::string& predicate_name, float threshold)
    -> common::Result<std::vector<std::string>, LTNError> {
    std::vector<std::string> satisfying_individuals;

    for (const auto& [name, individual] : individuals_) {
        auto result = evaluate_predicate(predicate_name, name);
        if (result.is_err()) {
            return common::Err<LTNError>(result.unwrap_err());
        }

        if (result.unwrap() >= threshold) {
            satisfying_individuals.push_back(name);
        }
    }

    return common::Ok<std::vector<std::string>>(std::move(satisfying_individuals));
}

auto LogicTensorNetwork::get_embedding(const std::string& individual_name)
    -> common::Result<std::vector<float>, LTNError> {
    auto individual_result = get_individual(individual_name);
    if (individual_result.is_err()) {
        return common::Err<LTNError>(individual_result.unwrap_err());
    }

    return common::Ok<std::vector<float>>(individual_result.unwrap()->embedding);
}

// ================================================================================================
// STATISTICS AND UTILITIES
// ================================================================================================

auto LogicTensorNetwork::get_statistics() const -> Statistics {
    Statistics stats;
    stats.num_individuals = individuals_.size();
    stats.num_predicates = predicates_.size();
    stats.num_formulas = formulas_.size();

    // Calculate total parameters
    stats.total_parameters = 0.0f;
    for (const auto& [name, individual] : individuals_) {
        stats.total_parameters += static_cast<float>(individual.embedding.size());

        // Calculate L2 norm of embedding
        float norm = 0.0f;
        for (float val : individual.embedding) {
            norm += val * val;
        }
        stats.embedding_norms.push_back(std::sqrt(norm));
    }

    for (const auto& [name, predicate] : predicates_) {
        stats.total_parameters +=
            static_cast<float>(predicate.weights.size() + predicate.bias.size());

        // Calculate L2 norm of predicate weights
        float norm = 0.0f;
        for (float val : predicate.weights) {
            norm += val * val;
        }
        for (float val : predicate.bias) {
            norm += val * val;
        }
        stats.predicate_norms.push_back(std::sqrt(norm));
    }

    // Calculate average formula satisfaction (placeholder)
    stats.average_formula_satisfaction = 0.8f;  // Would compute actual satisfaction

    return stats;
}

auto LogicTensorNetwork::export_model() const
    -> std::unordered_map<std::string, std::vector<float>> {
    std::unordered_map<std::string, std::vector<float>> model_data;

    // Export individual embeddings
    for (const auto& [name, individual] : individuals_) {
        model_data["individual_" + name] = individual.embedding;
    }

    // Export predicate parameters
    for (const auto& [name, predicate] : predicates_) {
        model_data["predicate_" + name + "_weights"] = predicate.weights;
        model_data["predicate_" + name + "_bias"] = predicate.bias;
    }

    return model_data;
}

auto LogicTensorNetwork::import_model(
    const std::unordered_map<std::string, std::vector<float>>& model_data)
    -> common::Result<std::monostate, LTNError> {
    // Import individual embeddings
    for (auto& [name, individual] : individuals_) {
        auto it = model_data.find("individual_" + name);
        if (it != model_data.end()) {
            if (it->second.size() == individual.embedding.size()) {
                individual.embedding = it->second;
            } else {
                return common::Err<LTNError>(LTNError::DIMENSION_MISMATCH);
            }
        }
    }

    // Import predicate parameters
    for (auto& [name, predicate] : predicates_) {
        auto weight_it = model_data.find("predicate_" + name + "_weights");
        auto bias_it = model_data.find("predicate_" + name + "_bias");

        if (weight_it != model_data.end()) {
            if (weight_it->second.size() == predicate.weights.size()) {
                predicate.weights = weight_it->second;
            } else {
                return common::Err<LTNError>(LTNError::DIMENSION_MISMATCH);
            }
        }

        if (bias_it != model_data.end()) {
            if (bias_it->second.size() == predicate.bias.size()) {
                predicate.bias = bias_it->second;
            } else {
                return common::Err<LTNError>(LTNError::DIMENSION_MISMATCH);
            }
        }
    }

    return common::Ok<std::monostate>(std::monostate{});
}

}  // namespace inference_lab::engines::neuro_symbolic
