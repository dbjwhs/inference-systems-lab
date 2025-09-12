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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../../../common/src/result.hpp"
#include "../../../common/src/type_system.hpp"
#include "fuzzy_logic.hpp"

namespace inference_lab::engines::neuro_symbolic {

// ================================================================================================
// DIFFERENTIABLE OPERATION INTERFACE
// ================================================================================================

/**
 * @brief Type-erased base interface for differentiable operations
 *
 * This provides a common interface for operations with different input/output types,
 * enabling heterogeneous containers of operations. Uses type erasure to hide
 * template parameters while maintaining type safety through runtime checks.
 */
class DifferentiableOperationBase {
  public:
    virtual ~DifferentiableOperationBase() = default;

    /**
     * @brief Get operation name for debugging/logging
     * @return Human-readable operation name
     */
    virtual auto name() const -> std::string = 0;

    /**
     * @brief Get input type information
     * @return Type information for runtime type checking
     */
    virtual auto input_type() const -> std::type_info const& = 0;

    /**
     * @brief Get output type information
     * @return Type information for runtime type checking
     */
    virtual auto output_type() const -> std::type_info const& = 0;

  protected:
    DifferentiableOperationBase() = default;
};

/**
 * @brief Templated interface for differentiable logical operations
 *
 * All differentiable operations implement this interface to provide:
 * - Forward computation with gradient tracking
 * - Backward gradient computation for optimization
 * - Consistent API for composition into larger networks
 */
template <typename InputType, typename OutputType>
class DifferentiableOperation : public DifferentiableOperationBase {
  public:
    virtual ~DifferentiableOperation() = default;

    /**
     * @brief Forward pass computation
     * @param inputs Input tensors/values
     * @return Output tensor/value with operation applied
     */
    virtual auto forward(const InputType& inputs) -> OutputType = 0;

    /**
     * @brief Backward pass gradient computation
     * @param output_grad Gradient from downstream operations
     * @return Input gradients for backpropagation
     */
    virtual auto backward(const OutputType& output_grad) -> InputType = 0;

    /**
     * @brief Get input type information
     * @return Type information for InputType
     */
    auto input_type() const -> std::type_info const& override { return typeid(InputType); }

    /**
     * @brief Get output type information
     * @return Type information for OutputType
     */
    auto output_type() const -> std::type_info const& override { return typeid(OutputType); }
};

// ================================================================================================
// GRADIENT COMPUTATION UTILITIES
// ================================================================================================

/**
 * @brief Memory-safe gradient computation context for tracking intermediate values
 *
 * Stores intermediate values during forward pass that are needed for
 * gradient computation in backward pass. Includes automatic memory management
 * to prevent uncontrolled growth.
 */
template <typename ValueType>
class GradientContext {
  public:
    // Maximum number of stored values to prevent memory exhaustion
    static constexpr std::size_t MAX_STORED_VALUES = 1000;

    /**
     * @brief Default constructor
     */
    GradientContext() = default;

    /**
     * @brief Copy constructor
     */
    GradientContext(const GradientContext& other) : stored_values_(other.stored_values_) {}

    /**
     * @brief Move constructor
     */
    GradientContext(GradientContext&& other) noexcept
        : stored_values_(std::move(other.stored_values_)) {}

    /**
     * @brief Copy assignment operator
     */
    GradientContext& operator=(const GradientContext& other) {
        if (this != &other) {
            stored_values_ = other.stored_values_;
        }
        return *this;
    }

    /**
     * @brief Move assignment operator
     */
    GradientContext& operator=(GradientContext&& other) noexcept {
        if (this != &other) {
            stored_values_ = std::move(other.stored_values_);
        }
        return *this;
    }

    /**
     * @brief Destructor with automatic cleanup
     */
    ~GradientContext() {
        clear();  // Ensure proper RAII cleanup
    }

    /**
     * @brief Store value for gradient computation
     * @param key Identifier for the value
     * @param value Value to store
     * @return True if stored successfully, false if storage limit reached
     */
    bool store(const std::string& key, const ValueType& value) {
        if (stored_values_.size() >= MAX_STORED_VALUES) {
            return false;  // Prevent unbounded memory growth
        }
        stored_values_.emplace(key, value);
        return true;
    }

    /**
     * @brief Retrieve stored value
     * @param key Identifier for the value
     * @return Stored value
     */
    auto get(const std::string& key) const -> const ValueType& { return stored_values_.at(key); }

    /**
     * @brief Check if value is stored
     * @param key Identifier to check
     * @return True if value exists
     */
    bool has(const std::string& key) const {
        return stored_values_.find(key) != stored_values_.end();
    }

    /**
     * @brief Clear all stored values (memory cleanup)
     */
    void clear() {
        stored_values_.clear();
        stored_values_.reserve(0);  // Release memory
    }

    /**
     * @brief Get current memory usage
     * @return Number of stored values
     */
    std::size_t size() const noexcept { return stored_values_.size(); }

    /**
     * @brief Check if context is at capacity
     * @return True if at maximum capacity
     */
    bool is_full() const noexcept { return stored_values_.size() >= MAX_STORED_VALUES; }

  private:
    std::unordered_map<std::string, ValueType> stored_values_;
};

/**
 * @brief Gradient bounds checking and clipping utilities
 *
 * Prevents gradient explosion and vanishing by applying appropriate bounds.
 */
namespace gradient_utils {

// Maximum allowed gradient magnitude to prevent explosion
constexpr float MAX_GRADIENT_MAGNITUDE = 10.0f;
// Minimum allowed gradient magnitude to prevent vanishing
constexpr float MIN_GRADIENT_MAGNITUDE = 1e-8f;

/**
 * @brief Clamp gradient values to prevent explosion/vanishing
 * @param gradient Input gradient value
 * @return Clamped gradient within safe bounds
 */
inline float clamp_gradient(float gradient) noexcept {
    if (std::isnan(gradient) || std::isinf(gradient)) {
        return 0.0f;  // Replace NaN/inf with zero
    }

    // Clamp to prevent explosion
    if (gradient > MAX_GRADIENT_MAGNITUDE) {
        return MAX_GRADIENT_MAGNITUDE;
    }
    if (gradient < -MAX_GRADIENT_MAGNITUDE) {
        return -MAX_GRADIENT_MAGNITUDE;
    }

    // Clamp to prevent vanishing (preserve sign)
    if (std::abs(gradient) < MIN_GRADIENT_MAGNITUDE) {
        return gradient < 0.0f ? -MIN_GRADIENT_MAGNITUDE : MIN_GRADIENT_MAGNITUDE;
    }

    return gradient;
}

/**
 * @brief Safely compute gradient with intermediate value checking
 * @param base Base gradient value
 * @param factor Multiplication factor (checked for safety)
 * @return Safely computed and clamped gradient
 */
inline float safe_gradient_multiply(float base, float factor) noexcept {
    // Check for problematic factor values first
    if (std::isnan(factor) || std::isinf(factor) || std::abs(factor) > 1e6f) {
        return clamp_gradient(base);  // Fall back to base value only
    }

    // Compute intermediate result
    float intermediate = base * factor;

    // Check intermediate result before final clamping
    if (std::isnan(intermediate) || std::isinf(intermediate)) {
        return clamp_gradient(base);  // Fall back to base value
    }

    return clamp_gradient(intermediate);
}

/**
 * @brief Safely compute complex gradient expression
 * @param output_grad Base output gradient
 * @param expr Complex expression result
 * @return Safely computed and clamped gradient
 */
inline float safe_gradient_expression(float output_grad, float expr) noexcept {
    return safe_gradient_multiply(output_grad, expr);
}

/**
 * @brief Apply gradient clipping to tensor gradients
 * @param input_grad Tensor gradient to clip
 */
template <typename TensorType>
void clip_tensor_gradients(TensorType& input_grad) {
    for (std::size_t i = 0; i < TensorType::size; ++i) {
        input_grad[i] = clamp_gradient(input_grad[i]);
    }
}

}  // namespace gradient_utils

// ================================================================================================
// DIFFERENTIABLE UNARY OPERATIONS
// ================================================================================================

/**
 * @brief Differentiable fuzzy negation: ¬x = 1 - x
 *
 * Gradient: ∂(¬x)/∂x = -1
 * This operation has constant gradient, making it well-behaved for optimization.
 */
template <typename ElementType, typename ShapeType>
class DifferentiableNot
    : public DifferentiableOperation<common::types::TypedTensor<ElementType, ShapeType>,
                                     common::types::TypedTensor<ElementType, ShapeType>> {
  public:
    using TensorType = common::types::TypedTensor<ElementType, ShapeType>;

    static_assert(std::is_same_v<ElementType, float>, "Differentiable ops require float tensors");

    auto forward(const TensorType& input) -> TensorType override {
        // Store input for backward pass
        if (!context_.store("input", input)) {
            // Context full - clear and retry
            context_.clear();
            context_.store("input", input);
        }

        // Compute ¬x = 1 - x
        return tensor_fuzzy_not(input);
    }

    auto backward(const TensorType& output_grad) -> TensorType override {
        // Gradient of (1 - x) is -1 for all elements
        auto input_grad = TensorType::zeros();
        for (std::size_t i = 0; i < TensorType::size; ++i) {
            input_grad[i] = gradient_utils::clamp_gradient(-output_grad[i]);
        }
        return input_grad;
    }

    auto name() const -> std::string override { return "DifferentiableNot"; }

  private:
    GradientContext<TensorType> context_;
};

/**
 * @brief Differentiable sigmoid activation: σ(x) = 1/(1 + exp(-x))
 *
 * Maps real values to (0,1) interval, useful for converting neural network
 * outputs to fuzzy truth values.
 * Gradient: ∂σ(x)/∂x = σ(x)(1 - σ(x))
 */
template <typename ElementType, typename ShapeType>
class DifferentiableSigmoid
    : public DifferentiableOperation<common::types::TypedTensor<ElementType, ShapeType>,
                                     common::types::TypedTensor<ElementType, ShapeType>> {
  public:
    using TensorType = common::types::TypedTensor<ElementType, ShapeType>;

    static_assert(std::is_same_v<ElementType, float>, "Differentiable ops require float tensors");

    auto forward(const TensorType& input) -> TensorType override {
        auto output = TensorType::zeros();
        for (std::size_t i = 0; i < TensorType::size; ++i) {
            output[i] = sigmoid_membership(input[i], 1.0f, 0.0f);
        }

        // Store output for backward pass (needed for gradient computation)
        if (!context_.store("output", output)) {
            context_.clear();
            context_.store("output", output);
        }
        return output;
    }

    auto backward(const TensorType& output_grad) -> TensorType override {
        const auto& output = context_.get("output");

        auto input_grad = TensorType::zeros();
        for (std::size_t i = 0; i < TensorType::size; ++i) {
            // Gradient: σ'(x) = σ(x)(1 - σ(x))
            input_grad[i] = gradient_utils::safe_gradient_expression(
                output_grad[i], output[i] * (1.0f - output[i]));
        }
        return input_grad;
    }

    auto name() const -> std::string override { return "DifferentiableSigmoid"; }

  private:
    GradientContext<TensorType> context_;
};

// ================================================================================================
// DIFFERENTIABLE BINARY OPERATIONS
// ================================================================================================

/**
 * @brief Differentiable fuzzy conjunction using product T-norm
 *
 * Operation: x ∧ y = x * y
 * Gradients: ∂(x*y)/∂x = y, ∂(x*y)/∂y = x
 *
 * Product T-norm is naturally differentiable and provides meaningful
 * gradients for both operands.
 */
template <typename ElementType, typename ShapeType>
class DifferentiableAnd
    : public DifferentiableOperation<std::tuple<common::types::TypedTensor<ElementType, ShapeType>,
                                                common::types::TypedTensor<ElementType, ShapeType>>,
                                     common::types::TypedTensor<ElementType, ShapeType>> {
  public:
    using TensorType = common::types::TypedTensor<ElementType, ShapeType>;
    using InputType = std::tuple<TensorType, TensorType>;

    static_assert(std::is_same_v<ElementType, float>, "Differentiable ops require float tensors");

    auto forward(const InputType& inputs) -> TensorType override {
        const auto& [input1, input2] = inputs;

        // Store inputs for backward pass
        if (!context_.store("input1", input1) || !context_.store("input2", input2)) {
            context_.clear();
            context_.store("input1", input1);
            context_.store("input2", input2);
        }

        // Compute element-wise product (fuzzy AND)
        return tensor_fuzzy_and(input1, input2);
    }

    auto backward(const TensorType& output_grad) -> InputType override {
        const auto& input1 = context_.get("input1");
        const auto& input2 = context_.get("input2");

        auto grad1 = TensorType::zeros();
        auto grad2 = TensorType::zeros();

        for (std::size_t i = 0; i < TensorType::size; ++i) {
            // ∂(x*y)/∂x = y, ∂(x*y)/∂y = x
            grad1[i] = gradient_utils::clamp_gradient(output_grad[i] * input2[i]);
            grad2[i] = gradient_utils::clamp_gradient(output_grad[i] * input1[i]);
        }

        return std::make_tuple(grad1, grad2);
    }

    auto name() const -> std::string override { return "DifferentiableAnd"; }

  private:
    GradientContext<TensorType> context_;
};

/**
 * @brief Differentiable fuzzy disjunction using probabilistic sum
 *
 * Operation: x ∨ y = x + y - x*y
 * Gradients: ∂(x + y - xy)/∂x = 1 - y, ∂(x + y - xy)/∂y = 1 - x
 *
 * Probabilistic sum maintains differentiability while preserving
 * logical semantics of disjunction.
 */
template <typename ElementType, typename ShapeType>
class DifferentiableOr
    : public DifferentiableOperation<std::tuple<common::types::TypedTensor<ElementType, ShapeType>,
                                                common::types::TypedTensor<ElementType, ShapeType>>,
                                     common::types::TypedTensor<ElementType, ShapeType>> {
  public:
    using TensorType = common::types::TypedTensor<ElementType, ShapeType>;
    using InputType = std::tuple<TensorType, TensorType>;

    static_assert(std::is_same_v<ElementType, float>, "Differentiable ops require float tensors");

    auto forward(const InputType& inputs) -> TensorType override {
        const auto& [input1, input2] = inputs;

        // Store inputs for backward pass
        if (!context_.store("input1", input1) || !context_.store("input2", input2)) {
            context_.clear();
            context_.store("input1", input1);
            context_.store("input2", input2);
        }

        // Compute element-wise probabilistic sum (fuzzy OR)
        return tensor_fuzzy_or(input1, input2);
    }

    auto backward(const TensorType& output_grad) -> InputType override {
        const auto& input1 = context_.get("input1");
        const auto& input2 = context_.get("input2");

        auto grad1 = TensorType::zeros();
        auto grad2 = TensorType::zeros();

        for (std::size_t i = 0; i < TensorType::size; ++i) {
            // ∂(x + y - xy)/∂x = 1 - y, ∂(x + y - xy)/∂y = 1 - x
            grad1[i] = gradient_utils::safe_gradient_expression(output_grad[i], 1.0f - input2[i]);
            grad2[i] = gradient_utils::safe_gradient_expression(output_grad[i], 1.0f - input1[i]);
        }

        return std::make_tuple(grad1, grad2);
    }

    auto name() const -> std::string override { return "DifferentiableOr"; }

  private:
    GradientContext<TensorType> context_;
};

/**
 * @brief Differentiable fuzzy implication: x → y = (1-x) + xy
 *
 * Uses the probabilistic interpretation: P(y|x) ≈ fuzzy_or(¬x, y)
 * Gradients computed using chain rule through fuzzy operations.
 */
template <typename ElementType, typename ShapeType>
class DifferentiableImplies
    : public DifferentiableOperation<std::tuple<common::types::TypedTensor<ElementType, ShapeType>,
                                                common::types::TypedTensor<ElementType, ShapeType>>,
                                     common::types::TypedTensor<ElementType, ShapeType>> {
  public:
    using TensorType = common::types::TypedTensor<ElementType, ShapeType>;
    using InputType = std::tuple<TensorType, TensorType>;

    static_assert(std::is_same_v<ElementType, float>, "Differentiable ops require float tensors");

    auto forward(const InputType& inputs) -> TensorType override {
        const auto& [input1, input2] = inputs;

        // Store inputs for backward pass
        if (!context_.store("input1", input1) || !context_.store("input2", input2)) {
            context_.clear();
            context_.store("input1", input1);
            context_.store("input2", input2);
        }

        // Compute element-wise implication: x → y = ¬x ∨ y
        auto result = TensorType::zeros();
        for (std::size_t i = 0; i < TensorType::size; ++i) {
            result[i] = fuzzy_implies(input1[i], input2[i]);
        }
        return result;
    }

    auto backward(const TensorType& output_grad) -> InputType override {
        const auto& input1 = context_.get("input1");
        const auto& input2 = context_.get("input2");

        auto grad1 = TensorType::zeros();
        auto grad2 = TensorType::zeros();

        for (std::size_t i = 0; i < TensorType::size; ++i) {
            // For x → y = (1-x) + xy - (1-x)xy:
            // ∂/∂x = -1 + y - y + xy = -1 + xy = x*y - (1-x)*(1-y)
            // ∂/∂y = x - (1-x)*(-1) = x + (1-x) = 1
            float x = input1[i];
            float y = input2[i];

            grad1[i] = gradient_utils::safe_gradient_expression(output_grad[i], y - 1.0f + x * y);
            grad2[i] = gradient_utils::safe_gradient_expression(output_grad[i], 1.0f - x);
        }

        return std::make_tuple(grad1, grad2);
    }

    auto name() const -> std::string override { return "DifferentiableImplies"; }

  private:
    GradientContext<TensorType> context_;
};

// ================================================================================================
// DIFFERENTIABLE QUANTIFIERS
// ================================================================================================

/**
 * @brief Differentiable universal quantification using smooth aggregation
 *
 * Uses soft minimum via log-sum-exp trick for numerical stability:
 * soft_min(x) = -log(∑ exp(-αx)) / α
 *
 * As α → ∞, approaches true minimum while maintaining differentiability.
 */
template <typename ElementType, typename ShapeType>
class DifferentiableForall
    : public DifferentiableOperation<common::types::TypedTensor<ElementType, ShapeType>,
                                     FuzzyValue> {
  public:
    using TensorType = common::types::TypedTensor<ElementType, ShapeType>;

    static_assert(std::is_same_v<ElementType, float>, "Differentiable ops require float tensors");

    explicit DifferentiableForall(float temperature = 10.0f) : temperature_(temperature) {}

    auto forward(const TensorType& input) -> FuzzyValue override {
        // Store input for backward pass
        if (!context_.store("input", input)) {
            // Context full - clear and retry
            context_.clear();
            context_.store("input", input);
        }

        // Compute soft minimum using log-sum-exp
        float max_val = *std::max_element(input.data(), input.data() + TensorType::size);
        float sum_exp = 0.0f;

        for (std::size_t i = 0; i < TensorType::size; ++i) {
            sum_exp += std::exp(-temperature_ * (input[i] - max_val));
        }

        float result = max_val - std::log(sum_exp) / temperature_;
        // Store result as member variable for backward pass
        stored_result_ = result;

        return clamp_fuzzy_value(result);
    }

    auto backward(const FuzzyValue& output_grad) -> TensorType override {
        const auto& input = context_.get("input");
        float result = stored_result_;

        auto input_grad = TensorType::zeros();

        // Gradient of soft minimum
        for (std::size_t i = 0; i < TensorType::size; ++i) {
            float weight = std::exp(-temperature_ * (input[i] - result));
            input_grad[i] = gradient_utils::clamp_gradient(output_grad * weight);
        }

        return input_grad;
    }

    auto name() const -> std::string override { return "DifferentiableForall"; }

  private:
    float temperature_;
    float stored_result_ = 0.0f;
    GradientContext<TensorType> context_;
};

/**
 * @brief Differentiable existential quantification using smooth aggregation
 *
 * Uses soft maximum via log-sum-exp trick:
 * soft_max(x) = log(∑ exp(αx)) / α
 *
 * As α → ∞, approaches true maximum while maintaining differentiability.
 */
template <typename ElementType, typename ShapeType>
class DifferentiableExists
    : public DifferentiableOperation<common::types::TypedTensor<ElementType, ShapeType>,
                                     FuzzyValue> {
  public:
    using TensorType = common::types::TypedTensor<ElementType, ShapeType>;

    static_assert(std::is_same_v<ElementType, float>, "Differentiable ops require float tensors");

    explicit DifferentiableExists(float temperature = 10.0f) : temperature_(temperature) {}

    auto forward(const TensorType& input) -> FuzzyValue override {
        // Store input for backward pass
        if (!context_.store("input", input)) {
            // Context full - clear and retry
            context_.clear();
            context_.store("input", input);
        }

        // Compute soft maximum using log-sum-exp
        float max_val = *std::max_element(input.data(), input.data() + TensorType::size);
        float sum_exp = 0.0f;

        for (std::size_t i = 0; i < TensorType::size; ++i) {
            sum_exp += std::exp(temperature_ * (input[i] - max_val));
        }

        float result = max_val + std::log(sum_exp) / temperature_;
        // Store result as member variable for backward pass
        stored_result_ = result;

        return clamp_fuzzy_value(result);
    }

    auto backward(const FuzzyValue& output_grad) -> TensorType override {
        const auto& input = context_.get("input");
        float result = stored_result_;

        auto input_grad = TensorType::zeros();

        // Gradient of soft maximum
        for (std::size_t i = 0; i < TensorType::size; ++i) {
            float weight = std::exp(temperature_ * (input[i] - result));
            input_grad[i] = gradient_utils::clamp_gradient(output_grad * weight);
        }

        return input_grad;
    }

    auto name() const -> std::string override { return "DifferentiableExists"; }

  private:
    float temperature_;
    float stored_result_ = 0.0f;
    GradientContext<TensorType> context_;
};

// ================================================================================================
// COMPOSITE DIFFERENTIABLE OPERATIONS
// ================================================================================================

/**
 * @brief Differentiable logical formula evaluation
 *
 * Combines multiple differentiable operations into a computational graph
 * that can represent complex logical formulas with gradient flow.
 */
template <typename ElementType, typename ShapeType>
class DifferentiableFormula {
  public:
    using TensorType = common::types::TypedTensor<ElementType, ShapeType>;

    /**
     * @brief Add binary operation to the formula
     * @param op_type Operation type ("and", "or", "implies")
     * @param left_input Left operand tensor
     * @param right_input Right operand tensor
     * @return Operation result tensor
     */
    auto add_binary_op(const std::string& op_type,
                       const TensorType& left_input,
                       const TensorType& right_input) -> TensorType {
        if (op_type == "and") {
            auto op = std::make_unique<DifferentiableAnd<ElementType, ShapeType>>();
            auto result = op->forward(std::make_tuple(left_input, right_input));
            operations_.push_back(std::move(op));
            return result;
        } else if (op_type == "or") {
            auto op = std::make_unique<DifferentiableOr<ElementType, ShapeType>>();
            auto result = op->forward(std::make_tuple(left_input, right_input));
            operations_.push_back(std::move(op));
            return result;
        } else if (op_type == "implies") {
            auto op = std::make_unique<DifferentiableImplies<ElementType, ShapeType>>();
            auto result = op->forward(std::make_tuple(left_input, right_input));
            operations_.push_back(std::move(op));
            return result;
        }

        throw std::invalid_argument("Unknown operation type: " + op_type);
    }

    /**
     * @brief Add unary operation to the formula
     * @param op_type Operation type ("not", "sigmoid")
     * @param input Input tensor
     * @return Operation result tensor
     */
    auto add_unary_op(const std::string& op_type, const TensorType& input) -> TensorType {
        if (op_type == "not") {
            auto op = std::make_unique<DifferentiableNot<ElementType, ShapeType>>();
            auto result = op->forward(input);
            unary_operations_.push_back(std::move(op));
            return result;
        } else if (op_type == "sigmoid") {
            auto op = std::make_unique<DifferentiableSigmoid<ElementType, ShapeType>>();
            auto result = op->forward(input);
            unary_operations_.push_back(std::move(op));
            return result;
        }

        throw std::invalid_argument("Unknown operation type: " + op_type);
    }

    /**
     * @brief Add quantifier operation to the formula
     * @param quantifier_type Quantifier type ("forall", "exists")
     * @param input Input tensor
     * @param temperature Smoothing parameter
     * @return Aggregated fuzzy value
     */
    auto add_quantifier(const std::string& quantifier_type,
                        const TensorType& input,
                        float temperature = 10.0f) -> FuzzyValue {
        if (quantifier_type == "forall") {
            auto op = std::make_unique<DifferentiableForall<ElementType, ShapeType>>(temperature);
            auto result = op->forward(input);
            quantifier_operations_.push_back(std::move(op));
            return result;
        } else if (quantifier_type == "exists") {
            auto op = std::make_unique<DifferentiableExists<ElementType, ShapeType>>(temperature);
            auto result = op->forward(input);
            quantifier_operations_.push_back(std::move(op));
            return result;
        }

        throw std::invalid_argument("Unknown quantifier type: " + quantifier_type);
    }

    /**
     * @brief Get total number of operations in the formula
     * @return Operation count
     */
    auto operation_count() const -> std::size_t {
        return operations_.size() + unary_operations_.size() + quantifier_operations_.size();
    }

    /**
     * @brief Clear all operations and reset the formula
     */
    void clear() {
        operations_.clear();
        unary_operations_.clear();
        quantifier_operations_.clear();
    }

  private:
    // Storage for different operation types to manage memory properly
    std::vector<std::unique_ptr<DifferentiableAnd<ElementType, ShapeType>>> operations_;
    std::vector<std::unique_ptr<DifferentiableNot<ElementType, ShapeType>>> unary_operations_;
    std::vector<std::unique_ptr<DifferentiableForall<ElementType, ShapeType>>>
        quantifier_operations_;
};

}  // namespace inference_lab::engines::neuro_symbolic
