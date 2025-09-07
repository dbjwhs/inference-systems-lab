/**
 * @file tensor_logic_bridge.hpp
 * @brief Bridge between TypedTensor system and logical operations
 *
 * This file provides the integration layer between the existing TypedTensor
 * infrastructure and the fuzzy logic/differentiable operations system. It enables
 * seamless conversion between numerical tensor representations and logical
 * truth values, supporting both forward inference and gradient-based learning.
 *
 * Key Features:
 * - Type-safe conversion between tensors and truth values
 * - Predicate evaluation over tensor data
 * - Integration with existing memory allocation and SIMD optimization
 * - Support for both static and dynamic tensor shapes
 * - Automatic broadcasting for element-wise logical operations
 * - Efficient batch processing of logical formulas
 *
 * Architecture Integration:
 * @code
 *   TypedTensor<float> ─────► LogicalTensor ─────► Differentiable Ops
 *   (numerical data)          (truth values)       (logical reasoning)
 *         ▲                        │                        │
 *         │                        │                        ▼
 *   Neural Networks ◄─────── Gradient Flow ◄───── Fuzzy Logic Results
 *   (backpropagation)        (optimization)        (continuous [0,1])
 * @endcode
 *
 * Example Usage:
 * @code
 * // Create logical tensor from neural network output
 * TypedTensor<float, Shape<10, 3>> features = neural_net.forward(input);
 * LogicalTensor<float, Shape<10, 3>> logical = LogicalTensor::from_tensor(features);
 *
 * // Define predicates over the tensor data
 * auto is_positive = logical.predicate([](float x) { return x > 0.0f; });
 * auto is_large = logical.predicate([](float x) { return x > 0.8f; });
 *
 * // Combine predicates with logical operations
 * auto complex_predicate = logical.logical_and(is_positive, is_large);
 *
 * // Quantify over dimensions
 * auto forall_features = complex_predicate.forall(1);  // ∀ over feature dim
 * auto exists_samples = forall_features.exists(0);     // ∃ over sample dim
 * @endcode
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "../../../common/src/result.hpp"
#include "../../../common/src/type_system.hpp"
#include "differentiable_ops.hpp"
#include "fuzzy_logic.hpp"

namespace inference_lab::engines::neuro_symbolic {

// ================================================================================================
// LOGICAL TENSOR WRAPPER
// ================================================================================================

/**
 * @brief Logical wrapper around TypedTensor for fuzzy logic operations
 *
 * This class wraps a TypedTensor and provides logical operations that maintain
 * the tensor structure while operating on fuzzy truth values. It serves as the
 * primary interface between numerical computation and logical reasoning.
 *
 * @tparam ElementType Element type (must be float for logical operations)
 * @tparam ShapeType Compile-time shape specification
 */
template <typename ElementType, typename ShapeType>
class LogicalTensor {
  public:
    using TensorType = common::types::TypedTensor<ElementType, ShapeType>;
    using element_type = ElementType;
    using shape_type = ShapeType;

    static constexpr std::size_t rank = ShapeType::rank;
    static constexpr std::size_t size = ShapeType::size;

    static_assert(std::is_same_v<ElementType, float>,
                  "LogicalTensor requires float element type for fuzzy operations");

    // ============================================================================================
    // CONSTRUCTION AND CONVERSION
    // ============================================================================================

    /**
     * @brief Construct from existing TypedTensor
     * @param tensor Source tensor (values should be in [0,1] for logical interpretation)
     */
    explicit LogicalTensor(TensorType tensor) : tensor_(std::move(tensor)) {
        // Optionally clamp values to [0,1] range for logical consistency
        for (std::size_t i = 0; i < size; ++i) {
            tensor_[i] = clamp_fuzzy_value(tensor_[i]);
        }
    }

    /**
     * @brief Create LogicalTensor from raw tensor data
     * @param tensor Source tensor
     * @return LogicalTensor wrapping the tensor
     */
    static auto from_tensor(TensorType tensor) -> LogicalTensor {
        return LogicalTensor(std::move(tensor));
    }

    /**
     * @brief Create LogicalTensor with all true values
     * @return LogicalTensor filled with 1.0 (true)
     */
    static auto all_true() -> LogicalTensor { return LogicalTensor(TensorType::filled(1.0f)); }

    /**
     * @brief Create LogicalTensor with all false values
     * @return LogicalTensor filled with 0.0 (false)
     */
    static auto all_false() -> LogicalTensor { return LogicalTensor(TensorType::zeros()); }

    /**
     * @brief Create LogicalTensor from predicate function applied to another tensor
     * @tparam SourceElementType Element type of source tensor
     * @param source Source tensor to evaluate predicate on
     * @param predicate Function mapping source elements to truth values
     * @return LogicalTensor with predicate results
     */
    template <typename SourceElementType>
    static auto from_predicate(
        const common::types::TypedTensor<SourceElementType, ShapeType>& source,
        std::function<float(SourceElementType)> predicate) -> LogicalTensor {
        auto result_data = std::make_unique<float[]>(size);
        for (std::size_t i = 0; i < size; ++i) {
            result_data[i] = clamp_fuzzy_value(predicate(source[i]));
        }

        return LogicalTensor(TensorType::from_data(std::move(result_data)));
    }

    // ============================================================================================
    // DATA ACCESS AND PROPERTIES
    // ============================================================================================

    /**
     * @brief Get underlying tensor
     * @return Reference to wrapped tensor
     */
    auto tensor() -> TensorType& { return tensor_; }
    auto tensor() const -> const TensorType& { return tensor_; }

    /**
     * @brief Access truth value at given index
     * @param index Element index
     * @return Truth value at index
     */
    auto operator[](std::size_t index) -> FuzzyValue& {
        assert(index < size);
        return tensor_[index];
    }
    auto operator[](std::size_t index) const -> const FuzzyValue& {
        assert(index < size);
        return tensor_[index];
    }

    /**
     * @brief Get raw data pointer
     * @return Pointer to underlying data
     */
    auto data() -> FuzzyValue* { return tensor_.data(); }
    auto data() const -> const FuzzyValue* { return tensor_.data(); }

    /**
     * @brief Get shape information
     * @return Shape of the logical tensor
     */
    static constexpr auto shape() -> ShapeType { return ShapeType{}; }

    // ============================================================================================
    // PREDICATE OPERATIONS
    // ============================================================================================

    /**
     * @brief Apply unary predicate to create new LogicalTensor
     * @param predicate Function mapping values to truth values
     * @return New LogicalTensor with predicate applied
     */
    auto predicate(std::function<float(float)> predicate) const -> LogicalTensor {
        auto result_data = std::make_unique<float[]>(size);
        for (std::size_t i = 0; i < size; ++i) {
            result_data[i] = clamp_fuzzy_value(predicate(tensor_[i]));
        }
        return LogicalTensor(TensorType::from_data(std::move(result_data)));
    }

    /**
     * @brief Create predicate testing if values are greater than threshold
     * @param threshold Comparison threshold
     * @return LogicalTensor with comparison results
     */
    auto greater_than(float threshold) const -> LogicalTensor {
        return predicate(
            [threshold](float x) { return sigmoid_membership(x - threshold, 10.0f, 0.0f); });
    }

    /**
     * @brief Create predicate testing if values are less than threshold
     * @param threshold Comparison threshold
     * @return LogicalTensor with comparison results
     */
    auto less_than(float threshold) const -> LogicalTensor {
        return predicate(
            [threshold](float x) { return sigmoid_membership(threshold - x, 10.0f, 0.0f); });
    }

    /**
     * @brief Create predicate testing if values are approximately equal to target
     * @param target Target value
     * @param tolerance Tolerance for equality (controls sharpness)
     * @return LogicalTensor with equality results
     */
    auto approximately_equal(float target, float tolerance = 0.1f) const -> LogicalTensor {
        return predicate(
            [target, tolerance](float x) { return gaussian_membership(x, target, tolerance); });
    }

    // ============================================================================================
    // LOGICAL OPERATIONS
    // ============================================================================================

    /**
     * @brief Element-wise logical negation
     * @return New LogicalTensor with negated values
     */
    auto logical_not() const -> LogicalTensor {
        auto result = tensor_fuzzy_not(tensor_);
        return LogicalTensor(std::move(result));
    }

    /**
     * @brief Element-wise logical conjunction
     * @param other Other LogicalTensor
     * @return New LogicalTensor with AND results
     */
    auto logical_and(const LogicalTensor& other) const -> LogicalTensor {
        auto result = tensor_fuzzy_and(tensor_, other.tensor_);
        return LogicalTensor(std::move(result));
    }

    /**
     * @brief Element-wise logical disjunction
     * @param other Other LogicalTensor
     * @return New LogicalTensor with OR results
     */
    auto logical_or(const LogicalTensor& other) const -> LogicalTensor {
        auto result = tensor_fuzzy_or(tensor_, other.tensor_);
        return LogicalTensor(std::move(result));
    }

    /**
     * @brief Element-wise logical implication
     * @param other Other LogicalTensor (consequent)
     * @return New LogicalTensor with implication results
     */
    auto logical_implies(const LogicalTensor& other) const -> LogicalTensor {
        auto result_data = std::make_unique<float[]>(size);
        for (std::size_t i = 0; i < size; ++i) {
            result_data[i] = fuzzy_implies(tensor_[i], other.tensor_[i]);
        }
        return LogicalTensor(TensorType::from_data(std::move(result_data)));
    }

    /**
     * @brief Element-wise logical biconditional
     * @param other Other LogicalTensor
     * @return New LogicalTensor with biconditional results
     */
    auto logical_biconditional(const LogicalTensor& other) const -> LogicalTensor {
        auto result_data = std::make_unique<float[]>(size);
        for (std::size_t i = 0; i < size; ++i) {
            result_data[i] = fuzzy_biconditional(tensor_[i], other.tensor_[i]);
        }
        return LogicalTensor(TensorType::from_data(std::move(result_data)));
    }

    // ============================================================================================
    // QUANTIFICATION AND AGGREGATION
    // ============================================================================================

    /**
     * @brief Universal quantification over all elements
     * @return Single fuzzy value representing ∀ over tensor
     */
    auto forall() const -> FuzzyValue { return tensor_fuzzy_forall(tensor_); }

    /**
     * @brief Existential quantification over all elements
     * @return Single fuzzy value representing ∃ over tensor
     */
    auto exists() const -> FuzzyValue { return tensor_fuzzy_exists(tensor_); }

    /**
     * @brief Smooth universal quantification with temperature
     * @param temperature Smoothing parameter (higher = sharper)
     * @return Smooth ∀ aggregation result
     */
    auto smooth_forall(float temperature = 10.0f) const -> FuzzyValue {
        std::vector<float> values(tensor_.data(), tensor_.data() + size);
        return smooth_fuzzy_forall(values, temperature);
    }

    /**
     * @brief Smooth existential quantification with temperature
     * @param temperature Smoothing parameter (higher = sharper)
     * @return Smooth ∃ aggregation result
     */
    auto smooth_exists(float temperature = 10.0f) const -> FuzzyValue {
        std::vector<float> values(tensor_.data(), tensor_.data() + size);
        return smooth_fuzzy_exists(values, temperature);
    }

    // ============================================================================================
    // STATISTICAL AND ANALYSIS OPERATIONS
    // ============================================================================================

    /**
     * @brief Calculate truth value statistics
     * @return Struct containing mean, variance, min, max of truth values
     */
    struct Statistics {
        float mean;
        float variance;
        float min_value;
        float max_value;
        float entropy;  // Shannon entropy of truth distribution
    };

    auto statistics() const -> Statistics {
        float sum = 0.0f;
        float min_val = 1.0f;
        float max_val = 0.0f;

        for (std::size_t i = 0; i < size; ++i) {
            float val = tensor_[i];
            sum += val;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        float mean = sum / static_cast<float>(size);

        // Calculate variance
        float variance_sum = 0.0f;
        for (std::size_t i = 0; i < size; ++i) {
            float diff = tensor_[i] - mean;
            variance_sum += diff * diff;
        }
        float variance = variance_sum / static_cast<float>(size);

        // Calculate Shannon entropy (treating as probability distribution)
        float entropy = 0.0f;
        for (std::size_t i = 0; i < size; ++i) {
            float p = tensor_[i];
            if (p > 0.0f) {
                entropy -= p * std::log2(p);
            }
            float q = 1.0f - p;
            if (q > 0.0f) {
                entropy -= q * std::log2(q);
            }
        }
        entropy /= static_cast<float>(size);

        return Statistics{mean, variance, min_val, max_val, entropy};
    }

    /**
     * @brief Count elements satisfying truth threshold
     * @param threshold Truth value threshold (default: 0.5)
     * @return Number of elements >= threshold
     */
    auto count_true(float threshold = 0.5f) const -> std::size_t {
        std::size_t count = 0;
        for (std::size_t i = 0; i < size; ++i) {
            if (tensor_[i] >= threshold) {
                ++count;
            }
        }
        return count;
    }

    /**
     * @brief Get proportion of elements satisfying truth threshold
     * @param threshold Truth value threshold (default: 0.5)
     * @return Proportion of true elements [0,1]
     */
    auto truth_proportion(float threshold = 0.5f) const -> float {
        return static_cast<float>(count_true(threshold)) / static_cast<float>(size);
    }

    // ============================================================================================
    // DIFFERENTIABLE OPERATIONS INTEGRATION
    // ============================================================================================

    /**
     * @brief Create differentiable AND operation
     * @param other Other LogicalTensor
     * @return Differentiable AND result
     */
    auto differentiable_and(const LogicalTensor& other) const
        -> std::tuple<LogicalTensor, std::unique_ptr<DifferentiableAnd<ElementType, ShapeType>>> {
        auto diff_op = std::make_unique<DifferentiableAnd<ElementType, ShapeType>>();
        auto result = diff_op->forward(std::make_tuple(tensor_, other.tensor_));

        return std::make_tuple(LogicalTensor(std::move(result)), std::move(diff_op));
    }

    /**
     * @brief Create differentiable OR operation
     * @param other Other LogicalTensor
     * @return Differentiable OR result
     */
    auto differentiable_or(const LogicalTensor& other) const
        -> std::tuple<LogicalTensor, std::unique_ptr<DifferentiableOr<ElementType, ShapeType>>> {
        auto diff_op = std::make_unique<DifferentiableOr<ElementType, ShapeType>>();
        auto result = diff_op->forward(std::make_tuple(tensor_, other.tensor_));

        return std::make_tuple(LogicalTensor(std::move(result)), std::move(diff_op));
    }

    /**
     * @brief Create differentiable NOT operation
     * @return Differentiable NOT result
     */
    auto differentiable_not() const
        -> std::tuple<LogicalTensor, std::unique_ptr<DifferentiableNot<ElementType, ShapeType>>> {
        auto diff_op = std::make_unique<DifferentiableNot<ElementType, ShapeType>>();
        auto result = diff_op->forward(tensor_);

        return std::make_tuple(LogicalTensor(std::move(result)), std::move(diff_op));
    }

  private:
    TensorType tensor_;  ///< Underlying tensor storing truth values
};

// ================================================================================================
// CONVENIENCE FUNCTIONS AND OPERATORS
// ================================================================================================

/**
 * @brief Logical AND operator overload
 * @param lhs Left operand
 * @param rhs Right operand
 * @return Result of logical AND
 */
template <typename ElementType, typename ShapeType>
auto operator&(const LogicalTensor<ElementType, ShapeType>& lhs,
               const LogicalTensor<ElementType, ShapeType>& rhs)
    -> LogicalTensor<ElementType, ShapeType> {
    return lhs.logical_and(rhs);
}

/**
 * @brief Logical OR operator overload
 * @param lhs Left operand
 * @param rhs Right operand
 * @return Result of logical OR
 */
template <typename ElementType, typename ShapeType>
auto operator|(const LogicalTensor<ElementType, ShapeType>& lhs,
               const LogicalTensor<ElementType, ShapeType>& rhs)
    -> LogicalTensor<ElementType, ShapeType> {
    return lhs.logical_or(rhs);
}

/**
 * @brief Logical NOT operator overload
 * @param operand Input LogicalTensor
 * @return Result of logical NOT
 */
template <typename ElementType, typename ShapeType>
auto operator!(const LogicalTensor<ElementType, ShapeType>& operand)
    -> LogicalTensor<ElementType, ShapeType> {
    return operand.logical_not();
}

// ================================================================================================
// UTILITY FUNCTIONS FOR TENSOR-LOGIC INTEGRATION
// ================================================================================================

/**
 * @brief Convert neural network output to logical truth values
 * @param nn_output Neural network output tensor
 * @param activation_type Activation function ("sigmoid", "tanh", "softmax")
 * @return LogicalTensor with truth values in [0,1]
 */
template <typename ElementType, typename ShapeType>
auto neural_to_logical(const common::types::TypedTensor<ElementType, ShapeType>& nn_output,
                       const std::string& activation_type = "sigmoid")
    -> LogicalTensor<ElementType, ShapeType> {
    static_assert(std::is_same_v<ElementType, float>, "Conversion requires float tensors");

    auto result_data = std::make_unique<float[]>(nn_output.size);

    if (activation_type == "sigmoid") {
        for (std::size_t i = 0; i < nn_output.size; ++i) {
            result_data[i] = sigmoid_membership(nn_output[i], 1.0f, 0.0f);
        }
    } else if (activation_type == "tanh") {
        for (std::size_t i = 0; i < nn_output.size; ++i) {
            // Convert tanh output [-1,1] to [0,1]
            result_data[i] = (std::tanh(nn_output[i]) + 1.0f) * 0.5f;
        }
    } else if (activation_type == "identity") {
        for (std::size_t i = 0; i < nn_output.size; ++i) {
            result_data[i] = clamp_fuzzy_value(nn_output[i]);
        }
    } else {
        throw std::invalid_argument("Unknown activation type: " + activation_type);
    }

    auto tensor =
        common::types::TypedTensor<ElementType, ShapeType>::from_data(std::move(result_data));
    return LogicalTensor<ElementType, ShapeType>::from_tensor(std::move(tensor));
}

/**
 * @brief Create LogicalTensor from probability distributions
 * @param probabilities Tensor containing probability values
 * @return LogicalTensor interpreting probabilities as truth values
 */
template <typename ElementType, typename ShapeType>
auto probabilities_to_logical(
    const common::types::TypedTensor<ElementType, ShapeType>& probabilities)
    -> LogicalTensor<ElementType, ShapeType> {
    static_assert(std::is_same_v<ElementType, float>, "Probabilities require float tensors");

    // Probabilities are already in [0,1], just ensure clamping
    auto result_data = std::make_unique<float[]>(probabilities.size);
    for (std::size_t i = 0; i < probabilities.size; ++i) {
        result_data[i] = clamp_fuzzy_value(probabilities[i]);
    }

    auto tensor =
        common::types::TypedTensor<ElementType, ShapeType>::from_data(std::move(result_data));
    return LogicalTensor<ElementType, ShapeType>::from_tensor(std::move(tensor));
}

}  // namespace inference_lab::engines::neuro_symbolic
