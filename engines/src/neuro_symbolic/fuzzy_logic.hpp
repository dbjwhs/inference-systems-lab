/**
 * @file fuzzy_logic.hpp
 * @brief Fuzzy logic operations with continuous truth values for differentiable reasoning
 *
 * This file implements fuzzy logic operations that serve as the foundation for
 * Logic Tensor Networks (LTNs). Unlike traditional boolean logic, fuzzy logic
 * operates on continuous truth values in [0,1], making it naturally compatible
 * with neural networks and gradient-based optimization.
 *
 * Key Features:
 * - Continuous truth values enabling differentiable logic operations
 * - T-norm and T-conorm operators (product, Lukasiewicz, Gödel)
 * - Fuzzy quantifiers (∀, ∃) with aggregation functions
 * - Smooth approximations ensuring gradient flow
 * - Integration with existing TypedTensor system for efficient computation
 *
 * Mathematical Foundation:
 * Fuzzy logic extends classical logic by replacing boolean values {0,1} with
 * continuous values in [0,1]. This enables:
 * - Gradual membership in sets/predicates
 * - Smooth logical connectives suitable for optimization
 * - Probabilistic interpretation of logical statements
 *
 * Example Usage:
 * @code
 * // Fuzzy truth values
 * FuzzyValue p = 0.8f;    // "mostly true"
 * FuzzyValue q = 0.3f;    // "somewhat false"
 *
 * // Fuzzy logical operations
 * auto conjunction = fuzzy_and(p, q);    // T-norm: 0.24 (product)
 * auto disjunction = fuzzy_or(p, q);     // T-conorm: 0.86
 * auto negation = fuzzy_not(p);          // 0.2
 * auto implication = fuzzy_implies(p, q); // 0.375
 *
 * // Fuzzy quantification over tensors
 * TypedTensor<float, Shape<10>> truth_values = ...;
 * auto forall_result = fuzzy_forall(truth_values);
 * auto exists_result = fuzzy_exists(truth_values);
 * @endcode
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "../../../common/src/result.hpp"
#include "../../../common/src/type_system.hpp"

namespace inference_lab::engines::neuro_symbolic {

// ================================================================================================
// FUZZY TRUTH VALUE TYPES
// ================================================================================================

/**
 * @brief Fundamental fuzzy truth value type
 *
 * Represents a continuous truth value in the interval [0,1] where:
 * - 0.0 represents "completely false"
 * - 1.0 represents "completely true"
 * - Values in between represent degrees of truth
 */
using FuzzyValue = float;

/**
 * @brief Validate that a value is a valid fuzzy truth value
 * @param value The value to validate
 * @return True if value is in [0,1], false otherwise
 */
constexpr bool is_valid_fuzzy_value(FuzzyValue value) noexcept {
    return value >= 0.0f && value <= 1.0f;
}

/**
 * @brief Clamp a value to the valid fuzzy range [0,1]
 * @param value The value to clamp
 * @return Value clamped to [0,1]
 */
constexpr FuzzyValue clamp_fuzzy_value(float value) noexcept {
    return std::max(0.0f, std::min(1.0f, value));
}

// ================================================================================================
// T-NORM AND T-CONORM OPERATORS
// ================================================================================================

/**
 * @brief T-norm operators for fuzzy conjunction (AND)
 *
 * T-norms are binary operations that generalize classical conjunction
 * to the fuzzy setting, satisfying:
 * - Commutativity: T(a,b) = T(b,a)
 * - Associativity: T(T(a,b),c) = T(a,T(b,c))
 * - Monotonicity: if a ≤ c and b ≤ d, then T(a,b) ≤ T(c,d)
 * - Boundary conditions: T(a,1) = a, T(a,0) = 0
 */
namespace tnorms {

/**
 * @brief Product T-norm: T(a,b) = a * b
 *
 * Most commonly used in Logic Tensor Networks due to its smooth
 * gradients and natural probabilistic interpretation.
 */
constexpr FuzzyValue product(FuzzyValue a, FuzzyValue b) noexcept {
    return a * b;
}

/**
 * @brief Lukasiewicz T-norm: T(a,b) = max(0, a + b - 1)
 *
 * Based on Lukasiewicz logic, provides different aggregation
 * behavior than product norm.
 */
constexpr FuzzyValue lukasiewicz(FuzzyValue a, FuzzyValue b) noexcept {
    return std::max(0.0f, a + b - 1.0f);
}

/**
 * @brief Minimum T-norm: T(a,b) = min(a, b)
 *
 * Also known as Gödel T-norm. Simple but not differentiable
 * at boundary points.
 */
constexpr FuzzyValue minimum(FuzzyValue a, FuzzyValue b) noexcept {
    return std::min(a, b);
}

/**
 * @brief Drastic T-norm: T(a,b) = min(a,b) if max(a,b) = 1, else 0
 *
 * Extreme T-norm that approaches classical logic behavior.
 */
constexpr FuzzyValue drastic(FuzzyValue a, FuzzyValue b) noexcept {
    if (std::max(a, b) == 1.0f) {
        return std::min(a, b);
    }
    return 0.0f;
}

}  // namespace tnorms

/**
 * @brief T-conorm operators for fuzzy disjunction (OR)
 *
 * T-conorms (S-norms) are binary operations that generalize classical
 * disjunction to the fuzzy setting. They are dual to T-norms via
 * De Morgan's laws: S(a,b) = 1 - T(1-a, 1-b)
 */
namespace tconorms {

/**
 * @brief Probabilistic sum: S(a,b) = a + b - a*b
 *
 * Dual to product T-norm, commonly used in probabilistic systems.
 */
constexpr FuzzyValue probabilistic_sum(FuzzyValue a, FuzzyValue b) noexcept {
    return a + b - a * b;
}

/**
 * @brief Lukasiewicz T-conorm: S(a,b) = min(1, a + b)
 *
 * Bounded sum operation, dual to Lukasiewicz T-norm.
 */
constexpr FuzzyValue bounded_sum(FuzzyValue a, FuzzyValue b) noexcept {
    return std::min(1.0f, a + b);
}

/**
 * @brief Maximum T-conorm: S(a,b) = max(a, b)
 *
 * Dual to minimum T-norm, simple but non-differentiable at boundaries.
 */
constexpr FuzzyValue maximum(FuzzyValue a, FuzzyValue b) noexcept {
    return std::max(a, b);
}

/**
 * @brief Drastic T-conorm: S(a,b) = max(a,b) if min(a,b) = 0, else 1
 *
 * Dual to drastic T-norm.
 */
constexpr FuzzyValue drastic(FuzzyValue a, FuzzyValue b) noexcept {
    if (std::min(a, b) == 0.0f) {
        return std::max(a, b);
    }
    return 1.0f;
}

}  // namespace tconorms

// ================================================================================================
// BASIC FUZZY LOGICAL OPERATIONS
// ================================================================================================

/**
 * @brief Fuzzy negation: ¬a = 1 - a
 * @param a Input fuzzy value
 * @return Negated fuzzy value
 */
constexpr FuzzyValue fuzzy_not(FuzzyValue a) noexcept {
    return 1.0f - a;
}

/**
 * @brief Fuzzy conjunction using product T-norm
 * @param a First operand
 * @param b Second operand
 * @return Fuzzy conjunction result
 */
constexpr FuzzyValue fuzzy_and(FuzzyValue a, FuzzyValue b) noexcept {
    return tnorms::product(a, b);
}

/**
 * @brief Fuzzy disjunction using probabilistic sum T-conorm
 * @param a First operand
 * @param b Second operand
 * @return Fuzzy disjunction result
 */
constexpr FuzzyValue fuzzy_or(FuzzyValue a, FuzzyValue b) noexcept {
    return tconorms::probabilistic_sum(a, b);
}

/**
 * @brief Fuzzy implication: a → b = ¬a ∨ b = (1-a) + ab
 * @param a Antecedent
 * @param b Consequent
 * @return Fuzzy implication result
 *
 * Uses the probabilistic interpretation: P(B|A) approximated by fuzzy operations.
 */
constexpr FuzzyValue fuzzy_implies(FuzzyValue a, FuzzyValue b) noexcept {
    return fuzzy_or(fuzzy_not(a), b);
}

/**
 * @brief Fuzzy biconditional: a ↔ b = (a → b) ∧ (b → a)
 * @param a First operand
 * @param b Second operand
 * @return Fuzzy biconditional result
 */
constexpr FuzzyValue fuzzy_biconditional(FuzzyValue a, FuzzyValue b) noexcept {
    return fuzzy_and(fuzzy_implies(a, b), fuzzy_implies(b, a));
}

/**
 * @brief Fuzzy exclusive or: a ⊕ b = (a ∨ b) ∧ ¬(a ∧ b)
 * @param a First operand
 * @param b Second operand
 * @return Fuzzy XOR result
 */
constexpr FuzzyValue fuzzy_xor(FuzzyValue a, FuzzyValue b) noexcept {
    return fuzzy_and(fuzzy_or(a, b), fuzzy_not(fuzzy_and(a, b)));
}

// ================================================================================================
// FUZZY QUANTIFIERS AND AGGREGATION
// ================================================================================================

/**
 * @brief Fuzzy universal quantification (∀) using product aggregation
 *
 * For a collection of truth values, computes ∀x P(x) as the product
 * of all individual truth values. This is differentiable and provides
 * meaningful gradients for optimization.
 *
 * @tparam Container Container type holding fuzzy values
 * @param values Collection of fuzzy truth values
 * @return Aggregated universal quantification result
 */
template <typename Container>
FuzzyValue fuzzy_forall(const Container& values) {
    FuzzyValue result = 1.0f;
    for (const auto& value : values) {
        result *= value;
    }
    return result;
}

/**
 * @brief Fuzzy existential quantification (∃) using probabilistic sum
 *
 * For a collection of truth values, computes ∃x P(x) as 1 - ∏(1 - xi).
 * This represents the probability that at least one element satisfies
 * the predicate.
 *
 * @tparam Container Container type holding fuzzy values
 * @param values Collection of fuzzy truth values
 * @return Aggregated existential quantification result
 */
template <typename Container>
FuzzyValue fuzzy_exists(const Container& values) {
    FuzzyValue complement_product = 1.0f;
    for (const auto& value : values) {
        complement_product *= (1.0f - value);
    }
    return 1.0f - complement_product;
}

/**
 * @brief Smooth approximation of fuzzy universal quantification
 *
 * Uses p-mean with large p to approximate minimum operation while
 * maintaining differentiability. As p → ∞, approaches min operation.
 *
 * @tparam Container Container type holding fuzzy values
 * @param values Collection of fuzzy truth values
 * @param p Smoothing parameter (larger = closer to min)
 * @return Smooth universal quantification result
 */
template <typename Container>
FuzzyValue smooth_fuzzy_forall(const Container& values, float p = 10.0f) {
    if (values.empty())
        return 1.0f;

    float sum = 0.0f;
    for (const auto& value : values) {
        sum += std::pow(value, p);
    }
    return std::pow(sum / values.size(), 1.0f / p);
}

/**
 * @brief Smooth approximation of fuzzy existential quantification
 *
 * Uses p-mean with large p to approximate maximum operation while
 * maintaining differentiability. As p → ∞, approaches max operation.
 *
 * @tparam Container Container type holding fuzzy values
 * @param values Collection of fuzzy truth values
 * @param p Smoothing parameter (larger = closer to max)
 * @return Smooth existential quantification result
 */
template <typename Container>
FuzzyValue smooth_fuzzy_exists(const Container& values, float p = 10.0f) {
    if (values.empty())
        return 0.0f;

    float sum = 0.0f;
    for (const auto& value : values) {
        sum += std::pow(value, p);
    }
    return std::pow(sum / values.size(), 1.0f / p);
}

// ================================================================================================
// TENSOR-BASED FUZZY OPERATIONS
// ================================================================================================

/**
 * @brief Apply fuzzy negation to all elements of a tensor
 * @tparam ElementType Tensor element type
 * @tparam ShapeType Tensor shape type
 * @param tensor Input tensor
 * @return New tensor with negated values
 */
template <typename ElementType, typename ShapeType>
auto tensor_fuzzy_not(const common::types::TypedTensor<ElementType, ShapeType>& tensor)
    -> common::types::TypedTensor<ElementType, ShapeType> {
    static_assert(std::is_same_v<ElementType, float>, "Fuzzy operations require float tensors");

    auto result = common::types::TypedTensor<ElementType, ShapeType>::zeros();
    for (std::size_t i = 0; i < tensor.size; ++i) {
        result[i] = fuzzy_not(tensor[i]);
    }
    return result;
}

/**
 * @brief Element-wise fuzzy AND between two tensors
 * @tparam ElementType Tensor element type
 * @tparam ShapeType Tensor shape type
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with element-wise AND results
 */
template <typename ElementType, typename ShapeType>
auto tensor_fuzzy_and(const common::types::TypedTensor<ElementType, ShapeType>& a,
                      const common::types::TypedTensor<ElementType, ShapeType>& b)
    -> common::types::TypedTensor<ElementType, ShapeType> {
    static_assert(std::is_same_v<ElementType, float>, "Fuzzy operations require float tensors");

    auto result = common::types::TypedTensor<ElementType, ShapeType>::zeros();
    for (std::size_t i = 0; i < a.size; ++i) {
        result[i] = fuzzy_and(a[i], b[i]);
    }
    return result;
}

/**
 * @brief Element-wise fuzzy OR between two tensors
 * @tparam ElementType Tensor element type
 * @tparam ShapeType Tensor shape type
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with element-wise OR results
 */
template <typename ElementType, typename ShapeType>
auto tensor_fuzzy_or(const common::types::TypedTensor<ElementType, ShapeType>& a,
                     const common::types::TypedTensor<ElementType, ShapeType>& b)
    -> common::types::TypedTensor<ElementType, ShapeType> {
    static_assert(std::is_same_v<ElementType, float>, "Fuzzy operations require float tensors");

    auto result = common::types::TypedTensor<ElementType, ShapeType>::zeros();
    for (std::size_t i = 0; i < a.size; ++i) {
        result[i] = fuzzy_or(a[i], b[i]);
    }
    return result;
}

/**
 * @brief Reduce tensor to single fuzzy value using universal quantification
 * @tparam ElementType Tensor element type
 * @tparam ShapeType Tensor shape type
 * @param tensor Input tensor
 * @return Single fuzzy value representing ∀ over all elements
 */
template <typename ElementType, typename ShapeType>
FuzzyValue tensor_fuzzy_forall(const common::types::TypedTensor<ElementType, ShapeType>& tensor) {
    static_assert(std::is_same_v<ElementType, float>, "Fuzzy operations require float tensors");

    FuzzyValue result = 1.0f;
    for (std::size_t i = 0; i < tensor.size; ++i) {
        result *= tensor[i];
    }
    return result;
}

/**
 * @brief Reduce tensor to single fuzzy value using existential quantification
 * @tparam ElementType Tensor element type
 * @tparam ShapeType Tensor shape type
 * @param tensor Input tensor
 * @return Single fuzzy value representing ∃ over all elements
 */
template <typename ElementType, typename ShapeType>
FuzzyValue tensor_fuzzy_exists(const common::types::TypedTensor<ElementType, ShapeType>& tensor) {
    static_assert(std::is_same_v<ElementType, float>, "Fuzzy operations require float tensors");

    FuzzyValue complement_product = 1.0f;
    for (std::size_t i = 0; i < tensor.size; ++i) {
        complement_product *= (1.0f - tensor[i]);
    }
    return 1.0f - complement_product;
}

// ================================================================================================
// FUZZY SET OPERATIONS
// ================================================================================================

/**
 * @brief Fuzzy membership function types
 */
enum class MembershipType {
    TRIANGULAR,   ///< Triangular membership function
    TRAPEZOIDAL,  ///< Trapezoidal membership function
    GAUSSIAN,     ///< Gaussian membership function
    SIGMOID       ///< Sigmoid membership function
};

/**
 * @brief Triangular fuzzy membership function
 * @param x Input value
 * @param a Left boundary
 * @param b Peak value
 * @param c Right boundary
 * @return Membership degree in [0,1]
 */
constexpr FuzzyValue triangular_membership(float x, float a, float b, float c) noexcept {
    if (x <= a || x >= c)
        return 0.0f;
    if (x <= b)
        return (x - a) / (b - a);
    return (c - x) / (c - b);
}

/**
 * @brief Gaussian fuzzy membership function
 * @param x Input value
 * @param center Center of the Gaussian
 * @param sigma Standard deviation
 * @return Membership degree in [0,1]
 */
inline FuzzyValue gaussian_membership(float x, float center, float sigma) noexcept {
    float diff = x - center;
    return std::exp(-(diff * diff) / (2.0f * sigma * sigma));
}

/**
 * @brief Sigmoid fuzzy membership function
 * @param x Input value
 * @param a Steepness parameter
 * @param c Center point
 * @return Membership degree in [0,1]
 */
inline FuzzyValue sigmoid_membership(float x, float a, float c) noexcept {
    return 1.0f / (1.0f + std::exp(-a * (x - c)));
}

}  // namespace inference_lab::engines::neuro_symbolic
