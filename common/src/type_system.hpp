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
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace inference_lab::common::types {

// ================================================================================================
// COMPILE-TIME SHAPE SYSTEM
// ================================================================================================

/**
 * @brief Compile-time tensor shape representation
 * @tparam Dims... Dimension sizes (use 0 for dynamic dimensions)
 *
 * Represents tensor shapes at compile time, enabling shape verification
 * and automatic shape inference for tensor operations.
 */
template <std::size_t... Dims>
struct Shape {
    static constexpr std::size_t rank = sizeof...(Dims);
    static constexpr std::array<std::size_t, rank> dimensions = {Dims...};

    // Calculate total number of elements at compile time
    static constexpr std::size_t size = (Dims * ... * 1);

    // Check if shape has any dynamic dimensions
    static constexpr bool is_static = ((Dims != 0) && ...);

    // Get dimension at compile time
    template <std::size_t Index>
    static constexpr std::size_t dim() {
        static_assert(Index < rank, "Dimension index out of bounds");
        return dimensions[Index];
    }

    // Get shape as runtime vector for compatibility
    static auto to_vector() -> std::vector<std::size_t> {
        return std::vector<std::size_t>{Dims...};
    }
};

/**
 * @brief Dynamic shape for runtime-determined dimensions
 */
class DynamicShape {
  public:
    explicit DynamicShape(std::initializer_list<std::size_t> dims) : dimensions_(dims) {}
    explicit DynamicShape(const std::vector<std::size_t>& dims) : dimensions_(dims) {}

    auto rank() const -> std::size_t { return dimensions_.size(); }
    auto size() const -> std::size_t {
        return std::accumulate(
            dimensions_.begin(), dimensions_.end(), std::size_t{1}, std::multiplies<>());
    }
    auto dim(std::size_t index) const -> std::size_t { return dimensions_[index]; }
    auto dimensions() const -> const std::vector<std::size_t>& { return dimensions_; }

    auto operator==(const DynamicShape& other) const -> bool {
        return dimensions_ == other.dimensions_;
    }

  private:
    std::vector<std::size_t> dimensions_;
};

// ================================================================================================
// SHAPE OPERATIONS AND METAPROGRAMMING
// ================================================================================================

/**
 * @brief Concatenate two shapes along a new axis
 */
template <typename Shape1, typename Shape2>
struct ConcatShapes;

template <std::size_t... Dims1, std::size_t... Dims2>
struct ConcatShapes<Shape<Dims1...>, Shape<Dims2...>> {
    using type = Shape<Dims1..., Dims2...>;
};

/**
 * @brief Reshape operation with compile-time verification
 */
template <typename OriginalShape, typename NewShape>
struct CanReshape {
    static constexpr bool value = OriginalShape::size == NewShape::size;
};

/**
 * @brief Matrix multiplication shape inference
 */
template <typename LeftShape, typename RightShape>
struct MatMulShape;

template <std::size_t M, std::size_t K1, std::size_t K2, std::size_t N>
struct MatMulShape<Shape<M, K1>, Shape<K2, N>> {
    static_assert(K1 == K2, "Matrix dimensions must match for multiplication");
    using type = Shape<M, N>;
};

/**
 * @brief Broadcast shape inference for element-wise operations
 */
template <typename Shape1, typename Shape2>
struct BroadcastShape;

template <std::size_t... Dims1, std::size_t... Dims2>
struct BroadcastShape<Shape<Dims1...>, Shape<Dims2...>> {
    // Simplified broadcast rule: shapes must be identical for now
    static_assert(sizeof...(Dims1) == sizeof...(Dims2), "Broadcasting requires same rank");
    static_assert(((Dims1 == Dims2) && ...),
                  "Broadcasting not yet implemented for different shapes");
    using type = Shape<Dims1...>;
};

// ================================================================================================
// STRONG TYPE ALIASES FOR ML CONCEPTS
// ================================================================================================

/**
 * @brief Base class for strong type aliases
 * @tparam T Underlying type
 * @tparam Tag Unique tag type for disambiguation
 */
template <typename T, typename Tag>
class StrongType {
  public:
    explicit StrongType(const T& value) : value_(value) {}
    explicit StrongType(T&& value) : value_(std::move(value)) {}

    auto get() const -> const T& { return value_; }
    auto get() -> T& { return value_; }

    auto operator*() const -> const T& { return value_; }
    auto operator*() -> T& { return value_; }

    auto operator->() const -> const T* { return &value_; }
    auto operator->() -> T* { return &value_; }

  private:
    T value_;
};

// ML-specific type tags
struct WeightsTag {};
struct BiasTag {};
struct ActivationTag {};
struct GradientTag {};
struct LearningRateTag {};
struct MomentumTag {};

/**
 * @brief Strong types for different ML tensor categories
 */
template <typename ElementType, typename ShapeType = DynamicShape>
using Weights = StrongType<ElementType, WeightsTag>;

template <typename ElementType, typename ShapeType = DynamicShape>
using Bias = StrongType<ElementType, BiasTag>;

template <typename ElementType, typename ShapeType = DynamicShape>
using Activation = StrongType<ElementType, ActivationTag>;

template <typename ElementType, typename ShapeType = DynamicShape>
using Gradient = StrongType<ElementType, GradientTag>;

using LearningRate = StrongType<float, LearningRateTag>;
using Momentum = StrongType<float, MomentumTag>;

// ================================================================================================
// COMPILE-TIME TENSOR TYPE WITH SHAPE VERIFICATION
// ================================================================================================

/**
 * @brief Compile-time tensor with shape verification
 * @tparam ElementType Tensor element type (float, int8_t, etc.)
 * @tparam ShapeType Compile-time shape specification
 *
 * Provides a tensor type that enforces shape constraints at compile time,
 * preventing dimension mismatches and enabling aggressive optimizations.
 */
template <typename ElementType, typename ShapeType>
class TypedTensor {
  public:
    using element_type = ElementType;
    using shape_type = ShapeType;

    static constexpr std::size_t rank = ShapeType::rank;
    static constexpr std::size_t size = ShapeType::size;
    static constexpr bool is_static = ShapeType::is_static;

    /**
     * @brief Construct tensor with compile-time shape verification
     */
    explicit TypedTensor(std::unique_ptr<ElementType[]> data) : data_(std::move(data)) {
        static_assert(ShapeType::size > 0, "Tensor must have positive size");
    }

    /**
     * @brief Create tensor from raw data with size verification
     */
    static auto from_data(std::unique_ptr<ElementType[]> data) -> TypedTensor {
        return TypedTensor(std::move(data));
    }

    /**
     * @brief Create zero-initialized tensor
     */
    static auto zeros() -> TypedTensor {
        auto data = std::make_unique<ElementType[]>(size);
        std::fill_n(data.get(), size, ElementType{0});
        return TypedTensor(std::move(data));
    }

    /**
     * @brief Create tensor filled with specific value
     */
    static auto filled(const ElementType& value) -> TypedTensor {
        auto data = std::make_unique<ElementType[]>(size);
        std::fill_n(data.get(), size, value);
        return TypedTensor(std::move(data));
    }

    /**
     * @brief Copy constructor
     */
    TypedTensor(const TypedTensor& other) : data_(std::make_unique<ElementType[]>(size)) {
        std::copy_n(other.data_.get(), size, data_.get());
    }

    /**
     * @brief Copy assignment operator
     */
    TypedTensor& operator=(const TypedTensor& other) {
        if (this != &other) {
            std::copy_n(other.data_.get(), size, data_.get());
        }
        return *this;
    }

    /**
     * @brief Move constructor
     */
    TypedTensor(TypedTensor&& other) noexcept : data_(std::move(other.data_)) {}

    /**
     * @brief Move assignment operator
     */
    TypedTensor& operator=(TypedTensor&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }

    // Data access
    auto data() const -> const ElementType* { return data_.get(); }
    auto data() -> ElementType* { return data_.get(); }

    auto operator[](std::size_t index) const -> const ElementType& {
        assert(index < size);
        return data_[index];
    }

    auto operator[](std::size_t index) -> ElementType& {
        assert(index < size);
        return data_[index];
    }

    // Shape information
    static constexpr auto shape() -> ShapeType { return ShapeType{}; }

    /**
     * @brief Get dimension size at compile time
     */
    template <std::size_t Dim>
    static constexpr std::size_t dim() {
        static_assert(Dim < ShapeType::rank, "Dimension index out of bounds");
        return ShapeType::template dim<Dim>();
    }

    /**
     * @brief Reshape tensor with compile-time verification
     */
    template <typename NewShape>
    auto reshape() const -> TypedTensor<ElementType, NewShape> {
        static_assert(CanReshape<ShapeType, NewShape>::value,
                      "Cannot reshape: total size must remain the same");

        auto new_data = std::make_unique<ElementType[]>(NewShape::size);
        std::copy_n(data_.get(), size, new_data.get());
        return TypedTensor<ElementType, NewShape>::from_data(std::move(new_data));
    }

    /**
     * @brief Element-wise addition with broadcast checking
     */
    template <typename OtherShape>
    auto operator+(const TypedTensor<ElementType, OtherShape>& other) const
        -> TypedTensor<ElementType, typename BroadcastShape<ShapeType, OtherShape>::type> {
        using ResultShape = typename BroadcastShape<ShapeType, OtherShape>::type;
        auto result = TypedTensor<ElementType, ResultShape>::zeros();

        for (std::size_t i = 0; i < size; ++i) {
            result.data()[i] = data_[i] + other.data()[i];
        }

        return result;
    }

    /**
     * @brief Matrix multiplication with shape verification
     */
    template <typename OtherShape>
    auto matmul(const TypedTensor<ElementType, OtherShape>& other) const
        -> TypedTensor<ElementType, typename MatMulShape<ShapeType, OtherShape>::type> {
        using ResultShape = typename MatMulShape<ShapeType, OtherShape>::type;
        static_assert(rank == 2 && OtherShape::rank == 2,
                      "Matrix multiplication requires 2D tensors");

        constexpr std::size_t M = ShapeType::template dim<0>();
        constexpr std::size_t K = ShapeType::template dim<1>();
        constexpr std::size_t N = OtherShape::template dim<1>();

        auto result = TypedTensor<ElementType, ResultShape>::zeros();

        // Simple matrix multiplication (can be optimized with BLAS)
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                ElementType sum{0};
                for (std::size_t k = 0; k < K; ++k) {
                    sum += data_[i * K + k] * other.data()[k * N + j];
                }
                result.data()[i * N + j] = sum;
            }
        }

        return result;
    }

  private:
    std::unique_ptr<ElementType[]> data_;
};

// ================================================================================================
// LAYER TYPE DEFINITIONS
// ================================================================================================

/**
 * @brief Compile-time layer interface
 * @tparam InputShape Input tensor shape
 * @tparam OutputShape Output tensor shape
 */
template <typename InputShape, typename OutputShape>
class Layer {
  public:
    using input_shape = InputShape;
    using output_shape = OutputShape;

    virtual ~Layer() = default;

    /**
     * @brief Forward pass with compile-time shape verification
     */
    virtual auto forward(const TypedTensor<float, InputShape>& input)
        -> TypedTensor<float, OutputShape> = 0;
};

/**
 * @brief Dense (fully connected) layer with compile-time shapes
 */
template <std::size_t InputSize, std::size_t OutputSize>
class DenseLayer : public Layer<Shape<InputSize>, Shape<OutputSize>> {
  public:
    using input_shape = Shape<InputSize>;
    using output_shape = Shape<OutputSize>;
    using weight_shape = Shape<InputSize, OutputSize>;
    using bias_shape = Shape<OutputSize>;

    DenseLayer()
        : weights_(TypedTensor<float, weight_shape>::zeros()),
          bias_(TypedTensor<float, bias_shape>::zeros()) {}

    auto forward(const TypedTensor<float, input_shape>& input)
        -> TypedTensor<float, output_shape> override {
        // Matrix-vector multiplication: output = input * weights + bias
        auto result = TypedTensor<float, output_shape>::zeros();

        for (std::size_t out = 0; out < OutputSize; ++out) {
            float sum = 0;
            for (std::size_t in = 0; in < InputSize; ++in) {
                sum += input[in] * weights_[in * OutputSize + out];
            }
            result[out] = sum + bias_[out];
        }

        return result;
    }

    // Weight and bias accessors
    auto weights() -> TypedTensor<float, weight_shape>& { return weights_; }
    auto bias() -> TypedTensor<float, bias_shape>& { return bias_; }

  private:
    TypedTensor<float, weight_shape> weights_;
    TypedTensor<float, bias_shape> bias_;
};

/**
 * @brief ReLU activation layer
 */
template <typename ShapeType>
class ReLULayer : public Layer<ShapeType, ShapeType> {
  public:
    auto forward(const TypedTensor<float, ShapeType>& input)
        -> TypedTensor<float, ShapeType> override {
        auto result = TypedTensor<float, ShapeType>::zeros();
        for (std::size_t i = 0; i < ShapeType::size; ++i) {
            result[i] = std::max(0.0f, input[i]);
        }
        return result;
    }
};

// ================================================================================================
// TYPE TRAITS AND CONCEPTS
// ================================================================================================

/**
 * @brief Type trait to check if type is a TypedTensor
 */
template <typename T>
struct is_typed_tensor : std::false_type {};

template <typename ElementType, typename ShapeType>
struct is_typed_tensor<TypedTensor<ElementType, ShapeType>> : std::true_type {};

template <typename T>
inline constexpr bool is_typed_tensor_v = is_typed_tensor<T>::value;

/**
 * @brief Type trait to check if type is a compile-time shape
 */
template <typename T>
struct is_static_shape : std::false_type {};

template <std::size_t... Dims>
struct is_static_shape<Shape<Dims...>> : std::true_type {};

template <typename T>
inline constexpr bool is_static_shape_v = is_static_shape<T>::value;

/**
 * @brief Type trait to extract element type from tensor
 */
template <typename T>
struct tensor_element_type;

template <typename ElementType, typename ShapeType>
struct tensor_element_type<TypedTensor<ElementType, ShapeType>> {
    using type = ElementType;
};

template <typename T>
using tensor_element_type_t = typename tensor_element_type<T>::type;

/**
 * @brief Type trait to extract shape type from tensor
 */
template <typename T>
struct tensor_shape_type;

template <typename ElementType, typename ShapeType>
struct tensor_shape_type<TypedTensor<ElementType, ShapeType>> {
    using type = ShapeType;
};

template <typename T>
using tensor_shape_type_t = typename tensor_shape_type<T>::type;

// ================================================================================================
// COMPILE-TIME MODEL BUILDER
// ================================================================================================

/**
 * @brief Compile-time model composition with automatic shape inference
 */
template <typename... Layers>
class Sequential {
  public:
    static constexpr std::size_t num_layers = sizeof...(Layers);

    Sequential(Layers... layers) : layers_(std::move(layers)...) {}

    /**
     * @brief Forward pass through all layers with compile-time shape tracking
     */
    template <typename InputTensor>
    auto forward(const InputTensor& input) {
        return forward_impl(input, std::index_sequence_for<Layers...>{});
    }

  private:
    std::tuple<Layers...> layers_;

    template <typename InputTensor, std::size_t... Is>
    auto forward_impl(const InputTensor& input, std::index_sequence<Is...>) {
        return forward_recursive<0>(input);
    }

    template <std::size_t Index, typename CurrentTensor>
    auto forward_recursive(const CurrentTensor& tensor) {
        if constexpr (Index < num_layers) {
            auto& layer = std::get<Index>(layers_);
            auto output = layer.forward(tensor);
            return forward_recursive<Index + 1>(output);
        } else {
            return tensor;
        }
    }
};

/**
 * @brief Factory function for creating sequential models
 */
template <typename... Layers>
auto make_sequential(Layers&&... layers) -> Sequential<std::decay_t<Layers>...> {
    return Sequential<std::decay_t<Layers>...>(std::forward<Layers>(layers)...);
}

// ================================================================================================
// AUTOMATIC DIFFERENTIATION FRAMEWORK (PLACEHOLDER)
// ================================================================================================

/**
 * @brief Dual number for automatic differentiation
 * @tparam T Scalar type (typically float or double)
 */
template <typename T>
struct Dual {
    T value;
    T gradient;

    Dual(T v, T g = T{0}) : value(v), gradient(g) {}

    // Arithmetic operations with automatic gradient computation
    auto operator+(const Dual& other) const -> Dual {
        return Dual(value + other.value, gradient + other.gradient);
    }

    auto operator*(const Dual& other) const -> Dual {
        return Dual(value * other.value, gradient * other.value + value * other.gradient);
    }

    // Common functions
    friend auto relu(const Dual& x) -> Dual {
        return x.value > T{0} ? Dual(x.value, x.gradient) : Dual(T{0}, T{0});
    }

    friend auto sigmoid(const Dual& x) -> Dual {
        T s = T{1} / (T{1} + std::exp(-x.value));
        return Dual(s, x.gradient * s * (T{1} - s));
    }
};

using DualFloat = Dual<float>;
using DualDouble = Dual<double>;

}  // namespace inference_lab::common::types
