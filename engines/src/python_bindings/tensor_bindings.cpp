/**
 * @file tensor_bindings.cpp
 * @brief Python bindings for tensor types and operations
 *
 * Provides zero-copy integration between NumPy arrays and C++ tensor types,
 * enabling efficient data exchange for ML workloads.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Note: These headers will be implemented in Phase 2
// #include <common/src/containers.hpp>
// #include <common/src/ml_types.hpp>

namespace py = pybind11;

/**
 * @brief Placeholder tensor class for Phase 1 setup
 *
 * This will be replaced with the actual MLTensor implementation in Phase 2.
 * For now, it demonstrates the binding patterns.
 */
class PlaceholderTensor {
  private:
    std::vector<float> data_;
    std::vector<size_t> shape_;

  public:
    PlaceholderTensor(const std::vector<size_t>& shape) : shape_(shape) {
        size_t total_size = 1;
        for (auto dim : shape)
            total_size *= dim;
        data_.resize(total_size, 0.0f);
    }

    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    void fill(float value) { std::fill(data_.begin(), data_.end(), value); }
};

/**
 * @brief Convert NumPy array to C++ tensor (zero-copy when possible)
 *
 * @param input NumPy array
 * @return PlaceholderTensor with shared or copied data
 */
PlaceholderTensor numpy_to_tensor(py::array_t<float> input) {
    py::buffer_info buf = input.request();

    // Extract shape information
    std::vector<size_t> shape(buf.shape.begin(), buf.shape.end());
    PlaceholderTensor tensor(shape);

    // Copy data (in Phase 2, this will be zero-copy when possible)
    if (buf.c_contiguous) {
        std::memcpy(tensor.data(), buf.ptr, tensor.size() * sizeof(float));
    } else {
        // Handle non-contiguous arrays
        auto src = static_cast<float*>(buf.ptr);
        auto dst = tensor.data();
        for (size_t i = 0; i < tensor.size(); ++i) {
            dst[i] = src[i];  // Simplified - real implementation needs stride handling
        }
    }

    return tensor;
}

/**
 * @brief Convert C++ tensor to NumPy array (zero-copy when possible)
 *
 * @param tensor C++ tensor
 * @return NumPy array view or copy
 */
py::array_t<float> tensor_to_numpy(const PlaceholderTensor& tensor) {
    // Create NumPy array with tensor's memory
    // In Phase 2, this will be true zero-copy with proper lifetime management
    auto result =
        py::array_t<float>(tensor.shape(),  // Shape
                           tensor.data()    // Data pointer
                           // Note: In production, need py::cast(tensor) to keep object alive
        );

    return result;
}

/**
 * @brief Bind tensor types and operations to Python
 *
 * Creates Python classes for tensor operations with NumPy integration.
 */
void bind_tensor_types(py::module& m) {
    // Create a submodule for tensor types
    py::module tensor_module = m.def_submodule("tensor", "Tensor types and operations");

    // Bind the placeholder tensor class
    py::class_<PlaceholderTensor>(tensor_module, "Tensor")
        .def(py::init<const std::vector<size_t>&>(), "Create tensor with given shape")
        .def("shape", &PlaceholderTensor::shape, "Get tensor shape")
        .def("size", &PlaceholderTensor::size, "Get total number of elements")
        .def("fill", &PlaceholderTensor::fill, "Fill tensor with value")
        .def("to_numpy", &tensor_to_numpy, "Convert to NumPy array")
        .def_static("from_numpy", &numpy_to_tensor, "Create tensor from NumPy array")
        .def("__repr__", [](const PlaceholderTensor& tensor) {
            std::string shape_str = "[";
            for (size_t i = 0; i < tensor.shape().size(); ++i) {
                if (i > 0)
                    shape_str += ", ";
                shape_str += std::to_string(tensor.shape()[i]);
            }
            shape_str += "]";
            return "Tensor(shape=" + shape_str + ", size=" + std::to_string(tensor.size()) + ")";
        });

    // Utility functions for tensor operations
    tensor_module.def(
        "zeros",
        [](const std::vector<size_t>& shape) {
            PlaceholderTensor tensor(shape);
            tensor.fill(0.0f);
            return tensor;
        },
        "Create tensor filled with zeros");

    tensor_module.def(
        "ones",
        [](const std::vector<size_t>& shape) {
            PlaceholderTensor tensor(shape);
            tensor.fill(1.0f);
            return tensor;
        },
        "Create tensor filled with ones");

    // Demonstrate NumPy integration
    tensor_module.def(
        "test_numpy_roundtrip",
        [](py::array_t<float> input) {
            // Convert NumPy -> C++ -> NumPy
            auto cpp_tensor = numpy_to_tensor(input);
            return tensor_to_numpy(cpp_tensor);
        },
        "Test NumPy <-> C++ tensor conversion");

    // Performance testing utility
    tensor_module.def(
        "benchmark_conversion",
        [](py::array_t<float> input, int iterations) {
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iterations; ++i) {
                auto cpp_tensor = numpy_to_tensor(input);
                auto numpy_result = tensor_to_numpy(cpp_tensor);
                // Force evaluation
                py::cast<py::array_t<float>>(numpy_result);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            return duration.count() / static_cast<double>(iterations);
        },
        "Benchmark NumPy conversion performance (microseconds per conversion)");
}
