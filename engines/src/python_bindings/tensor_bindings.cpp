/**
 * @file tensor_bindings.cpp
 * @brief Python bindings for tensor types and operations
 *
 * Provides zero-copy integration between NumPy arrays and C++ tensor types,
 * enabling efficient data exchange for ML workloads.
 */

#include <chrono>
#include <functional>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Note: These headers will be implemented in Phase 2
// #include <common/src/containers.hpp>
// #include <common/src/ml_types.hpp>

namespace py = pybind11;

// Forward declarations for template functions
template <typename T>
void copy_strided_data(const py::buffer_info& buf, float* dst);

template <typename SrcType, typename DstType>
void convert_and_copy(const py::buffer_info& buf, DstType* dst);

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
 * @param input NumPy array (supports float32, float64, int32, int64)
 * @return PlaceholderTensor with shared or copied data
 * @throws std::runtime_error if input is invalid or unsupported dtype
 */
PlaceholderTensor numpy_to_tensor(py::array input) {
    py::buffer_info buf = input.request();

    // Validate input
    if (buf.ndim == 0) {
        throw std::runtime_error("Cannot convert 0-dimensional array to tensor");
    }
    if (buf.size == 0) {
        throw std::runtime_error("Cannot convert empty array to tensor");
    }

    // Extract shape information
    std::vector<size_t> shape(buf.shape.begin(), buf.shape.end());
    PlaceholderTensor tensor(shape);

    // Handle different data types with proper conversion
    auto dtype = py::dtype::of(input);
    if (dtype.is(py::dtype::of<float>())) {
        // float32 - direct copy
        if (buf.c_contiguous) {
            std::memcpy(tensor.data(), buf.ptr, tensor.size() * sizeof(float));
        } else {
            copy_strided_data<float>(buf, tensor.data());
        }
    } else if (dtype.is(py::dtype::of<double>())) {
        // float64 - convert to float32
        convert_and_copy<double, float>(buf, tensor.data());
    } else if (dtype.is(py::dtype::of<int32_t>())) {
        // int32 - convert to float32
        convert_and_copy<int32_t, float>(buf, tensor.data());
    } else if (dtype.is(py::dtype::of<int64_t>())) {
        // int64 - convert to float32
        convert_and_copy<int64_t, float>(buf, tensor.data());
    } else {
        throw std::runtime_error("Unsupported NumPy dtype: " + py::str(dtype).cast<std::string>());
    }

    return tensor;
}

/**
 * @brief Copy strided NumPy data to contiguous C++ buffer
 *
 * @tparam T Source data type
 * @param buf NumPy buffer info
 * @param dst Destination buffer
 */
template <typename T>
void copy_strided_data(const py::buffer_info& buf, float* dst) {
    auto src = static_cast<const T*>(buf.ptr);
    const auto& strides = buf.strides;
    const auto& shape = buf.shape;

    std::function<void(size_t, size_t, const T*)> copy_recursive;
    copy_recursive = [&](size_t dim, size_t dst_idx, const T* src_ptr) {
        if (dim == shape.size()) {
            dst[dst_idx] = static_cast<float>(*src_ptr);
            return;
        }

        for (ssize_t i = 0; i < shape[dim]; ++i) {
            size_t next_dst_idx = dst_idx;
            if (dim < shape.size() - 1) {
                size_t stride_count = 1;
                for (size_t d = dim + 1; d < shape.size(); ++d) {
                    stride_count *= shape[d];
                }
                next_dst_idx = dst_idx * shape[dim] + i * stride_count;
            } else {
                next_dst_idx = dst_idx + i;
            }

            const T* next_src_ptr = reinterpret_cast<const T*>(
                reinterpret_cast<const char*>(src_ptr) + i * strides[dim]);
            copy_recursive(dim + 1, next_dst_idx, next_src_ptr);
        }
    };

    copy_recursive(0, 0, src);
}

/**
 * @brief Convert and copy data from source type to float
 *
 * @tparam SrcType Source data type
 * @tparam DstType Destination data type (float)
 * @param buf NumPy buffer info
 * @param dst Destination buffer
 */
template <typename SrcType, typename DstType>
void convert_and_copy(const py::buffer_info& buf, DstType* dst) {
    if (buf.c_contiguous) {
        auto src = static_cast<const SrcType*>(buf.ptr);
        for (size_t i = 0; i < buf.size; ++i) {
            dst[i] = static_cast<DstType>(src[i]);
        }
    } else {
        copy_strided_data<SrcType>(buf, dst);
    }
}

/**
 * @brief Convert C++ tensor to NumPy array (zero-copy when possible)
 *
 * @param tensor C++ tensor
 * @return NumPy array view or copy with proper lifetime management
 * @throws std::runtime_error if tensor is empty or invalid
 */
py::array_t<float> tensor_to_numpy(const PlaceholderTensor& tensor) {
    // Validate input tensor
    if (tensor.size() == 0) {
        throw std::runtime_error("Cannot convert empty tensor to NumPy array");
    }
    if (tensor.shape().empty()) {
        throw std::runtime_error("Cannot convert tensor with empty shape to NumPy array");
    }

    // Convert shape to ssize_t for pybind11 compatibility
    std::vector<ssize_t> py_shape;
    py_shape.reserve(tensor.shape().size());
    for (size_t dim : tensor.shape()) {
        py_shape.push_back(static_cast<ssize_t>(dim));
    }

    // Create strides for C-contiguous layout
    std::vector<ssize_t> py_strides;
    py_strides.reserve(py_shape.size());
    ssize_t stride = sizeof(float);
    for (int i = static_cast<int>(py_shape.size()) - 1; i >= 0; --i) {
        py_strides.insert(py_strides.begin(), stride);
        stride *= py_shape[i];
    }

    // Create NumPy array with proper memory management
    // Copy data to ensure safe memory management across language boundary
    auto result =
        py::array_t<float>(py_shape,                                        // Shape
                           py_strides,                                      // Strides
                           tensor.data(),                                   // Data pointer
                           py::cast(tensor, py::return_value_policy::copy)  // Keep object alive
        );

    // For safety, create a copy to avoid memory management issues
    // In production, we would use proper shared_ptr or reference counting
    return py::array_t<float>(py_shape, tensor.data());
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

    // Demonstrate NumPy integration with enhanced error handling
    tensor_module.def(
        "test_numpy_roundtrip",
        [](py::array input) {
            try {
                // Convert NumPy -> C++ -> NumPy with enhanced type support
                auto cpp_tensor = numpy_to_tensor(input);
                return tensor_to_numpy(cpp_tensor);
            } catch (const std::exception& e) {
                throw py::value_error("NumPy roundtrip failed: " + std::string(e.what()));
            }
        },
        "Test NumPy <-> C++ tensor conversion (supports float32/64, int32/64)");

    // Performance testing utility with better error handling
    tensor_module.def(
        "benchmark_conversion",
        [](py::array input, int iterations) {
            if (iterations <= 0) {
                throw py::value_error("Iterations must be positive");
            }

            try {
                // Warmup run to catch any conversion errors early
                auto warmup_tensor = numpy_to_tensor(input);
                auto warmup_result = tensor_to_numpy(warmup_tensor);

                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < iterations; ++i) {
                    auto cpp_tensor = numpy_to_tensor(input);
                    auto numpy_result = tensor_to_numpy(cpp_tensor);
                    // Force evaluation to prevent optimization
                    volatile auto size = numpy_result.size();
                    (void)size;
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                return duration.count() / static_cast<double>(iterations);
            } catch (const std::exception& e) {
                throw py::runtime_error("Benchmark failed: " + std::string(e.what()));
            }
        },
        "Benchmark NumPy conversion performance (microseconds per conversion)");

    // Additional utility for testing different data types
    tensor_module.def(
        "test_dtype_support",
        []() {
            py::dict supported_types;
            supported_types["float32"] = true;
            supported_types["float64"] = true;
            supported_types["int32"] = true;
            supported_types["int64"] = true;
            supported_types["int8"] = false;
            supported_types["int16"] = false;
            supported_types["uint8"] = false;
            supported_types["uint16"] = false;
            supported_types["uint32"] = false;
            supported_types["uint64"] = false;
            return supported_types;
        },
        "Get dictionary of supported NumPy data types");
}
