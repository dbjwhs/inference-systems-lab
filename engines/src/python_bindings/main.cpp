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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Forward declarations for binding functions
namespace py = pybind11;
using namespace pybind11::literals;

// Submodule binding functions
void bind_result_types(py::module& m);      // Implemented in result_bindings.cpp
void bind_tensor_types(py::module& m);      // Implemented in tensor_bindings.cpp
void bind_inference_engine(py::module& m);  // Implemented in inference_bindings.cpp
void bind_model_registry(py::module& m);    // Implemented in registry_bindings.cpp
// void bind_logging_system(py::module& m);  // Temporarily disabled

/**
 * @brief Main pybind11 module definition
 *
 * Creates the 'inference_lab' Python module with all C++ bindings.
 * This module provides Python access to the high-performance C++ inference engine.
 */
PYBIND11_MODULE(inference_lab, m) {
    m.doc() = "Inference Systems Laboratory - High-Performance ML Inference Engine";

    // Module metadata
    m.attr("__version__") = "0.1.0";
    m.attr("__author__") = "Inference Systems Laboratory";

    // Bind core types and utilities
    bind_result_types(m);
    bind_tensor_types(m);
    // bind_logging_system(m);  // Temporarily disabled

    // Bind inference engine interfaces
    bind_inference_engine(m);

    // Bind model registry for lifecycle management
    bind_model_registry(m);

    // Module-level utility functions
    m.def(
        "get_build_info",
        []() {
            return py::dict("version"_a = "0.1.0",
                            "build_type"_a = CMAKE_BUILD_TYPE,
                            "compiler"_a = CMAKE_CXX_COMPILER_ID,
                            "cpp_standard"_a = __cplusplus);
        },
        "Get build information for the C++ backend");
}
