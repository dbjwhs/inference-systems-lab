/**
 * @file main.cpp
 * @brief Main pybind11 module entry point for Inference Systems Laboratory
 *
 * This file serves as the primary entry point for the Python bindings of the
 * Inference Systems Laboratory. It aggregates all bindings from submodules.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Forward declarations for binding functions
namespace py = pybind11;

// Submodule binding functions (to be implemented)
void bind_result_types(py::module& m);
void bind_tensor_types(py::module& m);
void bind_inference_engine(py::module& m);
void bind_logging_system(py::module& m);

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
    bind_logging_system(m);

    // Bind inference engine interfaces
    bind_inference_engine(m);

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
