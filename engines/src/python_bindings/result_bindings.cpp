/**
 * @file result_bindings.cpp
 * @brief Python bindings for Result<T, E> error handling types
 *
 * Provides Python wrappers for the C++ Result<T, E> pattern, enabling
 * Rust-style error handling in Python with zero-overhead abstractions.
 */

#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include the actual Result<T, E> implementation
#include "../../common/src/result.hpp"

namespace py = pybind11;
using namespace inference_lab::common;

/**
 * @brief Python wrapper for Result<T, E> types
 *
 * Provides a Pythonic interface to the C++ Result<T, E> pattern while
 * maintaining the same error handling semantics.
 *
 * @tparam T The success type
 * @tparam E The error type
 */
template <typename T, typename E>
class PyResult {
  private:
    Result<T, E> result_{};

  public:
    explicit PyResult(Result<T, E> result) : result_(std::move(result)) {}

    /**
     * @brief Check if the result contains a success value
     * @return true if the result is Ok, false if Err
     */
    bool is_ok() const noexcept { return result_.is_ok(); }

    /**
     * @brief Check if the result contains an error value
     * @return true if the result is Err, false if Ok
     */
    bool is_err() const noexcept { return result_.is_err(); }

    /**
     * @brief Unwrap the success value (throws if error)
     * @return The success value
     * @throws py::value_error if the result contains an error
     */
    T unwrap() {
        if (result_.is_err()) {
            auto error_msg = format_error(result_.unwrap_err());
            throw py::value_error("Result contains error: " + error_msg);
        }
        return result_.unwrap();
    }

    /**
     * @brief Unwrap the error value (throws if success)
     * @return The error value
     * @throws py::value_error if the result contains a success value
     */
    E unwrap_err() {
        if (result_.is_ok()) {
            throw py::value_error("Result contains success value, not error");
        }
        return result_.unwrap_err();
    }

    /**
     * @brief Get the success value as optional
     * @return Optional containing the success value, or None if error
     */
    std::optional<T> ok() const { return result_.ok(); }

    /**
     * @brief Get the error value as optional
     * @return Optional containing the error value, or None if success
     */
    std::optional<E> err() const { return result_.err(); }

    /**
     * @brief Get success value or return default
     * @param default_value Value to return if result is error
     * @return Success value or default
     */
    T unwrap_or(const T& default_value) const { return result_.unwrap_or(default_value); }

  private:
    /**
     * @brief Format error for display (to be specialized for specific error types)
     */
    std::string format_error(const E& error) const {
        if constexpr (std::is_convertible_v<E, std::string>) {
            return std::string(error);
        } else {
            return "Error occurred";
        }
    }
};

/**
 * @brief Exception translation utilities for cross-language error handling
 */
namespace exception_translation {

/**
 * @brief Convert C++ exceptions to appropriate Python exceptions
 *
 * This function provides intelligent mapping from C++ exception types to
 * Python exception types, improving the Python user experience.
 */
void translate_cpp_exception(const std::exception& e) {
    std::string msg = e.what();

    // Map common C++ exceptions to appropriate Python exceptions
    if (msg.find("invalid") != std::string::npos || msg.find("Invalid") != std::string::npos) {
        throw py::value_error(msg);
    } else if (msg.find("not found") != std::string::npos ||
               msg.find("missing") != std::string::npos) {
        throw py::key_error(msg);
    } else if (msg.find("type") != std::string::npos || msg.find("cast") != std::string::npos) {
        throw py::type_error(msg);
    } else if (msg.find("index") != std::string::npos || msg.find("range") != std::string::npos) {
        throw py::index_error(msg);
    } else if (msg.find("tensor") != std::string::npos || msg.find("numpy") != std::string::npos) {
        throw py::value_error("Tensor operation failed: " + msg);
    } else if (msg.find("model") != std::string::npos) {
        throw py::runtime_error("Model operation failed: " + msg);
    } else if (msg.find("inference") != std::string::npos) {
        throw py::runtime_error("Inference operation failed: " + msg);
    } else {
        throw py::runtime_error(msg);
    }
}

/**
 * @brief Safe wrapper to execute C++ operations and handle Result<T, E>
 *
 * @tparam T Success type
 * @tparam E Error type
 * @param operation Lambda returning Result<T, E>
 * @return T Success value
 * @throws Python exception if Result contains error
 */
template <typename T, typename E>
T safe_unwrap(std::function<Result<T, E>()> operation) {
    try {
        auto result = operation();
        if (result.is_ok()) {
            return result.unwrap();
        } else {
            // Convert error to string and throw appropriate Python exception
            std::string error_msg;
            if constexpr (std::is_same_v<E, std::string>) {
                error_msg = result.unwrap_err();
            } else if constexpr (std::is_convertible_v<E, std::string>) {
                error_msg = std::string(result.unwrap_err());
            } else {
                error_msg = "Operation failed with unknown error type";
            }

            // Create a temporary exception to use translation logic
            std::runtime_error temp_exception(error_msg);
            translate_cpp_exception(temp_exception);

            // Fallback (should not reach here due to throw in translate_cpp_exception)
            throw py::runtime_error(error_msg);
        }
    } catch (const std::exception& e) {
        translate_cpp_exception(e);
        // Fallback (should not reach here)
        throw py::runtime_error(std::string("Unexpected error: ") + e.what());
    }
}

}  // namespace exception_translation

/**
 * @brief Enhanced Result wrapper with automatic exception translation
 */
template <typename T, typename E>
class SafeResult {
  private:
    Result<T, E> result_;

  public:
    explicit SafeResult(Result<T, E> result) : result_(std::move(result)) {}

    /**
     * @brief Get the value or raise appropriate Python exception
     */
    T get() {
        return exception_translation::safe_unwrap<T, E>([this]() { return result_; });
    }

    /**
     * @brief Check if operation was successful
     */
    bool is_success() const { return result_.is_ok(); }

    /**
     * @brief Get error message without throwing (returns empty string if success)
     */
    std::string get_error_message() const {
        if (result_.is_err()) {
            if constexpr (std::is_same_v<E, std::string>) {
                return result_.unwrap_err();
            } else if constexpr (std::is_convertible_v<E, std::string>) {
                return std::string(result_.unwrap_err());
            } else {
                return "Error details unavailable";
            }
        }
        return "";
    }

    /**
     * @brief Get the underlying Result for advanced operations
     */
    const Result<T, E>& get_result() const { return result_; }
};

/**
 * @brief Bind Result<T, E> types to Python
 *
 * Creates Python classes for common Result<T, E> instantiations used
 * throughout the inference system.
 */
static void bind_result_types(py::module& m) {
    // Create a submodule for result types
    py::module result_module = m.def_submodule("result", "Result<T, E> error handling types");

    // Bind common Result<T, E> instantiations

    // String result for configuration and file operations
    py::class_<PyResult<std::string, std::string>>(result_module, "StringResult")
        .def("is_ok", &PyResult<std::string, std::string>::is_ok)
        .def("is_err", &PyResult<std::string, std::string>::is_err)
        .def("unwrap", &PyResult<std::string, std::string>::unwrap)
        .def("unwrap_err", &PyResult<std::string, std::string>::unwrap_err)
        .def("ok", &PyResult<std::string, std::string>::ok)
        .def("err", &PyResult<std::string, std::string>::err)
        .def("unwrap_or", &PyResult<std::string, std::string>::unwrap_or)
        .def("__bool__", &PyResult<std::string, std::string>::is_ok)
        .def("__repr__", [](const PyResult<std::string, std::string>& self) {
            if (self.is_ok()) {
                return "Ok(\"" + *self.ok() + "\")";
            } else {
                return "Err(\"" + *self.err() + "\")";
            }
        });

    // Factory functions for creating results from C++
    result_module.def(
        "ok",
        [](const std::string& value) { return PyResult<std::string, std::string>(Ok(value)); },
        "Create a success result");

    result_module.def(
        "err",
        [](const std::string& error) { return PyResult<std::string, std::string>(Err(error)); },
        "Create an error result");

    // Helper functions
    result_module.def(
        "collect",
        [](const std::vector<PyResult<std::string, std::string>>& results) {
            std::vector<std::string> successes;
            for (const auto& result : results) {
                if (result.is_err()) {
                    return PyResult<std::vector<std::string>, std::string>(Err(*result.err()));
                }
                successes.push_back(*result.ok());
            }
            return PyResult<std::vector<std::string>, std::string>(Ok(std::move(successes)));
        },
        "Collect a vector of results into a single result");

    // Bind SafeResult wrapper for automatic exception translation
    py::class_<SafeResult<std::string, std::string>>(result_module, "SafeStringResult")
        .def("get",
             &SafeResult<std::string, std::string>::get,
             "Get value or raise Python exception")
        .def("is_success",
             &SafeResult<std::string, std::string>::is_success,
             "Check if operation succeeded")
        .def("get_error_message",
             &SafeResult<std::string, std::string>::get_error_message,
             "Get error message without throwing")
        .def("__bool__", &SafeResult<std::string, std::string>::is_success)
        .def("__repr__", [](const SafeResult<std::string, std::string>& self) {
            if (self.is_success()) {
                return "SafeResult(success=True)";
            } else {
                return "SafeResult(success=False, error=\"" + self.get_error_message() + "\")";
            }
        });

    py::class_<SafeResult<int, std::string>>(result_module, "SafeIntResult")
        .def("get", &SafeResult<int, std::string>::get, "Get value or raise Python exception")
        .def(
            "is_success", &SafeResult<int, std::string>::is_success, "Check if operation succeeded")
        .def("get_error_message",
             &SafeResult<int, std::string>::get_error_message,
             "Get error message without throwing")
        .def("__bool__", &SafeResult<int, std::string>::is_success)
        .def("__repr__", [](const SafeResult<int, std::string>& self) {
            if (self.is_success()) {
                return "SafeResult(success=True)";
            } else {
                return "SafeResult(success=False, error=\"" + self.get_error_message() + "\")";
            }
        });

    // Exception translation utilities
    result_module.def(
        "translate_exception",
        [](const std::string& error_msg) {
            std::runtime_error error(error_msg);
            exception_translation::translate_cpp_exception(error);
        },
        "Translate C++ error message to appropriate Python exception");

    result_module.def(
        "safe_execute",
        [](py::function func) -> SafeResult<py::object, std::string> {
            try {
                py::object result = func();
                return SafeResult<py::object, std::string>(Ok(result));
            } catch (const py::error_already_set& e) {
                return SafeResult<py::object, std::string>(Err(std::string(e.what())));
            } catch (const std::exception& e) {
                return SafeResult<py::object, std::string>(Err(std::string(e.what())));
            }
        },
        "Safely execute a Python function and wrap result");

    // Register global exception translators
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::out_of_range& e) {
            PyErr_SetString(PyExc_IndexError, e.what());
        } catch (const std::runtime_error& e) {
            // Use our intelligent exception translation
            try {
                exception_translation::translate_cpp_exception(e);
            } catch (const py::error_already_set&) {
                // Exception was successfully translated, let it propagate
                return;
            }
            // Fallback to runtime error
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const std::logic_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });
}
