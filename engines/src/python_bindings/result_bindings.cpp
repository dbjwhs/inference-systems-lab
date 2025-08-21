/**
 * @file result_bindings.cpp
 * @brief Python bindings for Result<T, E> error handling types
 *
 * Provides Python wrappers for the C++ Result<T, E> pattern, enabling
 * Rust-style error handling in Python with zero-overhead abstractions.
 */

#include <common/src/result.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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
    Result<T, E> result_;

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
 * @brief Bind Result<T, E> types to Python
 *
 * Creates Python classes for common Result<T, E> instantiations used
 * throughout the inference system.
 */
void bind_result_types(py::module& m) {
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
        [](const std::string& value) {
            return PyResult<std::string, std::string>(Result<std::string, std::string>::ok(value));
        },
        "Create a success result");

    result_module.def(
        "err",
        [](const std::string& error) {
            return PyResult<std::string, std::string>(Result<std::string, std::string>::err(error));
        },
        "Create an error result");

    // Helper functions
    result_module.def(
        "collect",
        [](const std::vector<PyResult<std::string, std::string>>& results) {
            std::vector<std::string> successes;
            for (const auto& result : results) {
                if (result.is_err()) {
                    return PyResult<std::vector<std::string>, std::string>(
                        Result<std::vector<std::string>, std::string>::err(*result.err()));
                }
                successes.push_back(*result.ok());
            }
            return PyResult<std::vector<std::string>, std::string>(
                Result<std::vector<std::string>, std::string>::ok(std::move(successes)));
        },
        "Collect a vector of results into a single result");
}
