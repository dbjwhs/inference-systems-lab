/**
 * @file logging_bindings.cpp
 * @brief Python bindings for the logging system
 *
 * Bridges the C++ structured logging system with Python logging,
 * enabling unified log handling across the Python-C++ boundary.
 */

#include <common/src/logging.hpp>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/**
 * @brief Python callback handler for C++ log messages
 *
 * Enables forwarding C++ log messages to Python logging infrastructure.
 */
class PythonLogHandler {
  private:
    static py::function python_logger_;
    static bool handler_installed_;

  public:
    /**
     * @brief Set Python logger function to receive C++ log messages
     *
     * @param logger Python function(level: int, message: str) -> None
     */
    static void set_python_logger(py::function logger) {
        python_logger_ = logger;
        handler_installed_ = true;
    }

    /**
     * @brief Forward C++ log message to Python
     *
     * @param level Log level (maps to Python logging levels)
     * @param message Formatted log message
     */
    static void log_to_python(LogLevel level, const std::string& message) {
        if (handler_installed_ && !python_logger_.is_none()) {
            try {
                py::gil_scoped_acquire acquire;
                python_logger_(static_cast<int>(level), message);
            } catch (const std::exception& e) {
                // Fallback to stderr if Python logging fails
                std::cerr << "[C++ LOG] " << message << std::endl;
            }
        }
    }

    /**
     * @brief Remove Python logger handler
     */
    static void clear_python_logger() {
        python_logger_ = py::function();
        handler_installed_ = false;
    }

    static bool has_python_handler() { return handler_installed_; }
};

// Static member definitions
py::function PythonLogHandler::python_logger_;
bool PythonLogHandler::handler_installed_ = false;

/**
 * @brief Python wrapper for C++ logger
 *
 * Provides a Python interface to the C++ logging system while maintaining
 * performance and thread safety.
 */
class PyLogger {
  private:
    std::string component_name_;

  public:
    explicit PyLogger(const std::string& component_name) : component_name_(component_name) {}

    void debug(const std::string& message) {
        LOG_DEBUG_PRINT_COMPONENT(component_name_, "{}", message);
        PythonLogHandler::log_to_python(LogLevel::DEBUG, "[" + component_name_ + "] " + message);
    }

    void info(const std::string& message) {
        LOG_INFO_PRINT_COMPONENT(component_name_, "{}", message);
        PythonLogHandler::log_to_python(LogLevel::INFO, "[" + component_name_ + "] " + message);
    }

    void warning(const std::string& message) {
        LOG_WARNING_PRINT_COMPONENT(component_name_, "{}", message);
        PythonLogHandler::log_to_python(LogLevel::WARNING, "[" + component_name_ + "] " + message);
    }

    void error(const std::string& message) {
        LOG_ERROR_PRINT_COMPONENT(component_name_, "{}", message);
        PythonLogHandler::log_to_python(LogLevel::ERROR, "[" + component_name_ + "] " + message);
    }

    void critical(const std::string& message) {
        LOG_CRITICAL_PRINT_COMPONENT(component_name_, "{}", message);
        PythonLogHandler::log_to_python(LogLevel::CRITICAL, "[" + component_name_ + "] " + message);
    }

    const std::string& component_name() const { return component_name_; }
};

/**
 * @brief Convert Python logging level to C++ LogLevel
 */
LogLevel python_to_cpp_log_level(int python_level) {
    // Python logging levels: DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
    if (python_level <= 10)
        return LogLevel::DEBUG;
    if (python_level <= 20)
        return LogLevel::INFO;
    if (python_level <= 30)
        return LogLevel::WARNING;
    if (python_level <= 40)
        return LogLevel::ERROR;
    return LogLevel::CRITICAL;
}

/**
 * @brief Convert C++ LogLevel to Python logging level
 */
int cpp_to_python_log_level(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:
            return 10;
        case LogLevel::INFO:
            return 20;
        case LogLevel::WARNING:
            return 30;
        case LogLevel::ERROR:
            return 40;
        case LogLevel::CRITICAL:
            return 50;
        default:
            return 20;
    }
}

/**
 * @brief Bind logging system to Python
 *
 * Creates Python classes for unified logging across C++ and Python components.
 */
void bind_logging_system(py::module& m) {
    // Create a submodule for logging
    py::module logging_module = m.def_submodule("logging", "Unified logging system");

    // Bind LogLevel enum
    py::enum_<LogLevel>(logging_module, "LogLevel")
        .value("DEBUG", LogLevel::DEBUG)
        .value("INFO", LogLevel::INFO)
        .value("WARNING", LogLevel::WARNING)
        .value("ERROR", LogLevel::ERROR)
        .value("CRITICAL", LogLevel::CRITICAL)
        .export_values();

    // Bind Python logger wrapper
    py::class_<PyLogger>(logging_module, "Logger")
        .def(py::init<const std::string&>(), "Create logger for component")
        .def("debug", &PyLogger::debug)
        .def("info", &PyLogger::info)
        .def("warning", &PyLogger::warning)
        .def("error", &PyLogger::error)
        .def("critical", &PyLogger::critical)
        .def("component_name", &PyLogger::component_name)
        .def("__repr__", [](const PyLogger& logger) {
            return "Logger(component=\"" + logger.component_name() + "\")";
        });

    // Bind log handler functions
    logging_module.def("set_python_handler",
                       &PythonLogHandler::set_python_logger,
                       "Set Python function to receive C++ log messages");

    logging_module.def("clear_python_handler",
                       &PythonLogHandler::clear_python_logger,
                       "Remove Python log handler");

    logging_module.def("has_python_handler",
                       &PythonLogHandler::has_python_handler,
                       "Check if Python log handler is installed");

    // Utility functions
    logging_module.def("cpp_to_python_level",
                       &cpp_to_python_log_level,
                       "Convert C++ LogLevel to Python logging level");

    logging_module.def("python_to_cpp_level",
                       &python_to_cpp_log_level,
                       "Convert Python logging level to C++ LogLevel");

    // Global logging functions (convenience)
    logging_module.def(
        "log_debug",
        [](const std::string& component, const std::string& message) {
            LOG_DEBUG_PRINT_COMPONENT(component, "{}", message);
        },
        "Log debug message from Python");

    logging_module.def(
        "log_info",
        [](const std::string& component, const std::string& message) {
            LOG_INFO_PRINT_COMPONENT(component, "{}", message);
        },
        "Log info message from Python");

    logging_module.def(
        "log_warning",
        [](const std::string& component, const std::string& message) {
            LOG_WARNING_PRINT_COMPONENT(component, "{}", message);
        },
        "Log warning message from Python");

    logging_module.def(
        "log_error",
        [](const std::string& component, const std::string& message) {
            LOG_ERROR_PRINT_COMPONENT(component, "{}", message);
        },
        "Log error message from Python");

    logging_module.def(
        "log_critical",
        [](const std::string& component, const std::string& message) {
            LOG_CRITICAL_PRINT_COMPONENT(component, "{}", message);
        },
        "Log critical message from Python");

    // Performance testing
    logging_module.def(
        "benchmark_logging",
        [](int iterations) {
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iterations; ++i) {
                LOG_INFO_PRINT("Benchmark message {}", i);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            return duration.count() / static_cast<double>(iterations);
        },
        "Benchmark logging performance (microseconds per log)");
}
