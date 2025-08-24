/**
 * @file registry_bindings.cpp
 * @brief Pybind11 bindings for ModelRegistry class
 *
 * Provides Python access to the C++ ModelRegistry implementation for unified
 * model lifecycle management across Python and C++ code.
 */

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "../model_registry.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace inference_lab::engines;

/**
 * @brief Bind ModelRegistry types and functions to Python
 *
 * This function creates Python bindings for:
 * - ModelStatus enum
 * - ModelType enum
 * - InferenceBackend enum (already bound in tensor_bindings)
 * - ModelInfo structure
 * - PerformanceMetrics structure
 * - ValidationResult structure
 * - ModelRegistry class with all methods
 * - RegistryError enum and exception handling
 */
void bind_model_registry(py::module& m) {
    // Bind ModelStatus enum
    py::enum_<ModelStatus>(m, "ModelStatus")
        .value("DEVELOPMENT", ModelStatus::DEVELOPMENT)
        .value("STAGING", ModelStatus::STAGING)
        .value("PRODUCTION", ModelStatus::PRODUCTION)
        .value("DEPRECATED", ModelStatus::DEPRECATED)
        .value("ARCHIVED", ModelStatus::ARCHIVED)
        .export_values();

    // Bind ModelType enum
    py::enum_<ModelType>(m, "ModelType")
        .value("ONNX", ModelType::ONNX)
        .value("TENSORRT", ModelType::TENSORRT)
        .value("PYTORCH", ModelType::PYTORCH)
        .value("RULE_BASED", ModelType::RULE_BASED)
        .value("OTHER", ModelType::OTHER)
        .export_values();

    // Bind RegistryError enum
    py::enum_<RegistryError>(m, "RegistryError")
        .value("DATABASE_CONNECTION_FAILED", RegistryError::DATABASE_CONNECTION_FAILED)
        .value("MODEL_NOT_FOUND", RegistryError::MODEL_NOT_FOUND)
        .value("MODEL_ALREADY_EXISTS", RegistryError::MODEL_ALREADY_EXISTS)
        .value("INVALID_MODEL_VERSION", RegistryError::INVALID_MODEL_VERSION)
        .value("SQL_EXECUTION_ERROR", RegistryError::SQL_EXECUTION_ERROR)
        .value("SCHEMA_MISMATCH", RegistryError::SCHEMA_MISMATCH)
        .value("PERMISSION_DENIED", RegistryError::PERMISSION_DENIED)
        .export_values();

    // Bind ModelInfo structure
    py::class_<ModelInfo>(m, "ModelInfo")
        .def(py::init<>())
        .def_readwrite("model_id", &ModelInfo::model_id)
        .def_readwrite("model_name", &ModelInfo::model_name)
        .def_readwrite("model_version", &ModelInfo::model_version)
        .def_readwrite("model_type", &ModelInfo::model_type)
        .def_readwrite("backend", &ModelInfo::backend)
        .def_readwrite("model_path", &ModelInfo::model_path)
        .def_readwrite("config_path", &ModelInfo::config_path)
        .def_readwrite("metadata_path", &ModelInfo::metadata_path)
        .def_readwrite("description", &ModelInfo::description)
        .def_readwrite("author", &ModelInfo::author)
        .def_readwrite("model_size_bytes", &ModelInfo::model_size_bytes)
        .def_readwrite("estimated_memory_mb", &ModelInfo::estimated_memory_mb)
        .def_readwrite("avg_inference_time_ms", &ModelInfo::avg_inference_time_ms)
        .def_readwrite("status", &ModelInfo::status)
        .def_readwrite("created_at", &ModelInfo::created_at)
        .def_readwrite("updated_at", &ModelInfo::updated_at)
        .def_readwrite("deployed_at", &ModelInfo::deployed_at)
        .def_readwrite("input_shape", &ModelInfo::input_shape)
        .def_readwrite("output_shape", &ModelInfo::output_shape)
        .def_readwrite("tags", &ModelInfo::tags)
        .def_readwrite("custom_metadata", &ModelInfo::custom_metadata);

    // Bind PerformanceMetrics structure
    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def(py::init<>())
        .def_readwrite("metric_id", &PerformanceMetrics::metric_id)
        .def_readwrite("model_id", &PerformanceMetrics::model_id)
        .def_readwrite("inference_time_ms", &PerformanceMetrics::inference_time_ms)
        .def_readwrite("preprocessing_time_ms", &PerformanceMetrics::preprocessing_time_ms)
        .def_readwrite("postprocessing_time_ms", &PerformanceMetrics::postprocessing_time_ms)
        .def_readwrite("memory_usage_mb", &PerformanceMetrics::memory_usage_mb)
        .def_readwrite("gpu_memory_mb", &PerformanceMetrics::gpu_memory_mb)
        .def_readwrite("throughput_qps", &PerformanceMetrics::throughput_qps)
        .def_readwrite("batch_size", &PerformanceMetrics::batch_size)
        .def_readwrite("recorded_at", &PerformanceMetrics::recorded_at);

    // Bind ValidationResult structure
    py::class_<ValidationResult>(m, "ValidationResult")
        .def(py::init<>())
        .def_readwrite("validation_id", &ValidationResult::validation_id)
        .def_readwrite("model_id", &ValidationResult::model_id)
        .def_readwrite("validation_type", &ValidationResult::validation_type)
        .def_readwrite("validation_status", &ValidationResult::validation_status)
        .def_readwrite("accuracy_score", &ValidationResult::accuracy_score)
        .def_readwrite("precision_score", &ValidationResult::precision_score)
        .def_readwrite("recall_score", &ValidationResult::recall_score)
        .def_readwrite("f1_score", &ValidationResult::f1_score)
        .def_readwrite("custom_metrics", &ValidationResult::custom_metrics)
        .def_readwrite("validated_at", &ValidationResult::validated_at);

    // Bind ModelRegistry class
    py::class_<ModelRegistry>(m, "ModelRegistry")
        // Factory method for creating registry instances
        .def_static(
            "create",
            [](const std::string& db_path) -> std::unique_ptr<ModelRegistry> {
                auto result = ModelRegistry::create(db_path);
                if (result.is_ok()) {
                    return std::move(result).unwrap();
                } else {
                    // Convert RegistryError to Python exception
                    auto error = result.unwrap_err();
                    throw py::value_error("Failed to create ModelRegistry: " + to_string(error));
                }
            },
            "Create a ModelRegistry instance",
            "db_path"_a)

        // Model registration and retrieval
        .def(
            "register_model",
            [](ModelRegistry& self, const ModelInfo& info) -> int {
                auto result = self.register_model(info);
                if (result.is_ok()) {
                    return result.unwrap();
                } else {
                    throw py::value_error("Failed to register model: " +
                                          to_string(result.unwrap_err()));
                }
            },
            "Register a new model in the registry",
            "info"_a)

        .def(
            "get_model",
            [](ModelRegistry& self, int model_id) -> ModelInfo {
                auto result = self.get_model(model_id);
                if (result.is_ok()) {
                    return result.unwrap();
                } else {
                    throw py::value_error("Failed to get model: " + to_string(result.unwrap_err()));
                }
            },
            "Get model by ID",
            "model_id"_a)

        .def(
            "get_model",
            [](ModelRegistry& self,
               const std::string& model_name,
               const std::string& model_version) -> ModelInfo {
                auto result = self.get_model(model_name, model_version);
                if (result.is_ok()) {
                    return result.unwrap();
                } else {
                    throw py::value_error("Failed to get model: " + to_string(result.unwrap_err()));
                }
            },
            "Get model by name and version",
            "model_name"_a,
            "model_version"_a)

        // Model listing and filtering
        .def(
            "list_models",
            [](ModelRegistry& self,
               std::optional<ModelStatus> status = std::nullopt,
               std::optional<ModelType> model_type = std::nullopt,
               std::optional<InferenceBackend> backend = std::nullopt) -> std::vector<ModelInfo> {
                auto result = self.list_models(status, model_type, backend);
                if (result.is_ok()) {
                    return result.unwrap();
                } else {
                    throw py::value_error("Failed to list models: " +
                                          to_string(result.unwrap_err()));
                }
            },
            "List models with optional filtering",
            "status"_a = py::none(),
            "model_type"_a = py::none(),
            "backend"_a = py::none())

        // Lifecycle management
        .def(
            "update_model_status",
            [](ModelRegistry& self, int model_id, ModelStatus new_status) {
                auto result = self.update_model_status(model_id, new_status);
                if (!result.is_ok()) {
                    throw py::value_error("Failed to update model status: " +
                                          to_string(result.unwrap_err()));
                }
            },
            "Update model lifecycle status",
            "model_id"_a,
            "new_status"_a)

        // Performance metrics
        .def(
            "record_performance_metrics",
            [](ModelRegistry& self, const PerformanceMetrics& metrics) -> int {
                auto result = self.record_performance_metrics(metrics);
                if (result.is_ok()) {
                    return result.unwrap();
                } else {
                    throw py::value_error("Failed to record metrics: " +
                                          to_string(result.unwrap_err()));
                }
            },
            "Record performance metrics for a model",
            "metrics"_a)

        // Validation results
        .def(
            "record_validation_result",
            [](ModelRegistry& self, const ValidationResult& validation) -> int {
                auto result = self.record_validation_result(validation);
                if (result.is_ok()) {
                    return result.unwrap();
                } else {
                    throw py::value_error("Failed to record validation: " +
                                          to_string(result.unwrap_err()));
                }
            },
            "Record validation results for a model",
            "validation"_a)

        // Utility methods
        .def(
            "get_latest_model_version",
            [](ModelRegistry& self, const std::string& model_name) -> std::string {
                auto result = self.get_latest_model_version(model_name);
                if (result.is_ok()) {
                    return result.unwrap();
                } else {
                    throw py::value_error("Failed to get latest version: " +
                                          to_string(result.unwrap_err()));
                }
            },
            "Get the latest version of a model",
            "model_name"_a)

        .def(
            "get_production_models",
            [](ModelRegistry& self) -> std::vector<ModelInfo> {
                auto result = self.get_production_models();
                if (result.is_ok()) {
                    return result.unwrap();
                } else {
                    throw py::value_error("Failed to get production models: " +
                                          to_string(result.unwrap_err()));
                }
            },
            "Get all production models")

        .def(
            "get_model_performance_summary",
            [](ModelRegistry& self, int model_id) -> std::unordered_map<std::string, double> {
                auto result = self.get_model_performance_summary(model_id);
                if (result.is_ok()) {
                    return result.unwrap();
                } else {
                    throw py::value_error("Failed to get performance summary: " +
                                          to_string(result.unwrap_err()));
                }
            },
            "Get performance summary for a model",
            "model_id"_a)

        .def(
            "cleanup_old_metrics",
            [](ModelRegistry& self, int days_to_keep = 30) -> int {
                auto result = self.cleanup_old_metrics(days_to_keep);
                if (result.is_ok()) {
                    return result.unwrap();
                } else {
                    throw py::value_error("Failed to cleanup metrics: " +
                                          to_string(result.unwrap_err()));
                }
            },
            "Clean up old performance metrics",
            "days_to_keep"_a = 30)

        // Connection status
        .def("is_connected", &ModelRegistry::is_connected, "Check if database connection is active")

        .def("get_db_path", &ModelRegistry::get_db_path, "Get the database file path");

    // Utility function for error string conversion
    m.def("registry_error_to_string", &to_string, "Convert RegistryError to string", "error"_a);
}
