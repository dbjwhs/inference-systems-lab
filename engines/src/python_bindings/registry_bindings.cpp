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

    // Bind InferenceBackend enum (from ml_types.hpp)
    py::enum_<InferenceBackend>(m, "InferenceBackend")
        .value("CPU_NATIVE", InferenceBackend::CPU_NATIVE)
        .value("TENSORRT_GPU", InferenceBackend::TENSORRT_GPU)
        .value("ONNX_RUNTIME", InferenceBackend::ONNX_RUNTIME)
        .value("RULE_BASED", InferenceBackend::RULE_BASED)
        .value("HYBRID_NEURAL_SYMBOLIC", InferenceBackend::HYBRID_NEURAL_SYMBOLIC)
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

        // Note: Additional methods not yet implemented:
        // - record_performance_metrics
        // - record_validation_result
        // - get_latest_model_version
        // - get_production_models
        // - get_model_performance_summary
        // - cleanup_old_metrics

        // Connection status
        .def("is_connected", &ModelRegistry::is_connected, "Check if database connection is active")

        .def("get_db_path", &ModelRegistry::get_db_path, "Get the database file path");

    // Utility function for error string conversion
    m.def("registry_error_to_string", &to_string, "Convert RegistryError to string", "error"_a);
}
