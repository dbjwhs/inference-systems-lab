// MIT License
// Copyright (c) 2025 Inference Systems Laboratory
//
// Model Registry C++ Interface - Unified model management across Python and C++

#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <sqlite3.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../common/src/ml_types.hpp"
#include "../../common/src/result.hpp"

namespace inference_lab::engines {

using namespace inference_lab::common;
using inference_lab::common::ml::InferenceBackend;

/**
 * @brief Model lifecycle status matching Python ModelStatus enum
 */
enum class ModelStatus { DEVELOPMENT, STAGING, PRODUCTION, DEPRECATED, ARCHIVED };

/**
 * @brief Model types supported by the registry
 */
enum class ModelType { ONNX, TENSORRT, PYTORCH, RULE_BASED, OTHER };

/**
 * @brief Model metadata structure
 *
 * Represents a model entry in the registry with all associated metadata
 */
struct ModelInfo {
    int model_id;
    std::string model_name;
    std::string model_version;
    ModelType model_type;
    InferenceBackend backend;
    std::string model_path;

    // Optional fields
    std::optional<std::string> config_path;
    std::optional<std::string> metadata_path;
    std::optional<std::string> description;
    std::optional<std::string> author;

    // Performance characteristics
    std::optional<int64_t> model_size_bytes;
    std::optional<int> estimated_memory_mb;
    std::optional<double> avg_inference_time_ms;

    // Status and timestamps
    ModelStatus status;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    std::optional<std::chrono::system_clock::time_point> deployed_at;

    // Shape information (stored as JSON strings in DB)
    std::optional<std::string> input_shape;
    std::optional<std::string> output_shape;
    std::optional<std::string> tags;
    std::optional<std::string> custom_metadata;
};

/**
 * @brief Performance metrics for a model
 */
struct PerformanceMetrics {
    int metric_id;
    int model_id;
    double inference_time_ms;
    std::optional<double> preprocessing_time_ms;
    std::optional<double> postprocessing_time_ms;
    std::optional<double> memory_usage_mb;
    std::optional<double> gpu_memory_mb;
    std::optional<double> throughput_qps;
    std::optional<int> batch_size;
    std::chrono::system_clock::time_point recorded_at;
};

/**
 * @brief Validation result for a model
 */
struct ValidationResult {
    int validation_id;
    int model_id;
    std::string validation_type;
    std::string validation_status;  // "passed", "failed", "warning"
    std::optional<double> accuracy_score;
    std::optional<double> precision_score;
    std::optional<double> recall_score;
    std::optional<double> f1_score;
    std::optional<std::string> custom_metrics;  // JSON string
    std::chrono::system_clock::time_point validated_at;
};

/**
 * @brief Registry error types
 */
enum class RegistryError {
    DATABASE_CONNECTION_FAILED,
    MODEL_NOT_FOUND,
    MODEL_ALREADY_EXISTS,
    INVALID_MODEL_VERSION,
    SQL_EXECUTION_ERROR,
    SCHEMA_MISMATCH,
    PERMISSION_DENIED
};

/**
 * @brief Convert RegistryError to string
 */
inline std::string to_string(RegistryError error) {
    switch (error) {
        case RegistryError::DATABASE_CONNECTION_FAILED:
            return "Database connection failed";
        case RegistryError::MODEL_NOT_FOUND:
            return "Model not found";
        case RegistryError::MODEL_ALREADY_EXISTS:
            return "Model with same name and version already exists";
        case RegistryError::INVALID_MODEL_VERSION:
            return "Invalid model version format";
        case RegistryError::SQL_EXECUTION_ERROR:
            return "SQL execution error";
        case RegistryError::SCHEMA_MISMATCH:
            return "Database schema mismatch";
        case RegistryError::PERMISSION_DENIED:
            return "Permission denied";
        default:
            return "Unknown registry error";
    }
}

/**
 * @brief C++ interface to the model registry database
 *
 * Provides RAII-compliant access to the SQLite model registry,
 * enabling C++ code to interact with the same database used by Python tools.
 *
 * Thread-safety: This class is NOT thread-safe. Use one instance per thread
 * or add external synchronization.
 *
 * Example usage:
 * @code
 * auto registry = ModelRegistry::create("model_registry.db");
 * if (registry.is_ok()) {
 *     auto model_result = registry.unwrap()->get_model("resnet50", "1.0.0");
 *     if (model_result.is_ok()) {
 *         auto model_info = model_result.unwrap();
 *         // Use model_info...
 *     }
 * }
 * @endcode
 */
class ModelRegistry {
  public:
    /**
     * @brief Create a model registry instance
     *
     * @param db_path Path to the SQLite database file
     * @return Result containing the registry or an error
     */
    static Result<std::unique_ptr<ModelRegistry>, RegistryError> create(const std::string& db_path);

    /**
     * @brief Destructor - closes database connection
     */
    ~ModelRegistry();

    // Disable copy, enable move
    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;
    ModelRegistry(ModelRegistry&&) noexcept;
    ModelRegistry& operator=(ModelRegistry&&) noexcept;

    /**
     * @brief Register a new model in the registry
     *
     * @param info Model information to register
     * @return Result containing the model ID or an error
     */
    Result<int, RegistryError> register_model(const ModelInfo& info);

    /**
     * @brief Get model information by ID
     *
     * @param model_id The model ID
     * @return Result containing model info or an error
     */
    Result<ModelInfo, RegistryError> get_model(int model_id);

    /**
     * @brief Get model information by name and version
     *
     * @param model_name The model name
     * @param model_version The model version
     * @return Result containing model info or an error
     */
    Result<ModelInfo, RegistryError> get_model(const std::string& model_name,
                                               const std::string& model_version);

    /**
     * @brief List all models with optional filtering
     *
     * @param status Optional status filter
     * @param model_type Optional type filter
     * @param backend Optional backend filter
     * @return Result containing list of models or an error
     */
    Result<std::vector<ModelInfo>, RegistryError> list_models(
        std::optional<ModelStatus> status = std::nullopt,
        std::optional<ModelType> model_type = std::nullopt,
        std::optional<InferenceBackend> backend = std::nullopt);

    /**
     * @brief Update model status (lifecycle management)
     *
     * @param model_id The model ID
     * @param new_status The new status
     * @return Result indicating success or error
     */
    Result<std::monostate, RegistryError> update_model_status(int model_id, ModelStatus new_status);

    /**
     * @brief Record performance metrics for a model
     *
     * @param metrics The performance metrics to record
     * @return Result containing the metric ID or an error
     */
    Result<int, RegistryError> record_performance_metrics(const PerformanceMetrics& metrics);

    /**
     * @brief Record validation results for a model
     *
     * @param result The validation result to record
     * @return Result containing the validation ID or an error
     */
    Result<int, RegistryError> record_validation_result(const ValidationResult& result);

    /**
     * @brief Get the latest version of a model by name
     *
     * @param model_name The model name
     * @return Result containing the version string or an error
     */
    Result<std::string, RegistryError> get_latest_model_version(const std::string& model_name);

    /**
     * @brief Get all production models
     *
     * @return Result containing list of production models or an error
     */
    Result<std::vector<ModelInfo>, RegistryError> get_production_models();

    /**
     * @brief Get performance summary for a model
     *
     * @param model_id The model ID
     * @return Result containing performance metrics or an error
     */
    Result<std::unordered_map<std::string, double>, RegistryError> get_model_performance_summary(
        int model_id);

    /**
     * @brief Clean up old performance metrics
     *
     * @param days_to_keep Number of days of metrics to retain
     * @return Result containing count of deleted records or an error
     */
    Result<int, RegistryError> cleanup_old_metrics(int days_to_keep = 30);

    /**
     * @brief Check if the database connection is valid
     *
     * @return True if connected, false otherwise
     */
    bool is_connected() const;

    /**
     * @brief Get the database path
     *
     * @return The path to the database file
     */
    const std::string& get_db_path() const { return db_path_; }

  private:
    /**
     * @brief Private constructor - use create() factory method
     */
    explicit ModelRegistry(const std::string& db_path);

    /**
     * @brief Initialize database connection and schema
     *
     * @return Result indicating success or error
     */
    Result<std::monostate, RegistryError> initialize();

    /**
     * @brief Execute a SQL query that returns data
     *
     * @param query The SQL query
     * @param callback Callback function for each row
     * @param user_data User data passed to callback
     * @return Result indicating success or error
     */
    Result<std::monostate, RegistryError> execute_query(const std::string& query,
                                                        int (*callback)(void*, int, char**, char**),
                                                        void* user_data);

    /**
     * @brief Execute a SQL statement that doesn't return data
     *
     * @param statement The SQL statement
     * @return Result indicating success or error
     */
    Result<std::monostate, RegistryError> execute_statement(const std::string& statement);

    /**
     * @brief Prepare and bind a SQL statement
     *
     * @param query The SQL query with placeholders
     * @return Prepared statement or nullptr on error
     */
    sqlite3_stmt* prepare_statement(const std::string& query);

    /**
     * @brief Convert timestamp string to time_point
     */
    static std::chrono::system_clock::time_point parse_timestamp(const std::string& timestamp);

    /**
     * @brief Convert time_point to timestamp string
     */
    static std::string format_timestamp(const std::chrono::system_clock::time_point& time_point);

    /**
     * @brief Convert ModelStatus enum to string
     */
    static std::string status_to_string(ModelStatus status);

    /**
     * @brief Convert string to ModelStatus enum
     */
    static ModelStatus string_to_status(const std::string& status);

    /**
     * @brief Convert ModelType enum to string
     */
    static std::string type_to_string(ModelType type);

    /**
     * @brief Convert string to ModelType enum
     */
    static ModelType string_to_type(const std::string& type);

    /**
     * @brief Convert InferenceBackend enum to string
     */
    static std::string backend_to_string(InferenceBackend backend);

    /**
     * @brief Convert string to InferenceBackend enum
     */
    static InferenceBackend string_to_backend(const std::string& backend);

    // Member variables
    std::string db_path_;
    sqlite3* db_;
    bool connected_;
};

}  // namespace inference_lab::engines
