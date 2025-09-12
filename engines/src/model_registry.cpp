// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#include "model_registry.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "../../common/src/logging.hpp"

namespace inference_lab::engines {

using namespace inference_lab::common;

// Static helper functions for SQLite callbacks
struct QueryCallbackData {
    std::vector<std::unordered_map<std::string, std::string>> results;
};

static int query_callback(void* user_data, int argc, char** argv, char** col_names) {
    auto* data = static_cast<QueryCallbackData*>(user_data);
    std::unordered_map<std::string, std::string> row;

    for (int i = 0; i < argc; i++) {
        row[col_names[i]] = argv[i] ? argv[i] : "";
    }

    data->results.push_back(std::move(row));
    return 0;
}

// Constructor
ModelRegistry::ModelRegistry(const std::string& db_path)
    : db_path_(db_path), db_(nullptr), connected_(false) {}

// Destructor
ModelRegistry::~ModelRegistry() {
    if (db_) {
        sqlite3_close(db_);
    }
}

// Move constructor
ModelRegistry::ModelRegistry(ModelRegistry&& other) noexcept
    : db_path_(std::move(other.db_path_)), db_(other.db_), connected_(other.connected_) {
    other.db_ = nullptr;
    other.connected_ = false;
}

// Move assignment
ModelRegistry& ModelRegistry::operator=(ModelRegistry&& other) noexcept {
    if (this != &other) {
        if (db_) {
            sqlite3_close(db_);
        }
        db_path_ = std::move(other.db_path_);
        db_ = other.db_;
        connected_ = other.connected_;
        other.db_ = nullptr;
        other.connected_ = false;
    }
    return *this;
}

// Factory method
Result<std::unique_ptr<ModelRegistry>, RegistryError> ModelRegistry::create(
    const std::string& db_path) {
    auto registry = std::unique_ptr<ModelRegistry>(new ModelRegistry(db_path));

    auto init_result = registry->initialize();
    if (init_result.is_err()) {
        return Err(init_result.unwrap_err());
    }

    return Ok(std::move(registry));
}

// Initialize database connection
Result<std::monostate, RegistryError> ModelRegistry::initialize() {
    // Open database connection
    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        LOG_ERROR_PRINT("Failed to open database: {}", sqlite3_errmsg(db_));
        return Err(RegistryError::DATABASE_CONNECTION_FAILED);
    }

    // Enable foreign keys
    auto result = execute_statement("PRAGMA foreign_keys = ON");
    if (result.is_err()) {
        return Err(result.unwrap_err());
    }

    // Check if schema exists by querying for models table
    QueryCallbackData data;
    rc = sqlite3_exec(db_,
                      "SELECT name FROM sqlite_master WHERE type='table' AND name='models'",
                      query_callback,
                      &data,
                      nullptr);

    if (rc != SQLITE_OK || data.results.empty()) {
        // Schema doesn't exist - try to load it
        std::string schema_path = db_path_ + "/../model_registry_schema.sql";
        std::ifstream schema_file(schema_path);

        if (schema_file.is_open()) {
            std::stringstream buffer;
            buffer << schema_file.rdbuf();

            char* err_msg = nullptr;
            rc = sqlite3_exec(db_, buffer.str().c_str(), nullptr, nullptr, &err_msg);

            if (rc != SQLITE_OK) {
                LOG_ERROR_PRINT("Failed to create schema: {}", err_msg);
                sqlite3_free(err_msg);
                return Err(RegistryError::SCHEMA_MISMATCH);
            }
        } else {
            LOG_WARNING_PRINT("Schema file not found, assuming database is already initialized");
        }
    }

    connected_ = true;
    return Ok(std::monostate{});
}

// Register a new model
Result<int, RegistryError> ModelRegistry::register_model(const ModelInfo& info) {
    if (!connected_) {
        return Err(RegistryError::DATABASE_CONNECTION_FAILED);
    }

    std::stringstream query;
    query
        << "INSERT INTO models (model_name, model_version, model_type, backend, model_path, status";

    // Add optional fields if present
    if (info.config_path)
        query << ", config_path";
    if (info.description)
        query << ", description";
    if (info.author)
        query << ", author";
    if (info.model_size_bytes)
        query << ", model_size_bytes";
    if (info.estimated_memory_mb)
        query << ", estimated_memory_mb";
    if (info.input_shape)
        query << ", input_shape";
    if (info.output_shape)
        query << ", output_shape";
    if (info.tags)
        query << ", tags";
    if (info.custom_metadata)
        query << ", custom_metadata";

    query << ") VALUES (?, ?, ?, ?, ?, ?";

    if (info.config_path)
        query << ", ?";
    if (info.description)
        query << ", ?";
    if (info.author)
        query << ", ?";
    if (info.model_size_bytes)
        query << ", ?";
    if (info.estimated_memory_mb)
        query << ", ?";
    if (info.input_shape)
        query << ", ?";
    if (info.output_shape)
        query << ", ?";
    if (info.tags)
        query << ", ?";
    if (info.custom_metadata)
        query << ", ?";

    query << ")";

    sqlite3_stmt* stmt = prepare_statement(query.str());
    if (!stmt) {
        return Err(RegistryError::SQL_EXECUTION_ERROR);
    }

    // Bind parameters
    int param_index = 1;
    sqlite3_bind_text(stmt, param_index++, info.model_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, param_index++, info.model_version.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(
        stmt, param_index++, type_to_string(info.model_type).c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(
        stmt, param_index++, backend_to_string(info.backend).c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, param_index++, info.model_path.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(
        stmt, param_index++, status_to_string(info.status).c_str(), -1, SQLITE_STATIC);

    // Bind optional parameters
    if (info.config_path) {
        sqlite3_bind_text(stmt, param_index++, info.config_path->c_str(), -1, SQLITE_STATIC);
    }
    if (info.description) {
        sqlite3_bind_text(stmt, param_index++, info.description->c_str(), -1, SQLITE_STATIC);
    }
    if (info.author) {
        sqlite3_bind_text(stmt, param_index++, info.author->c_str(), -1, SQLITE_STATIC);
    }
    if (info.model_size_bytes) {
        sqlite3_bind_int64(stmt, param_index++, *info.model_size_bytes);
    }
    if (info.estimated_memory_mb) {
        sqlite3_bind_int(stmt, param_index++, *info.estimated_memory_mb);
    }
    if (info.input_shape) {
        sqlite3_bind_text(stmt, param_index++, info.input_shape->c_str(), -1, SQLITE_STATIC);
    }
    if (info.output_shape) {
        sqlite3_bind_text(stmt, param_index++, info.output_shape->c_str(), -1, SQLITE_STATIC);
    }
    if (info.tags) {
        sqlite3_bind_text(stmt, param_index++, info.tags->c_str(), -1, SQLITE_STATIC);
    }
    if (info.custom_metadata) {
        sqlite3_bind_text(stmt, param_index++, info.custom_metadata->c_str(), -1, SQLITE_STATIC);
    }

    // Execute statement
    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        LOG_ERROR_PRINT("Failed to register model: {}", sqlite3_errmsg(db_));
        sqlite3_finalize(stmt);

        // Check if it's a unique constraint violation
        if (std::string(sqlite3_errmsg(db_)).find("UNIQUE") != std::string::npos) {
            return Err(RegistryError::MODEL_ALREADY_EXISTS);
        }
        return Err(RegistryError::SQL_EXECUTION_ERROR);
    }

    int model_id = static_cast<int>(sqlite3_last_insert_rowid(db_));
    sqlite3_finalize(stmt);

    LOG_INFO_PRINT(
        "Registered model: {} v{} (ID: {})", info.model_name, info.model_version, model_id);

    return Ok(model_id);
}

// Get model by ID
Result<ModelInfo, RegistryError> ModelRegistry::get_model(int model_id) {
    if (!connected_) {
        return Err(RegistryError::DATABASE_CONNECTION_FAILED);
    }

    std::string query = "SELECT * FROM models WHERE model_id = ?";
    sqlite3_stmt* stmt = prepare_statement(query);
    if (!stmt) {
        return Err(RegistryError::SQL_EXECUTION_ERROR);
    }

    sqlite3_bind_int(stmt, 1, model_id);

    int rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        ModelInfo info;
        info.model_id = model_id;

        // Extract required fields
        info.model_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        info.model_version = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        info.model_type =
            string_to_type(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)));
        info.backend =
            string_to_backend(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4)));
        info.model_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));

        // Extract optional fields
        if (sqlite3_column_type(stmt, 6) != SQLITE_NULL) {
            info.config_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        }
        if (sqlite3_column_type(stmt, 7) != SQLITE_NULL) {
            info.metadata_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        }

        // Extract status
        const char* status_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 16));
        if (status_str) {
            info.status = string_to_status(status_str);
        }

        sqlite3_finalize(stmt);
        return Ok(info);
    }

    sqlite3_finalize(stmt);
    return Err(RegistryError::MODEL_NOT_FOUND);
}

// Get model by name and version
Result<ModelInfo, RegistryError> ModelRegistry::get_model(const std::string& model_name,
                                                          const std::string& model_version) {
    if (!connected_) {
        return Err(RegistryError::DATABASE_CONNECTION_FAILED);
    }

    std::string query = "SELECT model_id FROM models WHERE model_name = ? AND model_version = ?";
    sqlite3_stmt* stmt = prepare_statement(query);
    if (!stmt) {
        return Err(RegistryError::SQL_EXECUTION_ERROR);
    }

    sqlite3_bind_text(stmt, 1, model_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, model_version.c_str(), -1, SQLITE_STATIC);

    int rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        int model_id = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);
        return get_model(model_id);
    }

    sqlite3_finalize(stmt);
    return Err(RegistryError::MODEL_NOT_FOUND);
}

// List models with filtering
Result<std::vector<ModelInfo>, RegistryError> ModelRegistry::list_models(
    std::optional<ModelStatus> status,
    std::optional<ModelType> model_type,
    std::optional<InferenceBackend> backend) {
    if (!connected_) {
        return Err(RegistryError::DATABASE_CONNECTION_FAILED);
    }

    std::stringstream query;
    query << "SELECT model_id FROM models WHERE 1=1";

    if (status) {
        query << " AND status = '" << status_to_string(*status) << "'";
    }
    if (model_type) {
        query << " AND model_type = '" << type_to_string(*model_type) << "'";
    }
    if (backend) {
        query << " AND backend = '" << backend_to_string(*backend) << "'";
    }

    query << " ORDER BY created_at DESC";

    QueryCallbackData data;
    int rc = sqlite3_exec(db_, query.str().c_str(), query_callback, &data, nullptr);

    if (rc != SQLITE_OK) {
        return Err(RegistryError::SQL_EXECUTION_ERROR);
    }

    std::vector<ModelInfo> models;
    for (const auto& row : data.results) {
        int model_id = std::stoi(row.at("model_id"));
        auto model_result = get_model(model_id);
        if (model_result.is_ok()) {
            models.push_back(model_result.unwrap());
        }
    }

    return Ok(models);
}

// Update model status
Result<std::monostate, RegistryError> ModelRegistry::update_model_status(int model_id,
                                                                         ModelStatus new_status) {
    if (!connected_) {
        return Err(RegistryError::DATABASE_CONNECTION_FAILED);
    }

    std::string query;
    if (new_status == ModelStatus::PRODUCTION) {
        query = "UPDATE models SET status = ?, deployed_at = CURRENT_TIMESTAMP WHERE model_id = ?";
    } else if (new_status == ModelStatus::DEPRECATED) {
        query =
            "UPDATE models SET status = ?, deprecated_at = CURRENT_TIMESTAMP WHERE model_id = ?";
    } else {
        query = "UPDATE models SET status = ? WHERE model_id = ?";
    }

    sqlite3_stmt* stmt = prepare_statement(query);
    if (!stmt) {
        return Err(RegistryError::SQL_EXECUTION_ERROR);
    }

    sqlite3_bind_text(stmt, 1, status_to_string(new_status).c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, model_id);

    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        return Err(RegistryError::SQL_EXECUTION_ERROR);
    }

    LOG_INFO_PRINT("Updated model {} status to {}", model_id, status_to_string(new_status));
    return Ok(std::monostate{});
}

// Get production models
Result<std::vector<ModelInfo>, RegistryError> ModelRegistry::get_production_models() {
    return list_models(ModelStatus::PRODUCTION, std::nullopt, std::nullopt);
}

// Get latest model version
Result<std::string, RegistryError> ModelRegistry::get_latest_model_version(
    const std::string& model_name) {
    if (!connected_) {
        return Err(RegistryError::DATABASE_CONNECTION_FAILED);
    }

    std::string query = "SELECT MAX(model_version) FROM models WHERE model_name = ?";
    sqlite3_stmt* stmt = prepare_statement(query);
    if (!stmt) {
        return Err(RegistryError::SQL_EXECUTION_ERROR);
    }

    sqlite3_bind_text(stmt, 1, model_name.c_str(), -1, SQLITE_STATIC);

    int rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW && sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
        std::string version = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        sqlite3_finalize(stmt);
        return Ok(version);
    }

    sqlite3_finalize(stmt);
    return Err(RegistryError::MODEL_NOT_FOUND);
}

// Check connection status
bool ModelRegistry::is_connected() const {
    return connected_ && db_ != nullptr;
}

// Prepare SQL statement
sqlite3_stmt* ModelRegistry::prepare_statement(const std::string& query) {
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        LOG_ERROR_PRINT("Failed to prepare statement: {}", sqlite3_errmsg(db_));
        return nullptr;
    }

    return stmt;
}

// Execute statement
Result<std::monostate, RegistryError> ModelRegistry::execute_statement(
    const std::string& statement) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, statement.c_str(), nullptr, nullptr, &err_msg);

    if (rc != SQLITE_OK) {
        LOG_ERROR_PRINT("SQL execution error: {}", err_msg);
        sqlite3_free(err_msg);
        return Err(RegistryError::SQL_EXECUTION_ERROR);
    }

    return Ok(std::monostate{});
}

// Enum conversion functions
std::string ModelRegistry::status_to_string(ModelStatus status) {
    switch (status) {
        case ModelStatus::DEVELOPMENT:
            return "development";
        case ModelStatus::STAGING:
            return "staging";
        case ModelStatus::PRODUCTION:
            return "production";
        case ModelStatus::DEPRECATED:
            return "deprecated";
        case ModelStatus::ARCHIVED:
            return "archived";
        default:
            return "development";
    }
}

ModelStatus ModelRegistry::string_to_status(const std::string& status) {
    if (status == "staging")
        return ModelStatus::STAGING;
    if (status == "production")
        return ModelStatus::PRODUCTION;
    if (status == "deprecated")
        return ModelStatus::DEPRECATED;
    if (status == "archived")
        return ModelStatus::ARCHIVED;
    return ModelStatus::DEVELOPMENT;
}

std::string ModelRegistry::type_to_string(ModelType type) {
    switch (type) {
        case ModelType::ONNX:
            return "onnx";
        case ModelType::TENSORRT:
            return "tensorrt";
        case ModelType::PYTORCH:
            return "pytorch";
        case ModelType::RULE_BASED:
            return "rule_based";
        case ModelType::OTHER:
            return "other";
        default:
            return "other";
    }
}

ModelType ModelRegistry::string_to_type(const std::string& type) {
    if (type == "onnx")
        return ModelType::ONNX;
    if (type == "tensorrt")
        return ModelType::TENSORRT;
    if (type == "pytorch")
        return ModelType::PYTORCH;
    if (type == "rule_based")
        return ModelType::RULE_BASED;
    return ModelType::OTHER;
}

std::string ModelRegistry::backend_to_string(InferenceBackend backend) {
    switch (backend) {
        case InferenceBackend::RULE_BASED:
            return "RULE_BASED";
        case InferenceBackend::TENSORRT_GPU:
            return "TENSORRT_GPU";
        case InferenceBackend::ONNX_RUNTIME:
            return "ONNX_RUNTIME";
        case InferenceBackend::HYBRID_NEURAL_SYMBOLIC:
            return "HYBRID_NEURAL_SYMBOLIC";
        default:
            return "RULE_BASED";
    }
}

InferenceBackend ModelRegistry::string_to_backend(const std::string& backend) {
    if (backend == "TENSORRT_GPU")
        return InferenceBackend::TENSORRT_GPU;
    if (backend == "ONNX_RUNTIME")
        return InferenceBackend::ONNX_RUNTIME;
    if (backend == "HYBRID_NEURAL_SYMBOLIC")
        return InferenceBackend::HYBRID_NEURAL_SYMBOLIC;
    return InferenceBackend::RULE_BASED;
}

}  // namespace inference_lab::engines
