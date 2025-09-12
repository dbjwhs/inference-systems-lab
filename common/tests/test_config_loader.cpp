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

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "config_loader.hpp"
#include "logging.hpp"

namespace inference_lab {

class ConfigLoaderTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create test configuration directory
        test_dir_ = std::filesystem::temp_directory_path() / "inference_lab_test_config";
        std::filesystem::create_directories(test_dir_);

        config_path_ = test_dir_ / "test_config.yaml";
        schema_path_ = test_dir_ / "test_schema.json";

        createTestConfig();
        createTestSchema();

        // Set test environment variable
        std::setenv("TEST_DB_PATH", "/tmp/test_registry.db", 1);
        std::setenv("INFERENCE_LAB_ENV", "development", 1);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
        std::unsetenv("TEST_DB_PATH");
        std::unsetenv("INFERENCE_LAB_ENV");
    }

    void createTestConfig() {
        std::ofstream config_file(config_path_);
        config_file << R"(schema_version: "1.0.0"
application:
  name: "inference-systems-lab"
  version: "0.1.0"
  environment: "development"
  debug_mode: true
  settings:
    max_threads: 4
    memory_limit_mb: 1024
    enable_profiling: true
    enable_metrics_collection: true
    config_validation_strict: true

logging:
  level: "INFO"
  outputs:
    console:
      enabled: true
      format: "{timestamp} [{level}] [{thread}] {message}"
      color_enabled: true
    file:
      enabled: true
      path: "logs/inference_lab.log"
      format: "{timestamp} [{level}] [{thread}] {message}"
      max_size_mb: 50
      backup_count: 3
      rotation: "size"

registry:
  database:
    type: "sqlite"
    path: "${TEST_DB_PATH}"
    connection:
      timeout_seconds: 30
      pool_size: 5
      max_overflow: 10
      echo_sql: false

engines:
  default_backend: "RULE_BASED"
  default_precision: "FP32"
  rule_based:
    enabled: true
    max_rules: 1000
    max_facts: 10000
    inference_strategy: "forward_chaining"

features:
  experimental:
    neural_symbolic_hybrid: false
    distributed_inference: false
    advanced_caching: true
  beta:
    web_interface: true
    api_versioning: true
    batch_inference: true

# Environment-specific overrides
environments:
  development:
    logging:
      level: "DEBUG"
    application:
      debug_mode: true
  production:
    logging:
      level: "WARNING"
    application:
      debug_mode: false
)";
        config_file.close();
    }

    void createTestSchema() {
        nlohmann::json schema = nlohmann::json::parse(R"({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "schema_version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+\\.\\d+$"
    },
    "application": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "version": {"type": "string"},
        "environment": {
          "type": "string",
          "enum": ["development", "staging", "production"]
        },
        "debug_mode": {"type": "boolean"},
        "settings": {
          "type": "object",
          "properties": {
            "max_threads": {"type": "integer", "minimum": 1},
            "memory_limit_mb": {"type": "integer", "minimum": 64}
          }
        }
      },
      "required": ["name", "version", "environment"]
    }
  },
  "required": ["schema_version", "application"]
})");

        std::ofstream schema_file(schema_path_);
        schema_file << schema.dump(2);
        schema_file.close();
    }

    std::filesystem::path test_dir_;
    std::filesystem::path config_path_;
    std::filesystem::path schema_path_;
};

TEST_F(ConfigLoaderTest, BasicConfigurationLoading) {
    ConfigurationLoader loader(config_path_, schema_path_, "development");

    auto result = loader.load();
    ASSERT_TRUE(result.is_ok());

    auto config = result.unwrap();
    EXPECT_FALSE(config.empty());
}

TEST_F(ConfigLoaderTest, TypeSafeConfigAccess) {
    ConfigurationLoader loader(config_path_, schema_path_, "development");
    loader.load();

    // Test string access
    auto name_result = loader.get("application.name");
    ASSERT_TRUE(name_result.is_ok());
    EXPECT_EQ(name_result.unwrap(), "inference-systems-lab");

    // Test boolean access
    auto debug_result = loader.get_bool("application.debug_mode");
    ASSERT_TRUE(debug_result.is_ok());
    EXPECT_TRUE(debug_result.unwrap());

    // Test integer access
    auto threads_result = loader.get_int("application.settings.max_threads");
    ASSERT_TRUE(threads_result.is_ok());
    EXPECT_EQ(threads_result.unwrap(), 4);

    // Test with default values
    auto missing_result = loader.get("missing.key", "default_value");
    ASSERT_TRUE(missing_result.is_ok());
    EXPECT_EQ(missing_result.unwrap(), "default_value");

    auto missing_bool_result = loader.get_bool("missing.bool", true);
    ASSERT_TRUE(missing_bool_result.is_ok());
    EXPECT_TRUE(missing_bool_result.unwrap());
}

TEST_F(ConfigLoaderTest, EnvironmentVariableResolution) {
    ConfigurationLoader loader(config_path_, schema_path_, "development");
    loader.load();

    // Test environment variable resolution
    auto db_path_result = loader.get("registry.database.path");
    ASSERT_TRUE(db_path_result.is_ok());
    EXPECT_EQ(db_path_result.unwrap(), "/tmp/test_registry.db");
}

TEST_F(ConfigLoaderTest, EnvironmentSpecificOverrides) {
    // Test development environment
    ConfigurationLoader dev_loader(config_path_, schema_path_, "development");
    dev_loader.load();

    auto dev_debug = dev_loader.get_bool("application.debug_mode");
    ASSERT_TRUE(dev_debug.is_ok());
    EXPECT_TRUE(dev_debug.unwrap());

    auto dev_log_level = dev_loader.get("logging.level");
    ASSERT_TRUE(dev_log_level.is_ok());
    EXPECT_EQ(dev_log_level.unwrap(), "DEBUG");  // Overridden from INFO to DEBUG

    // Test production environment
    ConfigurationLoader prod_loader(config_path_, schema_path_, "production");
    prod_loader.load();

    auto prod_debug = prod_loader.get_bool("application.debug_mode");
    ASSERT_TRUE(prod_debug.is_ok());
    EXPECT_FALSE(prod_debug.unwrap());

    auto prod_log_level = prod_loader.get("logging.level");
    ASSERT_TRUE(prod_log_level.is_ok());
    EXPECT_EQ(prod_log_level.unwrap(), "WARNING");  // Overridden from INFO to WARNING
}

TEST_F(ConfigLoaderTest, KeyExistenceAndListing) {
    ConfigurationLoader loader(config_path_, schema_path_, "development");
    loader.load();

    // Test key existence
    EXPECT_TRUE(loader.has("application.name"));
    EXPECT_TRUE(loader.has("features.experimental.advanced_caching"));
    EXPECT_FALSE(loader.has("non.existent.key"));

    // Test key listing
    auto root_keys = loader.keys();
    EXPECT_FALSE(root_keys.empty());

    auto app_keys = loader.keys("application");
    EXPECT_FALSE(app_keys.empty());

    // Check for specific keys
    auto app_keys_set = std::set<std::string>(app_keys.begin(), app_keys.end());
    EXPECT_TRUE(app_keys_set.find("application.name") != app_keys_set.end());
    EXPECT_TRUE(app_keys_set.find("application.version") != app_keys_set.end());
}

TEST_F(ConfigLoaderTest, FeatureFlagAccess) {
    ConfigurationLoader loader(config_path_, schema_path_, "development");
    loader.load();

    // Test experimental features
    auto experimental_result = loader.get_dict("features.experimental");
    ASSERT_TRUE(experimental_result.is_ok());

    auto experimental_features = experimental_result.unwrap();
    EXPECT_FALSE(experimental_features.empty());

    auto caching_result = loader.get_bool("features.experimental.advanced_caching");
    ASSERT_TRUE(caching_result.is_ok());
    EXPECT_TRUE(caching_result.unwrap());

    // Test beta features
    auto web_interface_result = loader.get_bool("features.beta.web_interface");
    ASSERT_TRUE(web_interface_result.is_ok());
    EXPECT_TRUE(web_interface_result.unwrap());
}

TEST_F(ConfigLoaderTest, ConfigurationProperties) {
    ConfigurationLoader loader(config_path_, schema_path_, "development");
    loader.load();

    EXPECT_EQ(loader.environment_name(), "development");
    EXPECT_TRUE(loader.is_development());
    EXPECT_FALSE(loader.is_staging());
    EXPECT_FALSE(loader.is_production());
    EXPECT_TRUE(loader.debug_mode());
}

TEST_F(ConfigLoaderTest, ErrorHandling) {
    // Test with non-existent config file
    ConfigurationLoader bad_loader("/non/existent/config.yaml", schema_path_, "development");
    auto result = bad_loader.load();
    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), ConfigError::FileNotFound);

    // Test accessing key before loading
    ConfigurationLoader unloaded_loader(config_path_, schema_path_, "development");
    auto name_result = unloaded_loader.get("application.name");
    EXPECT_TRUE(name_result.is_err());
    EXPECT_EQ(name_result.unwrap_err(), ConfigError::KeyNotFound);

    // Test type mismatch
    ConfigurationLoader loader(config_path_, schema_path_, "development");
    loader.load();

    auto type_mismatch_result = loader.get_int("application.name");  // String as int
    EXPECT_TRUE(type_mismatch_result.is_err());
    EXPECT_EQ(type_mismatch_result.unwrap_err(), ConfigError::TypeMismatch);
}

TEST_F(ConfigLoaderTest, GlobalInstanceManagement) {
    // Test global instance
    auto& global_loader = ConfigurationLoader::instance();

    // Set a custom instance
    auto custom_loader =
        std::make_unique<ConfigurationLoader>(config_path_, schema_path_, "development");
    custom_loader->load();

    ConfigurationLoader::set_instance(std::move(custom_loader));

    auto& new_global_loader = ConfigurationLoader::instance();
    EXPECT_EQ(new_global_loader.environment_name(), "development");
}

TEST_F(ConfigLoaderTest, EnvironmentVariableWithDefaults) {
    // Create config with environment variable that has a default
    std::filesystem::path env_config_path = test_dir_ / "env_test_config.yaml";
    std::ofstream config_file(env_config_path);
    config_file << R"(schema_version: "1.0.0"
application:
  name: "test-app"
  version: "1.0.0"
  environment: "development"

test:
  with_default: "${MISSING_VAR:default_value}"
  without_default: "${TEST_DB_PATH}"
  nested:
    config: "${TEST_DB_PATH}/nested"
)";
    config_file.close();

    ConfigurationLoader loader(env_config_path, schema_path_, "development");
    loader.load();

    // Test environment variable with default
    auto with_default_result = loader.get("test.with_default");
    ASSERT_TRUE(with_default_result.is_ok());
    EXPECT_EQ(with_default_result.unwrap(), "default_value");

    // Test regular environment variable
    auto without_default_result = loader.get("test.without_default");
    ASSERT_TRUE(without_default_result.is_ok());
    EXPECT_EQ(without_default_result.unwrap(), "/tmp/test_registry.db");

    // Test nested path with environment variable
    auto nested_result = loader.get("test.nested.config");
    ASSERT_TRUE(nested_result.is_ok());
    EXPECT_EQ(nested_result.unwrap(), "/tmp/test_registry.db/nested");
}

}  // namespace inference_lab
