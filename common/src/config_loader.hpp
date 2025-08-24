#pragma once

#include <any>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <variant>
#include <vector>

#include "result.hpp"

using namespace inference_lab;

namespace inference_lab {

enum class ConfigError {
    FileNotFound,
    ParseError,
    ValidationError,
    KeyNotFound,
    TypeMismatch,
    EnvironmentVariableNotFound,
    SchemaLoadError,
    InvalidPath
};

class ConfigurationLoader {
  public:
    using ConfigValue = std::variant<std::string,
                                     std::int64_t,
                                     double,
                                     bool,
                                     std::map<std::string, std::any>,
                                     std::vector<std::any>>;

    explicit ConfigurationLoader(
        const std::optional<std::filesystem::path>& config_path = std::nullopt,
        const std::optional<std::filesystem::path>& schema_path = std::nullopt,
        const std::optional<std::string>& environment = std::nullopt,
        bool validate_strict = true);

    auto load(const std::optional<std::filesystem::path>& config_path = std::nullopt)
        -> Result<std::map<std::string, std::any>, ConfigError>;

    auto get(const std::string& key,
             const std::optional<std::string>& default_value = std::nullopt) const
        -> Result<std::string, ConfigError>;

    auto get_bool(const std::string& key,
                  const std::optional<bool>& default_value = std::nullopt) const
        -> Result<bool, ConfigError>;

    auto get_int(const std::string& key,
                 const std::optional<std::int64_t>& default_value = std::nullopt) const
        -> Result<std::int64_t, ConfigError>;

    auto get_float(const std::string& key,
                   const std::optional<double>& default_value = std::nullopt) const
        -> Result<double, ConfigError>;

    auto get_dict(const std::string& key) const
        -> Result<std::map<std::string, std::any>, ConfigError>;

    auto get_list(const std::string& key) const -> Result<std::vector<std::any>, ConfigError>;

    auto has(const std::string& key) const -> bool;

    auto keys(const std::optional<std::string>& prefix = std::nullopt) const
        -> std::vector<std::string>;

    auto validate(const std::optional<std::map<std::string, std::any>>& config = std::nullopt) const
        -> std::vector<std::string>;

    // Environment properties
    auto environment_name() const -> const std::string& { return environment_; }
    auto is_development() const -> bool { return environment_ == "development"; }
    auto is_staging() const -> bool { return environment_ == "staging"; }
    auto is_production() const -> bool { return environment_ == "production"; }
    auto debug_mode() const -> bool;

    // Global instance management
    static auto instance() -> ConfigurationLoader&;
    static auto set_instance(std::unique_ptr<ConfigurationLoader> loader) -> void;

  private:
    auto resolve_environment_variables(const std::string& text) const -> std::string;
    auto resolve_config_recursive(std::map<std::string, std::any>& config) const -> void;
    auto apply_environment_overrides(std::map<std::string, std::any>& config) const -> void;
    auto merge_configs(std::map<std::string, std::any>& base,
                       const std::map<std::string, std::any>& overlay) const -> void;
    auto find_nested_value(const std::string& key_path) const -> std::optional<std::any>;
    auto auto_discover_config_path() const -> std::optional<std::filesystem::path>;
    auto auto_discover_schema_path() const -> std::optional<std::filesystem::path>;

    std::filesystem::path config_path_;
    std::optional<std::filesystem::path> schema_path_;
    std::string environment_;
    bool validate_strict_;
    bool is_loaded_;
    std::map<std::string, std::any> config_;
    std::regex env_var_pattern_;

    static std::unique_ptr<ConfigurationLoader> global_instance_;
};

class EnvironmentResolver {
  public:
    auto resolve(std::map<std::string, std::any>& config) const -> void;

  private:
    auto resolve_value(std::any& value) const -> void;
    auto resolve_string(std::string& text) const -> void;
    auto get_env_var(const std::string& var_name,
                     const std::optional<std::string>& default_value) const
        -> std::optional<std::string>;

    std::regex env_var_pattern_{R"(\$\{([^}:]+)(?::([^}]*))?\})"};
};

class ConfigValidator {
  public:
    explicit ConfigValidator(
        const std::optional<std::filesystem::path>& schema_path = std::nullopt);

    auto validate(const std::map<std::string, std::any>& config, bool strict = true) const
        -> std::vector<std::string>;

  private:
    std::optional<std::filesystem::path> schema_path_;
    bool schema_loaded_;
};

}  // namespace inference_lab