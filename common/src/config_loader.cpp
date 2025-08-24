#include "config_loader.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "logging.hpp"

#ifdef YAML_CONFIG_AVAILABLE
    #include <yaml-cpp/yaml.h>
#endif

// Try to include nlohmann/json if available
#ifdef __has_include
    #if __has_include(<nlohmann/json.hpp>)
        #include <nlohmann/json.hpp>
        #define JSON_AVAILABLE 1
    #endif
#endif

namespace inference_lab {

namespace {
#ifdef YAML_CONFIG_AVAILABLE
auto convert_yaml_to_any(const YAML::Node& node) -> std::any {
    if (node.IsScalar()) {
        const auto& scalar = node.as<std::string>();

        // Try to parse as bool
        if (scalar == "true" || scalar == "True")
            return true;
        if (scalar == "false" || scalar == "False")
            return false;

        // Try to parse as integer
        try {
            return static_cast<std::int64_t>(std::stoll(scalar));
        } catch (...) {}

        // Try to parse as float
        try {
            return std::stod(scalar);
        } catch (...) {}

        // Return as string
        return scalar;
    } else if (node.IsSequence()) {
        std::vector<std::any> result;
        for (const auto& item : node) {
            result.push_back(convert_yaml_to_any(item));
        }
        return result;
    } else if (node.IsMap()) {
        std::map<std::string, std::any> result;
        for (const auto& pair : node) {
            result[pair.first.as<std::string>()] = convert_yaml_to_any(pair.second);
        }
        return result;
    }
    return std::string{};
}

auto any_to_string(const std::any& value) -> std::string {
    try {
        if (value.type() == typeid(std::string)) {
            return std::any_cast<std::string>(value);
        } else if (value.type() == typeid(std::int64_t)) {
            return std::to_string(std::any_cast<std::int64_t>(value));
        } else if (value.type() == typeid(double)) {
            return std::to_string(std::any_cast<double>(value));
        } else if (value.type() == typeid(bool)) {
            return std::any_cast<bool>(value) ? "true" : "false";
        }
    } catch (...) {}
    return "";
}

template <typename T>
auto any_cast_safe(const std::any& value) -> std::optional<T> {
    try {
        return std::any_cast<T>(value);
    } catch (...) {
        return std::nullopt;
    }
}
}

std::unique_ptr<ConfigurationLoader> ConfigurationLoader::global_instance_;

ConfigurationLoader::ConfigurationLoader(const std::optional<std::filesystem::path>& config_path,
                                         const std::optional<std::filesystem::path>& schema_path,
                                         const std::optional<std::string>& environment,
                                         bool validate_strict)
    : config_path_(config_path ? *config_path : std::filesystem::path{}),
      schema_path_(schema_path),
      environment_(environment
                       ? *environment
                       : (std::getenv("INFERENCE_LAB_ENV") ? std::getenv("INFERENCE_LAB_ENV")
                                                           : "development")),
      validate_strict_(validate_strict),
      is_loaded_(false),
      env_var_pattern_(R"(\$\{([^}:]+)(?::([^}]*))?\})") {
    if (!config_path) {
        if (auto discovered = auto_discover_config_path()) {
            config_path_ = *discovered;
        }
    }

    if (!schema_path_) {
        schema_path_ = auto_discover_schema_path();
    }
}

auto ConfigurationLoader::load(const std::optional<std::filesystem::path>& config_path)
    -> Result<std::map<std::string, std::any>, ConfigError> {
    if (config_path) {
        config_path_ = *config_path;
    }

    if (config_path_.empty() || !std::filesystem::exists(config_path_)) {
        Logger::error("Configuration file not found: {}", config_path_.string());
        return Err(ConfigError::FileNotFound);
    }

    try {
        YAML::Node yaml_node = YAML::LoadFile(config_path_.string());
        config_ = std::any_cast<std::map<std::string, std::any>>(convert_yaml_to_any(yaml_node));

        Logger::info("Loaded configuration from {}", config_path_.string());

        // Apply environment-specific overrides
        apply_environment_overrides(config_);

        // Resolve environment variables
        resolve_config_recursive(config_);

        // Validate configuration
        if (schema_path_ && std::filesystem::exists(*schema_path_)) {
            ConfigValidator validator(schema_path_);
            auto validation_errors = validator.validate(config_, validate_strict_);

            if (!validation_errors.empty()) {
                Logger::warning("Configuration validation issues: {} errors",
                                validation_errors.size());
                for (size_t i = 0; i < std::min(size_t{3}, validation_errors.size()); ++i) {
                    Logger::warning("  - {}", validation_errors[i]);
                }
                if (validation_errors.size() > 3) {
                    Logger::warning("  ... and {} more errors", validation_errors.size() - 3);
                }
            } else {
                Logger::info("Configuration validation passed");
            }
        }

        is_loaded_ = true;
        return Ok(config_);

    } catch (const YAML::Exception& e) {
        Logger::error("Failed to parse YAML configuration: {}", e.what());
        return Err(ConfigError::ParseError);
    } catch (const std::exception& e) {
        Logger::error("Failed to load configuration: {}", e.what());
        return Err(ConfigError::ParseError);
    }
}

auto ConfigurationLoader::get(const std::string& key,
                              const std::optional<std::string>& default_value) const
    -> Result<std::string, ConfigError> {
    if (auto value = find_nested_value(key)) {
        return Ok(any_to_string(*value));
    }

    if (default_value) {
        return Ok(*default_value);
    }

    return Err(ConfigError::KeyNotFound);
}

auto ConfigurationLoader::get_bool(const std::string& key,
                                   const std::optional<bool>& default_value) const
    -> Result<bool, ConfigError> {
    if (auto value = find_nested_value(key)) {
        if (auto bool_val = any_cast_safe<bool>(*value)) {
            return Ok(*bool_val);
        }

        // Try to convert string to bool
        auto str_val = any_to_string(*value);
        std::transform(str_val.begin(), str_val.end(), str_val.begin(), ::tolower);
        if (str_val == "true" || str_val == "1" || str_val == "yes") {
            return Ok(true);
        } else if (str_val == "false" || str_val == "0" || str_val == "no") {
            return Ok(false);
        }

        return Err(ConfigError::TypeMismatch);
    }

    if (default_value) {
        return Ok(*default_value);
    }

    return Err(ConfigError::KeyNotFound);
}

auto ConfigurationLoader::get_int(const std::string& key,
                                  const std::optional<std::int64_t>& default_value) const
    -> Result<std::int64_t, ConfigError> {
    if (auto value = find_nested_value(key)) {
        if (auto int_val = any_cast_safe<std::int64_t>(*value)) {
            return Ok(*int_val);
        }

        // Try to convert string to int
        try {
            auto str_val = any_to_string(*value);
            return Ok(std::stoll(str_val));
        } catch (...) {
            return Err(ConfigError::TypeMismatch);
        }
    }

    if (default_value) {
        return Ok(*default_value);
    }

    return Err(ConfigError::KeyNotFound);
}

auto ConfigurationLoader::get_float(const std::string& key,
                                    const std::optional<double>& default_value) const
    -> Result<double, ConfigError> {
    if (auto value = find_nested_value(key)) {
        if (auto float_val = any_cast_safe<double>(*value)) {
            return Ok(*float_val);
        }

        // Try to convert string to float
        try {
            auto str_val = any_to_string(*value);
            return Ok(std::stod(str_val));
        } catch (...) {
            return Err(ConfigError::TypeMismatch);
        }
    }

    if (default_value) {
        return Ok(*default_value);
    }

    return Err(ConfigError::KeyNotFound);
}

auto ConfigurationLoader::get_dict(const std::string& key) const
    -> Result<std::map<std::string, std::any>, ConfigError> {
    if (auto value = find_nested_value(key)) {
        if (auto dict_val = any_cast_safe<std::map<std::string, std::any>>(*value)) {
            return Ok(*dict_val);
        }
        return Err(ConfigError::TypeMismatch);
    }

    return Err(ConfigError::KeyNotFound);
}

auto ConfigurationLoader::get_list(const std::string& key) const
    -> Result<std::vector<std::any>, ConfigError> {
    if (auto value = find_nested_value(key)) {
        if (auto list_val = any_cast_safe<std::vector<std::any>>(*value)) {
            return Ok(*list_val);
        }
        return Err(ConfigError::TypeMismatch);
    }

    return Err(ConfigError::KeyNotFound);
}

auto ConfigurationLoader::has(const std::string& key) const -> bool {
    return find_nested_value(key).has_value();
}

auto ConfigurationLoader::keys(const std::optional<std::string>& prefix) const
    -> std::vector<std::string> {
    std::vector<std::string> result;

    auto collect_keys = [&](const std::map<std::string, std::any>& dict,
                            const std::string& current_prefix) {
        auto collect_recursive = [&](const std::map<std::string, std::any>& d,
                                     const std::string& p,
                                     auto& self) -> void {
            for (const auto& [key, value] : d) {
                std::string full_key = p.empty() ? key : p + "." + key;
                result.push_back(full_key);

                if (auto nested_dict = any_cast_safe<std::map<std::string, std::any>>(value)) {
                    self(*nested_dict, full_key, self);
                }
            }
        };
        collect_recursive(dict, current_prefix, collect_recursive);
    };

    if (prefix) {
        if (auto nested_dict = get_dict(*prefix)) {
            collect_keys(*nested_dict, *prefix);
        }
    } else {
        collect_keys(config_, "");
    }

    return result;
}

auto ConfigurationLoader::validate(const std::optional<std::map<std::string, std::any>>& config)
    const -> std::vector<std::string> {
    if (!schema_path_ || !std::filesystem::exists(*schema_path_)) {
        return {"Schema file not found or not specified"};
    }

    ConfigValidator validator(schema_path_);
    return validator.validate(config ? *config : config_, validate_strict_);
}

auto ConfigurationLoader::debug_mode() const -> bool {
    return get_bool("application.debug_mode", false).unwrap_or(false);
}

auto ConfigurationLoader::instance() -> ConfigurationLoader& {
    if (!global_instance_) {
        global_instance_ = std::make_unique<ConfigurationLoader>();
        global_instance_->load();
    }
    return *global_instance_;
}

auto ConfigurationLoader::set_instance(std::unique_ptr<ConfigurationLoader> loader) -> void {
    global_instance_ = std::move(loader);
}

auto ConfigurationLoader::resolve_environment_variables(const std::string& text) const
    -> std::string {
    EnvironmentResolver resolver;
    std::string result = text;

    std::smatch match;
    std::string::const_iterator start = text.cbegin();
    std::string output;

    while (std::regex_search(start, text.cend(), match, env_var_pattern_)) {
        output.append(start, match[0].first);

        std::string var_name = match[1].str();
        std::string default_value = match.size() > 2 ? match[2].str() : "";

        if (const char* env_value = std::getenv(var_name.c_str())) {
            output.append(env_value);
        } else if (!default_value.empty()) {
            output.append(default_value);
        } else {
            Logger::warning("Environment variable not found: {}", var_name);
            output.append(match[0].str());  // Keep original placeholder
        }

        start = match[0].second;
    }
    output.append(start, text.cend());

    return output;
}

auto ConfigurationLoader::resolve_config_recursive(std::map<std::string, std::any>& config) const
    -> void {
    EnvironmentResolver resolver;
    resolver.resolve(config);
}

auto ConfigurationLoader::apply_environment_overrides(std::map<std::string, std::any>& config) const
    -> void {
    // Look for environment-specific configuration section
    std::string env_key = "environments." + environment_;

    if (auto env_overrides = find_nested_value(env_key)) {
        if (auto env_dict = any_cast_safe<std::map<std::string, std::any>>(*env_overrides)) {
            Logger::info("Applying {} environment overrides", environment_);
            merge_configs(config, *env_dict);
        }
    }
}

auto ConfigurationLoader::merge_configs(std::map<std::string, std::any>& base,
                                        const std::map<std::string, std::any>& overlay) const
    -> void {
    for (const auto& [key, value] : overlay) {
        if (auto base_dict = any_cast_safe<std::map<std::string, std::any>>(base[key])) {
            if (auto overlay_dict = any_cast_safe<std::map<std::string, std::any>>(value)) {
                merge_configs(*base_dict, *overlay_dict);
                continue;
            }
        }
        base[key] = value;
    }
}

auto ConfigurationLoader::find_nested_value(const std::string& key_path) const
    -> std::optional<std::any> {
    std::vector<std::string> keys;
    std::stringstream ss(key_path);
    std::string key;

    while (std::getline(ss, key, '.')) {
        keys.push_back(key);
    }

    const std::map<std::string, std::any>* current_dict = &config_;

    for (size_t i = 0; i < keys.size() - 1; ++i) {
        auto it = current_dict->find(keys[i]);
        if (it == current_dict->end()) {
            return std::nullopt;
        }

        if (auto nested_dict = any_cast_safe<std::map<std::string, std::any>>(it->second)) {
            current_dict = &(*nested_dict);
        } else {
            return std::nullopt;
        }
    }

    auto it = current_dict->find(keys.back());
    if (it != current_dict->end()) {
        return it->second;
    }

    return std::nullopt;
}

auto ConfigurationLoader::auto_discover_config_path() const
    -> std::optional<std::filesystem::path> {
    std::vector<std::filesystem::path> search_paths = {
        "config/inference_lab_config.yaml",
        "../config/inference_lab_config.yaml",
        "inference_lab_config.yaml",
        std::filesystem::path(std::getenv("HOME") ? std::getenv("HOME") : "") / ".config" /
            "inference_lab" / "config.yaml"};

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            Logger::info("Auto-discovered configuration file: {}", path.string());
            return path;
        }
    }

    return std::nullopt;
}

auto ConfigurationLoader::auto_discover_schema_path() const
    -> std::optional<std::filesystem::path> {
    std::vector<std::filesystem::path> search_paths = {"config/schema/inference_config.json",
                                                       "../config/schema/inference_config.json",
                                                       "inference_config_schema.json"};

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            Logger::info("Auto-discovered schema file: {}", path.string());
            return path;
        }
    }

    return std::nullopt;
}

// EnvironmentResolver implementation
auto EnvironmentResolver::resolve(std::map<std::string, std::any>& config) const -> void {
    for (auto& [key, value] : config) {
        resolve_value(value);
    }
}

auto EnvironmentResolver::resolve_value(std::any& value) const -> void {
    if (value.type() == typeid(std::string)) {
        auto str_value = std::any_cast<std::string>(value);
        resolve_string(str_value);
        value = str_value;
    } else if (value.type() == typeid(std::map<std::string, std::any>)) {
        auto dict_value = std::any_cast<std::map<std::string, std::any>>(value);
        resolve(dict_value);
        value = dict_value;
    } else if (value.type() == typeid(std::vector<std::any>)) {
        auto list_value = std::any_cast<std::vector<std::any>>(value);
        for (auto& item : list_value) {
            resolve_value(item);
        }
        value = list_value;
    }
}

auto EnvironmentResolver::resolve_string(std::string& text) const -> void {
    std::smatch match;
    std::string::const_iterator start = text.cbegin();
    std::string output;

    while (std::regex_search(start, text.cend(), match, env_var_pattern_)) {
        output.append(start, match[0].first);

        std::string var_name = match[1].str();
        std::optional<std::string> default_value =
            match.size() > 2 ? std::make_optional(match[2].str()) : std::nullopt;

        if (auto env_value = get_env_var(var_name, default_value)) {
            output.append(*env_value);
        } else {
            output.append(match[0].str());  // Keep original placeholder
        }

        start = match[0].second;
    }
    output.append(start, text.cend());

    text = output;
}

auto EnvironmentResolver::get_env_var(const std::string& var_name,
                                      const std::optional<std::string>& default_value) const
    -> std::optional<std::string> {
    if (const char* env_value = std::getenv(var_name.c_str())) {
        return std::string(env_value);
    }

    return default_value;
}

// ConfigValidator implementation
ConfigValidator::ConfigValidator(const std::optional<std::filesystem::path>& schema_path)
    : schema_path_(schema_path), schema_loaded_(false) {
    if (schema_path_ && std::filesystem::exists(*schema_path_)) {
        schema_loaded_ = true;
    }
}

auto ConfigValidator::validate(const std::map<std::string, std::any>& config, bool strict) const
    -> std::vector<std::string> {
    std::vector<std::string> errors;

    if (!schema_loaded_) {
        if (strict) {
            errors.push_back("Schema validation is enabled but schema file is not available");
        }
        return errors;
    }

    // Basic validation - check required fields
    auto check_required_field = [&](const std::string& field_path) {
        // Simple dot-notation path checking
        // This is a basic implementation - a full JSON Schema validator would be more comprehensive
        bool found = false;

        std::vector<std::string> keys;
        std::stringstream ss(field_path);
        std::string key;
        while (std::getline(ss, key, '.')) {
            keys.push_back(key);
        }

        const std::map<std::string, std::any>* current = &config;
        for (const auto& k : keys) {
            auto it = current->find(k);
            if (it == current->end()) {
                break;
            }

            if (k == keys.back()) {
                found = true;
                break;
            }

            try {
                current = &std::any_cast<const std::map<std::string, std::any>&>(it->second);
            } catch (...) {
                break;
            }
        }

        if (!found) {
            errors.push_back("Required field missing: " + field_path);
        }
    };

    // Check essential required fields
    check_required_field("schema_version");
    check_required_field("application.name");
    check_required_field("application.version");
    check_required_field("application.environment");

    return errors;
}

}  // namespace inference_lab
