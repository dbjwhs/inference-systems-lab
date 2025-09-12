// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "result.hpp"

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

// Simplified configuration loader that doesn't require YAML
class SimpleConfigurationLoader {
  public:
    explicit SimpleConfigurationLoader(
        const std::optional<std::filesystem::path>& config_path = std::nullopt,
        const std::string& environment = "development");

    auto load() -> common::Result<bool, ConfigError>;

    auto get(const std::string& key, const std::string& default_value = "") const -> std::string;
    auto get_bool(const std::string& key, bool default_value = false) const -> bool;
    auto get_int(const std::string& key, int default_value = 0) const -> int;

    auto environment_name() const -> const std::string& { return environment_; }
    auto is_development() const -> bool { return environment_ == "development"; }
    auto is_staging() const -> bool { return environment_ == "staging"; }
    auto is_production() const -> bool { return environment_ == "production"; }

  private:
    std::filesystem::path config_path_;
    std::string environment_;
    std::map<std::string, std::string> config_;
};

}  // namespace inference_lab
