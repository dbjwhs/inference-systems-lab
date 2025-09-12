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
