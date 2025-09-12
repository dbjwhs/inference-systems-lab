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

#include "config_loader_simple.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "logging.hpp"

namespace inference_lab {

SimpleConfigurationLoader::SimpleConfigurationLoader(
    const std::optional<std::filesystem::path>& config_path, const std::string& environment)
    : environment_(environment) {
    if (config_path) {
        config_path_ = *config_path;
    } else {
        // Auto-discover configuration file
        std::vector<std::filesystem::path> search_paths = {"config/inference_lab_config.yaml",
                                                           "../config/inference_lab_config.yaml",
                                                           "inference_lab_config.yaml"};

        for (const auto& path : search_paths) {
            if (std::filesystem::exists(path)) {
                config_path_ = path;
                break;
            }
        }
    }
}

auto SimpleConfigurationLoader::load() -> common::Result<bool, ConfigError> {
    if (config_path_.empty() || !std::filesystem::exists(config_path_)) {
        // Configuration file not found, using defaults

        // Load default configuration
        config_["application.name"] = "inference-systems-lab";
        config_["application.version"] = "0.1.0";
        config_["application.environment"] = environment_;
        config_["application.debug_mode"] = environment_ == "development" ? "true" : "false";
        config_["logging.level"] = environment_ == "development" ? "DEBUG" : "INFO";

        return common::Ok(true);
    }

    // Configuration file found but YAML parsing not available in this build
    // Using default configuration values

    // Load default configuration
    config_["application.name"] = "inference-systems-lab";
    config_["application.version"] = "0.1.0";
    config_["application.environment"] = environment_;
    config_["application.debug_mode"] = environment_ == "development" ? "true" : "false";
    config_["logging.level"] = environment_ == "development" ? "DEBUG" : "INFO";

    return common::Ok(true);
}

auto SimpleConfigurationLoader::get(const std::string& key, const std::string& default_value) const
    -> std::string {
    auto it = config_.find(key);
    if (it != config_.end()) {
        return it->second;
    }
    return default_value;
}

auto SimpleConfigurationLoader::get_bool(const std::string& key, bool default_value) const -> bool {
    auto value = get(key);
    if (value.empty()) {
        return default_value;
    }

    return value == "true" || value == "1" || value == "yes";
}

auto SimpleConfigurationLoader::get_int(const std::string& key, int default_value) const -> int {
    auto value = get(key);
    if (value.empty()) {
        return default_value;
    }

    try {
        return std::stoi(value);
    } catch (...) {
        return default_value;
    }
}

}  // namespace inference_lab
