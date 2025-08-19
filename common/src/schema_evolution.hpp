// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file schema_evolution.hpp
 * @brief Schema versioning and evolution utilities for Cap'n Proto inference types
 *
 * This file provides comprehensive support for schema versioning and evolution,
 * allowing safe migration between different versions of the inference engine's
 * data formats. Key features include:
 *
 * - Semantic versioning support with compatibility checking
 * - Automatic migration between compatible schema versions
 * - Validation of schema evolution rules and constraints
 * - Backward compatibility layer for older data formats
 * - Migration path planning and execution
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Include generated Cap'n Proto headers
#include <capnp/message.h>

#include "inference_types.hpp"
#include "schemas/inference_types.capnp.h"

namespace inference_lab::common::evolution {

/**
 * @class SchemaVersion
 * @brief Represents a semantic version of the schema with compatibility information
 */
class SchemaVersion {
  public:
    /**
     * @brief Construct a new SchemaVersion
     * @param major Major version number (breaking changes)
     * @param minor Minor version number (backward compatible features)
     * @param patch Patch version number (backward compatible fixes)
     * @param schemaHash Optional hash of the schema definition for integrity checking
     */
    SchemaVersion(uint32_t major,
                  uint32_t minor,
                  uint32_t patch,
                  const std::string& schema_hash = "");

    /**
     * @brief Parse version from string format (e.g., "1.2.3")
     * @param versionString String in format "major.minor.patch"
     * @return SchemaVersion object if parsing succeeds, nullopt otherwise
     */
    static auto from_string(const std::string& version_string) -> std::optional<SchemaVersion>;

    // Accessor methods
    auto get_major() const -> uint32_t { return major_; }
    auto get_minor() const -> uint32_t { return minor_; }
    auto get_patch() const -> uint32_t { return patch_; }
    auto get_version_string() const -> std::string;
    auto get_schema_hash() const -> const std::string& { return schema_hash_; }

    // Compatibility checking methods

    /**
     * @brief Check if this version is compatible with another version
     * @param other Version to check compatibility with
     * @return true if versions are compatible (can read each other's data)
     */
    auto is_compatible_with(const SchemaVersion& other) const -> bool;

    /**
     * @brief Check if this version can read data from an older version
     * @param older Older version to check forward compatibility with
     * @return true if this version can read older version's data
     */
    auto is_forward_compatible_with(const SchemaVersion& older) const -> bool;

    /**
     * @brief Check if this version's data can be read by a newer version
     * @param newer Newer version to check backward compatibility with
     * @return true if newer version can read this version's data
     */
    auto is_backward_compatible_with(const SchemaVersion& newer) const -> bool;

    // Comparison operators for sorting and ordering
    auto operator==(const SchemaVersion& other) const -> bool;
    auto operator!=(const SchemaVersion& other) const -> bool;
    auto operator<(const SchemaVersion& other) const -> bool;
    auto operator<=(const SchemaVersion& other) const -> bool;
    auto operator>(const SchemaVersion& other) const -> bool;
    auto operator>=(const SchemaVersion& other) const -> bool;

    /**
     * @brief Generate string representation for debugging
     * @return String in format "major.minor.patch [hash]"
     */
    auto to_string() const -> std::string;

    // Cap'n Proto interoperability
    explicit SchemaVersion(schemas::SchemaVersion::Reader reader);
    void write_to(schemas::SchemaVersion::Builder builder) const;

  private:
    uint32_t major_{};
    uint32_t minor_{};
    uint32_t patch_{};
    std::string schema_hash_{};
};

/**
 * @class MigrationPath
 * @brief Describes how to migrate data between two schema versions
 */
class MigrationPath {
  public:
    /**
     * @enum Strategy
     * @brief Migration strategies for different types of schema changes
     */
    enum class Strategy : std::uint8_t {  // NOLINT(performance-enum-size) - false positive, uint8_t
                                          // is correct
        DIRECT_MAPPING,                   ///< Direct field mapping (no data loss)
        TRANSFORMATION,                   ///< Field transformation required
        DEFAULT_VALUES,                   ///< Default values for new fields
        CUSTOM_LOGIC,                     ///< Custom migration logic required
        LOSSY                             ///< Data may be lost in migration
    };

    /**
     * @brief Construct a new MigrationPath
     * @param fromVersion Source schema version
     * @param toVersion Target schema version
     * @param strategy Migration strategy to use
     * @param reversible Whether this migration can be reversed
     * @param description Human-readable description of the migration
     */
    MigrationPath(const SchemaVersion& from_version,
                  const SchemaVersion& to_version,
                  Strategy strategy,
                  bool reversible = false,
                  const std::string& description = "");

    // Accessor methods
    auto get_from_version() const -> const SchemaVersion& { return from_version_; }
    auto get_to_version() const -> const SchemaVersion& { return to_version_; }
    auto get_strategy() const -> Strategy { return strategy_; }
    auto is_reversible() const -> bool { return reversible_; }
    auto get_description() const -> const std::string& { return description_; }
    auto get_warnings() const -> const std::vector<std::string>& { return warnings_; }

    /**
     * @brief Add a warning about this migration
     * @param warning Warning message to add
     */
    void add_warning(const std::string& warning);

    /**
     * @brief Check if this migration path can handle the version transition
     * @param from Source version
     * @param to Target version
     * @return true if this path can migrate from 'from' to 'to'
     */
    auto can_migrate(const SchemaVersion& from, const SchemaVersion& to) const -> bool;

    /**
     * @brief Generate string representation for debugging
     */
    auto to_string() const -> std::string;

    // Cap'n Proto interoperability
    explicit MigrationPath(schemas::MigrationPath::Reader reader);
    void write_to(schemas::MigrationPath::Builder builder) const;

  private:
    SchemaVersion from_version_;
    SchemaVersion to_version_;
    Strategy strategy_;
    bool reversible_;
    std::string description_{};
    std::vector<std::string> warnings_{};
};

/**
 * @class SchemaEvolutionManager
 * @brief Manages schema versioning and data migration for the inference engine
 */
class SchemaEvolutionManager {
  public:
    /**
     * @brief Construct a new SchemaEvolutionManager
     * @param currentVersion Current schema version being used
     */
    explicit SchemaEvolutionManager(const SchemaVersion& current_version);

    /**
     * @brief Register a migration path between two schema versions
     * @param path Migration path to register
     */
    void register_migration_path(const MigrationPath& path);

    /**
     * @brief Check if data with a given schema version can be read
     * @param dataVersion Version of the data to check
     * @return true if data can be read (possibly with migration)
     */
    auto can_read_version(const SchemaVersion& data_version) const -> bool;

    /**
     * @brief Find the migration path needed to read data from a specific version
     * @param fromVersion Version of the data to migrate from
     * @return Migration path if available, nullopt if no path exists
     */
    auto find_migration_path(const SchemaVersion& from_version) const
        -> std::optional<MigrationPath>;

    /**
     * @brief Get all supported versions that can be read by this manager
     * @return Vector of all supported schema versions
     */
    auto get_supported_versions() const -> std::vector<SchemaVersion>;

    /**
     * @brief Validate that a schema evolution is safe and follows best practices
     * @param evolution Schema evolution metadata to validate
     * @return Vector of validation errors (empty if valid)
     */
    auto validate_evolution(const schemas::SchemaEvolution::Reader& evolution) const
        -> std::vector<std::string>;

    /**
     * @brief Create schema evolution metadata for the current state
     * @return SchemaEvolution structure with current version and migration paths
     */
    auto create_evolution_metadata(capnp::MessageBuilder& message) const
        -> schemas::SchemaEvolution::Builder;

    // Migration execution methods

    /**
     * @brief Migrate a Fact from an older schema version to the current version
     * @param fact Fact to migrate
     * @param sourceVersion Version the fact was created with
     * @return Migrated fact if successful, nullopt if migration failed
     */
    auto migrate_fact(const Fact& fact, const SchemaVersion& source_version) const
        -> std::optional<Fact>;

    /**
     * @brief Migrate a Rule from an older schema version to the current version
     * @param rule Rule to migrate
     * @param sourceVersion Version the rule was created with
     * @return Migrated rule if successful, nullopt if migration failed
     */
    auto migrate_rule(const Rule& rule, const SchemaVersion& source_version) const
        -> std::optional<Rule>;

    /**
     * @brief Generate a compatibility matrix showing relationships between versions
     * @return String representation of the compatibility matrix
     */
    auto generate_compatibility_matrix() const -> std::string;

    // Current version info
    auto get_current_version() const -> const SchemaVersion& { return current_version_; }

  private:
    SchemaVersion current_version_;
    std::vector<MigrationPath> migration_paths_{};
    std::unordered_map<std::string, size_t> path_index_{};  // version_string -> index in
                                                            // migrationPaths_

    /**
     * @brief Apply default values migration strategy
     * @param sourceVersion Source schema version
     * @return true if migration succeeded
     */
    auto apply_default_values_migration(const SchemaVersion& source_version) const -> bool;

    /**
     * @brief Apply field transformation migration strategy
     * @param sourceVersion Source schema version
     * @return true if migration succeeded
     */
    auto apply_transformation_migration(const SchemaVersion& source_version) const -> bool;
};

/**
 * @class VersionValidator
 * @brief Utility class for validating schema versions and evolution rules
 */
class VersionValidator {
  public:
    /**
     * @brief Validate that a schema version follows semantic versioning rules
     * @param version Version to validate
     * @return Vector of validation errors (empty if valid)
     */
    static auto validate_version(const SchemaVersion& version) -> std::vector<std::string>;

    /**
     * @brief Validate that a migration path follows evolution best practices
     * @param path Migration path to validate
     * @return Vector of validation errors (empty if valid)
     */
    static auto validate_migration_path(const MigrationPath& path) -> std::vector<std::string>;

    /**
     * @brief Check if a version transition is safe according to semantic versioning rules
     * @param from Source version
     * @param to Target version
     * @return true if transition is safe, false otherwise
     */
    static auto is_safe_transition(const SchemaVersion& from, const SchemaVersion& to) -> bool;

    /**
     * @brief Generate warnings for potentially risky schema changes
     * @param from Source version
     * @param to Target version
     * @return Vector of warning messages
     */
    static auto generate_warnings(const SchemaVersion& from, const SchemaVersion& to)
        -> std::vector<std::string>;
};

/**
 * @class SchemaRegistry
 * @brief Central registry for tracking schema versions and their definitions
 */
class SchemaRegistry {
  public:
    /**
     * @brief Get the singleton instance of the schema registry
     */
    static auto get_instance() -> SchemaRegistry&;

    /**
     * @brief Register a schema version with its definition hash
     * @param version Schema version to register
     * @param schemaHash Hash of the schema definition
     */
    void register_schema(const SchemaVersion& version, const std::string& schema_hash);

    /**
     * @brief Get the current active schema version
     */
    auto get_current_schema() const -> const SchemaVersion&;

    /**
     * @brief Set the current active schema version
     * @param version Version to set as current
     */
    void set_current_schema(const SchemaVersion& version);

    /**
     * @brief Check if a schema version is registered
     * @param version Version to check
     * @return true if version is registered
     */
    auto is_registered(const SchemaVersion& version) const -> bool;

    /**
     * @brief Get all registered schema versions
     * @return Vector of all registered versions, sorted by version number
     */
    auto get_all_versions() const -> std::vector<SchemaVersion>;

  private:
    SchemaRegistry() = default;

    std::vector<SchemaVersion> registered_versions_{};
    SchemaVersion current_version_{1, 0, 0};  // Default to 1.0.0
};

// Convenience constants for the current schema version
constexpr uint32_t CURRENT_SCHEMA_MAJOR = 1;
constexpr uint32_t CURRENT_SCHEMA_MINOR = 0;
constexpr uint32_t CURRENT_SCHEMA_PATCH = 0;

/**
 * @brief Get the current schema version as a constant
 * @return SchemaVersion representing the current version (1.0.0)
 */
inline auto get_current_schema_version() -> SchemaVersion {
    return SchemaVersion(CURRENT_SCHEMA_MAJOR, CURRENT_SCHEMA_MINOR, CURRENT_SCHEMA_PATCH);
}

}  // namespace inference_lab::common::evolution
