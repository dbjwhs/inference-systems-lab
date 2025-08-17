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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Include generated Cap'n Proto headers
#include <capnp/message.h>
#include <capnp/serialize.h>

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
                  const std::string& schemaHash = "");

    /**
     * @brief Parse version from string format (e.g., "1.2.3")
     * @param versionString String in format "major.minor.patch"
     * @return SchemaVersion object if parsing succeeds, nullopt otherwise
     */
    static std::optional<SchemaVersion> fromString(const std::string& versionString);

    // Accessor methods
    uint32_t getMajor() const { return major_; }
    uint32_t getMinor() const { return minor_; }
    uint32_t getPatch() const { return patch_; }
    std::string getVersionString() const;
    const std::string& getSchemaHash() const { return schemaHash_; }

    // Compatibility checking methods

    /**
     * @brief Check if this version is compatible with another version
     * @param other Version to check compatibility with
     * @return true if versions are compatible (can read each other's data)
     */
    bool isCompatibleWith(const SchemaVersion& other) const;

    /**
     * @brief Check if this version can read data from an older version
     * @param older Older version to check forward compatibility with
     * @return true if this version can read older version's data
     */
    bool isForwardCompatibleWith(const SchemaVersion& older) const;

    /**
     * @brief Check if this version's data can be read by a newer version
     * @param newer Newer version to check backward compatibility with
     * @return true if newer version can read this version's data
     */
    bool isBackwardCompatibleWith(const SchemaVersion& newer) const;

    // Comparison operators for sorting and ordering
    bool operator==(const SchemaVersion& other) const;
    bool operator!=(const SchemaVersion& other) const;
    bool operator<(const SchemaVersion& other) const;
    bool operator<=(const SchemaVersion& other) const;
    bool operator>(const SchemaVersion& other) const;
    bool operator>=(const SchemaVersion& other) const;

    /**
     * @brief Generate string representation for debugging
     * @return String in format "major.minor.patch [hash]"
     */
    std::string toString() const;

    // Cap'n Proto interoperability
    explicit SchemaVersion(schemas::SchemaVersion::Reader reader);
    void writeTo(schemas::SchemaVersion::Builder builder) const;

  private:
    uint32_t major_;
    uint32_t minor_;
    uint32_t patch_;
    std::string schemaHash_;
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
    enum class Strategy {
        DirectMapping,   ///< Direct field mapping (no data loss)
        Transformation,  ///< Field transformation required
        DefaultValues,   ///< Default values for new fields
        CustomLogic,     ///< Custom migration logic required
        Lossy            ///< Data may be lost in migration
    };

    /**
     * @brief Construct a new MigrationPath
     * @param fromVersion Source schema version
     * @param toVersion Target schema version
     * @param strategy Migration strategy to use
     * @param reversible Whether this migration can be reversed
     * @param description Human-readable description of the migration
     */
    MigrationPath(const SchemaVersion& fromVersion,
                  const SchemaVersion& toVersion,
                  Strategy strategy,
                  bool reversible = false,
                  const std::string& description = "");

    // Accessor methods
    const SchemaVersion& getFromVersion() const { return fromVersion_; }
    const SchemaVersion& getToVersion() const { return toVersion_; }
    Strategy getStrategy() const { return strategy_; }
    bool isReversible() const { return reversible_; }
    const std::string& getDescription() const { return description_; }
    const std::vector<std::string>& getWarnings() const { return warnings_; }

    /**
     * @brief Add a warning about this migration
     * @param warning Warning message to add
     */
    void addWarning(const std::string& warning);

    /**
     * @brief Check if this migration path can handle the version transition
     * @param from Source version
     * @param to Target version
     * @return true if this path can migrate from 'from' to 'to'
     */
    bool canMigrate(const SchemaVersion& from, const SchemaVersion& to) const;

    /**
     * @brief Generate string representation for debugging
     */
    std::string toString() const;

    // Cap'n Proto interoperability
    explicit MigrationPath(schemas::MigrationPath::Reader reader);
    void writeTo(schemas::MigrationPath::Builder builder) const;

  private:
    SchemaVersion fromVersion_;
    SchemaVersion toVersion_;
    Strategy strategy_;
    bool reversible_;
    std::string description_;
    std::vector<std::string> warnings_;
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
    explicit SchemaEvolutionManager(const SchemaVersion& currentVersion);

    /**
     * @brief Register a migration path between two schema versions
     * @param path Migration path to register
     */
    void registerMigrationPath(const MigrationPath& path);

    /**
     * @brief Check if data with a given schema version can be read
     * @param dataVersion Version of the data to check
     * @return true if data can be read (possibly with migration)
     */
    bool canReadVersion(const SchemaVersion& dataVersion) const;

    /**
     * @brief Find the migration path needed to read data from a specific version
     * @param fromVersion Version of the data to migrate from
     * @return Migration path if available, nullopt if no path exists
     */
    std::optional<MigrationPath> findMigrationPath(const SchemaVersion& fromVersion) const;

    /**
     * @brief Get all supported versions that can be read by this manager
     * @return Vector of all supported schema versions
     */
    std::vector<SchemaVersion> getSupportedVersions() const;

    /**
     * @brief Validate that a schema evolution is safe and follows best practices
     * @param evolution Schema evolution metadata to validate
     * @return Vector of validation errors (empty if valid)
     */
    std::vector<std::string> validateEvolution(
        const schemas::SchemaEvolution::Reader& evolution) const;

    /**
     * @brief Create schema evolution metadata for the current state
     * @return SchemaEvolution structure with current version and migration paths
     */
    schemas::SchemaEvolution::Builder createEvolutionMetadata(capnp::MessageBuilder& message) const;

    // Migration execution methods

    /**
     * @brief Migrate a Fact from an older schema version to the current version
     * @param fact Fact to migrate
     * @param sourceVersion Version the fact was created with
     * @return Migrated fact if successful, nullopt if migration failed
     */
    std::optional<Fact> migrateFact(const Fact& fact, const SchemaVersion& sourceVersion) const;

    /**
     * @brief Migrate a Rule from an older schema version to the current version
     * @param rule Rule to migrate
     * @param sourceVersion Version the rule was created with
     * @return Migrated rule if successful, nullopt if migration failed
     */
    std::optional<Rule> migrateRule(const Rule& rule, const SchemaVersion& sourceVersion) const;

    /**
     * @brief Generate a compatibility matrix showing relationships between versions
     * @return String representation of the compatibility matrix
     */
    std::string generateCompatibilityMatrix() const;

    // Current version info
    const SchemaVersion& getCurrentVersion() const { return currentVersion_; }

  private:
    SchemaVersion currentVersion_;
    std::vector<MigrationPath> migrationPaths_;
    std::unordered_map<std::string, size_t> pathIndex_;  // version_string -> index in
                                                         // migrationPaths_

    /**
     * @brief Apply default values migration strategy
     * @param sourceVersion Source schema version
     * @return true if migration succeeded
     */
    bool applyDefaultValuesMigration(const SchemaVersion& sourceVersion) const;

    /**
     * @brief Apply field transformation migration strategy
     * @param sourceVersion Source schema version
     * @return true if migration succeeded
     */
    bool applyTransformationMigration(const SchemaVersion& sourceVersion) const;
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
    static std::vector<std::string> validateVersion(const SchemaVersion& version);

    /**
     * @brief Validate that a migration path follows evolution best practices
     * @param path Migration path to validate
     * @return Vector of validation errors (empty if valid)
     */
    static std::vector<std::string> validateMigrationPath(const MigrationPath& path);

    /**
     * @brief Check if a version transition is safe according to semantic versioning rules
     * @param from Source version
     * @param to Target version
     * @return true if transition is safe, false otherwise
     */
    static bool isSafeTransition(const SchemaVersion& from, const SchemaVersion& to);

    /**
     * @brief Generate warnings for potentially risky schema changes
     * @param from Source version
     * @param to Target version
     * @return Vector of warning messages
     */
    static std::vector<std::string> generateWarnings(const SchemaVersion& from,
                                                     const SchemaVersion& to);
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
    static SchemaRegistry& getInstance();

    /**
     * @brief Register a schema version with its definition hash
     * @param version Schema version to register
     * @param schemaHash Hash of the schema definition
     */
    void registerSchema(const SchemaVersion& version, const std::string& schemaHash);

    /**
     * @brief Get the current active schema version
     */
    const SchemaVersion& getCurrentSchema() const;

    /**
     * @brief Set the current active schema version
     * @param version Version to set as current
     */
    void setCurrentSchema(const SchemaVersion& version);

    /**
     * @brief Check if a schema version is registered
     * @param version Version to check
     * @return true if version is registered
     */
    bool isRegistered(const SchemaVersion& version) const;

    /**
     * @brief Get all registered schema versions
     * @return Vector of all registered versions, sorted by version number
     */
    std::vector<SchemaVersion> getAllVersions() const;

  private:
    SchemaRegistry() = default;

    std::vector<SchemaVersion> registeredVersions_;
    SchemaVersion currentVersion_{1, 0, 0};  // Default to 1.0.0
};

// Convenience constants for the current schema version
constexpr uint32_t CURRENT_SCHEMA_MAJOR = 1;
constexpr uint32_t CURRENT_SCHEMA_MINOR = 0;
constexpr uint32_t CURRENT_SCHEMA_PATCH = 0;

/**
 * @brief Get the current schema version as a constant
 * @return SchemaVersion representing the current version (1.0.0)
 */
inline SchemaVersion getCurrentSchemaVersion() {
    return SchemaVersion(CURRENT_SCHEMA_MAJOR, CURRENT_SCHEMA_MINOR, CURRENT_SCHEMA_PATCH);
}

}  // namespace inference_lab::common::evolution
