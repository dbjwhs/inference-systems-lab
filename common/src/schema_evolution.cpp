// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#include "schema_evolution.hpp"

#include <algorithm>
#include <cassert>
#include <regex>
#include <sstream>

#include "inference_types.hpp"

namespace inference_lab::common::evolution {

// SchemaVersion implementation
SchemaVersion::SchemaVersion(uint32_t major,
                             uint32_t minor,
                             uint32_t patch,
                             const std::string& schemaHash)
    : major_(major), minor_(minor), patch_(patch), schema_hash_(schemaHash) {}

std::optional<SchemaVersion> SchemaVersion::from_string(const std::string& versionString) {
    std::regex versionRegex(R"((\d+)\.(\d+)\.(\d+))");
    std::smatch matches;

    if (std::regex_match(versionString, matches, versionRegex)) {
        try {
            uint32_t major = std::stoul(matches[1].str());
            uint32_t minor = std::stoul(matches[2].str());
            uint32_t patch = std::stoul(matches[3].str());
            return SchemaVersion(major, minor, patch);
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::string SchemaVersion::get_version_string() const {
    std::ostringstream oss;
    oss << major_ << "." << minor_ << "." << patch_;
    return oss.str();
}

bool SchemaVersion::is_compatible_with(const SchemaVersion& other) const {
    // Same major version means compatible
    return major_ == other.major_;
}

bool SchemaVersion::is_forward_compatible_with(const SchemaVersion& older) const {
    // Can read older data if major version is same and this version is newer or equal
    return major_ == older.major_ &&
           (minor_ > older.minor_ || (minor_ == older.minor_ && patch_ >= older.patch_));
}

bool SchemaVersion::is_backward_compatible_with(const SchemaVersion& newer) const {
    // Newer version can read this data if major is same and newer is actually newer
    return newer.is_forward_compatible_with(*this);
}

bool SchemaVersion::operator==(const SchemaVersion& other) const {
    return major_ == other.major_ && minor_ == other.minor_ && patch_ == other.patch_;
}

bool SchemaVersion::operator!=(const SchemaVersion& other) const {
    return !(*this == other);
}

bool SchemaVersion::operator<(const SchemaVersion& other) const {
    if (major_ != other.major_)
        return major_ < other.major_;
    if (minor_ != other.minor_)
        return minor_ < other.minor_;
    return patch_ < other.patch_;
}

bool SchemaVersion::operator<=(const SchemaVersion& other) const {
    return *this < other || *this == other;
}

bool SchemaVersion::operator>(const SchemaVersion& other) const {
    return other < *this;
}

bool SchemaVersion::operator>=(const SchemaVersion& other) const {
    return *this > other || *this == other;
}

std::string SchemaVersion::to_string() const {
    std::ostringstream oss;
    oss << get_version_string();
    if (!schema_hash_.empty()) {
        oss << " [" << schema_hash_.substr(0, 8) << "...]";
    }
    return oss.str();
}

SchemaVersion::SchemaVersion(schemas::SchemaVersion::Reader reader)
    : major_(reader.getMajor()),
      minor_(reader.getMinor()),
      patch_(reader.getPatch()),
      schema_hash_(reader.getSchemaHash()) {}

void SchemaVersion::write_to(schemas::SchemaVersion::Builder builder) const {
    builder.setMajor(major_);
    builder.setMinor(minor_);
    builder.setPatch(patch_);
    builder.setVersionString(get_version_string());
    builder.setMinCompatibleMajor(major_);  // For now, only same major version is compatible
    builder.setMinCompatibleMinor(0);
    builder.setSchemaHash(schema_hash_);
}

// MigrationPath implementation
MigrationPath::MigrationPath(const SchemaVersion& fromVersion,
                             const SchemaVersion& toVersion,
                             Strategy strategy,
                             bool reversible,
                             const std::string& description)
    : from_version_(fromVersion),
      to_version_(toVersion),
      strategy_(strategy),
      reversible_(reversible),
      description_(description) {}

void MigrationPath::add_warning(const std::string& warning) {
    warnings_.push_back(warning);
}

bool MigrationPath::can_migrate(const SchemaVersion& from, const SchemaVersion& to) const {
    return from_version_ == from && to_version_ == to;
}

std::string MigrationPath::to_string() const {
    std::ostringstream oss;
    oss << from_version_.to_string() << " -> " << to_version_.to_string();
    oss << " (" << static_cast<int>(strategy_) << ")";
    if (reversible_)
        oss << " [reversible]";
    if (!description_.empty())
        oss << ": " << description_;
    return oss.str();
}

MigrationPath::MigrationPath(schemas::MigrationPath::Reader reader)
    : from_version_(reader.getFromVersion()),
      to_version_(reader.getToVersion()),
      strategy_(static_cast<Strategy>(reader.getStrategy())),
      reversible_(reader.getReversible()),
      description_(reader.getDescription()) {
    auto warningsReader = reader.getWarnings();
    for (const auto& warning : warningsReader) {
        warnings_.push_back(warning);
    }
}

void MigrationPath::write_to(schemas::MigrationPath::Builder builder) const {
    from_version_.write_to(builder.initFromVersion());
    to_version_.write_to(builder.initToVersion());
    builder.setStrategy(static_cast<schemas::MigrationStrategy>(strategy_));
    builder.setReversible(reversible_);
    builder.setDescription(description_);

    auto warningsBuilder = builder.initWarnings(warnings_.size());
    for (size_t i = 0; i < warnings_.size(); ++i) {
        warningsBuilder.set(i, warnings_[i]);
    }
}

// SchemaEvolutionManager implementation
SchemaEvolutionManager::SchemaEvolutionManager(const SchemaVersion& currentVersion)
    : current_version_(currentVersion) {}

void SchemaEvolutionManager::register_migration_path(const MigrationPath& path) {
    migration_paths_.push_back(path);

    // Update index for quick lookup
    std::string key = path.get_from_version().get_version_string() + "->" +
                      path.get_to_version().get_version_string();
    path_index_[key] = migration_paths_.size() - 1;
}

bool SchemaEvolutionManager::can_read_version(const SchemaVersion& dataVersion) const {
    // Can always read current version
    if (dataVersion == current_version_) {
        return true;
    }

    // Check if migration path exists
    return find_migration_path(dataVersion).has_value();
}

std::optional<MigrationPath> SchemaEvolutionManager::find_migration_path(
    const SchemaVersion& fromVersion) const {
    // Direct migration to current version
    for (const auto& path : migration_paths_) {
        if (path.can_migrate(fromVersion, current_version_)) {
            return path;
        }
    }

    // TODO: Implement multi-step migration path finding
    // For now, only support direct migrations
    return std::nullopt;
}

std::vector<SchemaVersion> SchemaEvolutionManager::get_supported_versions() const {
    std::vector<SchemaVersion> versions;
    versions.push_back(current_version_);

    for (const auto& path : migration_paths_) {
        if (path.get_to_version() == current_version_) {
            versions.push_back(path.get_from_version());
        }
    }

    // Remove duplicates and sort
    std::sort(versions.begin(), versions.end());
    versions.erase(std::unique(versions.begin(), versions.end()), versions.end());

    return versions;
}

std::vector<std::string> SchemaEvolutionManager::validate_evolution(
    const schemas::SchemaEvolution::Reader& evolution) const {
    std::vector<std::string> errors;

    SchemaVersion currentVer(evolution.getCurrentVersion());

    // Validate current version
    auto versionErrors = VersionValidator::validate_version(currentVer);
    errors.insert(errors.end(), versionErrors.begin(), versionErrors.end());

    // Validate migration paths
    auto migrationPaths = evolution.getMigrationPaths();
    for (const auto& pathReader : migrationPaths) {
        MigrationPath path(pathReader);
        auto pathErrors = VersionValidator::validate_migration_path(path);
        errors.insert(errors.end(), pathErrors.begin(), pathErrors.end());
    }

    return errors;
}

schemas::SchemaEvolution::Builder SchemaEvolutionManager::create_evolution_metadata(
    capnp::MessageBuilder& message) const {
    auto builder = message.getRoot<schemas::SchemaEvolution>();

    // Set current version
    current_version_.write_to(builder.initCurrentVersion());

    // Set supported versions
    auto supportedVersions = get_supported_versions();
    auto supportedBuilder = builder.initSupportedVersions(supportedVersions.size());
    for (size_t i = 0; i < supportedVersions.size(); ++i) {
        supportedVersions[i].write_to(supportedBuilder[i]);
    }

    // Set migration paths
    auto pathsBuilder = builder.initMigrationPaths(migration_paths_.size());
    for (size_t i = 0; i < migration_paths_.size(); ++i) {
        migration_paths_[i].write_to(pathsBuilder[i]);
    }

    // Set timestamp
    builder.setEvolutionTimestamp(std::time(nullptr) * 1000);  // milliseconds

    return builder;
}

std::optional<Fact> SchemaEvolutionManager::migrate_fact(const Fact& fact,
                                                         const SchemaVersion& sourceVersion) const {
    if (sourceVersion == current_version_) {
        return fact;  // No migration needed
    }

    auto migrationPath = find_migration_path(sourceVersion);
    if (!migrationPath) {
        return std::nullopt;  // No migration path available
    }

    // For now, implement basic migration strategies
    switch (migrationPath->get_strategy()) {
        case MigrationPath::Strategy::DIRECT_MAPPING:
            return fact;  // No changes needed

        case MigrationPath::Strategy::DEFAULT_VALUES:
            // Add default schema version if not present
            return fact;  // Return as-is for now

        default:
            return std::nullopt;  // Unsupported migration strategy
    }
}

std::optional<Rule> SchemaEvolutionManager::migrate_rule(const Rule& rule,
                                                         const SchemaVersion& sourceVersion) const {
    if (sourceVersion == current_version_) {
        return rule;  // No migration needed
    }

    auto migrationPath = find_migration_path(sourceVersion);
    if (!migrationPath) {
        return std::nullopt;  // No migration path available
    }

    // For now, implement basic migration strategies
    switch (migrationPath->get_strategy()) {
        case MigrationPath::Strategy::DIRECT_MAPPING:
            return rule;  // No changes needed

        case MigrationPath::Strategy::DEFAULT_VALUES:
            // Add default schema version if not present
            return rule;  // Return as-is for now

        default:
            return std::nullopt;  // Unsupported migration strategy
    }
}

std::string SchemaEvolutionManager::generate_compatibility_matrix() const {
    std::ostringstream oss;
    auto supportedVersions = get_supported_versions();

    oss << "Schema Compatibility Matrix\n";
    oss << "Current version: " << current_version_.to_string() << "\n\n";

    oss << "Supported versions:\n";
    for (const auto& version : supportedVersions) {
        oss << "  " << version.to_string();
        if (version == current_version_) {
            oss << " (current)";
        }
        oss << "\n";
    }

    oss << "\nMigration paths:\n";
    for (const auto& path : migration_paths_) {
        oss << "  " << path.to_string() << "\n";
    }

    return oss.str();
}

bool SchemaEvolutionManager::apply_default_values_migration(
    [[maybe_unused]] const SchemaVersion& sourceVersion) const {
    // Implementation for default values migration
    return true;  // Placeholder
}

bool SchemaEvolutionManager::apply_transformation_migration(
    [[maybe_unused]] const SchemaVersion& sourceVersion) const {
    // Implementation for transformation migration
    return true;  // Placeholder
}

// VersionValidator implementation
std::vector<std::string> VersionValidator::validate_version(const SchemaVersion& version) {
    std::vector<std::string> errors;

    // Validate semantic versioning rules
    if (version.get_major() == 0 && version.get_minor() == 0 && version.get_patch() == 0) {
        errors.push_back("Version 0.0.0 is not valid");
    }

    return errors;
}

std::vector<std::string> VersionValidator::validate_migration_path(const MigrationPath& path) {
    std::vector<std::string> errors;

    const auto& from = path.get_from_version();
    const auto& to = path.get_to_version();

    // Validate version ordering
    if (from >= to) {
        errors.push_back("Migration path must go from older to newer version");
    }

    // Validate major version changes
    if (from.get_major() != to.get_major()) {
        if (path.get_strategy() != MigrationPath::Strategy::CUSTOM_LOGIC &&
            path.get_strategy() != MigrationPath::Strategy::LOSSY) {
            errors.push_back("Major version changes require custom logic or lossy migration");
        }
    }

    return errors;
}

bool VersionValidator::is_safe_transition(const SchemaVersion& from, const SchemaVersion& to) {
    // Safe if only minor or patch versions increase
    return from.get_major() == to.get_major() && from <= to;
}

std::vector<std::string> VersionValidator::generate_warnings(const SchemaVersion& from,
                                                             const SchemaVersion& to) {
    std::vector<std::string> warnings;

    if (from.get_major() != to.get_major()) {
        warnings.push_back("Major version change may break backward compatibility");
    }

    if (to.get_minor() > from.get_minor() + 1) {
        warnings.push_back("Skipping minor versions may indicate missing migration paths");
    }

    return warnings;
}

// SchemaRegistry implementation
SchemaRegistry& SchemaRegistry::get_instance() {
    static SchemaRegistry instance;
    return instance;
}

void SchemaRegistry::register_schema(const SchemaVersion& version,
                                     [[maybe_unused]] const std::string& schemaHash) {
    auto it = std::find(registered_versions_.begin(), registered_versions_.end(), version);
    if (it == registered_versions_.end()) {
        registered_versions_.push_back(version);
        std::sort(registered_versions_.begin(), registered_versions_.end());
    }
}

const SchemaVersion& SchemaRegistry::get_current_schema() const {
    return current_version_;
}

void SchemaRegistry::set_current_schema(const SchemaVersion& version) {
    current_version_ = version;
}

bool SchemaRegistry::is_registered(const SchemaVersion& version) const {
    return std::find(registered_versions_.begin(), registered_versions_.end(), version) !=
           registered_versions_.end();
}

std::vector<SchemaVersion> SchemaRegistry::get_all_versions() const {
    return registered_versions_;
}

}  // namespace inference_lab::common::evolution
