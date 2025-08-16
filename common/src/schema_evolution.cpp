// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#include "schema_evolution.hpp"
#include "inference_types.hpp"
#include <sstream>
#include <algorithm>
#include <regex>
#include <cassert>

namespace inference_lab::common::evolution {

// SchemaVersion implementation
SchemaVersion::SchemaVersion(uint32_t major, uint32_t minor, uint32_t patch, 
                           const std::string& schemaHash)
    : major_(major), minor_(minor), patch_(patch), schemaHash_(schemaHash) {
}

std::optional<SchemaVersion> SchemaVersion::fromString(const std::string& versionString) {
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

std::string SchemaVersion::getVersionString() const {
    std::ostringstream oss;
    oss << major_ << "." << minor_ << "." << patch_;
    return oss.str();
}

bool SchemaVersion::isCompatibleWith(const SchemaVersion& other) const {
    // Same major version means compatible
    return major_ == other.major_;
}

bool SchemaVersion::isForwardCompatibleWith(const SchemaVersion& older) const {
    // Can read older data if major version is same and this version is newer or equal
    return major_ == older.major_ && 
           (minor_ > older.minor_ || (minor_ == older.minor_ && patch_ >= older.patch_));
}

bool SchemaVersion::isBackwardCompatibleWith(const SchemaVersion& newer) const {
    // Newer version can read this data if major is same and newer is actually newer
    return newer.isForwardCompatibleWith(*this);
}

bool SchemaVersion::operator==(const SchemaVersion& other) const {
    return major_ == other.major_ && minor_ == other.minor_ && patch_ == other.patch_;
}

bool SchemaVersion::operator!=(const SchemaVersion& other) const {
    return !(*this == other);
}

bool SchemaVersion::operator<(const SchemaVersion& other) const {
    if (major_ != other.major_) return major_ < other.major_;
    if (minor_ != other.minor_) return minor_ < other.minor_;
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

std::string SchemaVersion::toString() const {
    std::ostringstream oss;
    oss << getVersionString();
    if (!schemaHash_.empty()) {
        oss << " [" << schemaHash_.substr(0, 8) << "...]";
    }
    return oss.str();
}

SchemaVersion::SchemaVersion(schemas::SchemaVersion::Reader reader)
    : major_(reader.getMajor())
    , minor_(reader.getMinor())
    , patch_(reader.getPatch())
    , schemaHash_(reader.getSchemaHash()) {
}

void SchemaVersion::writeTo(schemas::SchemaVersion::Builder builder) const {
    builder.setMajor(major_);
    builder.setMinor(minor_);
    builder.setPatch(patch_);
    builder.setVersionString(getVersionString());
    builder.setMinCompatibleMajor(major_); // For now, only same major version is compatible
    builder.setMinCompatibleMinor(0);
    builder.setSchemaHash(schemaHash_);
}

// MigrationPath implementation
MigrationPath::MigrationPath(const SchemaVersion& fromVersion, const SchemaVersion& toVersion,
                           Strategy strategy, bool reversible, const std::string& description)
    : fromVersion_(fromVersion)
    , toVersion_(toVersion)
    , strategy_(strategy)
    , reversible_(reversible)
    , description_(description) {
}

void MigrationPath::addWarning(const std::string& warning) {
    warnings_.push_back(warning);
}

bool MigrationPath::canMigrate(const SchemaVersion& from, const SchemaVersion& to) const {
    return fromVersion_ == from && toVersion_ == to;
}

std::string MigrationPath::toString() const {
    std::ostringstream oss;
    oss << fromVersion_.toString() << " -> " << toVersion_.toString();
    oss << " (" << static_cast<int>(strategy_) << ")";
    if (reversible_) oss << " [reversible]";
    if (!description_.empty()) oss << ": " << description_;
    return oss.str();
}

MigrationPath::MigrationPath(schemas::MigrationPath::Reader reader)
    : fromVersion_(reader.getFromVersion())
    , toVersion_(reader.getToVersion())
    , strategy_(static_cast<Strategy>(reader.getStrategy()))
    , reversible_(reader.getReversible())
    , description_(reader.getDescription()) {
    
    auto warningsReader = reader.getWarnings();
    for (const auto& warning : warningsReader) {
        warnings_.push_back(warning);
    }
}

void MigrationPath::writeTo(schemas::MigrationPath::Builder builder) const {
    fromVersion_.writeTo(builder.initFromVersion());
    toVersion_.writeTo(builder.initToVersion());
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
    : currentVersion_(currentVersion) {
}

void SchemaEvolutionManager::registerMigrationPath(const MigrationPath& path) {
    migrationPaths_.push_back(path);
    
    // Update index for quick lookup
    std::string key = path.getFromVersion().getVersionString() + "->" + 
                     path.getToVersion().getVersionString();
    pathIndex_[key] = migrationPaths_.size() - 1;
}

bool SchemaEvolutionManager::canReadVersion(const SchemaVersion& dataVersion) const {
    // Can always read current version
    if (dataVersion == currentVersion_) {
        return true;
    }
    
    // Check if migration path exists
    return findMigrationPath(dataVersion).has_value();
}

std::optional<MigrationPath> SchemaEvolutionManager::findMigrationPath(const SchemaVersion& fromVersion) const {
    // Direct migration to current version
    for (const auto& path : migrationPaths_) {
        if (path.canMigrate(fromVersion, currentVersion_)) {
            return path;
        }
    }
    
    // TODO: Implement multi-step migration path finding
    // For now, only support direct migrations
    return std::nullopt;
}

std::vector<SchemaVersion> SchemaEvolutionManager::getSupportedVersions() const {
    std::vector<SchemaVersion> versions;
    versions.push_back(currentVersion_);
    
    for (const auto& path : migrationPaths_) {
        if (path.getToVersion() == currentVersion_) {
            versions.push_back(path.getFromVersion());
        }
    }
    
    // Remove duplicates and sort
    std::sort(versions.begin(), versions.end());
    versions.erase(std::unique(versions.begin(), versions.end()), versions.end());
    
    return versions;
}

std::vector<std::string> SchemaEvolutionManager::validateEvolution(const schemas::SchemaEvolution::Reader& evolution) const {
    std::vector<std::string> errors;
    
    SchemaVersion currentVer(evolution.getCurrentVersion());
    
    // Validate current version
    auto versionErrors = VersionValidator::validateVersion(currentVer);
    errors.insert(errors.end(), versionErrors.begin(), versionErrors.end());
    
    // Validate migration paths
    auto migrationPaths = evolution.getMigrationPaths();
    for (const auto& pathReader : migrationPaths) {
        MigrationPath path(pathReader);
        auto pathErrors = VersionValidator::validateMigrationPath(path);
        errors.insert(errors.end(), pathErrors.begin(), pathErrors.end());
    }
    
    return errors;
}

schemas::SchemaEvolution::Builder SchemaEvolutionManager::createEvolutionMetadata(capnp::MessageBuilder& message) const {
    auto builder = message.getRoot<schemas::SchemaEvolution>();
    
    // Set current version
    currentVersion_.writeTo(builder.initCurrentVersion());
    
    // Set supported versions
    auto supportedVersions = getSupportedVersions();
    auto supportedBuilder = builder.initSupportedVersions(supportedVersions.size());
    for (size_t i = 0; i < supportedVersions.size(); ++i) {
        supportedVersions[i].writeTo(supportedBuilder[i]);
    }
    
    // Set migration paths
    auto pathsBuilder = builder.initMigrationPaths(migrationPaths_.size());
    for (size_t i = 0; i < migrationPaths_.size(); ++i) {
        migrationPaths_[i].writeTo(pathsBuilder[i]);
    }
    
    // Set timestamp
    builder.setEvolutionTimestamp(std::time(nullptr) * 1000); // milliseconds
    
    return builder;
}

std::optional<Fact> SchemaEvolutionManager::migrateFact(const Fact& fact, const SchemaVersion& sourceVersion) const {
    if (sourceVersion == currentVersion_) {
        return fact; // No migration needed
    }
    
    auto migrationPath = findMigrationPath(sourceVersion);
    if (!migrationPath) {
        return std::nullopt; // No migration path available
    }
    
    // For now, implement basic migration strategies
    switch (migrationPath->getStrategy()) {
        case MigrationPath::Strategy::DirectMapping:
            return fact; // No changes needed
            
        case MigrationPath::Strategy::DefaultValues:
            // Add default schema version if not present
            return fact; // Return as-is for now
            
        default:
            return std::nullopt; // Unsupported migration strategy
    }
}

std::optional<Rule> SchemaEvolutionManager::migrateRule(const Rule& rule, const SchemaVersion& sourceVersion) const {
    if (sourceVersion == currentVersion_) {
        return rule; // No migration needed
    }
    
    auto migrationPath = findMigrationPath(sourceVersion);
    if (!migrationPath) {
        return std::nullopt; // No migration path available
    }
    
    // For now, implement basic migration strategies
    switch (migrationPath->getStrategy()) {
        case MigrationPath::Strategy::DirectMapping:
            return rule; // No changes needed
            
        case MigrationPath::Strategy::DefaultValues:
            // Add default schema version if not present
            return rule; // Return as-is for now
            
        default:
            return std::nullopt; // Unsupported migration strategy
    }
}

std::string SchemaEvolutionManager::generateCompatibilityMatrix() const {
    std::ostringstream oss;
    auto supportedVersions = getSupportedVersions();
    
    oss << "Schema Compatibility Matrix\n";
    oss << "Current version: " << currentVersion_.toString() << "\n\n";
    
    oss << "Supported versions:\n";
    for (const auto& version : supportedVersions) {
        oss << "  " << version.toString();
        if (version == currentVersion_) {
            oss << " (current)";
        }
        oss << "\n";
    }
    
    oss << "\nMigration paths:\n";
    for (const auto& path : migrationPaths_) {
        oss << "  " << path.toString() << "\n";
    }
    
    return oss.str();
}

bool SchemaEvolutionManager::applyDefaultValuesMigration(const SchemaVersion& sourceVersion) const {
    // Implementation for default values migration
    return true; // Placeholder
}

bool SchemaEvolutionManager::applyTransformationMigration(const SchemaVersion& sourceVersion) const {
    // Implementation for transformation migration
    return true; // Placeholder
}

// VersionValidator implementation
std::vector<std::string> VersionValidator::validateVersion(const SchemaVersion& version) {
    std::vector<std::string> errors;
    
    // Validate semantic versioning rules
    if (version.getMajor() == 0 && version.getMinor() == 0 && version.getPatch() == 0) {
        errors.push_back("Version 0.0.0 is not valid");
    }
    
    return errors;
}

std::vector<std::string> VersionValidator::validateMigrationPath(const MigrationPath& path) {
    std::vector<std::string> errors;
    
    const auto& from = path.getFromVersion();
    const auto& to = path.getToVersion();
    
    // Validate version ordering
    if (from >= to) {
        errors.push_back("Migration path must go from older to newer version");
    }
    
    // Validate major version changes
    if (from.getMajor() != to.getMajor()) {
        if (path.getStrategy() != MigrationPath::Strategy::CustomLogic &&
            path.getStrategy() != MigrationPath::Strategy::Lossy) {
            errors.push_back("Major version changes require custom logic or lossy migration");
        }
    }
    
    return errors;
}

bool VersionValidator::isSafeTransition(const SchemaVersion& from, const SchemaVersion& to) {
    // Safe if only minor or patch versions increase
    return from.getMajor() == to.getMajor() && from <= to;
}

std::vector<std::string> VersionValidator::generateWarnings(const SchemaVersion& from, const SchemaVersion& to) {
    std::vector<std::string> warnings;
    
    if (from.getMajor() != to.getMajor()) {
        warnings.push_back("Major version change may break backward compatibility");
    }
    
    if (to.getMinor() > from.getMinor() + 1) {
        warnings.push_back("Skipping minor versions may indicate missing migration paths");
    }
    
    return warnings;
}

// SchemaRegistry implementation
SchemaRegistry& SchemaRegistry::getInstance() {
    static SchemaRegistry instance;
    return instance;
}

void SchemaRegistry::registerSchema(const SchemaVersion& version, const std::string& schemaHash) {
    auto it = std::find(registeredVersions_.begin(), registeredVersions_.end(), version);
    if (it == registeredVersions_.end()) {
        registeredVersions_.push_back(version);
        std::sort(registeredVersions_.begin(), registeredVersions_.end());
    }
}

const SchemaVersion& SchemaRegistry::getCurrentSchema() const {
    return currentVersion_;
}

void SchemaRegistry::setCurrentSchema(const SchemaVersion& version) {
    currentVersion_ = version;
}

bool SchemaRegistry::isRegistered(const SchemaVersion& version) const {
    return std::find(registeredVersions_.begin(), registeredVersions_.end(), version) != registeredVersions_.end();
}

std::vector<SchemaVersion> SchemaRegistry::getAllVersions() const {
    return registeredVersions_;
}

} // namespace inference_lab::common::evolution