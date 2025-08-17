// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file schema_evolution_demo.cpp
 * @brief Demonstration and test of schema versioning and evolution capabilities
 *
 * This program demonstrates:
 * - Creating schema versions and checking compatibility
 * - Setting up migration paths between versions
 * - Validating schema evolution rules
 * - Testing backward compatibility scenarios
 */

#include <cassert>
#include <iostream>

#include "../src/inference_types.hpp"
#include "../src/schema_evolution.hpp"

using namespace inference_lab::common;
using namespace inference_lab::common::evolution;

static void test_schema_versioning() {
    std::cout << "=== Testing Schema Versioning ===\n";

    // Test version creation and comparison
    SchemaVersion v1_0_0(1, 0, 0);
    SchemaVersion v1_1_0(1, 1, 0);
    SchemaVersion v1_1_1(1, 1, 1);
    SchemaVersion v2_0_0(2, 0, 0);

    std::cout << "Created versions: " << v1_0_0.toString() << ", " << v1_1_0.toString() << ", "
              << v1_1_1.toString() << ", " << v2_0_0.toString() << "\n";

    // Test compatibility checks
    assert(v1_0_0.isCompatibleWith(v1_1_0));   // Same major version
    assert(!v1_0_0.isCompatibleWith(v2_0_0));  // Different major version

    // Test forward compatibility
    assert(v1_1_0.isForwardCompatibleWith(v1_0_0));   // Can read older
    assert(!v1_0_0.isForwardCompatibleWith(v1_1_0));  // Cannot read newer

    // Test version parsing
    auto parsed = SchemaVersion::fromString("1.2.3");
    assert(parsed.has_value());
    assert(parsed->getMajor() == 1);
    assert(parsed->getMinor() == 2);
    assert(parsed->getPatch() == 3);

    std::cout << "✅ Schema versioning tests passed\n\n";
}

static void test_migration_paths() {
    std::cout << "=== Testing Migration Paths ===\n";

    SchemaVersion v1_0_0(1, 0, 0);
    SchemaVersion v1_1_0(1, 1, 0);
    SchemaVersion v2_0_0(2, 0, 0);

    // Create migration paths
    MigrationPath path1(v1_0_0,
                        v1_1_0,
                        MigrationPath::Strategy::DefaultValues,
                        true,
                        "Added optional schema version fields");

    MigrationPath path2(v1_1_0,
                        v2_0_0,
                        MigrationPath::Strategy::CustomLogic,
                        false,
                        "Major refactoring of data structures");

    std::cout << "Migration path 1: " << path1.toString() << "\n";
    std::cout << "Migration path 2: " << path2.toString() << "\n";

    // Test migration capability
    assert(path1.canMigrate(v1_0_0, v1_1_0));
    assert(!path1.canMigrate(v1_0_0, v2_0_0));

    std::cout << "✅ Migration paths tests passed\n\n";
}

static void test_schema_evolution_manager() {
    std::cout << "=== Testing Schema Evolution Manager ===\n";

    SchemaVersion current_version(1, 1, 0);
    SchemaEvolutionManager manager(current_version);

    // Register migration paths
    SchemaVersion v1_0_0(1, 0, 0);
    MigrationPath path(v1_0_0,
                       current_version,
                       MigrationPath::Strategy::DefaultValues,
                       true,
                       "Migrate from initial version to current");

    manager.registerMigrationPath(path);

    // Test version reading capability
    assert(manager.canReadVersion(current_version));  // Current version
    assert(manager.canReadVersion(v1_0_0));           // Supported via migration

    SchemaVersion v2_0_0(2, 0, 0);
    assert(!manager.canReadVersion(v2_0_0));  // No migration path

    // Test supported versions
    auto supported = manager.getSupportedVersions();
    std::cout << "Supported versions: ";
    for (const auto& version : supported) {
        std::cout << version.toString() << " ";
    }
    std::cout << "\n";

    // Generate compatibility matrix
    std::cout << "\n" << manager.generateCompatibilityMatrix() << "\n";

    std::cout << "✅ Schema evolution manager tests passed\n\n";
}

static void test_version_validator() {
    std::cout << "=== Testing Version Validator ===\n";

    // Test invalid version
    SchemaVersion invalid_version(0, 0, 0);
    auto errors = VersionValidator::validateVersion(invalid_version);
    assert(!errors.empty());
    std::cout << "Invalid version errors: " << errors[0] << "\n";

    // Test valid version
    SchemaVersion valid_version(1, 0, 0);
    errors = VersionValidator::validateVersion(valid_version);
    assert(errors.empty());

    // Test safe transition
    SchemaVersion v1_0_0(1, 0, 0);
    SchemaVersion v1_1_0(1, 1, 0);
    SchemaVersion v2_0_0(2, 0, 0);

    assert(VersionValidator::isSafeTransition(v1_0_0, v1_1_0));   // Minor increment
    assert(!VersionValidator::isSafeTransition(v1_0_0, v2_0_0));  // Major increment

    // Test migration path validation
    MigrationPath safe_path(v1_0_0, v1_1_0, MigrationPath::Strategy::DefaultValues);
    errors = VersionValidator::validateMigrationPath(safe_path);
    assert(errors.empty());

    MigrationPath unsafe_path(v1_1_0, v1_0_0, MigrationPath::Strategy::DirectMapping);  // Backwards
    errors = VersionValidator::validateMigrationPath(unsafe_path);
    assert(!errors.empty());
    std::cout << "Unsafe migration error: " << errors[0] << "\n";

    std::cout << "✅ Version validator tests passed\n\n";
}

static void test_schema_registry() {
    std::cout << "=== Testing Schema Registry ===\n";

    auto& registry = SchemaRegistry::getInstance();

    SchemaVersion v1_0_0(1, 0, 0, "hash1");
    SchemaVersion v1_1_0(1, 1, 0, "hash2");

    // Register schemas
    registry.registerSchema(v1_0_0, "hash1");
    registry.registerSchema(v1_1_0, "hash2");

    // Test registration check
    assert(registry.isRegistered(v1_0_0));
    assert(registry.isRegistered(v1_1_0));

    SchemaVersion v2_0_0(2, 0, 0);
    assert(!registry.isRegistered(v2_0_0));

    // Set current schema
    registry.setCurrentSchema(v1_1_0);
    assert(registry.getCurrentSchema() == v1_1_0);

    // Get all versions
    auto all_versions = registry.getAllVersions();
    std::cout << "Registered versions: ";
    for (const auto& version : all_versions) {
        std::cout << version.toString() << " ";
    }
    std::cout << "\n";

    std::cout << "✅ Schema registry tests passed\n\n";
}

static void test_data_migration() {
    std::cout << "=== Testing Data Migration ===\n";

    // Create a fact without schema version (simulates old data)
    std::vector<Value> args = {Value::fromText("socrates")};
    Fact old_fact(1, "isHuman", args, 1.0, 1234567890);

    std::cout << "Original fact: " << old_fact.toString() << "\n";

    // Set up migration manager
    SchemaVersion current_version(1, 1, 0);
    SchemaVersion old_version(1, 0, 0);
    SchemaEvolutionManager manager(current_version);

    MigrationPath path(old_version,
                       current_version,
                       MigrationPath::Strategy::DefaultValues,
                       true,
                       "Add schema version field with default value");
    manager.registerMigrationPath(path);

    // Migrate the fact
    auto migrated_fact = manager.migrateFact(old_fact, old_version);
    assert(migrated_fact.has_value());

    std::cout << "Migrated fact: " << migrated_fact->toString() << "\n";

    std::cout << "✅ Data migration tests passed\n\n";
}

static void demonstrate_evolution_scenarios() {
    std::cout << "=== Demonstrating Evolution Scenarios ===\n";

    // Scenario 1: Adding optional fields (backward compatible)
    std::cout << "Scenario 1: Adding optional schema version field\n";
    SchemaVersion v1_0_0(1, 0, 0);
    SchemaVersion v1_1_0(1, 1, 0);

    MigrationPath add_fields(v1_0_0,
                             v1_1_0,
                             MigrationPath::Strategy::DefaultValues,
                             true,
                             "Added optional schemaVersion field to Facts and Rules");

    auto warnings = VersionValidator::generateWarnings(v1_0_0, v1_1_0);
    std::cout << "Migration: " << add_fields.toString() << "\n";
    if (warnings.empty()) {
        std::cout << "✅ Safe migration - no warnings\n";
    }

    // Scenario 2: Major version change (breaking compatibility)
    std::cout << "\nScenario 2: Major restructuring\n";
    SchemaVersion v2_0_0(2, 0, 0);

    MigrationPath major_change(v1_1_0,
                               v2_0_0,
                               MigrationPath::Strategy::CustomLogic,
                               false,
                               "Restructured data model - breaking changes");
    major_change.addWarning("May lose some metadata during migration");
    major_change.addWarning("Custom migration logic required");

    warnings = VersionValidator::generateWarnings(v1_1_0, v2_0_0);
    std::cout << "Migration: " << major_change.toString() << "\n";
    std::cout << "Warnings:\n";
    for (const auto& warning : major_change.getWarnings()) {
        std::cout << "  ⚠️  " << warning << "\n";
    }
    for (const auto& warning : warnings) {
        std::cout << "  ⚠️  " << warning << "\n";
    }

    std::cout << "\n✅ Evolution scenarios demonstrated\n\n";
}

auto main() -> int {
    std::cout << "Schema Evolution and Versioning Demo\n";
    std::cout << "====================================\n\n";

    try {
        test_schema_versioning();
        test_migration_paths();
        test_schema_evolution_manager();
        test_version_validator();
        test_schema_registry();
        test_data_migration();
        demonstrate_evolution_scenarios();

        std::cout << "🎉 All tests passed! Schema evolution system is working correctly.\n";

        // Show current schema version
        auto current_schema = getCurrentSchemaVersion();
        std::cout << "\nCurrent schema version: " << current_schema.toString() << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception\n";
        return 1;
    }
}
