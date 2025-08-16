# MIT License
# Copyright (c) 2025 dbjwhs
#
# This software is provided "as is" without warranty of any kind, express or implied.
# The authors are not liable for any damages arising from the use of this software.

@0x85150b117366d14b;

# Core inference engine data types schema
# Defines serializable structures for facts, rules, queries, and results
# Schema version: 1.0.0

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("inference_lab::common::schemas");

# Schema versioning and evolution support
struct SchemaVersion {
    major @0 :UInt32;
    minor @1 :UInt32; 
    patch @2 :UInt32;
    
    # Semantic version string (e.g., "1.2.3")
    versionString @3 :Text;
    
    # Compatibility information
    minCompatibleMajor @4 :UInt32;
    minCompatibleMinor @5 :UInt32;
    
    # Schema hash for integrity checking
    schemaHash @6 :Text;
}

# Schema evolution metadata
struct SchemaEvolution {
    # Current schema version
    currentVersion @0 :SchemaVersion;
    
    # List of supported versions for backward compatibility
    supportedVersions @1 :List(SchemaVersion);
    
    # Migration paths available
    migrationPaths @2 :List(MigrationPath);
    
    # Schema evolution timestamp
    evolutionTimestamp @3 :UInt64;
}

# Describes how to migrate between schema versions
struct MigrationPath {
    fromVersion @0 :SchemaVersion;
    toVersion @1 :SchemaVersion;
    
    # Migration strategy
    strategy @2 :MigrationStrategy;
    
    # Whether this migration is reversible
    reversible @3 :Bool = false;
    
    # Migration metadata
    description @4 :Text;
    warnings @5 :List(Text);
}

enum MigrationStrategy {
    # Direct field mapping (no data loss)
    directMapping @0;
    
    # Field transformation required
    transformation @1;
    
    # Default values for new fields
    defaultValues @2;
    
    # Custom migration logic required
    customLogic @3;
    
    # Data may be lost in migration
    lossy @4;
}

# Basic data types that can appear in facts and rules
struct Value {
    union {
        # Primitive types
        int64Value @0 :Int64;
        float64Value @1 :Float64;
        textValue @2 :Text;
        boolValue @3 :Bool;
        
        # Complex types  
        listValue @4 :List(Value);
        structValue @5 :List(Field);
    }
}

struct Field {
    name @0 :Text;
    value @1 :Value;
}

# Represents a single fact in the knowledge base
struct Fact {
    # Unique identifier for this fact
    id @0 :UInt64;
    
    # Predicate name (e.g., "isHuman", "livesIn", "hasProperty")
    predicate @1 :Text;
    
    # Arguments/parameters for this fact
    # e.g., for "isHuman(socrates)", args would contain "socrates"
    args @2 :List(Value);
    
    # Optional confidence/weight for probabilistic reasoning
    confidence @3 :Float64 = 1.0;
    
    # Timestamp when this fact was asserted
    timestamp @4 :UInt64;
    
    # Optional metadata
    metadata @5 :List(Field);
    
    # Schema version this fact was created with
    schemaVersion @6 :SchemaVersion;
}

# Represents a rule in the inference engine
struct Rule {
    # Unique identifier for this rule
    id @0 :UInt64;
    
    # Human-readable name for debugging
    name @1 :Text;
    
    # Conditions that must be satisfied (AND logic)
    # e.g., for "if isHuman(X) and hasPet(X, dog) then lovesAnimals(X)"
    conditions @2 :List(Condition);
    
    # Conclusions to assert if conditions are met
    conclusions @3 :List(Conclusion);
    
    # Rule priority for conflict resolution
    priority @4 :Int32 = 0;
    
    # Optional confidence for this rule
    confidence @5 :Float64 = 1.0;
    
    # Optional metadata
    metadata @6 :List(Field);
    
    # Schema version this rule was created with
    schemaVersion @7 :SchemaVersion;
}

# A condition within a rule
struct Condition {
    # Predicate to match
    predicate @0 :Text;
    
    # Arguments with variables and constants
    # Variables start with uppercase, constants are literals
    args @1 :List(Value);
    
    # Whether this is a negation (NOT condition)
    negated @2 :Bool = false;
}

# A conclusion within a rule
struct Conclusion {
    # Predicate to assert
    predicate @0 :Text;
    
    # Arguments (may contain variables from conditions)
    args @1 :List(Value);
    
    # Confidence for this conclusion
    confidence @2 :Float64 = 1.0;
}

# Represents a query to the inference engine
struct Query {
    # Unique identifier for this query
    id @0 :UInt64;
    
    # Query type
    type @1 :QueryType;
    
    # The goal to prove or find
    goal @2 :Condition;
    
    # Maximum number of results to return
    maxResults @3 :UInt32 = 100;
    
    # Query timeout in milliseconds
    timeoutMs @4 :UInt32 = 5000;
    
    # Optional query metadata
    metadata @5 :List(Field);
}

enum QueryType {
    # Find all facts that match the goal
    findAll @0;
    
    # Check if goal can be proven (true/false)
    prove @1;
    
    # Find first N solutions
    findFirst @2;
    
    # Explain how a goal can be proven
    explain @3;
}

# Results returned from a query
struct QueryResult {
    # Original query ID
    queryId @0 :UInt64;
    
    # Whether the query succeeded
    success @1 :Bool;
    
    # Results found
    results @2 :List(Result);
    
    # Total execution time in microseconds
    executionTimeUs @3 :UInt64;
    
    # Number of inference steps performed
    inferenceSteps @4 :UInt64;
    
    # Optional error message if query failed
    error @5 :Text;
    
    # Result metadata
    metadata @6 :List(Field);
}

# A single result from a query
struct Result {
    # The fact that satisfies the query
    fact @0 :Fact;
    
    # Variable bindings if query contained variables
    bindings @1 :List(Binding);
    
    # Confidence in this result
    confidence @2 :Float64;
    
    # Proof trace (for explain queries)
    proof @3 :List(ProofStep);
}

# Variable binding in a result
struct Binding {
    variable @0 :Text;
    value @1 :Value;
}

# Step in a proof trace
struct ProofStep {
    # Rule that was applied
    ruleId @0 :UInt64;
    
    # Rule name for debugging
    ruleName @1 :Text;
    
    # Facts that matched the rule conditions
    matchedFacts @2 :List(UInt64);  # Fact IDs
    
    # New facts derived from this step
    derivedFacts @3 :List(Fact);
}

# Knowledge base snapshot for serialization
struct KnowledgeBase {
    # All facts in the knowledge base
    facts @0 :List(Fact);
    
    # All rules in the knowledge base
    rules @1 :List(Rule);
    
    # Deprecated: Legacy version field (use schemaEvolution instead)
    version @2 :UInt64;
    timestamp @3 :UInt64;
    metadata @4 :List(Field);
    
    # Schema evolution information
    schemaEvolution @5 :SchemaEvolution;
    
    # Data format version (separate from schema version)
    dataFormatVersion @6 :UInt32 = 1;
    
    # Integrity information
    checksum @7 :Text;
    
    # Compression information if applicable
    compressionInfo @8 :CompressionInfo;
}

# Compression metadata for serialized data
struct CompressionInfo {
    algorithm @0 :CompressionAlgorithm;
    originalSize @1 :UInt64;
    compressedSize @2 :UInt64;
    compressionRatio @3 :Float64;
}

enum CompressionAlgorithm {
    none @0;
    gzip @1;
    lz4 @2;
    zstd @3;
}

# Backward compatibility support structures
struct CompatibilityLayer {
    # Source version being migrated from
    sourceVersion @0 :SchemaVersion;
    
    # Target version being migrated to  
    targetVersion @1 :SchemaVersion;
    
    # Field mappings for compatibility
    fieldMappings @2 :List(FieldMapping);
    
    # Default values for new fields
    defaultValues @3 :List(DefaultValue);
    
    # Transformation rules
    transformationRules @4 :List(TransformationRule);
    
    # Compatibility warnings
    warnings @5 :List(Text);
}

# Maps fields between schema versions
struct FieldMapping {
    sourceField @0 :Text;
    targetField @1 :Text;
    mappingType @2 :FieldMappingType;
    
    # Optional transformation expression
    transformation @3 :Text;
}

enum FieldMappingType {
    direct @0;        # Direct 1:1 mapping
    renamed @1;       # Field was renamed
    split @2;         # One field split into multiple
    merged @3;        # Multiple fields merged into one
    transformed @4;   # Data transformation required
    deprecated @5;    # Field no longer exists
}

# Default values for new fields in schema evolution
struct DefaultValue {
    fieldName @0 :Text;
    fieldType @1 :Text;
    defaultValue @2 :Value;
    description @3 :Text;
}

# Transformation rules for complex field evolution
struct TransformationRule {
    ruleId @0 :UInt64;
    name @1 :Text;
    description @2 :Text;
    
    # Input fields required for transformation
    inputFields @3 :List(Text);
    
    # Output fields produced by transformation
    outputFields @4 :List(Text);
    
    # Transformation logic (implementation-specific)
    logic @5 :Text;
    
    # Whether this transformation is reversible
    reversible @6 :Bool = false;
}

# Version compatibility matrix
struct CompatibilityMatrix {
    # Current schema version
    currentVersion @0 :SchemaVersion;
    
    # Supported compatibility relationships
    compatibilityEntries @1 :List(CompatibilityEntry);
    
    # Matrix generation timestamp
    generatedAt @2 :UInt64;
}

struct CompatibilityEntry {
    version @0 :SchemaVersion;
    compatibility @1 :CompatibilityLevel;
    migrationPath @2 :MigrationPath;
    notes @3 :Text;
}

enum CompatibilityLevel {
    # Fully compatible - no migration needed
    fullCompatible @0;
    
    # Forward compatible - newer can read older
    forwardCompatible @1;
    
    # Backward compatible - older can read newer  
    backwardCompatible @2;
    
    # Compatible with migration
    migrationRequired @3;
    
    # Incompatible
    incompatible @4;
}