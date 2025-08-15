# MIT License
# Copyright (c) 2025 dbjwhs
#
# This software is provided "as is" without warranty of any kind, express or implied.
# The authors are not liable for any damages arising from the use of this software.

@0x85150b117366d14b;

# Core inference engine data types schema
# Defines serializable structures for facts, rules, queries, and results

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("inference_lab::common::schemas");

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
    
    # Metadata about the knowledge base
    version @2 :UInt64;
    timestamp @3 :UInt64;
    metadata @4 :List(Field);
}