// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file inference_types_demo.cpp
 * @brief Comprehensive demonstration of the C++ inference types interface
 * 
 * This demo showcases all the major features of the C++ wrapper interface
 * for Cap'n Proto inference types. It demonstrates:
 * 
 * 1. Creating Facts using builders with different argument types
 * 2. Creating Rules with conditions and conclusions using fluent interface
 * 3. Creating Queries of different types (FindAll, Prove, Explain)
 * 4. Serialization and deserialization to/from binary Cap'n Proto format
 * 5. Working with complex nested Values (lists, structs)
 * 6. Type checking and safe value extraction
 * 
 * The demo creates a small knowledge base about ancient Greek philosophers
 * and demonstrates how the inference types would be used in a real system.
 */

#include "../src/inference_types.hpp"
#include "../src/inference_builders.hpp"
#include <iostream>

using namespace inference_lab::common;
using namespace inference_lab::common::builders;

int main() {
    std::cout << "=== Inference Types C++ Interface Demo ===\n\n";
    
    //=========================================================================
    // Section 1: Creating Facts
    //=========================================================================
    // This section demonstrates various ways to create facts using the builder
    // pattern. Facts represent basic knowledge like "socrates is human" or
    // "socrates lives in athens".
    
    std::cout << "1. Creating Facts:\n";
    
    auto socrates_fact = FactBuilder("isHuman")
        .withArg("socrates")
        .build();
    std::cout << "   " << socrates_fact.toString() << "\n";
    
    auto plato_fact = FactBuilder("isHuman")
        .withArg("plato")
        .withConfidence(0.95)
        .withMetadata("source", Value::fromText("philosophy_db"))
        .build();
    std::cout << "   " << plato_fact.toString() << "\n";
    
    auto age_fact = FactBuilder("age")
        .withArg("socrates")
        .withArg(static_cast<int64_t>(70))
        .build();
    std::cout << "   " << age_fact.toString() << "\n";
    
    auto lives_in_fact = FactBuilder("livesIn")
        .withArg("socrates")
        .withArg("athens")
        .build();
    std::cout << "   " << lives_in_fact.toString() << "\n\n";
    
    //=========================================================================
    // Section 2: Creating Rules  
    //=========================================================================
    // This section demonstrates creating inference rules using the fluent
    // interface. Rules define logical relationships like "if X is human, then X is mortal".
    // The builder allows complex rules with multiple conditions and conclusions.
    
    std::cout << "2. Creating Rules:\n";
    
    auto mortality_rule = rule("mortality_rule")
        .when("isHuman").withVariable("X")
        .then("isMortal").withVariable("X")
        .withPriority(10)
        .build();
    std::cout << "   " << mortality_rule.toString() << "\n";
    
    auto wisdom_rule = rule("wisdom_rule")
        .when("isHuman").withVariable("X")
        .when("age").withVariable("X").withArg(static_cast<int64_t>(60))  // age(X, Y) where Y >= 60
        .then("isWise").withVariable("X")
        .withConfidence(0.8)
        .build();
    std::cout << "   " << wisdom_rule.toString() << "\n";
    
    auto philosopher_rule = rule("philosopher_rule")
        .when("isHuman").withVariable("X")
        .when("livesIn").withVariable("X").withArg("athens")
        .whenNot("isIgnorant").withVariable("X")
        .then("isPhilosopher").withVariable("X")
        .withConfidence(0.9)
        .build();
    std::cout << "   " << philosopher_rule.toString() << "\n\n";
    
    //=========================================================================
    // Section 3: Creating Queries
    //=========================================================================
    // This section shows how to create different types of queries that can be
    // sent to the inference engine. Each query type serves a different purpose:
    // - FindAll: Find all facts matching a pattern
    // - Prove: Check if something can be proven true/false
    // - Explain: Get a proof trace showing how something was derived
    
    std::cout << "3. Creating Queries:\n";
    
    auto find_all_humans = QueryBuilder::findAll()
        .goal("isHuman").withVariable("X")
        .build();
    std::cout << "   " << find_all_humans.toString() << "\n";
    
    auto prove_mortal = QueryBuilder::prove()
        .goal("isMortal").withArg("socrates")
        .build();
    std::cout << "   " << prove_mortal.toString() << "\n";
    
    auto explain_wisdom = QueryBuilder::explain()
        .goal("isWise").withVariable("X")
        .maxResults(10)
        .timeout(3000)
        .build();
    std::cout << "   " << explain_wisdom.toString() << "\n\n";
    
    //=========================================================================
    // Section 4: Serialization and Deserialization
    //=========================================================================
    // This section demonstrates how to serialize C++ objects to binary Cap'n Proto
    // format for storage or transmission, and how to deserialize them back.
    // It also shows JSON-like text serialization for debugging.
    
    std::cout << "4. Serialization:\n";
    
    // Serialize to binary
    auto serialized_fact = Serializer::serialize(socrates_fact);
    std::cout << "   Serialized fact size: " << serialized_fact.size() << " bytes\n";
    
    // Deserialize from binary
    auto deserialized_fact = Serializer::deserializeFact(serialized_fact);
    if (deserialized_fact) {
        std::cout << "   Deserialized: " << deserialized_fact->toString() << "\n";
    }
    
    // JSON representation
    std::cout << "   JSON representation:\n" << Serializer::toJson(socrates_fact) << "\n\n";
    
    //=========================================================================
    // Section 5: Complex Value Types
    //=========================================================================
    // This section shows how to work with complex Value types like lists and
    // structured objects. These allow representing more sophisticated data
    // than just primitive types.
    
    std::cout << "5. Complex Values:\n";
    
    // List value
    std::vector<Value> students = {
        Value::fromText("plato"),
        Value::fromText("aristotle"),
        Value::fromText("xenophon")
    };
    auto teaching_fact = fact("teaches", "socrates", Value::fromList(students));
    std::cout << "   " << teaching_fact.toString() << "\n";
    
    // Struct value
    std::unordered_map<std::string, Value> person_info = {
        {"name", Value::fromText("socrates")},
        {"age", Value::fromInt64(70)},
        {"city", Value::fromText("athens")},
        {"is_philosopher", Value::fromBool(true)}
    };
    auto person_fact = fact("person", Value::fromStruct(person_info));
    std::cout << "   " << person_fact.toString() << "\n\n";
    
    //=========================================================================
    // Section 6: Type Checking and Safe Value Extraction
    //=========================================================================
    // This section demonstrates the type safety features of the Value class,
    // including type checking methods and safe extraction that returns nullopt
    // instead of throwing exceptions on type mismatches.
    
    std::cout << "6. Value Type Operations:\n";
    
    Value int_val = Value::fromInt64(42);
    Value text_val = Value::fromText("hello");
    
    std::cout << "   int_val.isInt64(): " << (int_val.isInt64() ? "true" : "false") << "\n";
    std::cout << "   text_val.isText(): " << (text_val.isText() ? "true" : "false") << "\n";
    
    // Safe extraction
    auto maybe_int = text_val.tryAsInt64();
    std::cout << "   text_val.tryAsInt64(): " << (maybe_int ? "has value" : "nullopt") << "\n";
    
    auto maybe_text = text_val.tryAsText();
    std::cout << "   text_val.tryAsText(): " << (maybe_text ? *maybe_text : "nullopt") << "\n\n";
    
    std::cout << "=== Demo Complete ===\n";
    
    return 0;
}