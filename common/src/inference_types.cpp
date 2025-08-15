// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file inference_types.cpp
 * @brief Implementation of C++ wrapper classes for Cap'n Proto inference types
 * 
 * This file contains the implementation of all the wrapper classes defined in
 * inference_types.hpp. The implementations handle:
 * - Type-safe value creation and extraction
 * - Conversion between C++ objects and Cap'n Proto format
 * - String representation generation for debugging
 * - Memory management and resource cleanup
 * - Error handling for type mismatches and invalid operations
 */

#include "inference_types.hpp"
#include <sstream>
#include <stdexcept>
#include <chrono>

namespace inference_lab::common {

//=============================================================================
// Value Implementation
//=============================================================================

// Factory methods for creating typed Value objects
// These use the private constructor to set the discriminant type and then
// initialize the appropriate storage member

Value Value::fromInt64(int64_t value) {
    Value v(Type::Int64);
    v.int64_value_ = value;
    return v;
}

Value Value::fromFloat64(double value) {
    Value v(Type::Float64);
    v.float64_value_ = value;
    return v;
}

Value Value::fromText(const std::string& value) {
    Value v(Type::Text);
    v.text_value_ = value;  // Copy string into text_value_ member
    return v;
}

Value Value::fromBool(bool value) {
    Value v(Type::Bool);
    v.bool_value_ = value;
    return v;
}

Value Value::fromList(const std::vector<Value>& values) {
    Value v(Type::List);
    v.list_value_ = values;  // Deep copy of the vector and its Value elements
    return v;
}

Value Value::fromStruct(const std::unordered_map<std::string, Value>& fields) {
    Value v(Type::Struct);
    v.struct_value_ = fields;  // Deep copy of the map and its Value elements
    return v;
}

// Type checking methods - these are simple discriminant checks
// They're const and noexcept since they only read the type_ field

bool Value::isInt64() const { return type_ == Type::Int64; }
bool Value::isFloat64() const { return type_ == Type::Float64; }
bool Value::isText() const { return type_ == Type::Text; }
bool Value::isBool() const { return type_ == Type::Bool; }
bool Value::isList() const { return type_ == Type::List; }
bool Value::isStruct() const { return type_ == Type::Struct; }

// Unsafe extraction methods - these throw exceptions on type mismatch
// Use these when you're certain of the type or want fail-fast behavior

int64_t Value::asInt64() const {
    if (type_ != Type::Int64) throw std::runtime_error("Value is not int64");
    return int64_value_;
}

double Value::asFloat64() const {
    if (type_ != Type::Float64) throw std::runtime_error("Value is not float64");
    return float64_value_;
}

std::string Value::asText() const {
    if (type_ != Type::Text) throw std::runtime_error("Value is not text");
    return text_value_;  // Returns copy of the string
}

bool Value::asBool() const {
    if (type_ != Type::Bool) throw std::runtime_error("Value is not bool");
    return bool_value_;
}

std::vector<Value> Value::asList() const {
    if (type_ != Type::List) throw std::runtime_error("Value is not list");
    return list_value_;  // Returns copy of the vector
}

std::unordered_map<std::string, Value> Value::asStruct() const {
    if (type_ != Type::Struct) throw std::runtime_error("Value is not struct");
    return struct_value_;  // Returns copy of the map
}

// Safe extraction methods - these return nullopt on type mismatch
// Use these when you want to handle type mismatches gracefully

std::optional<int64_t> Value::tryAsInt64() const {
    return type_ == Type::Int64 ? std::optional<int64_t>(int64_value_) : std::nullopt;
}

std::optional<double> Value::tryAsFloat64() const {
    return type_ == Type::Float64 ? std::optional<double>(float64_value_) : std::nullopt;
}

std::optional<std::string> Value::tryAsText() const {
    return type_ == Type::Text ? std::optional<std::string>(text_value_) : std::nullopt;
}

std::optional<bool> Value::tryAsBool() const {
    return type_ == Type::Bool ? std::optional<bool>(bool_value_) : std::nullopt;
}

std::optional<std::vector<Value>> Value::tryAsList() const {
    return type_ == Type::List ? std::optional<std::vector<Value>>(list_value_) : std::nullopt;
}

std::optional<std::unordered_map<std::string, Value>> Value::tryAsStruct() const {
    return type_ == Type::Struct ? std::optional<std::unordered_map<std::string, Value>>(struct_value_) : std::nullopt;
}

std::string Value::toString() const {
    switch (type_) {
        case Type::Int64: return std::to_string(int64_value_);
        case Type::Float64: return std::to_string(float64_value_);
        case Type::Text: return "\"" + text_value_ + "\"";
        case Type::Bool: return bool_value_ ? "true" : "false";
        case Type::List: {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < list_value_.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << list_value_[i].toString();
            }
            ss << "]";
            return ss.str();
        }
        case Type::Struct: {
            std::stringstream ss;
            ss << "{";
            bool first = true;
            for (const auto& [key, value] : struct_value_) {
                if (!first) ss << ", ";
                ss << "\"" << key << "\": " << value.toString();
                first = false;
            }
            ss << "}";
            return ss.str();
        }
    }
    return "<unknown>";
}

Value::Value(schemas::Value::Reader reader) {
    switch (reader.which()) {
        case schemas::Value::INT64_VALUE:
            type_ = Type::Int64;
            int64_value_ = reader.getInt64Value();
            break;
        case schemas::Value::FLOAT64_VALUE:
            type_ = Type::Float64;
            float64_value_ = reader.getFloat64Value();
            break;
        case schemas::Value::TEXT_VALUE:
            type_ = Type::Text;
            text_value_ = reader.getTextValue().cStr();
            break;
        case schemas::Value::BOOL_VALUE:
            type_ = Type::Bool;
            bool_value_ = reader.getBoolValue();
            break;
        case schemas::Value::LIST_VALUE: {
            type_ = Type::List;
            auto list = reader.getListValue();
            list_value_.reserve(list.size());
            for (auto item : list) {
                list_value_.emplace_back(item);
            }
            break;
        }
        case schemas::Value::STRUCT_VALUE: {
            type_ = Type::Struct;
            auto fields = reader.getStructValue();
            for (auto field : fields) {
                struct_value_[field.getName().cStr()] = Value(field.getValue());
            }
            break;
        }
    }
}

void Value::writeTo(schemas::Value::Builder builder) const {
    switch (type_) {
        case Type::Int64:
            builder.setInt64Value(int64_value_);
            break;
        case Type::Float64:
            builder.setFloat64Value(float64_value_);
            break;
        case Type::Text:
            builder.setTextValue(text_value_);
            break;
        case Type::Bool:
            builder.setBoolValue(bool_value_);
            break;
        case Type::List: {
            auto list = builder.initListValue(list_value_.size());
            for (size_t i = 0; i < list_value_.size(); ++i) {
                list_value_[i].writeTo(list[i]);
            }
            break;
        }
        case Type::Struct: {
            auto fields = builder.initStructValue(struct_value_.size());
            size_t i = 0;
            for (const auto& [key, value] : struct_value_) {
                fields[i].setName(key);
                value.writeTo(fields[i].getValue());
                ++i;
            }
            break;
        }
    }
}

// Fact implementation
Fact::Fact(uint64_t id, const std::string& predicate, const std::vector<Value>& args,
           double confidence, uint64_t timestamp)
    : id_(id), predicate_(predicate), args_(args), confidence_(confidence), timestamp_(timestamp) {
    if (timestamp_ == 0) {
        timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
}

void Fact::setMetadata(const std::string& key, const Value& value) {
    metadata_[key] = value;
}

std::optional<Value> Fact::getMetadata(const std::string& key) const {
    auto it = metadata_.find(key);
    return it != metadata_.end() ? std::optional<Value>(it->second) : std::nullopt;
}

std::string Fact::toString() const {
    std::stringstream ss;
    ss << predicate_ << "(";
    for (size_t i = 0; i < args_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << args_[i].toString();
    }
    ss << ")";
    if (confidence_ != 1.0) {
        ss << " [confidence: " << confidence_ << "]";
    }
    return ss.str();
}

Fact::Fact(schemas::Fact::Reader reader) 
    : id_(reader.getId())
    , predicate_(reader.getPredicate().cStr())
    , confidence_(reader.getConfidence())
    , timestamp_(reader.getTimestamp()) {
    
    auto args = reader.getArgs();
    args_.reserve(args.size());
    for (auto arg : args) {
        args_.emplace_back(arg);
    }
    
    auto metadata = reader.getMetadata();
    for (auto field : metadata) {
        metadata_[field.getName().cStr()] = Value(field.getValue());
    }
}

void Fact::writeTo(schemas::Fact::Builder builder) const {
    builder.setId(id_);
    builder.setPredicate(predicate_);
    builder.setConfidence(confidence_);
    builder.setTimestamp(timestamp_);
    
    auto args = builder.initArgs(args_.size());
    for (size_t i = 0; i < args_.size(); ++i) {
        args_[i].writeTo(args[i]);
    }
    
    auto metadata = builder.initMetadata(metadata_.size());
    size_t i = 0;
    for (const auto& [key, value] : metadata_) {
        metadata[i].setName(key);
        value.writeTo(metadata[i].getValue());
        ++i;
    }
}

// Rule::Condition implementation
std::string Rule::Condition::toString() const {
    std::stringstream ss;
    if (negated) ss << "NOT ";
    ss << predicate << "(";
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << args[i].toString();
    }
    ss << ")";
    return ss.str();
}

// Rule::Conclusion implementation
std::string Rule::Conclusion::toString() const {
    std::stringstream ss;
    ss << predicate << "(";
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << args[i].toString();
    }
    ss << ")";
    if (confidence != 1.0) {
        ss << " [confidence: " << confidence << "]";
    }
    return ss.str();
}

// Rule implementation
Rule::Rule(uint64_t id, const std::string& name,
           const std::vector<Condition>& conditions,
           const std::vector<Conclusion>& conclusions,
           int32_t priority, double confidence)
    : id_(id), name_(name), conditions_(conditions), conclusions_(conclusions),
      priority_(priority), confidence_(confidence) {}

std::string Rule::toString() const {
    std::stringstream ss;
    ss << name_ << ": IF ";
    for (size_t i = 0; i < conditions_.size(); ++i) {
        if (i > 0) ss << " AND ";
        ss << conditions_[i].toString();
    }
    ss << " THEN ";
    for (size_t i = 0; i < conclusions_.size(); ++i) {
        if (i > 0) ss << " AND ";
        ss << conclusions_[i].toString();
    }
    if (priority_ != 0) {
        ss << " [priority: " << priority_ << "]";
    }
    return ss.str();
}

Rule::Rule(schemas::Rule::Reader reader)
    : id_(reader.getId())
    , name_(reader.getName().cStr())
    , priority_(reader.getPriority())
    , confidence_(reader.getConfidence()) {
    
    auto conditions = reader.getConditions();
    conditions_.reserve(conditions.size());
    for (auto cond : conditions) {
        Condition c;
        c.predicate = cond.getPredicate().cStr();
        c.negated = cond.getNegated();
        auto args = cond.getArgs();
        c.args.reserve(args.size());
        for (auto arg : args) {
            c.args.emplace_back(arg);
        }
        conditions_.push_back(std::move(c));
    }
    
    auto conclusions = reader.getConclusions();
    conclusions_.reserve(conclusions.size());
    for (auto concl : conclusions) {
        Conclusion c;
        c.predicate = concl.getPredicate().cStr();
        c.confidence = concl.getConfidence();
        auto args = concl.getArgs();
        c.args.reserve(args.size());
        for (auto arg : args) {
            c.args.emplace_back(arg);
        }
        conclusions_.push_back(std::move(c));
    }
}

void Rule::writeTo(schemas::Rule::Builder builder) const {
    builder.setId(id_);
    builder.setName(name_);
    builder.setPriority(priority_);
    builder.setConfidence(confidence_);
    
    auto conditions = builder.initConditions(conditions_.size());
    for (size_t i = 0; i < conditions_.size(); ++i) {
        conditions[i].setPredicate(conditions_[i].predicate);
        conditions[i].setNegated(conditions_[i].negated);
        auto args = conditions[i].initArgs(conditions_[i].args.size());
        for (size_t j = 0; j < conditions_[i].args.size(); ++j) {
            conditions_[i].args[j].writeTo(args[j]);
        }
    }
    
    auto conclusions = builder.initConclusions(conclusions_.size());
    for (size_t i = 0; i < conclusions_.size(); ++i) {
        conclusions[i].setPredicate(conclusions_[i].predicate);
        conclusions[i].setConfidence(conclusions_[i].confidence);
        auto args = conclusions[i].initArgs(conclusions_[i].args.size());
        for (size_t j = 0; j < conclusions_[i].args.size(); ++j) {
            conclusions_[i].args[j].writeTo(args[j]);
        }
    }
}

// Query implementation
Query::Query(uint64_t id, Type type, const Rule::Condition& goal,
             uint32_t maxResults, uint32_t timeoutMs)
    : id_(id), type_(type), goal_(goal), maxResults_(maxResults), timeoutMs_(timeoutMs) {}

std::string Query::toString() const {
    std::stringstream ss;
    ss << "Query[" << id_ << "]: ";
    switch (type_) {
        case Type::FindAll: ss << "FIND_ALL "; break;
        case Type::Prove: ss << "PROVE "; break;
        case Type::FindFirst: ss << "FIND_FIRST "; break;
        case Type::Explain: ss << "EXPLAIN "; break;
    }
    ss << goal_.toString();
    return ss.str();
}

// Serializer implementation
std::vector<uint8_t> Serializer::serialize(const Fact& fact) {
    ::capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<schemas::Fact>();
    fact.writeTo(builder);
    
    auto words = capnp::messageToFlatArray(message);
    auto bytes = words.asBytes();
    return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

std::vector<uint8_t> Serializer::serialize(const Rule& rule) {
    ::capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<schemas::Rule>();
    rule.writeTo(builder);
    
    auto words = capnp::messageToFlatArray(message);
    auto bytes = words.asBytes();
    return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

std::optional<Fact> Serializer::deserializeFact(const std::vector<uint8_t>& data) {
    try {
        kj::ArrayPtr<const capnp::word> words = 
            kj::arrayPtr(reinterpret_cast<const capnp::word*>(data.data()), 
                        data.size() / sizeof(capnp::word));
        
        capnp::FlatArrayMessageReader message(words);
        auto reader = message.getRoot<schemas::Fact>();
        return Fact(reader);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<Rule> Serializer::deserializeRule(const std::vector<uint8_t>& data) {
    try {
        kj::ArrayPtr<const capnp::word> words = 
            kj::arrayPtr(reinterpret_cast<const capnp::word*>(data.data()), 
                        data.size() / sizeof(capnp::word));
        
        capnp::FlatArrayMessageReader message(words);
        auto reader = message.getRoot<schemas::Rule>();
        return Rule(reader);
    } catch (...) {
        return std::nullopt;
    }
}

std::string Serializer::toJson(const Fact& fact) {
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"id\": " << fact.getId() << ",\n";
    ss << "  \"predicate\": \"" << fact.getPredicate() << "\",\n";
    ss << "  \"args\": [";
    const auto& args = fact.getArgs();
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << args[i].toString();
    }
    ss << "],\n";
    ss << "  \"confidence\": " << fact.getConfidence() << ",\n";
    ss << "  \"timestamp\": " << fact.getTimestamp() << "\n";
    ss << "}";
    return ss.str();
}

std::string Serializer::toJson(const Rule& rule) {
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"id\": " << rule.getId() << ",\n";
    ss << "  \"name\": \"" << rule.getName() << "\",\n";
    ss << "  \"conditions\": [";
    const auto& conditions = rule.getConditions();
    for (size_t i = 0; i < conditions.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "\"" << conditions[i].toString() << "\"";
    }
    ss << "],\n";
    ss << "  \"conclusions\": [";
    const auto& conclusions = rule.getConclusions();
    for (size_t i = 0; i < conclusions.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "\"" << conclusions[i].toString() << "\"";
    }
    ss << "],\n";
    ss << "  \"priority\": " << rule.getPriority() << "\n";
    ss << "}";
    return ss.str();
}

} // namespace inference_lab::common