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

#include <chrono>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <capnp/serialize.h>

namespace inference_lab::common {

//=============================================================================
// Value Implementation
//=============================================================================

// Factory methods for creating typed Value objects
// These use the private constructor to set the discriminant type and then
// initialize the appropriate storage member

auto Value::from_int64(int64_t value) -> Value {
    Value v(Type::INT64);
    v.int64_value_ = value;
    return v;
}

auto Value::from_float64(double value) -> Value {
    Value v(Type::FLOAT64);
    v.float64_value_ = value;
    return v;
}

auto Value::from_text(const std::string& value) -> Value {
    Value v(Type::TEXT);
    v.text_value_ = value;  // Copy string into text_value_ member
    return v;
}

auto Value::from_bool(bool value) -> Value {
    Value v(Type::BOOL);
    v.bool_value_ = value;
    return v;
}

auto Value::from_list(const std::vector<Value>& values) -> Value {
    Value v(Type::LIST);
    v.list_value_ = values;  // Deep copy of the vector and its Value elements
    return v;
}

auto Value::from_struct(const std::unordered_map<std::string, Value>& fields) -> Value {
    Value v(Type::STRUCT);
    v.struct_value_ = fields;  // Deep copy of the map and its Value elements
    return v;
}

// Type checking methods - these are simple discriminant checks
// They're const and noexcept since they only read the type_ field

auto Value::is_int64() const -> bool {
    return type_ == Type::INT64;
}
auto Value::is_float64() const -> bool {
    return type_ == Type::FLOAT64;
}
auto Value::is_text() const -> bool {
    return type_ == Type::TEXT;
}
auto Value::is_bool() const -> bool {
    return type_ == Type::BOOL;
}
auto Value::is_list() const -> bool {
    return type_ == Type::LIST;
}
auto Value::is_struct() const -> bool {
    return type_ == Type::STRUCT;
}

// Unsafe extraction methods - these throw exceptions on type mismatch
// Use these when you're certain of the type or want fail-fast behavior

auto Value::as_int64() const -> int64_t {
    if (type_ != Type::INT64)
        throw std::runtime_error("Value is not int64");
    return int64_value_;
}

auto Value::as_float64() const -> double {
    if (type_ != Type::FLOAT64)
        throw std::runtime_error("Value is not float64");
    return float64_value_;
}

auto Value::as_text() const -> std::string {
    if (type_ != Type::TEXT)
        throw std::runtime_error("Value is not text");
    return text_value_;  // Returns copy of the string
}

auto Value::as_bool() const -> bool {
    if (type_ != Type::BOOL)
        throw std::runtime_error("Value is not bool");
    return bool_value_;
}

auto Value::as_list() const -> std::vector<Value> {
    if (type_ != Type::LIST)
        throw std::runtime_error("Value is not list");
    return list_value_;  // Returns copy of the vector
}

auto Value::as_struct() const -> std::unordered_map<std::string, Value> {
    if (type_ != Type::STRUCT)
        throw std::runtime_error("Value is not struct");
    return struct_value_;  // Returns copy of the map
}

// Safe extraction methods - these return nullopt on type mismatch
// Use these when you want to handle type mismatches gracefully

auto Value::try_as_int64() const -> std::optional<int64_t> {
    return type_ == Type::INT64 ? std::optional<int64_t>(int64_value_) : std::nullopt;
}

auto Value::try_as_float64() const -> std::optional<double> {
    return type_ == Type::FLOAT64 ? std::optional<double>(float64_value_) : std::nullopt;
}

auto Value::try_as_text() const -> std::optional<std::string> {
    return type_ == Type::TEXT ? std::optional<std::string>(text_value_) : std::nullopt;
}

auto Value::try_as_bool() const -> std::optional<bool> {
    return type_ == Type::BOOL ? std::optional<bool>(bool_value_) : std::nullopt;
}

auto Value::try_as_list() const -> std::optional<std::vector<Value>> {
    return type_ == Type::LIST ? std::optional<std::vector<Value>>(list_value_) : std::nullopt;
}

auto Value::try_as_struct() const -> std::optional<std::unordered_map<std::string, Value>> {
    return type_ == Type::STRUCT
               ? std::optional<std::unordered_map<std::string, Value>>(struct_value_)
               : std::nullopt;
}

auto Value::to_string() const -> std::string {
    switch (type_) {
        case Type::INT64:
            return std::to_string(int64_value_);
        case Type::FLOAT64:
            return std::to_string(float64_value_);
        case Type::TEXT:
            return "\"" + text_value_ + "\"";
        case Type::BOOL:
            return bool_value_ ? "true" : "false";
        case Type::LIST: {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < list_value_.size(); ++i) {
                if (i > 0)
                    ss << ", ";
                ss << list_value_[i].to_string();
            }
            ss << "]";
            return ss.str();
        }
        case Type::STRUCT: {
            std::stringstream ss;
            ss << "{";
            bool first = true;
            for (const auto& [key, value] : struct_value_) {
                if (!first)
                    ss << ", ";
                ss << "\"" << key << "\": " << value.to_string();
                first = false;
            }
            ss << "}";
            return ss.str();
        }
    }
    return "<unknown>";
}

Value::Value(schemas::Value::Reader reader) : text_value_{}, list_value_{}, struct_value_{} {
    switch (reader.which()) {
        case schemas::Value::INT64_VALUE:
            type_ = Type::INT64;
            int64_value_ = reader.getInt64Value();
            break;
        case schemas::Value::FLOAT64_VALUE:
            type_ = Type::FLOAT64;
            float64_value_ = reader.getFloat64Value();
            break;
        case schemas::Value::TEXT_VALUE:
            type_ = Type::TEXT;
            text_value_ = reader.getTextValue().cStr();
            break;
        case schemas::Value::BOOL_VALUE:
            type_ = Type::BOOL;
            bool_value_ = reader.getBoolValue();
            break;
        case schemas::Value::LIST_VALUE: {
            type_ = Type::LIST;
            auto list = reader.getListValue();
            list_value_.reserve(list.size());
            for (auto item : list) {
                list_value_.emplace_back(item);
            }
            break;
        }
        case schemas::Value::STRUCT_VALUE: {
            type_ = Type::STRUCT;
            auto fields = reader.getStructValue();
            for (auto field : fields) {
                struct_value_[field.getName().cStr()] = Value(field.getValue());
            }
            break;
        }
    }
}

auto Value::write_to(schemas::Value::Builder builder) const -> void {
    switch (type_) {
        case Type::INT64:
            builder.setInt64Value(int64_value_);
            break;
        case Type::FLOAT64:
            builder.setFloat64Value(float64_value_);
            break;
        case Type::TEXT:
            builder.setTextValue(text_value_);
            break;
        case Type::BOOL:
            builder.setBoolValue(bool_value_);
            break;
        case Type::LIST: {
            auto list = builder.initListValue(list_value_.size());
            for (size_t i = 0; i < list_value_.size(); ++i) {
                list_value_[i].write_to(list[i]);
            }
            break;
        }
        case Type::STRUCT: {
            auto fields = builder.initStructValue(struct_value_.size());
            size_t i = 0;
            for (const auto& [key, value] : struct_value_) {
                fields[i].setName(key);
                value.write_to(fields[i].getValue());
                ++i;
            }
            break;
        }
    }
}

// Fact implementation
Fact::Fact(uint64_t id,
           const std::string& predicate,
           const std::vector<Value>& args,
           double confidence,
           uint64_t timestamp)
    : id_(id), predicate_(predicate), args_(args), confidence_(confidence), timestamp_(timestamp) {
    if (timestamp_ == 0) {
        timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    }
}

auto Fact::set_metadata(const std::string& key, const Value& value) -> void {
    metadata_[key] = value;
}

auto Fact::get_metadata(const std::string& key) const -> std::optional<Value> {
    auto it = metadata_.find(key);
    return it != metadata_.end() ? std::optional<Value>(it->second) : std::nullopt;
}

auto Fact::to_string() const -> std::string {
    std::stringstream ss;
    ss << predicate_ << "(";
    for (size_t i = 0; i < args_.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << args_[i].to_string();
    }
    ss << ")";
    if (confidence_ != 1.0) {
        ss << " [confidence: " << confidence_ << "]";
    }
    return ss.str();
}

Fact::Fact(schemas::Fact::Reader reader)
    : id_(reader.getId()),
      predicate_(reader.getPredicate().cStr()),
      confidence_(reader.getConfidence()),
      timestamp_(reader.getTimestamp()) {
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

auto Fact::write_to(schemas::Fact::Builder builder) const -> void {
    builder.setId(id_);
    builder.setPredicate(predicate_);
    builder.setConfidence(confidence_);
    builder.setTimestamp(timestamp_);

    auto args = builder.initArgs(args_.size());
    for (size_t i = 0; i < args_.size(); ++i) {
        args_[i].write_to(args[i]);
    }

    auto metadata = builder.initMetadata(metadata_.size());
    size_t i = 0;
    for (const auto& [key, value] : metadata_) {
        metadata[i].setName(key);
        value.write_to(metadata[i].getValue());
        ++i;
    }
}

// Rule::Condition implementation
auto Rule::Condition::to_string() const -> std::string {
    std::stringstream ss;
    if (negated_)
        ss << "NOT ";
    ss << predicate_ << "(";
    for (size_t i = 0; i < args_.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << args_[i].to_string();
    }
    ss << ")";
    return ss.str();
}

// Rule::Conclusion implementation
auto Rule::Conclusion::to_string() const -> std::string {
    std::stringstream ss;
    ss << predicate_ << "(";
    for (size_t i = 0; i < args_.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << args_[i].to_string();
    }
    ss << ")";
    if (confidence_ != 1.0) {
        ss << " [confidence: " << confidence_ << "]";
    }
    return ss.str();
}

// Rule implementation
Rule::Rule(uint64_t id,
           const std::string& name,
           const std::vector<Condition>& conditions,
           const std::vector<Conclusion>& conclusions,
           int32_t priority,
           double confidence)
    : id_(id),
      name_(name),
      conditions_(conditions),
      conclusions_(conclusions),
      priority_(priority),
      confidence_(confidence) {}

auto Rule::to_string() const -> std::string {
    std::stringstream ss;
    ss << name_ << ": IF ";
    for (size_t i = 0; i < conditions_.size(); ++i) {
        if (i > 0)
            ss << " AND ";
        ss << conditions_[i].to_string();
    }
    ss << " THEN ";
    for (size_t i = 0; i < conclusions_.size(); ++i) {
        if (i > 0)
            ss << " AND ";
        ss << conclusions_[i].to_string();
    }
    if (priority_ != 0) {
        ss << " [priority: " << priority_ << "]";
    }
    return ss.str();
}

Rule::Rule(schemas::Rule::Reader reader)
    : id_(reader.getId()),
      name_(reader.getName().cStr()),
      priority_(reader.getPriority()),
      confidence_(reader.getConfidence()) {
    auto conditions = reader.getConditions();
    conditions_.reserve(conditions.size());
    for (auto cond : conditions) {
        Condition c;
        c.predicate_ = cond.getPredicate().cStr();
        c.negated_ = cond.getNegated();
        auto args = cond.getArgs();
        c.args_.reserve(args.size());
        for (auto arg : args) {
            c.args_.emplace_back(arg);
        }
        conditions_.push_back(std::move(c));
    }

    auto conclusions = reader.getConclusions();
    conclusions_.reserve(conclusions.size());
    for (auto concl : conclusions) {
        Conclusion c;
        c.predicate_ = concl.getPredicate().cStr();
        c.confidence_ = concl.getConfidence();
        auto args = concl.getArgs();
        c.args_.reserve(args.size());
        for (auto arg : args) {
            c.args_.emplace_back(arg);
        }
        conclusions_.push_back(std::move(c));
    }
}

auto Rule::write_to(schemas::Rule::Builder builder) const -> void {
    builder.setId(id_);
    builder.setName(name_);
    builder.setPriority(priority_);
    builder.setConfidence(confidence_);

    auto conditions = builder.initConditions(conditions_.size());
    for (size_t i = 0; i < conditions_.size(); ++i) {
        conditions[i].setPredicate(conditions_[i].predicate_);
        conditions[i].setNegated(conditions_[i].negated_);
        auto args = conditions[i].initArgs(conditions_[i].args_.size());
        for (size_t j = 0; j < conditions_[i].args_.size(); ++j) {
            conditions_[i].args_[j].write_to(args[j]);
        }
    }

    auto conclusions = builder.initConclusions(conclusions_.size());
    for (size_t i = 0; i < conclusions_.size(); ++i) {
        conclusions[i].setPredicate(conclusions_[i].predicate_);
        conclusions[i].setConfidence(conclusions_[i].confidence_);
        auto args = conclusions[i].initArgs(conclusions_[i].args_.size());
        for (size_t j = 0; j < conclusions_[i].args_.size(); ++j) {
            conclusions_[i].args_[j].write_to(args[j]);
        }
    }
}

// Query implementation
Query::Query(
    uint64_t id, Type type, const Rule::Condition& goal, uint32_t max_results, uint32_t timeout_ms)
    : id_(id), type_(type), goal_(goal), max_results_(max_results), timeout_ms_(timeout_ms) {}

auto Query::to_string() const -> std::string {
    std::stringstream ss;
    ss << "Query[" << id_ << "]: ";
    switch (type_) {
        case Type::FIND_ALL:
            ss << "FIND_ALL ";
            break;
        case Type::PROVE:
            ss << "PROVE ";
            break;
        case Type::FIND_FIRST:
            ss << "FIND_FIRST ";
            break;
        case Type::EXPLAIN:
            ss << "EXPLAIN ";
            break;
    }
    ss << goal_.to_string();
    return ss.str();
}

// Serializer implementation
auto Serializer::serialize(const Fact& fact) -> std::vector<uint8_t> {
    ::capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<schemas::Fact>();
    fact.write_to(builder);

    kj::Array<capnp::word> words = capnp::messageToFlatArray(message);
    kj::ArrayPtr<const kj::byte> bytes = words.asBytes();
    return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

auto Serializer::serialize(const Rule& rule) -> std::vector<uint8_t> {
    ::capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<schemas::Rule>();
    rule.write_to(builder);

    kj::Array<capnp::word> words = capnp::messageToFlatArray(message);
    kj::ArrayPtr<const kj::byte> bytes = words.asBytes();
    return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

auto Serializer::deserialize_fact(const std::vector<uint8_t>& data) -> std::optional<Fact> {
    try {
        kj::ArrayPtr<const capnp::word> words = kj::arrayPtr(
            reinterpret_cast<const capnp::word*>(data.data()), data.size() / sizeof(capnp::word));

        ::capnp::FlatArrayMessageReader message(words);
        auto reader = message.getRoot<schemas::Fact>();
        return Fact(reader);
    } catch (...) {
        return std::nullopt;
    }
}

auto Serializer::deserialize_rule(const std::vector<uint8_t>& data) -> std::optional<Rule> {
    try {
        kj::ArrayPtr<const capnp::word> words = kj::arrayPtr(
            reinterpret_cast<const capnp::word*>(data.data()), data.size() / sizeof(capnp::word));

        ::capnp::FlatArrayMessageReader message(words);
        auto reader = message.getRoot<schemas::Rule>();
        return Rule(reader);
    } catch (...) {
        return std::nullopt;
    }
}

auto Serializer::to_json(const Fact& fact) -> std::string {
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"id\": " << fact.get_id() << ",\n";
    ss << "  \"predicate\": \"" << fact.get_predicate() << "\",\n";
    ss << "  \"args\": [";
    const auto& args = fact.get_args();
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << args[i].to_string();
    }
    ss << "],\n";
    ss << "  \"confidence\": " << fact.get_confidence() << ",\n";
    ss << "  \"timestamp\": " << fact.get_timestamp() << "\n";
    ss << "}";
    return ss.str();
}

auto Serializer::to_json(const Rule& rule) -> std::string {
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"id\": " << rule.get_id() << ",\n";
    ss << "  \"name\": \"" << rule.get_name() << "\",\n";
    ss << "  \"conditions\": [";
    const auto& conditions = rule.get_conditions();
    for (size_t i = 0; i < conditions.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << "\"" << conditions[i].to_string() << "\"";
    }
    ss << "],\n";
    ss << "  \"conclusions\": [";
    const auto& conclusions = rule.get_conclusions();
    for (size_t i = 0; i < conclusions.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << "\"" << conclusions[i].to_string() << "\"";
    }
    ss << "],\n";
    ss << "  \"priority\": " << rule.get_priority() << "\n";
    ss << "}";
    return ss.str();
}

}  // namespace inference_lab::common
