#include <memory>

#include <gtest/gtest.h>

#include "../moe_config.hpp"

namespace engines::mixture_experts {

class MoEConfigTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a basic valid configuration
        config_ = MoESystemConfig{};  // Uses defaults
    }

    MoESystemConfig config_;
};

TEST_F(MoEConfigTest, DefaultConfigurationIsValid) {
    EXPECT_TRUE(config_.validate()) << "Default configuration should be valid";

    // Check some key defaults
    EXPECT_EQ(config_.num_experts, 8u);
    EXPECT_EQ(config_.expert_capacity, 2u);
    EXPECT_EQ(config_.top_k_experts, 2u);
    EXPECT_GT(config_.memory_pool_size_mb, 0u);
}

TEST_F(MoEConfigTest, CreateDevelopmentConfig) {
    auto dev_config = MoESystemConfig::create_development_config();
    EXPECT_TRUE(dev_config.validate()) << "Development config should be valid";
    EXPECT_TRUE(dev_config.enable_debug_logging)
        << "Development config should enable debug logging";
    EXPECT_LT(dev_config.memory_pool_size_mb, 200u) << "Development config should use less memory";
}

TEST_F(MoEConfigTest, CreateProductionConfig) {
    auto prod_config = MoESystemConfig::create_production_config();
    EXPECT_TRUE(prod_config.validate()) << "Production config should be valid";
    EXPECT_FALSE(prod_config.enable_debug_logging)
        << "Production config should disable debug logging";
    EXPECT_GT(prod_config.memory_pool_size_mb, 400u) << "Production config should use more memory";
    EXPECT_TRUE(prod_config.enable_parameter_compression) << "Production should use compression";
}

TEST_F(MoEConfigTest, CreateLightweightConfig) {
    auto light_config = MoESystemConfig::create_lightweight_config();
    EXPECT_TRUE(light_config.validate()) << "Lightweight config should be valid";
    EXPECT_LT(light_config.memory_pool_size_mb, 100u)
        << "Lightweight config should minimize memory";
    EXPECT_LT(light_config.num_experts, 8u) << "Lightweight config should use fewer experts";
}

TEST_F(MoEConfigTest, CreatePerformanceConfig) {
    auto perf_config = MoESystemConfig::create_performance_config();
    EXPECT_TRUE(perf_config.validate()) << "Performance config should be valid";
    EXPECT_TRUE(perf_config.enable_simd_optimization) << "Performance config should enable SIMD";
    EXPECT_GT(perf_config.max_concurrent_requests, 100u)
        << "Performance config should handle more requests";
}

TEST_F(MoEConfigTest, InvalidConfigurationZeroExperts) {
    config_.num_experts = 0;
    EXPECT_FALSE(config_.validate()) << "Zero experts should be invalid";
}

TEST_F(MoEConfigTest, InvalidConfigurationExpertCapacityTooHigh) {
    config_.expert_capacity = 20;  // Much higher than num_experts=8
    EXPECT_FALSE(config_.validate()) << "Expert capacity > num_experts should be invalid";
}

TEST_F(MoEConfigTest, InvalidConfigurationNegativeThresholds) {
    config_.sparsity_threshold = -0.1f;  // Negative threshold
    EXPECT_FALSE(config_.validate()) << "Negative sparsity threshold should be invalid";
}

TEST_F(MoEConfigTest, ConfigurationToStringIsReadable) {
    auto config_str = config_.to_string();
    EXPECT_FALSE(config_str.empty()) << "Configuration string should not be empty";
    EXPECT_NE(config_str.find("num_experts"), std::string::npos)
        << "Configuration string should contain key parameters";
    EXPECT_NE(config_str.find("expert_capacity"), std::string::npos)
        << "Configuration string should contain expert capacity";
}

TEST_F(MoEConfigTest, ValidatorValidatesValidConfiguration) {
    auto validation_result = MoEConfigValidator::validate_configuration(config_);
    EXPECT_TRUE(validation_result.is_configuration_valid())
        << "Valid configuration should pass validation";
    EXPECT_TRUE(validation_result.errors.empty()) << "Should have no errors";
}

TEST_F(MoEConfigTest, ValidatorDetectsInvalidConfiguration) {
    config_.num_experts = 0;  // Make config invalid

    auto validation_result = MoEConfigValidator::validate_configuration(config_);
    EXPECT_FALSE(validation_result.is_configuration_valid())
        << "Invalid configuration should fail validation";
    EXPECT_FALSE(validation_result.errors.empty()) << "Should have errors";
}

TEST_F(MoEConfigTest, ValidatorProducesWarningsForSuboptimalConfig) {
    config_.memory_pool_size_mb = 10;        // Very small memory pool
    config_.max_concurrent_requests = 1000;  // Very high request load

    auto validation_result = MoEConfigValidator::validate_configuration(config_);
    // Should still be valid but with warnings
    EXPECT_TRUE(validation_result.is_configuration_valid());
    EXPECT_FALSE(validation_result.warnings.empty())
        << "Should have warnings for suboptimal config";
}

TEST_F(MoEConfigTest, ValidatorProvidesRecommendations) {
    config_.enable_simd_optimization = false;      // Suboptimal for performance
    config_.enable_parameter_compression = false;  // Suboptimal for memory

    auto validation_result = MoEConfigValidator::validate_configuration(config_);
    EXPECT_FALSE(validation_result.recommendations.empty())
        << "Should provide recommendations for improvement";
}

TEST_F(MoEConfigTest, ValidationReportIsReadable) {
    config_.num_experts = 0;  // Create invalid config

    auto validation_result = MoEConfigValidator::validate_configuration(config_);
    auto report = validation_result.get_validation_report();

    EXPECT_FALSE(report.empty()) << "Validation report should not be empty";
    EXPECT_NE(report.find("Error"), std::string::npos) << "Report should mention errors";
}

TEST_F(MoEConfigTest, SystemCapabilityValidation) {
    auto capability_result = MoEConfigValidator::validate_against_system_capabilities(config_);
    // This should generally pass unless system is severely constrained
    EXPECT_TRUE(capability_result.is_configuration_valid() || !capability_result.warnings.empty())
        << "System capability validation should either pass or provide warnings";
}

TEST_F(MoEConfigTest, OptimizationRecommendations) {
    auto optimized_config = MoEConfigValidator::recommend_optimizations(config_);
    EXPECT_TRUE(optimized_config.validate()) << "Optimized config should be valid";

    // Optimized config should generally be different from original
    // (unless original was already optimal)
    bool has_changes =
        (optimized_config.memory_pool_size_mb != config_.memory_pool_size_mb) ||
        (optimized_config.max_concurrent_requests != config_.max_concurrent_requests) ||
        (optimized_config.enable_simd_optimization != config_.enable_simd_optimization);

    // Allow for case where original config was already optimal
    EXPECT_TRUE(has_changes || optimized_config.to_string() == config_.to_string())
        << "Optimization should either improve config or leave optimal config unchanged";
}

TEST_F(MoEConfigTest, PerformanceTargetConstants) {
    // Verify constants align with roadmap specifications
    EXPECT_EQ(MoEConstants::TARGET_EFFICIENCY_MIN, 15.0f);
    EXPECT_EQ(MoEConstants::TARGET_EFFICIENCY_MAX, 25.0f);
    EXPECT_EQ(MoEConstants::TARGET_P50_LATENCY_MS, 75.0f);
    EXPECT_EQ(MoEConstants::TARGET_P95_LATENCY_MS, 150.0f);
    EXPECT_EQ(MoEConstants::TARGET_EXPERT_SELECTION_MS, 5.0f);
}

TEST_F(MoEConfigTest, MemoryEstimation) {
    // Test memory requirement estimation
    auto estimated_memory = MoEConfigValidator::estimate_memory_requirements(config_);
    EXPECT_GT(estimated_memory, 0u) << "Should estimate non-zero memory requirements";
    EXPECT_LE(estimated_memory, config_.memory_pool_size_mb * config_.num_experts * 2)
        << "Estimate should be reasonable relative to configuration";
}

TEST_F(MoEConfigTest, PerformanceCharacteristicsEstimation) {
    auto [expected_latency, expected_throughput] =
        MoEConfigValidator::estimate_performance_characteristics(config_);

    EXPECT_GT(expected_latency, 0.0f) << "Should estimate positive latency";
    EXPECT_GT(expected_throughput, 0.0f) << "Should estimate positive throughput";

    // Latency estimates should be reasonable for default config
    EXPECT_LT(expected_latency, 500.0f) << "Estimated latency should be reasonable";
}

TEST_F(MoEConfigTest, ConfigurationBoundaryValues) {
    // Test edge cases for configuration parameters

    // Minimum valid configuration
    config_.num_experts = 1;
    config_.expert_capacity = 1;
    config_.top_k_experts = 1;
    config_.memory_pool_size_mb = 1;

    auto validation_result = MoEConfigValidator::validate_configuration(config_);
    EXPECT_TRUE(validation_result.is_configuration_valid())
        << "Minimal valid configuration should pass";

    // Maximum reasonable configuration
    config_.num_experts = 256;
    config_.expert_capacity = 8;
    config_.top_k_experts = 8;
    config_.memory_pool_size_mb = 10000;

    validation_result = MoEConfigValidator::validate_configuration(config_);
    EXPECT_TRUE(validation_result.is_configuration_valid())
        << "Large but valid configuration should pass";
}

TEST_F(MoEConfigTest, ConfigurationConsistencyCheck) {
    // Verify that related parameters are consistent
    config_.expert_capacity = 4;
    config_.top_k_experts = 2;  // Should be <= expert_capacity

    EXPECT_TRUE(config_.validate()) << "Consistent capacity/top_k should be valid";

    config_.top_k_experts = 6;  // Now > expert_capacity
    EXPECT_FALSE(config_.validate()) << "Inconsistent capacity/top_k should be invalid";
}

}  // namespace engines::mixture_experts
