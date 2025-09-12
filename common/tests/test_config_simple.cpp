// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#include <gtest/gtest.h>

#include "config_loader_simple.hpp"

namespace inference_lab {

class SimpleConfigLoaderTest : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SimpleConfigLoaderTest, BasicConfigurationLoading) {
    SimpleConfigurationLoader loader(std::nullopt, "development");

    auto result = loader.load();
    ASSERT_TRUE(result.is_ok());
    EXPECT_TRUE(result.unwrap());
}

TEST_F(SimpleConfigLoaderTest, DefaultValues) {
    SimpleConfigurationLoader loader(std::nullopt, "development");
    loader.load();

    // Test default values
    EXPECT_EQ(loader.get("application.name"), "inference-systems-lab");
    EXPECT_EQ(loader.get("application.version"), "0.1.0");
    EXPECT_EQ(loader.get("application.environment"), "development");

    // Test boolean access
    EXPECT_TRUE(loader.get_bool("application.debug_mode"));

    // Test missing key with default
    EXPECT_EQ(loader.get("missing.key", "default"), "default");
    EXPECT_FALSE(loader.get_bool("missing.bool", false));
    EXPECT_EQ(loader.get_int("missing.int", 42), 42);
}

TEST_F(SimpleConfigLoaderTest, EnvironmentProperties) {
    SimpleConfigurationLoader dev_loader(std::nullopt, "development");
    dev_loader.load();

    EXPECT_EQ(dev_loader.environment_name(), "development");
    EXPECT_TRUE(dev_loader.is_development());
    EXPECT_FALSE(dev_loader.is_staging());
    EXPECT_FALSE(dev_loader.is_production());

    SimpleConfigurationLoader prod_loader(std::nullopt, "production");
    prod_loader.load();

    EXPECT_EQ(prod_loader.environment_name(), "production");
    EXPECT_FALSE(prod_loader.is_development());
    EXPECT_FALSE(prod_loader.is_staging());
    EXPECT_TRUE(prod_loader.is_production());
}

}  // namespace inference_lab
