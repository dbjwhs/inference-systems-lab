// MIT License
// Copyright (c) 2025 dbjwhs
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
