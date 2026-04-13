/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <pytorch/tokenizers/truncation.h>
#include <nlohmann/json.hpp>
#include <vector>

namespace tokenizers {

TEST(TruncationTest, TestMaxLength) {
  nlohmann::json j = R"({
    "max_length": 5,
    "strategy": "LongestFirst",
    "stride": 0,
    "direction": "Right"
  })"_json;

  TruncationConfig config;
  config.parse_json(j);
  auto truncation = config.create();

  std::vector<uint64_t> tokens = {1, 2, 3, 4, 5, 6, 7};
  // truncate with 0 added tokens
  auto truncated = truncation->truncate(tokens, 0);

  EXPECT_EQ(truncated.size(), 5);
  EXPECT_EQ(truncated, std::vector<uint64_t>({1, 2, 3, 4, 5}));
}

TEST(TruncationTest, TestMaxLengthWithAddedTokens) {
  nlohmann::json j = R"({
    "max_length": 5,
    "strategy": "LongestFirst",
    "stride": 0,
    "direction": "Right"
  })"_json;

  TruncationConfig config;
  config.parse_json(j);
  auto truncation = config.create();

  std::vector<uint64_t> tokens = {1, 2, 3, 4, 5, 6, 7};
  // truncate with 2 added tokens (e.g. BOS + EOS)
  // effective max length should be 5 - 2 = 3
  auto truncated = truncation->truncate(tokens, 2);

  EXPECT_EQ(truncated.size(), 3);
  EXPECT_EQ(truncated, std::vector<uint64_t>({1, 2, 3}));
}

TEST(TruncationTest, TestTruncateLeft) {
  nlohmann::json j = R"({
    "max_length": 3,
    "direction": "Left"
  })"_json;

  TruncationConfig config;
  config.parse_json(j);
  auto truncation = config.create();

  std::vector<uint64_t> tokens = {1, 2, 3, 4, 5};
  auto truncated = truncation->truncate(tokens, 0);

  EXPECT_EQ(truncated.size(), 3);
  // Should keep the end: 3, 4, 5
  EXPECT_EQ(truncated, std::vector<uint64_t>({3, 4, 5}));
}

TEST(TruncationTest, TestNoTruncationNeeded) {
  nlohmann::json j = R"({
    "max_length": 10
  })"_json;

  TruncationConfig config;
  config.parse_json(j);
  auto truncation = config.create();

  std::vector<uint64_t> tokens = {1, 2, 3};
  auto truncated = truncation->truncate(tokens, 0);

  EXPECT_EQ(truncated.size(), 3);
  EXPECT_EQ(truncated, tokens);
}

TEST(TruncationTest, TestEffectiveMaxLengthZero) {
  nlohmann::json j = R"({
    "max_length": 2
  })"_json;

  TruncationConfig config;
  config.parse_json(j);
  auto truncation = config.create();

  std::vector<uint64_t> tokens = {1, 2, 3};
  // Added tokens (3) > max_length (2), effective max length 0
  auto truncated = truncation->truncate(tokens, 3);

  EXPECT_TRUE(truncated.empty());
}

TEST(TruncationTest, TestLongestFirstPair) {
  nlohmann::json j = R"({
        "max_length": 6,
        "strategy": "LongestFirst"
    })"_json;
  TruncationConfig config;
  config.parse_json(j);
  auto truncation = config.create();

  std::vector<uint64_t> a = {1, 2, 3, 4, 5}; // len 5
  std::vector<uint64_t> b = {10, 20}; // len 2
  // Total len 7, need to truncate to 6.
  // LongestFirst should take 1 from 'a' because it's longer.

  auto result = truncation->truncate_pair(a, b, 0);
  EXPECT_EQ(result.first.size(), 4);
  EXPECT_EQ(result.second.size(), 2);
  EXPECT_EQ(result.first, std::vector<uint64_t>({1, 2, 3, 4}));
}

TEST(TruncationTest, TestStride) {
  nlohmann::json j = R"({
        "max_length": 5,
        "stride": 2
    })"_json;
  // If your API returns a "TruncatedPath" or similar structure
  // containing the overflow, verify the stride overlap here.
  // Currently we only return the primary truncated vector.
}

TEST(TruncationTest, TestOnlyFirstStrategy) {
  nlohmann::json j = R"({
        "max_length": 5,
        "strategy": "OnlyFirst"
    })"_json;
  TruncationConfig config;
  config.parse_json(j);
  auto truncation = config.create();

  std::vector<uint64_t> a = {1, 2, 3};
  std::vector<uint64_t> b = {10, 20, 30};
  // Total 6. Even though we need 5, OnlyFirst can only cut 'a'.
  auto result = truncation->truncate_pair(a, b, 0);
  EXPECT_EQ(result.first.size(), 2);
  EXPECT_EQ(result.second.size(), 3);
}

} // namespace tokenizers
