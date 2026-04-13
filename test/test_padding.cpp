/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <pytorch/tokenizers/padding.h>
#include <nlohmann/json.hpp>

namespace tokenizers {

TEST(PaddingTest, TestBatchLongestRightPadding) {
  nlohmann::json j = R"({
    "strategy": "BatchLongest",
    "direction": "Right",
    "pad_to_multiple_of": null,
    "pad_id": 0,
    "pad_type_id": 0,
    "pad_token": "[PAD]"
  })"_json;

  PaddingConfig config;
  config.parse_json(j);
  auto padding = config.create();

  std::vector<uint64_t> tokens = {1, 2, 3};
  auto padded = padding->pad(tokens);

  // BatchLongest with no multiple should not change length of a single sequence
  EXPECT_EQ(padded.size(), 3);
  EXPECT_EQ(padded, tokens);
}

TEST(PaddingTest, TestBatchLongestWithMultiple) {
  nlohmann::json j = R"({
    "strategy": "BatchLongest",
    "direction": "Right",
    "pad_to_multiple_of": 77,
    "pad_id": 49407,
    "pad_type_id": 0,
    "pad_token": "<|endoftext|>"
  })"_json;

  PaddingConfig config;
  config.parse_json(j);
  auto padding = config.create();

  std::vector<uint64_t> tokens = {1, 2, 3};
  auto padded = padding->pad(tokens);

  // Length 3 should be padded to 77
  EXPECT_EQ(padded.size(), 77);
  EXPECT_EQ(padded[0], 1);
  EXPECT_EQ(padded[1], 2);
  EXPECT_EQ(padded[2], 3);
  for (size_t i = 3; i < 77; ++i) {
    EXPECT_EQ(padded[i], 49407);
  }
}

TEST(PaddingTest, TestFixedStrategy) {
  nlohmann::json j = R"({
    "strategy": { "Fixed": 10 },
    "direction": "Right",
    "pad_id": 0
  })"_json;

  PaddingConfig config;
  config.parse_json(j);
  auto padding = config.create();

  std::vector<uint64_t> tokens = {1, 2, 3};
  auto padded = padding->pad(tokens);

  EXPECT_EQ(padded.size(), 10);
  for (size_t i = 3; i < 10; ++i) {
    EXPECT_EQ(padded[i], 0);
  }
}

TEST(PaddingTest, TestLeftPadding) {
  nlohmann::json j = R"({
    "strategy": { "Fixed": 5 },
    "direction": "Left",
    "pad_id": 1
  })"_json;

  PaddingConfig config;
  config.parse_json(j);
  auto padding = config.create();

  std::vector<uint64_t> tokens = {10, 20};
  auto padded = padding->pad(tokens);

  EXPECT_EQ(padded.size(), 5);
  EXPECT_EQ(padded[0], 1);
  EXPECT_EQ(padded[1], 1);
  EXPECT_EQ(padded[2], 1);
  EXPECT_EQ(padded[3], 10);
  EXPECT_EQ(padded[4], 20);
}

TEST(PaddingTest, TestPadToMultipleOfWithFixed) {
  nlohmann::json j = R"({
    "strategy": { "Fixed": 5 },
    "pad_to_multiple_of": 8,
    "pad_id": 0
  })"_json;

  PaddingConfig config;
  config.parse_json(j);
  auto padding = config.create();

  std::vector<uint64_t> tokens = {1, 2};
  auto padded = padding->pad(tokens);

  // Fixed 5, but pad_to_multiple_of 8 means target length is 8
  EXPECT_EQ(padded.size(), 8);
}

TEST(PaddingTest, TestNoPaddingIfLonger) {
  nlohmann::json j = R"({
    "strategy": { "Fixed": 5 },
    "pad_id": 0
  })"_json;

  PaddingConfig config;
  config.parse_json(j);
  auto padding = config.create();

  std::vector<uint64_t> tokens = {1, 2, 3, 4, 5, 6, 7};
  auto padded = padding->pad(tokens);

  EXPECT_EQ(padded.size(), 7);
  EXPECT_EQ(padded, tokens);
}

TEST(PaddingTest, TestAlreadyAMultiple) {
  nlohmann::json j = R"({
    "strategy": "BatchLongest",
    "pad_to_multiple_of": 8,
    "pad_id": 0
  })"_json;
  PaddingConfig config;
  config.parse_json(j);
  auto padding = config.create();

  // Length 8 is already a multiple of 8.
  std::vector<uint64_t> tokens(8, 1);
  auto padded = padding->pad(tokens);
  EXPECT_EQ(padded.size(), 8);
}

TEST(PaddingTest, TestFixedAndMultipleInteraction) {
  nlohmann::json j = R"({
    "strategy": { "Fixed": 10 },
    "pad_to_multiple_of": 8,
    "pad_id": 0
  })"_json;
  PaddingConfig config;
  config.parse_json(j);
  auto padding = config.create();

  std::vector<uint64_t> tokens = {1, 2, 3};
  auto padded = padding->pad(tokens);

  // Should pad to 16 (Max(10, tokens.size()) rounded up to multiple of 8)
  EXPECT_EQ(padded.size(), 16);
}

TEST(PaddingTest, TestAttentionMaskGeneration) {
  // Setup Right Padding
  auto padding =
      PaddingConfig("BatchLongest").set_pad_id(0).set_fixed_size(5).create();

  std::vector<uint64_t> tokens = {10, 20};
  auto padded = padding->pad(tokens);
  // Assuming your API has a way to get the mask
  auto mask = padding->generate_mask(tokens, padded.size());

  std::vector<uint32_t> expected_mask = {1, 1, 0, 0, 0};
  EXPECT_EQ(mask, expected_mask);
}

} // namespace tokenizers
