/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/normalizer.h>

using namespace tokenizers;

TEST(NormalizerTest, ReplaceNormalizerBasic) {
  // Test basic string replacement
  ReplaceNormalizer normalizer(" ", "▁");
  std::string input = "Hello World Test";
  std::string expected = "Hello▁World▁Test";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, ReplaceNormalizerNoMatch) {
  // Test when pattern doesn't match
  ReplaceNormalizer normalizer("xyz", "▁");
  std::string input = "Hello World";
  std::string expected = "Hello World";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, ReplaceNormalizerMultipleMatches) {
  // Test multiple matches
  ReplaceNormalizer normalizer("a", "X");
  std::string input = "banana";
  std::string expected = "bXnXnX";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, PrependNormalizerBasic) {
  // Test basic prepending
  PrependNormalizer normalizer("_");
  std::string input = "Hello";
  std::string expected = "_Hello";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, PrependNormalizerEmptyInput) {
  // Test prepend with empty input (should return empty)
  PrependNormalizer normalizer("_");
  std::string input = "";
  std::string expected = "";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, LowercaseNormalizerBasic) {
  // Test basic lowercasing
  LowercaseNormalizer normalizer;
  EXPECT_EQ(normalizer.normalize("HELLO WORLD"), "hello world");
  EXPECT_EQ(normalizer.normalize("Hello World"), "hello world");
  EXPECT_EQ(normalizer.normalize("hello world"), "hello world");
  // Test with accents (should lowercase but NOT strip them, LowercaseNormalizer
  // only lowercases)
  EXPECT_EQ(normalizer.normalize("HÉLLO"), "héllo");
}

TEST(NormalizerTest, NormalizerConfigLowercase) {
  // Test JSON parsing for Lowercase normalizer
  nlohmann::json config = {{"type", "Lowercase"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  EXPECT_EQ(normalizer->normalize("HELLO"), "hello");
}

TEST(NormalizerTest, NormalizerConfigPrepend) {
  // Test JSON parsing for Prepend normalizer
  nlohmann::json config = {{"type", "Prepend"}, {"prepend", "_"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "Hello";
  std::string expected = "_Hello";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, NormalizerConfigFromJson) {
  // Test JSON parsing for Replace normalizer
  nlohmann::json config = {
      {"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", "▁"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "Hello World Test";
  std::string expected = "Hello▁World▁Test";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, NormalizerConfigFromJsonRegex) {
  // Test JSON parsing for Replace normalizer with regex
  nlohmann::json config = {
      {"type", "Replace"}, {"pattern", {{"Regex", "\\s+"}}}, {"content", "_"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "Hello   World\t\tTest";
  std::string expected = "Hello_World_Test";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, SequenceNormalizer) {
  // Test sequence of normalizers
  std::vector<Normalizer::Ptr> normalizers;
  normalizers.push_back(std::make_shared<ReplaceNormalizer>(" ", "▁"));
  normalizers.push_back(std::make_shared<ReplaceNormalizer>("a", "X"));

  SequenceNormalizer seq_normalizer(normalizers);

  std::string input = "banana split";
  std::string expected = "bXnXnX▁split";
  std::string result = seq_normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, SequenceNormalizerFromConfig) {
  // Test sequence normalizer from config
  nlohmann::json config = {
      {"type", "Sequence"},
      {"normalizers",
       {{{"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", "▁"}},
        {{"type", "Replace"},
         {"pattern", {{"String", "a"}}},
         {"content", "X"}}}}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "banana split";
  std::string expected = "bXnXnX▁split";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, EmptyInput) {
  // Test with empty input
  ReplaceNormalizer normalizer(" ", "▁");
  std::string input = "";
  std::string expected = "";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, ConfigBuilder) {
  // Test config builder pattern
  auto normalizer =
      NormalizerConfig("Replace").set_pattern(" ").set_content("▁").create();

  std::string input = "Hello World";
  std::string expected = "Hello▁World";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

// -- BertNormalizer Exhaustive Tests ------------------------------------------

TEST(NormalizerTest, BertNormalizerAccentDefaulting) {
  // Logic: strip_accents should inherit 'lowercase' value if not set.
  // Test case: lowercase=true, strip_accents=unset -> should strip
  auto norm1 = NormalizerConfig("BertNormalizer").set_lowercase(true).create();
  EXPECT_EQ(norm1->normalize("Héllò"), "hello");

  // Test case: lowercase=false, strip_accents=unset -> should NOT strip
  auto norm2 = NormalizerConfig("BertNormalizer").set_lowercase(false).create();
  EXPECT_EQ(norm2->normalize("Héllò"), "Héllò");
}

TEST(NormalizerTest, BertNormalizerControlChars) {
  auto normalizer = NormalizerConfig("BertNormalizer")
                        .set_clean_text(true)
                        .set_lowercase(false)
                        .create();

  // 1. Remove NULL and Replacement Character
  // Using std::string constructor to include the null byte
  std::string input = "A";
  input.push_back('\0');
  input +=
      "B\xEF\xBF\xBD"
      "C";
  EXPECT_EQ(normalizer->normalize(input), "ABC");

  // 2. Remove C0/C1 Control chars but KEEP \n, \t, \r (mapping them to space)
  // \x01 is a control char, \x1F is a control char
  // \n and \t should become spaces
  EXPECT_EQ(
      normalizer->normalize(
          "start\x01\n\t\x1F"
          "end"),
      "start  end");
}

TEST(NormalizerTest, BertNormalizerCJKExhaustive) {
  auto normalizer = NormalizerConfig("BertNormalizer")
                        .set_handle_chinese_chars(true)
                        .set_lowercase(false)
                        .create();

  // 1. Consecutive CJK characters (should have spaces between each)
  // \u4e00 and \u4e01 are CJK
  EXPECT_EQ(normalizer->normalize("\u4e00\u4e01"), " \u4e00  \u4e01 ");

  // 2. CJK at the very start and end of string
  EXPECT_EQ(normalizer->normalize("\u4e00"), " \u4e00 ");

  // 3. CJK mixed with non-CJK
  EXPECT_EQ(normalizer->normalize("A\u4e00Z"), "A \u4e00 Z");
}

TEST(NormalizerTest, BertNormalizerDecomposition) {
  auto normalizer = NormalizerConfig("BertNormalizer")
                        .set_lowercase(true)
                        .set_strip_accents(true)
                        .create();

  // Test "complex" accents like cedilla and tilde
  // ç -> c, ñ -> n
  EXPECT_EQ(normalizer->normalize("FAÇADE"), "facade");
  EXPECT_EQ(normalizer->normalize("cañón"), "canon");

  // German Eszett is lowercased but standard BERT keeps it or handles via
  // WordPiece
  EXPECT_EQ(normalizer->normalize("STRASSE"), "strasse");
}

TEST(NormalizerTest, BertNormalizerWhitespaceCollapsing) {
  auto normalizer =
      NormalizerConfig("BertNormalizer").set_clean_text(true).create();

  // Each whitespace char is handled by mapping to a single space.
  EXPECT_EQ(normalizer->normalize("hello \t\n\r world"), "hello     world");
}

TEST(NormalizerTest, BertNormalizerJsonIncomplete) {
  // Test that parsing JSON with missing fields uses defaults
  // Default BERT: clean=true, handle_cjk=true, lower=true, accents=null
  nlohmann::json config = {{"type", "BertNormalizer"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  // Should lowercase and strip accents by default
  EXPECT_EQ(normalizer->normalize("HÉLLO"), "hello");
}
