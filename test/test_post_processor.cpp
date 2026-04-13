/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/post_processor.h>
#include <vector>

using namespace tokenizers;
using json = nlohmann::json;

TEST(PostProcessorTest, TemplateProcessingSingleSequence) {
  // Setup: [CLS] $0 [SEP]
  Template single_template = {
      Piece::SpecialToken("[CLS]", 0),
      Piece::Sequence(SequenceId::A, 0),
      Piece::SpecialToken("[SEP]", 0)};
  Template pair_template; // Empty

  std::map<std::string, SpecialToken> special_tokens;
  special_tokens["[CLS]"] = {"[CLS]", {101}, {"[CLS]"}};
  special_tokens["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};

  auto processor = std::make_shared<TemplateProcessing>(
      single_template, pair_template, special_tokens);

  std::vector<uint64_t> input = {1, 2, 3};
  std::vector<uint64_t> expected = {101, 1, 2, 3, 102};

  auto output = processor->process(input);

  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, TemplateProcessingPairSequence) {
  // Setup: [CLS] $A [SEP] $B [SEP]
  Template single_template;
  Template pair_template = {
      Piece::SpecialToken("[CLS]", 0),
      Piece::Sequence(SequenceId::A, 0),
      Piece::SpecialToken("[SEP]", 0),
      Piece::Sequence(SequenceId::B, 1),
      Piece::SpecialToken("[SEP]", 0)};

  std::map<std::string, SpecialToken> special_tokens;
  special_tokens["[CLS]"] = {"[CLS]", {101}, {"[CLS]"}};
  special_tokens["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};

  auto processor = std::make_shared<TemplateProcessing>(
      single_template, pair_template, special_tokens);

  std::vector<uint64_t> input_a = {1, 2};
  std::vector<uint64_t> input_b = {3, 4};
  std::vector<uint64_t> expected = {101, 1, 2, 102, 3, 4, 102};

  auto output = processor->process(input_a, input_b);

  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, SequenceProcessing) {
  // Processor 1: Prepend 101
  Template t1 = {
      Piece::SpecialToken("[CLS]", 0), Piece::Sequence(SequenceId::A, 0)};
  std::map<std::string, SpecialToken> st1;
  st1["[CLS]"] = {"[CLS]", {101}, {"[CLS]"}};
  auto p1 = std::make_shared<TemplateProcessing>(t1, Template{}, st1);

  // Processor 2: Append 102
  Template t2 = {
      Piece::Sequence(SequenceId::A, 0), Piece::SpecialToken("[SEP]", 0)};
  std::map<std::string, SpecialToken> st2;
  st2["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};
  auto p2 = std::make_shared<TemplateProcessing>(t2, Template{}, st2);

  auto seq_processor =
      std::make_shared<Sequence>(std::vector<PostProcessor::Ptr>{p1, p2});

  std::vector<uint64_t> input = {5, 6};
  // p1 -> {101, 5, 6}
  // p2 -> {101, 5, 6, 102}
  std::vector<uint64_t> expected = {101, 5, 6, 102};

  auto output = seq_processor->process(input);

  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, ConfigParsing) {
  // Mimic parsing a simplified config
  json config_json = {
      {"type", "TemplateProcessing"},
      {"single", {"[CLS]", "$0", "[SEP]"}},
      {"pair", {"[CLS]", "$A:0", "[SEP]", "$B:1", "[SEP]"}},
      {"special_tokens",
       {{"[CLS]", {{"ids", {101}}}}, {"[SEP]", {{"ids", {102}}}}}}};

  auto processor = PostProcessorConfig().parse_json(config_json).create();
  ASSERT_NE(processor, nullptr);

  std::vector<uint64_t> input = {10, 20};
  std::vector<uint64_t> expected = {101, 10, 20, 102};

  auto output = processor->process(input);
  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, EmptyInput) {
  Template single_template = {
      Piece::SpecialToken("[CLS]", 0),
      Piece::Sequence(SequenceId::A, 0),
      Piece::SpecialToken("[SEP]", 0)};
  std::map<std::string, SpecialToken> special_tokens;
  special_tokens["[CLS]"] = {"[CLS]", {101}, {"[CLS]"}};
  special_tokens["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};

  auto processor = std::make_shared<TemplateProcessing>(
      single_template, Template{}, special_tokens);

  std::vector<uint64_t> input = {};
  std::vector<uint64_t> expected = {101, 102};

  auto output = processor->process(input);
  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, MultipleSequencesInTemplate) {
  // Setup: $0 [SEP] $0
  Template single_template = {
      Piece::Sequence(SequenceId::A, 0),
      Piece::SpecialToken("[SEP]", 0),
      Piece::Sequence(SequenceId::A, 0)};
  std::map<std::string, SpecialToken> special_tokens;
  special_tokens["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};

  auto processor = std::make_shared<TemplateProcessing>(
      single_template, Template{}, special_tokens);

  std::vector<uint64_t> input = {1};
  std::vector<uint64_t> expected = {1, 102, 1};

  auto output = processor->process(input);
  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, NestedSequenceProcessing) {
  // p1: [101, $0]
  Template t1 = {
      Piece::SpecialToken("[CLS]", 0), Piece::Sequence(SequenceId::A, 0)};
  std::map<std::string, SpecialToken> st1;
  st1["[CLS]"] = {"[CLS]", {101}, {"[CLS]"}};
  auto p1 = std::make_shared<TemplateProcessing>(t1, Template{}, st1);

  // p2: [$0, 102]
  Template t2 = {
      Piece::Sequence(SequenceId::A, 0), Piece::SpecialToken("[SEP]", 0)};
  std::map<std::string, SpecialToken> st2;
  st2["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};
  auto p2 = std::make_shared<TemplateProcessing>(t2, Template{}, st2);

  auto inner_seq =
      std::make_shared<Sequence>(std::vector<PostProcessor::Ptr>{p1, p2});

  // p3: [200, $0, 201]
  Template t3 = {
      Piece::SpecialToken("200", 0),
      Piece::Sequence(SequenceId::A, 0),
      Piece::SpecialToken("201", 0)};
  std::map<std::string, SpecialToken> st3;
  st3["200"] = {"200", {200}, {"200"}};
  st3["201"] = {"201", {201}, {"201"}};
  auto p3 = std::make_shared<TemplateProcessing>(t3, Template{}, st3);

  auto outer_seq = std::make_shared<Sequence>(
      std::vector<PostProcessor::Ptr>{inner_seq, p3});

  std::vector<uint64_t> input = {5};
  // inner_seq(input) -> {101, 5, 102}
  // p3({101, 5, 102}) -> {200, 101, 5, 102, 201}
  std::vector<uint64_t> expected = {200, 101, 5, 102, 201};

  auto output = outer_seq->process(input);
  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, BertProcessing) {
  // [CLS] = 101, [SEP] = 102
  auto processor = std::make_shared<BertProcessing>(
      std::pair<std::string, uint64_t>{"[SEP]", 102},
      std::pair<std::string, uint64_t>{"[CLS]", 101});

  EXPECT_EQ(processor->added_tokens(false), 2);
  EXPECT_EQ(processor->added_tokens(true), 3);

  // Single: [CLS] $0 [SEP]
  std::vector<uint64_t> input = {12, 14};
  std::vector<uint64_t> expected = {101, 12, 14, 102};
  EXPECT_EQ(processor->process(input), expected);

  // Pair: [CLS] $A [SEP] $B [SEP]
  std::vector<uint64_t> input_b = {15};
  std::vector<uint64_t> expected_pair = {101, 12, 14, 102, 15, 102};
  EXPECT_EQ(processor->process(input, input_b), expected_pair);

  // No special tokens
  EXPECT_EQ(processor->process(input, false), input);
  std::vector<uint64_t> expected_no_special_pair = {12, 14, 15};
  EXPECT_EQ(
      processor->process(input, input_b, false), expected_no_special_pair);
}

TEST(PostProcessorTest, BertProcessingConfig) {
  json config_json = {
      {"type", "BertProcessing"},
      {"sep", {"[SEP]", 102}},
      {"cls", {"[CLS]", 101}}};

  auto processor = PostProcessorConfig().parse_json(config_json).create();
  ASSERT_NE(processor, nullptr);

  std::vector<uint64_t> input = {10};
  std::vector<uint64_t> expected = {101, 10, 102};

  EXPECT_EQ(processor->process(input), expected);
}

TEST(PostProcessorTest, RobertaProcessing) {
  // [CLS] = 0, [SEP] = 2
  auto processor = std::make_shared<RobertaProcessing>(
      std::pair<std::string, uint64_t>{"</s>", 2},
      std::pair<std::string, uint64_t>{"<s>", 0},
      true,
      true);

  EXPECT_EQ(processor->added_tokens(false), 2);
  EXPECT_EQ(processor->added_tokens(true), 4);

  // Single: <s> $0 </s>
  std::vector<uint64_t> input = {12, 14};
  std::vector<uint64_t> expected = {0, 12, 14, 2};
  EXPECT_EQ(processor->process(input), expected);

  // Pair: <s> $A </s> </s> $B </s>
  std::vector<uint64_t> input_b = {15};
  std::vector<uint64_t> expected_pair = {0, 12, 14, 2, 2, 15, 2};
  EXPECT_EQ(processor->process(input, input_b), expected_pair);

  // No special tokens
  EXPECT_EQ(processor->process(input, false), input);
  std::vector<uint64_t> expected_no_special_pair = {12, 14, 15};
  EXPECT_EQ(
      processor->process(input, input_b, false), expected_no_special_pair);
}

TEST(PostProcessorTest, RobertaProcessingConfig) {
  json config_json = {
      {"type", "RobertaProcessing"},
      {"sep", {"</s>", 2}},
      {"cls", {"<s>", 0}},
      {"trim_offsets", true},
      {"add_prefix_space", true}};

  auto processor = PostProcessorConfig().parse_json(config_json).create();
  ASSERT_NE(processor, nullptr);

  std::vector<uint64_t> input = {10};
  std::vector<uint64_t> expected = {0, 10, 2};

  EXPECT_EQ(processor->process(input), expected);
}

class PostProcessorExhaustiveTest : public ::testing::Test {
 protected:
  // BERT Constants
  const uint64_t B_CLS = 101;
  const uint64_t B_SEP = 102;

  // RoBERTa Constants
  const uint64_t R_CLS = 0;
  const uint64_t R_SEP = 2;
};

// -- BERT EXHAUSTIVE TESTS --

TEST_F(PostProcessorExhaustiveTest, BertProcessingExhaustive) {
  auto processor = std::make_shared<BertProcessing>(
      std::pair<std::string, uint64_t>{"[SEP]", B_SEP},
      std::pair<std::string, uint64_t>{"[CLS]", B_CLS});

  // 1. Single sequence: [CLS] A [SEP]
  std::vector<uint64_t> a = {10, 20};
  EXPECT_EQ(processor->process(a), std::vector<uint64_t>({101, 10, 20, 102}));

  // 2. Pair sequence: [CLS] A [SEP] B [SEP]
  std::vector<uint64_t> b = {30};
  EXPECT_EQ(
      processor->process(a, b),
      std::vector<uint64_t>({101, 10, 20, 102, 30, 102}));

  // 3. Empty Sequence A in Pair: [CLS] [SEP] B [SEP]
  EXPECT_EQ(
      processor->process(std::vector<uint64_t>{}, b),
      std::vector<uint64_t>({101, 102, 30, 102}));

  // 4. Empty Sequence B in Pair: [CLS] A [SEP] [SEP]
  EXPECT_EQ(
      processor->process(a, std::vector<uint64_t>{}),
      std::vector<uint64_t>({101, 10, 20, 102, 102}));

  // 5. Total added tokens count
  EXPECT_EQ(processor->added_tokens(false), 2); // CLS + SEP
  EXPECT_EQ(processor->added_tokens(true), 3); // CLS + SEP + SEP

  // 6. Special tokens disabled: should just concatenate for pairs
  EXPECT_EQ(processor->process(a, false), a);
  EXPECT_EQ(
      processor->process(a, b, false), std::vector<uint64_t>({10, 20, 30}));
}

// -- ROBERTA EXHAUSTIVE TESTS --

TEST_F(PostProcessorExhaustiveTest, RobertaProcessingExhaustive) {
  auto processor = std::make_shared<RobertaProcessing>(
      std::pair<std::string, uint64_t>{"</s>", R_SEP},
      std::pair<std::string, uint64_t>{"<s>", R_CLS},
      true, // trim_offsets
      true // add_prefix_space
  );

  // 1. Single sequence: <s> A </s>
  std::vector<uint64_t> a = {10, 20};
  EXPECT_EQ(processor->process(a), std::vector<uint64_t>({0, 10, 20, 2}));

  // 2. Pair sequence (THE DOUBLE SEP): <s> A </s> </s> B </s>
  // This is the most critical difference from BERT
  std::vector<uint64_t> b = {30};
  std::vector<uint64_t> expected_pair = {0, 10, 20, 2, 2, 30, 2};
  EXPECT_EQ(processor->process(a, b), expected_pair);

  // 3. Verify middle indices specifically for the double </s>
  auto output = processor->process(a, b);
  ASSERT_EQ(output.size(), 7);
  EXPECT_EQ(output[3], 2); // First </s>
  EXPECT_EQ(output[4], 2); // Second </s>

  // 4. Empty Inputs in Pair
  // Sequence A empty: <s> </s> </s> B </s>
  EXPECT_EQ(
      processor->process(std::vector<uint64_t>{}, b),
      std::vector<uint64_t>({0, 2, 2, 30, 2}));
  // Sequence B empty: <s> A </s> </s> </s>
  EXPECT_EQ(
      processor->process(a, std::vector<uint64_t>{}),
      std::vector<uint64_t>({0, 10, 20, 2, 2, 2}));

  // 5. Total added tokens count
  EXPECT_EQ(processor->added_tokens(false), 2); // <s> ... </s>
  EXPECT_EQ(processor->added_tokens(true), 4); // <s> ... </s> </s> ... </s>

  // 6. Special tokens disabled
  EXPECT_EQ(
      processor->process(a, b, false), std::vector<uint64_t>({10, 20, 30}));
}

// -- CONFIG & ARCHITECTURE VALIDATION --

TEST_F(PostProcessorExhaustiveTest, ConfigFactoryValidation) {
  // Ensure the factory produces the correct concrete types
  PostProcessorConfig bert_cfg("BertProcessing");
  bert_cfg.set_sep({"[SEP]", 102}).set_cls({"[CLS]", 101});
  auto p_bert = bert_cfg.create();

  PostProcessorConfig rob_cfg("RobertaProcessing");
  rob_cfg.set_sep({"</s>", 2}).set_cls({"<s>", 0});
  auto p_rob = rob_cfg.create();

  ASSERT_NE(p_bert, nullptr);
  ASSERT_NE(p_rob, nullptr);

  // Validate structural logic parity via the base class pointer
  std::vector<uint64_t> in = {5};
  EXPECT_EQ(p_bert->process(in).size(), 3); // [101, 5, 102]
  EXPECT_EQ(p_rob->process(in).size(), 3); // [0, 5, 2]
}
