/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/bpe_model.h>
#include <pytorch/tokenizers/model.h>

#include <fstream>

namespace tokenizers {

namespace {
// Helper to create a temporary file with given content
class TempFile {
 public:
  TempFile(const std::string& content) {
    path_ = std::tmpnam(nullptr);
    path_ += ".json";
    std::ofstream f(path_);
    f << content;
  }
  ~TempFile() {
    std::remove(path_.c_str());
  }
  const std::string& path() const {
    return path_;
  }

 private:
  std::string path_;
};
} // namespace

class ModelTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(ModelTest, TestBPEModelConfigParsing) {
  const char* json_str = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "b": 1,
        "c": 2,
        "ab": 3
      },
      "merges": [
        "a b"
      ],
      "byte_fallback": true,
      "unk_token": "[UNK]"
    },
    "added_tokens": [
      {"id": 4, "content": "[UNK]"},
      {"id": 5, "content": "<s>"}
    ]
  })";

  nlohmann::json j = nlohmann::json::parse(json_str);
  ModelConfig config;
  config.parse_json(j);

  EXPECT_EQ(config.type, "BPE");
  EXPECT_TRUE(config.token_pairs.has_value());
  EXPECT_EQ(config.token_pairs->size(), 4);
  EXPECT_TRUE(config.merges.has_value());
  EXPECT_EQ(config.merges->size(), 1);
  EXPECT_EQ(config.merges->at(0), "a b");
  EXPECT_TRUE(config.byte_fallback.has_value());
  EXPECT_TRUE(*config.byte_fallback);
  EXPECT_TRUE(config.unk_token.has_value());
  EXPECT_EQ(*config.unk_token, "[UNK]");
  EXPECT_TRUE(config.special_token_pairs.has_value());
  EXPECT_EQ(config.special_token_pairs->size(), 2);
}

TEST_F(ModelTest, TestBPEModelCreation) {
  const char* json_str = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "b": 1,
        "ab": 2,
        "[UNK]": 3
      },
      "merges": [
        "a b"
      ]
    },
    "added_tokens": [
      {"id": 3, "content": "[UNK]"},
      {"id": 4, "content": "<s>"}
    ]
  })";

  nlohmann::json j = nlohmann::json::parse(json_str);
  ModelConfig config;
  config.parse_json(j);
  config.set_bos_token("<s>");

  auto model = config.create();
  ASSERT_NE(model, nullptr);
  EXPECT_TRUE(model->is_loaded());
  EXPECT_EQ(
      model->vocab_size(), 5); // 3 regular (a, b, ab) + 2 special ([UNK], <s>)
  EXPECT_EQ(model->bos_token_id(), 4);
}

TEST_F(ModelTest, TestBPEGreedyMergePriority) {
  // Test ambiguity and rank order
  const char* json_str = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "aa": 1,
        "aaa": 2
      },
      "merges": [
        "a a",
        "aa a"
      ]
    },
    "added_tokens": []
  })";

  nlohmann::json j = nlohmann::json::parse(json_str);
  ModelConfig config;
  config.parse_json(j);
  auto model = config.create();
  ASSERT_NE(model, nullptr);

  // For "aaa":
  // 1. "a a" is rank 0. Pairs are (a,a) at index 0 and 1.
  // 2. Best pair is first available (index 0). Result: "aa", "a"
  // 3. Next pairs: (aa, a) which is rank 1. Result: "aaa"
  auto result = model->tokenize("aaa");
  ASSERT_TRUE(result.ok());
  std::vector<uint64_t> expected = {2}; // "aaa"
  EXPECT_EQ(result.get(), expected);
}

TEST_F(ModelTest, TestBPEOverlapTies) {
  // Test "aaaa" with merge ("a a")
  const char* json_str = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "aa": 1
      },
      "merges": [
        "a a"
      ]
    },
    "added_tokens": []
  })";

  nlohmann::json j = nlohmann::json::parse(json_str);
  ModelConfig config;
  config.parse_json(j);
  auto model = config.create();
  ASSERT_NE(model, nullptr);

  // "aaaa" -> ["a", "a", "a", "a"]
  // Merge 1: ["aa", "a", "a"] (first pair)
  // Merge 2: ["aa", "aa"] (next available pair)
  auto result = model->tokenize("aaaa");
  ASSERT_TRUE(result.ok());
  std::vector<uint64_t> expected = {1, 1}; // ["aa", "aa"]
  EXPECT_EQ(result.get(), expected);
}

TEST_F(ModelTest, TestBPEUTF8ByteFallback) {
  // Test multi-byte character fallback (💩 is F0 9F 92 A9)
  const char* json_str = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "<0xF0>": 1,
        "<0x9F>": 2,
        "<0x92>": 3,
        "<0xA9>": 4
      },
      "merges": [],
      "byte_fallback": true
    },
    "added_tokens": []
  })";

  nlohmann::json j = nlohmann::json::parse(json_str);
  ModelConfig config;
  config.parse_json(j);
  auto model = config.create();
  ASSERT_NE(model, nullptr);

  auto result = model->tokenize("💩");
  ASSERT_TRUE(result.ok());
  std::vector<uint64_t> expected = {1, 2, 3, 4};
  EXPECT_EQ(result.get(), expected);
}

TEST_F(ModelTest, TestVocabSpecialTokenPriority) {
  // If same string is in vocab and added_tokens, special must win
  const char* json_str = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "<s>": 1
      },
      "merges": []
    },
    "added_tokens": [
      {"id": 500, "content": "<s>"}
    ]
  })";

  nlohmann::json j = nlohmann::json::parse(json_str);
  ModelConfig config;
  config.parse_json(j);
  auto model = config.create();
  ASSERT_NE(model, nullptr);

  // "<s>" should resolve to 500, not 1
  auto result = model->tokenize("<s>");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get()[0], 500);

  EXPECT_TRUE(model->is_special_token(500));
}

TEST_F(ModelTest, TestBPEModelSpecialTokens) {
  const char* json_str = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "b": 1,
        "<s>": 2
      },
      "merges": []
    },
    "added_tokens": [
      {"id": 2, "content": "<s>"}
    ]
  })";

  nlohmann::json j = nlohmann::json::parse(json_str);
  ModelConfig config;
  config.parse_json(j);
  auto model = config.create();
  ASSERT_NE(model, nullptr);

  // tokenize "a<s>b"
  // It should split by special token first
  auto result = model->tokenize("a<s>b");
  ASSERT_TRUE(result.ok());
  std::vector<uint64_t> expected = {0, 2, 1};
  EXPECT_EQ(result.get(), expected);

  EXPECT_TRUE(model->is_special_token(2));
  EXPECT_FALSE(model->is_special_token(0));
}

TEST_F(ModelTest, TestBPEModelUnkToken) {
  const char* json_str = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "[UNK]": 1
      },
      "merges": [],
      "unk_token": "[UNK]"
    },
    "added_tokens": [
      {"id": 1, "content": "[UNK]"}
    ]
  })";

  nlohmann::json j = nlohmann::json::parse(json_str);
  ModelConfig config;
  config.parse_json(j);
  auto model = config.create();
  ASSERT_NE(model, nullptr);

  // 'z' is not in vocab and no byte fallback. Should use [UNK]
  auto result = model->tokenize("z");
  ASSERT_TRUE(result.ok());
  std::vector<uint64_t> expected = {1};
  EXPECT_EQ(result.get(), expected);
}

TEST_F(ModelTest, TestIdPieceMapping) {
  const char* json_str = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "hello": 0,
        "world": 1
      },
      "merges": []
    },
    "added_tokens": [
      {"id": 2, "content": "<s>"}
    ]
  })";

  ModelConfig config;
  config.parse_json(nlohmann::json::parse(json_str));
  auto model = config.create();

  EXPECT_EQ(model->id_to_piece(0).get(), "hello");
  EXPECT_EQ(model->id_to_piece(2).get(), "<s>");
  EXPECT_EQ(model->piece_to_id("world").get(), 1);
  EXPECT_EQ(model->piece_to_id("<s>").get(), 2);

  EXPECT_EQ(model->id_to_piece(99).error(), Error::OutOfRange);
  EXPECT_EQ(model->piece_to_id("missing").error(), Error::OutOfRange);
}

TEST_F(ModelTest, TestModelDiscovery) {
  // Discovery of BOS/EOS from added_tokens by name
  const char* json_str = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1},
      "merges": []
    },
    "added_tokens": [
      {"id": 2, "content": "<|begin_of_text|>"},
      {"id": 3, "content": "<|end_of_text|>"}
    ]
  })";

  ModelConfig config;
  config.parse_json(nlohmann::json::parse(json_str));
  auto model = config.create();

  EXPECT_EQ(model->bos_token_id(), 2);
  EXPECT_EQ(model->eos_token_id(), 3);
}

TEST_F(ModelTest, TestModelExtraConfig) {
  // Discovery from extra config files
  const char* tokenizer_json = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0},
      "merges": []
    },
    "added_tokens": [
      {"id": 1, "content": "<s>"},
      {"id": 2, "content": "</s>"}
    ]
  })";

  const char* config_json = R"({
    "bos_token": "<s>",
    "eos_token": "</s>"
  })";

  TempFile t_json(tokenizer_json);
  TempFile c_json(config_json);

  ModelConfig config;
  config.parse_json(nlohmann::json::parse(tokenizer_json));
  config.set_model_config_path(c_json.path());

  auto model = config.create();

  EXPECT_EQ(model->bos_token_id(), 1);

  EXPECT_EQ(model->eos_token_id(), 2);
}

TEST_F(ModelTest, TestWordPieceModelCreation) {
  const char* json_str = R"({
  
      "model": {
  
        "type": "WordPiece",
  
        "vocab": {
  
          "[UNK]": 0,
  
          "foo": 1,
  
          "##bar": 2
  
        },
  
        "unk_token": "[UNK]",
  
        "continuing_subword_prefix": "##",
  
        "max_input_chars_per_word": 100
  
      },
  
      "added_tokens": []
  
    })";

  nlohmann::json j = nlohmann::json::parse(json_str);

  ModelConfig config;

  config.parse_json(j);

  auto model = config.create();

  ASSERT_NE(model, nullptr);

  EXPECT_EQ(model->vocab_size(), 3);

  // "foobar" -> "foo" + "##bar"

  auto result = model->tokenize("foobar");

  ASSERT_TRUE(result.ok());

  std::vector<uint64_t> expected = {1, 2};

  EXPECT_EQ(result.get(), expected);
}

TEST_F(ModelTest, TestWordPieceMaxChars) {
  const char* json_str = R"({
  
      "model": {
  
        "type": "WordPiece",
  
        "vocab": {"[UNK]": 0},
  
        "unk_token": "[UNK]",
  
        "max_input_chars_per_word": 3
  
      }
  
    })";

  ModelConfig config;

  config.parse_json(nlohmann::json::parse(json_str));

  auto model = config.create();

  // "abcd" is 4 chars > 3 -> [UNK]

  auto result = model->tokenize("abcd");

  ASSERT_TRUE(result.ok());

  EXPECT_EQ(result.get()[0], 0);

  // "abc" is 3 chars <= 3 -> [UNK] because vocab doesn't have it, but not
  // length failure

  // (Logic will try to find "abc" in vocab, fail, and return unk)
}

} // namespace tokenizers

