/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

// Local
#include <pytorch/tokenizers/pre_tokenizer.h>
#include <unicode.h>

// Standard
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Third Party
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace tokenizers {

// PreTokenizerConfig //////////////////////////////////////////////////////////

PreTokenizerConfig::PreTokenizerConfig(std::string type)
    : type(std::move(type)) {}

PreTokenizer::Ptr PreTokenizerConfig::create() const {
  // NOTE: These types must line up with the type strings found in the
  //  tokenizers library
  //  https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/mod.rs#L73
  if (type == "Split") {
    if (!pattern) {
      throw std::runtime_error(
          "Missing pattern for PreTokenizer of type Split");
    }

    // Validate behavior parameter, if missing set to default "Removed"
    std::string behavior_str = behavior.value_or("Removed");
    if (behavior_str != "MergedWithPrevious" && behavior_str != "Isolated" &&
        behavior_str != "Removed") {
      throw std::runtime_error(
          "Unsupported behavior '" + behavior_str +
          "' for Split PreTokenizer. Only 'MergedWithPrevious', 'Removed' and 'Isolated' are supported.");
    }

    // Validate invert parameter
    const bool invert_flag = invert.value_or(false);
    const bool delimiter_flag = is_delimiter.value_or(false);
    if (invert_flag && delimiter_flag) {
      throw std::runtime_error(
          "invert=true is not supported for Split PreTokenizer with a String pattern.");
    }

    return PreTokenizer::Ptr(
        new RegexPreTokenizer(*pattern, delimiter_flag, behavior_str));
  }
  if (type == "Digits") {
    return PreTokenizer::Ptr(
        new DigitsPreTokenizer(individual_digits.value_or(false)));
  }
  if (type == "ByteLevel") {
    // Default use_regex to true for backwards compatibility
    // When use_regex is false, ByteLevel only does byte encoding, no regex split
    return PreTokenizer::Ptr(new ByteLevelPreTokenizer(
        add_prefix_space.value_or(true),
        pattern.value_or(""),
        use_regex.value_or(true)));
  }
  if (type == "Sequence") {
    if (!pretokenizers || pretokenizers->empty()) {
      throw std::runtime_error(
          "Missing pretokenizers for PreTokenizer of type Sequence");
    }
    std::vector<PreTokenizer::Ptr> pretoks;
    for (const auto& cfg : *pretokenizers) {
      pretoks.push_back(cfg.create());
    }
    return PreTokenizer::Ptr(new SequencePreTokenizer(pretoks));
  }
  if (type == "BertPreTokenizer") {
    return PreTokenizer::Ptr(new BertPreTokenizer());
  }
  throw std::runtime_error("Unsupported PreTokenizer type: " + type);
}

PreTokenizerConfig& PreTokenizerConfig::parse_json(const json& json_config) {
  type = json_config.at("type");
  if (type == "Split") {
    try {
      set_pattern(json_config.at("pattern").at("Regex"));
      set_is_delimiter(false);
    } catch (json::out_of_range&) {
      // "Regex" is not there, check "String", which is a delimiter
      std::string delimiter = json_config.at("pattern").at("String");
      // For string patterns, escape regex special characters to treat them as
      // literal strings (same as Rust's regex::escape)
      set_pattern(IRegex::escape(delimiter));
      set_is_delimiter(true);
    }

    // Parse behavior and invert fields
    if (json_config.contains("behavior")) {
      set_behavior(json_config["behavior"]);
    }

    if (json_config.contains("invert")) {
      set_invert(json_config["invert"]);
    }
  } else if (type == "Digits") {
    if (json_config.contains("individual_digits")) {
      set_individual_digits(json_config["individual_digits"]);
    }
  } else if (type == "ByteLevel") {
    if (json_config.contains("add_prefix_space")) {
      set_add_prefix_space(json_config["add_prefix_space"]);
    }
    if (json_config.contains("use_regex")) {
      set_use_regex(json_config["use_regex"]);
    }
    // TODO: trim_offsets
  } else if (type == "Sequence") {
    std::vector<PreTokenizerConfig> cfgs;
    for (const auto& entry : json_config.at("pretokenizers")) {
      cfgs.push_back(PreTokenizerConfig().parse_json(entry));
    }
    set_pretokenizers(std::move(cfgs));
  } else if (type == "BertPreTokenizer") {
    // BertPreTokenizer has no additional configuration parameters
  } else {
    throw std::runtime_error("Unsupported PreTokenizer type: " + type);
  }
  return *this;
}

// RegexPreTokenizer
// ///////////////////////////////////////////////////////////

std::unique_ptr<IRegex> RegexPreTokenizer::create_regex_(
    const std::string& pattern) {
  assert(!pattern.empty());
  auto regex_result = create_regex(pattern);
  if (!regex_result.ok()) {
    throw std::runtime_error(
        "Error: " + std::to_string(static_cast<int>(regex_result.error())));
  }
  return std::move(regex_result.get());
}

std::vector<std::string> RegexPreTokenizer::pre_tokenize(
    const std::string& input) const {
  if (!regex_) {
    return {};
  }

  static int regex_debug_count = 0;
  if (regex_debug_count < 3) {
    std::cout << "[RegexPreTokenizer] Input: \"" << input.substr(0, 50) << "..." << "\"" << std::endl;
    regex_debug_count++;
  }

  std::vector<std::string> results;
  auto matches = regex_->find_all(input);

  if (!is_delimiter_) {
    // Original behavior: return the matches themselves
    for (const auto& match : matches) {
      std::string piece = input.substr(match.start, match.end - match.start);
      results.push_back(piece);
    }
  } else {
    // Delimiter behavior
    if (matches.empty()) {
      // No matches found, return the entire input
      results.push_back(input);
      return results;
    }

    if (behavior_ == "MergedWithPrevious") {
      // MergedWithPrevious: Include delimiter with previous token
      // Example: "the-final--countdown" with delimiter "-"
      // -> ["the-", "final-", "-", "countdown"]
      size_t last_end = 0;

      for (size_t i = 0; i < matches.size(); ++i) {
        const auto& match = matches[i];

        // Add text before the match plus the delimiter
        if (match.start > last_end) {
          std::string token = input.substr(last_end, match.end - last_end);
          results.push_back(token);
        } else {
          // Only delimiter, no preceding text
          std::string delimiter =
              input.substr(match.start, match.end - match.start);
          results.push_back(delimiter);
        }

        last_end = match.end;
      }

      // Add remaining text after the last match (if any)
      if (last_end < input.length()) {
        results.push_back(input.substr(last_end));
      }
    } else if (behavior_ == "Isolated") {
      // Isolated: Keep delimiters as separate tokens
      // Example: "the-final--countdown" with delimiter "-"
      // -> ["the", "-", "final", "-", "-", "countdown"]
      size_t last_end = 0;
      for (const auto& match : matches) {
        // Add text before the match (if any)
        if (match.start > last_end) {
          results.push_back(input.substr(last_end, match.start - last_end));
        }

        // Add the delimiter itself as a separate token
        std::string delimiter =
            input.substr(match.start, match.end - match.start);
        results.push_back(delimiter);

        last_end = match.end;
      }

      // Add remaining text after the last match (if any)
      if (last_end < input.length()) {
        results.push_back(input.substr(last_end));
      }
    } else if (behavior_ == "Removed" || behavior_.empty()) {
      // Default delimiter behavior (split on delimiters, remove delimiters)
      size_t last_end = 0;
      for (const auto& match : matches) {
        // Add text before the match (if any)
        if (match.start > last_end) {
          results.push_back(input.substr(last_end, match.start - last_end));
        }
        last_end = match.end;
      }

      // Add remaining text after the last match (if any)
      if (last_end < input.length()) {
        results.push_back(input.substr(last_end));
      }
    }
  }
  return results;
}

// ByteLevelPreTokenizer
// ///////////////////////////////////////////////////////

//////////////////
// Impl Details //
//////////////////
namespace {

// Standard GPT2 regex
// https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
constexpr char GPT2_EXPR[] =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

} // namespace

//////////////////
// Construction //
//////////////////

ByteLevelPreTokenizer::ByteLevelPreTokenizer(
    bool add_prefix_space,
    const std::string& pattern,
    bool use_regex)
    : pattern_(pattern.empty() ? GPT2_EXPR : pattern),
      add_prefix_space_(add_prefix_space),
      use_regex_(use_regex) {}

std::vector<std::string> ByteLevelPreTokenizer::pre_tokenize(
    const std::string& input) const {
  // If use_regex is false, skip regex splitting and only apply byte encoding
  if (!use_regex_) {
    std::string encoded;
    for (unsigned char byte : input) {
      encoded += unicode_byte_to_utf8(byte);
    }
    return {encoded};
  }

  // Add the prefix space if configured to do so.
  std::string formatted_input = input;
  if (add_prefix_space_ && !formatted_input.empty() &&
      formatted_input[0] != ' ') {
    formatted_input.insert(formatted_input.begin(), ' ');
  }

  // Split using regex
  // Note: unicode_regex_split automatically applies byte encoding
  auto pieces = unicode_regex_split(formatted_input, {pattern_});

  return pieces;
}

// SequencePreTokenizer
// ////////////////////////////////////////////////////////

SequencePreTokenizer::SequencePreTokenizer(
    std::vector<PreTokenizer::Ptr> pre_tokenizers)
    : pre_tokenizers_(std::move(pre_tokenizers)) {}

std::vector<std::string> SequencePreTokenizer::pre_tokenize(
    const std::string& input) const {
  std::vector<std::string> pieces{std::string(input)};
  for (const auto& pre_tokenizer : pre_tokenizers_) {
    std::vector<std::string> new_pieces;
    for (const auto& piece : pieces) {
      for (const auto& subpiece : pre_tokenizer->pre_tokenize(piece)) {
        new_pieces.push_back(subpiece);
      }
    }
    pieces = std::move(new_pieces);
  }
  return pieces;
}

// BertPreTokenizer
// ////////////////////////////////////////////////////////

namespace {
bool is_bert_punc(uint32_t cp) {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
    return true;
  }
  return unicode_cpt_flags(cp).is_punctuation;
}

bool is_cjk(uint32_t c) {
  return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
      (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) ||
      (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B920 && c <= 0x2CEAF) ||
      (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
}
} // namespace

std::vector<std::string> BertPreTokenizer::pre_tokenize(
    const std::string& input) const {
  // 1. Split by whitespace (behavior: Removed)
  std::vector<std::string> words;
  std::vector<uint32_t> cpts = unicode_cpts_from_utf8(input);
  std::vector<uint32_t> current_word;
  for (uint32_t cp : cpts) {
    if (unicode_cpt_flags(cp).is_whitespace) {
      if (!current_word.empty()) {
        std::string s;
        for (uint32_t wcp : current_word) {
          s += unicode_cpt_to_utf8(wcp);
        }
        words.push_back(s);
        current_word.clear();
      }
    } else {
      current_word.push_back(cp);
    }
  }
  if (!current_word.empty()) {
    std::string s;
    for (uint32_t wcp : current_word) {
      s += unicode_cpt_to_utf8(wcp);
    }
    words.push_back(s);
  }

  // 2. Split by punctuation and CJK (behavior: Isolated)
  std::vector<std::string> final_tokens;
  for (const auto& word : words) {
    std::vector<uint32_t> word_cpts = unicode_cpts_from_utf8(word);
    std::vector<uint32_t> current_token;
    for (uint32_t cp : word_cpts) {
      if (is_bert_punc(cp) || is_cjk(cp)) {
        if (!current_token.empty()) {
          std::string s;
          for (uint32_t tcp : current_token) {
            s += unicode_cpt_to_utf8(tcp);
          }
          final_tokens.push_back(s);
          current_token.clear();
        }
        final_tokens.push_back(unicode_cpt_to_utf8(cp));
      } else {
        current_token.push_back(cp);
      }
    }
    if (!current_token.empty()) {
      std::string s;
      for (uint32_t tcp : current_token) {
        s += unicode_cpt_to_utf8(tcp);
      }
      final_tokens.push_back(s);
    }
  }

  return final_tokens;
}

} // namespace tokenizers
