/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

// Local
#include <pytorch/tokenizers/normalizer.h>
#include <pytorch/tokenizers/regex.h>
#include <pytorch/tokenizers/unicode-nfc.h>

// Third Party
#include <unicode.h>

// Standard
#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Third Party
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace tokenizers {

// NormalizerConfig ////////////////////////////////////////////////////////////

NormalizerConfig::NormalizerConfig(std::string type) : type(std::move(type)) {}

Normalizer::Ptr NormalizerConfig::create() const {
  // NOTE: These types must line up with the type strings found in the
  //  tokenizers library
  //  https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/normalizers/mod.rs
  if (type == "Replace") {
    if (!pattern) {
      throw std::runtime_error(
          "Missing pattern for Normalizer of type Replace");
    }
    if (!content) {
      throw std::runtime_error(
          "Missing content for Normalizer of type Replace");
    }
    return Normalizer::Ptr(new ReplaceNormalizer(*pattern, *content));
  }
  if (type == "Prepend") {
    if (!prepend) {
      throw std::runtime_error(
          "Missing prepend for Normalizer of type Prepend");
    }
    return Normalizer::Ptr(new PrependNormalizer(*prepend));
  }
  if (type == "Sequence") {
    if (!normalizers || normalizers->empty()) {
      throw std::runtime_error(
          "Missing normalizers for Normalizer of type Sequence");
    }
    std::vector<Normalizer::Ptr> norms;
    for (const auto& cfg : *normalizers) {
      norms.push_back(cfg.create());
    }
    return Normalizer::Ptr(new SequenceNormalizer(norms));
  }
  if (type == "BertNormalizer") {
    return Normalizer::Ptr(new BertNormalizer(
        clean_text.value_or(true),
        handle_chinese_chars.value_or(true),
        lowercase.value_or(true),
        strip_accents));
  }
  if (type == "NFC") {
    return Normalizer::Ptr(new NFCNormalizer());
  }
  if (type == "Lowercase") {
    return Normalizer::Ptr(new LowercaseNormalizer());
  }
  throw std::runtime_error("Unsupported Normalizer type: " + type);
}

NormalizerConfig& NormalizerConfig::parse_json(const json& json_config) {
  type = json_config.at("type");
  if (type == "Replace") {
    try {
      set_pattern(json_config.at("pattern").at("Regex"));
    } catch (json::out_of_range&) {
      // "Regex" is not there, check "String", which is a literal string
      std::string literal = json_config.at("pattern").at("String");
      // For string patterns, escape regex special characters to treat them as
      // literal strings (same as Rust's regex::escape)
      set_pattern(IRegex::escape(literal));
    }
    set_content(json_config.at("content"));
  } else if (type == "Prepend") {
    set_prepend(json_config.at("prepend"));
  } else if (type == "Sequence") {
    std::vector<NormalizerConfig> cfgs;
    for (const auto& entry : json_config.at("normalizers")) {
      cfgs.push_back(NormalizerConfig().parse_json(entry));
    }
    set_normalizers(std::move(cfgs));
  } else if (type == "NFC") {
    // NFC normalizer has no additional configuration parameters
    TK_LOG(
        Info,
        "Using NFC normalizer. Please notice that our implementation may not handle all edge cases.");
  } else if (type == "Lowercase") {
    // Lowercase normalizer has no additional configuration parameters
  } else if (type == "BertNormalizer") {
    if (json_config.contains("clean_text") &&
        !json_config.at("clean_text").is_null()) {
      set_clean_text(json_config.at("clean_text"));
    }
    if (json_config.contains("handle_chinese_chars") &&
        !json_config.at("handle_chinese_chars").is_null()) {
      set_handle_chinese_chars(json_config.at("handle_chinese_chars"));
    }
    if (json_config.contains("lowercase") &&
        !json_config.at("lowercase").is_null()) {
      set_lowercase(json_config.at("lowercase"));
    }
    if (json_config.contains("strip_accents") &&
        !json_config.at("strip_accents").is_null()) {
      set_strip_accents(json_config.at("strip_accents"));
    }
  } else {
    throw std::runtime_error("Unsupported Normalizer type: " + type);
  }
  return *this;
}

// ReplaceNormalizer
// ///////////////////////////////////////////////////////////

std::unique_ptr<IRegex> ReplaceNormalizer::create_regex_(
    const std::string& pattern) {
  assert(!pattern.empty());
  auto regex_result = create_regex(pattern);
  if (!regex_result.ok()) {
    std::string error =
        "Error: " + std::to_string(static_cast<int>(regex_result.error()));
    throw std::runtime_error(error);
  }
  return std::move(regex_result.get());
}

std::string ReplaceNormalizer::normalize(const std::string& input) const {
  if (!regex_)
    return input;

  std::string result = input;
  auto matches = regex_->find_all(result);

  // Process matches in reverse order to avoid offset issues
  for (auto it = matches.rbegin(); it != matches.rend(); ++it) {
    const auto& match = *it;
    result.replace(match.start, match.end - match.start, content_);
  }

  return result;
}

// PrependNormalizer
// ///////////////////////////////////////////////////////////

std::string PrependNormalizer::normalize(const std::string& input) const {
  if (input.empty()) {
    return "";
  }
  return prepend_ + input;
}

// SequenceNormalizer
// //////////////////////////////////////////////////////////

SequenceNormalizer::SequenceNormalizer(std::vector<Normalizer::Ptr> normalizers)
    : normalizers_(std::move(normalizers)) {}

std::string SequenceNormalizer::normalize(const std::string& input) const {
  std::string result = input;
  for (const auto& normalizer : normalizers_) {
    result = normalizer->normalize(result);
  }
  return result;
}

// NFCNormalizer
// ///////////////////////////////////////////////////////////////

std::string NFCNormalizer::normalize(const std::string& input) const {
  // Use our proper NFC normalization implementation
  return unicode_normalize_nfc_utf8(input);
}

// LowercaseNormalizer
// ///////////////////////////////////////////////////////////////

std::string LowercaseNormalizer::normalize(const std::string& input) const {
  // Convert UTF-8 string to codepoints
  auto codepoints = unicode_cpts_from_utf8(input);

  // Lowercase each codepoint
  for (auto& cp : codepoints) {
    cp = unicode_tolower(cp);
  }

  // Convert back to UTF-8 string
  std::string result;
  for (uint32_t cpt : codepoints) {
    result += unicode_cpt_to_utf8(cpt);
  }

  return result;
}

// BertNormalizer
// ///////////////////////////////////////////////////////////////

namespace {

bool is_bert_whitespace(uint32_t c) {
  switch (c) {
    case '\t':
    case '\n':
    case '\r':
      return true;
  }
  return unicode_cpt_flags(c).is_whitespace;
}

bool is_bert_control(uint32_t c) {
  switch (c) {
    case '\t':
    case '\n':
    case '\r':
      return false;
  }
  return unicode_cpt_flags(c).is_control;
}

bool is_chinese_char(uint32_t c) {
  return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
      (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) ||
      (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B920 && c <= 0x2CEAF) ||
      (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
}

} // namespace

std::string BertNormalizer::normalize(const std::string& input) const {
  std::vector<uint32_t> cpts = unicode_cpts_from_utf8(input);

  if (clean_text_) {
    std::vector<uint32_t> cleaned;
    cleaned.reserve(cpts.size());
    for (uint32_t cp : cpts) {
      if (cp == 0 || cp == 0xfffd || is_bert_control(cp)) {
        continue;
      }
      if (is_bert_whitespace(cp)) {
        cleaned.push_back(' ');
      } else {
        cleaned.push_back(cp);
      }
    }
    cpts = std::move(cleaned);
  }

  if (handle_chinese_chars_) {
    std::vector<uint32_t> handled;
    handled.reserve(cpts.size() * 3);
    for (uint32_t cp : cpts) {
      if (is_chinese_char(cp)) {
        handled.push_back(' ');
        handled.push_back(cp);
        handled.push_back(' ');
      } else {
        handled.push_back(cp);
      }
    }
    cpts = std::move(handled);
  }

  const bool do_strip =
      strip_accents_.has_value() ? *strip_accents_ : lowercase_;
  if (do_strip) {
    cpts = unicode_cpts_normalize_nfd(cpts);
    std::vector<uint32_t> stripped;
    stripped.reserve(cpts.size());
    for (uint32_t cp : cpts) {
      if (!unicode_cpt_flags(cp).is_accent_mark) {
        stripped.push_back(cp);
      }
    }
    cpts = std::move(stripped);
  }

  if (lowercase_) {
    for (auto& cp : cpts) {
      cp = unicode_tolower(cp);
    }
  }

  std::string result;
  for (uint32_t cp : cpts) {
    result += unicode_cpt_to_utf8(cp);
  }

  return result;
}

} // namespace tokenizers