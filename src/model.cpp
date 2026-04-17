/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/model.h>

#include <fstream>
#include <iostream>

#include <pytorch/tokenizers/bpe_model.h>
#include <pytorch/tokenizers/log.h>
#include <pytorch/tokenizers/map_utils.h>
#include <pytorch/tokenizers/wordpiece_model.h>

namespace tokenizers {

namespace {
// Helper to extract token string from either string or object format
std::string extract_token_string(const nlohmann::json& token_json) {
  if (token_json.is_string()) {
    return token_json.get<std::string>();
  } else if (token_json.is_object() && token_json.contains("content")) {
    return token_json["content"].get<std::string>();
  }
  return "";
};

Error parse_special_tokens(
    ModelConfig& config,
    const nlohmann::json& json_config) {
  try {
    if (json_config.contains("added_tokens")) {
      const auto& added_tokens = json_config.at("added_tokens");
      std::vector<std::pair<std::string, uint64_t>> sp_pairs;
      for (const auto& entry : added_tokens) {
        std::string content = entry.at("content").get<std::string>();
        sp_pairs.emplace_back(content, entry.at("id").get<uint64_t>());
        if (entry.value("rstrip", false)) {
          config.rstrip_tokens.insert(content);
        }
        if (entry.value("lstrip", false)) {
          config.lstrip_tokens.insert(content);
        }
      }
      config.set_special_token_pairs(std::move(sp_pairs));
    }
  } catch (const std::exception& e) {
    TK_LOG(Info, "Could not parse special tokens: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

void parse_tokens(ModelConfig& config, const nlohmann::json& model_json) {
  if (model_json.contains("vocab")) {
    const auto& vocab = model_json.at("vocab");
    std::vector<std::pair<std::string, uint64_t>> pairs;
    for (const auto& entry : vocab.items()) {
      pairs.emplace_back(entry.key(), entry.value().get<uint64_t>());
    }
    config.set_token_pairs(std::move(pairs));
  }
}

void parse_merges(ModelConfig& config, const nlohmann::json& model_json) {
  if (model_json.contains("merges")) {
    const auto& merges_json = model_json.at("merges");
    std::vector<std::string> m_vec;
    for (const auto& m : merges_json) {
      if (m.is_string()) {
        m_vec.push_back(m.get<std::string>());
      } else if (m.is_array() && m.size() == 2) {
        std::string first = m[0].get<std::string>();
        std::string second = m[1].get<std::string>();
        m_vec.push_back(first + " " + second);
      }
    }
    config.set_merges(std::move(m_vec));
  }
}

// Common setup for sequence tokens (BOS, EOS, UNK)
struct SequenceTokenIds {
  std::optional<uint64_t> unk_token_id;
  std::optional<uint64_t> bos_token_id;
  std::optional<uint64_t> eos_token_id;
};

SequenceTokenIds resolve_sequence_tokens(
    const ModelConfig& config,
    const detail::TokenMap& token_map,
    const detail::TokenMap& special_token_map) {
  SequenceTokenIds ids;
  std::string final_unk = config.unk_token.value_or("");
  std::string final_bos = config.bos_token.value_or("");
  std::string final_eos = config.eos_token.value_or("");

  auto check_extra = [&](const std::string& path) {
    if (path.empty())
      return;
    std::ifstream f(path);
    if (!f)
      return;
    try {
      nlohmann::json j = nlohmann::json::parse(f);
      if (final_bos.empty()) {
        if (j.contains("bos_token"))
          final_bos = extract_token_string(j["bos_token"]);
        else if (j.contains("cls_token"))
          final_bos = extract_token_string(j["cls_token"]);
      }
      if (final_eos.empty()) {
        if (j.contains("eos_token"))
          final_eos = extract_token_string(j["eos_token"]);
        else if (j.contains("sep_token"))
          final_eos = extract_token_string(j["sep_token"]);
      }
      if (final_unk.empty() && j.contains("unk_token")) {
        if (!j["unk_token"].is_null()) {
          final_unk = extract_token_string(j["unk_token"]);
        }
      }
    } catch (...) {
    }
  };

  check_extra(config.special_tokens_map_path.value_or(""));
  check_extra(config.model_config_path.value_or(""));

  // Fallback discovery: search by name
  if (final_bos.empty() || final_eos.empty()) {
    std::vector<std::string_view> bos_c, eos_c;
    // Search special tokens
    for (size_t i = 0; i < special_token_map.size(); ++i) {
      const auto& [token, _] = special_token_map.getElement(i);
      if (final_bos.empty() &&
          (token.find("bos") != std::string::npos ||
           token.find("begin") != std::string::npos ||
           token.find("cls") != std::string::npos))
        bos_c.push_back(token);
      if (final_eos.empty() &&
          (token.find("eos") != std::string::npos ||
           token.find("end") != std::string::npos ||
           token.find("sep") != std::string::npos))
        eos_c.push_back(token);
    }
    if (final_bos.empty() && bos_c.size() == 1) {
      final_bos = std::string(bos_c[0]);
    }
    if (final_eos.empty() && eos_c.size() == 1) {
      final_eos = std::string(eos_c[0]);
    }
  }

  // UNK priority list
  if (final_unk.empty()) {
    for (const auto& name : {"<unk>", "[UNK]", "<|endoftext|>"}) {
      if (special_token_map.tryGetInteger(name) ||
          token_map.tryGetInteger(name)) {
        final_unk = name;
        break;
      }
    }
  }

  // Resolve strings to IDs (check both maps)
  auto resolve = [&](const std::string& s) -> std::optional<uint64_t> {
    if (s.empty())
      return std::nullopt;
    auto id = special_token_map.tryGetInteger(s);
    if (!id) {
      id = token_map.tryGetInteger(s);
    }
    return id;
  };

  ids.unk_token_id = resolve(final_unk);
  ids.bos_token_id = resolve(final_bos);
  ids.eos_token_id = resolve(final_eos);

  return ids;
}

} // namespace

ModelConfig& ModelConfig::parse_json(const nlohmann::json& json_config) {
  if (json_config.contains("model") && !json_config.at("model").is_null()) {
    const auto& model_json = json_config.at("model");
    if (model_json.contains("type")) {
      type = model_json.at("type").get<std::string>();
    }

    // Common vocab parsing
    parse_tokens(*this, model_json);

    // Common config parsing
    if (model_json.contains("unk_token") &&
        !model_json.at("unk_token").is_null()) {
      set_unk_token(model_json.at("unk_token").get<std::string>());
    }

    if (type == "BPE") {
      parse_merges(*this, model_json);
      if (model_json.contains("byte_fallback")) {
        set_byte_fallback(model_json.at("byte_fallback").get<bool>());
      }
    } else if (type == "WordPiece") {
      if (model_json.contains("continuing_subword_prefix")) {
        set_continuing_subword_prefix(
            model_json.at("continuing_subword_prefix").get<std::string>());
      }
      if (model_json.contains("max_input_chars_per_word")) {
        set_max_input_chars_per_word(
            model_json.at("max_input_chars_per_word").get<size_t>());
      }
    }
  }

  parse_special_tokens(*this, json_config);

  return *this;
}

Model::Ptr ModelConfig::create() const {
  const auto& raw_token_pairs =
      token_pairs.value_or(std::vector<std::pair<std::string, uint64_t>>());
  const auto& raw_special_pairs = special_token_pairs.value_or(
      std::vector<std::pair<std::string, uint64_t>>());

  // Filter out special tokens from vocab if they were included
  std::vector<std::pair<std::string, uint64_t>> model_token_pairs;
  for (const auto& tp : raw_token_pairs) {
    bool is_special = false;
    for (const auto& sp : raw_special_pairs) {
      if (sp.second == tp.second) {
        is_special = true;
        break;
      }
    }
    if (!is_special) {
      model_token_pairs.push_back(tp);
    }
  }

  auto token_map_res = detail::build_token_map(std::move(model_token_pairs));
  auto special_token_map_res = detail::build_token_map(raw_special_pairs);

  if (!token_map_res.ok() || !special_token_map_res.ok()) {
    return nullptr;
  }

  auto token_map = std::move(*token_map_res);
  auto special_token_map = std::move(*special_token_map_res);

  // Resolve sequence tokens
  auto ids = resolve_sequence_tokens(*this, token_map, special_token_map);

  if (type == "BPE") {
    // Build special token regex
    std::unique_ptr<IRegex> special_token_regex;
    auto special_token_regex_res =
        detail::build_special_token_regex(special_token_map);
    if (special_token_regex_res.ok()) {
      special_token_regex = std::move(*special_token_regex_res);
    }

    // Build merge ranks from merges
    std::optional<detail::TokenMap> merge_ranks;
    detail::MergeMap merge_map;
    const auto& raw_merges = merges.value_or(std::vector<std::string>());
    for (size_t i = 0; i < raw_merges.size(); ++i) {
      const std::string& m = raw_merges[i];
      if (m.rfind("#version", 0) == 0)
        continue;
      auto space_pos = m.find(' ');
      if (space_pos != std::string::npos) {
        std::string first = m.substr(0, space_pos);
        std::string second = m.substr(space_pos + 1);
        auto first_id = token_map.tryGetInteger(first);
        auto second_id = token_map.tryGetInteger(second);
        if (first_id && second_id) {
          std::string merged = first + second;
          auto merged_id = token_map.tryGetInteger(merged);
          if (merged_id) {
            merge_map.emplace(
                std::make_pair(*first_id, *second_id),
                std::make_pair(static_cast<uint32_t>(i), *merged_id));
          }
        }
      }
    }
    auto merge_ranks_res = detail::build_merge_ranks_map(merge_map, token_map);
    if (merge_ranks_res.ok()) {
      merge_ranks.emplace(std::move(*merge_ranks_res));
    }

    return std::make_shared<BPEModel>(
        std::move(token_map),
        std::move(special_token_map),
        std::move(merge_ranks),
        std::move(special_token_regex),
        byte_fallback.value_or(false),
        ids.unk_token_id,
        ids.bos_token_id,
        ids.eos_token_id,
        rstrip_tokens,
        lstrip_tokens);

  } else if (type == "WordPiece") {
    std::string unk = unk_token.value_or("[UNK]");
    return std::make_shared<WordPieceModel>(
        std::move(token_map),
        std::move(special_token_map),
        unk,
        continuing_subword_prefix.value_or("##"),
        max_input_chars_per_word.value_or(100),
        ids.unk_token_id,
        ids.bos_token_id,
        ids.eos_token_id);
  }

  return nullptr;
}

} // namespace tokenizers
