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
#include <pytorch/tokenizers/map_utils.h>
#include <pytorch/tokenizers/log.h>

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

void parse_special_tokens(ModelConfig& config, const nlohmann::json& json_config) {
  if (json_config.contains("added_tokens")) {
    const auto& added_tokens = json_config.at("added_tokens");
    std::vector<std::pair<std::string, uint64_t>> sp_pairs;
    for (const auto& entry : added_tokens) {
      sp_pairs.emplace_back(
          entry.at("content").get<std::string>(),
          entry.at("id").get<uint64_t>());
    }
    config.set_special_token_pairs(std::move(sp_pairs));
  }
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

} // namespace

ModelConfig& ModelConfig::parse_json(const nlohmann::json& json_config) {
  if (json_config.contains("model") && !json_config.at("model").is_null()) {
    const auto& model_json = json_config.at("model");
    if (model_json.contains("type")) {
      type = model_json.at("type").get<std::string>();
    }

    if (type == "BPE") {
      parse_tokens(*this, model_json);
      parse_merges(*this, model_json);

      if (model_json.contains("byte_fallback")) {
        set_byte_fallback(model_json.at("byte_fallback").get<bool>());
      }
      if (model_json.contains("unk_token") &&
          !model_json.at("unk_token").is_null()) {
        set_unk_token(model_json.at("unk_token").get<std::string>());
      }
    }
  }

  parse_special_tokens(*this, json_config);

  return *this;
}

Model::Ptr ModelConfig::create() const {
  if (type == "BPE") {
    const auto& raw_token_pairs =
        token_pairs.value_or(std::vector<std::pair<std::string, uint64_t>>());
    const auto& raw_special_pairs = special_token_pairs.value_or(
        std::vector<std::pair<std::string, uint64_t>>());

    // Filter out special tokens from vocab if they were included
    std::vector<std::pair<std::string, uint64_t>> bpe_token_pairs;
    for (const auto& tp : raw_token_pairs) {
      bool is_special = false;
      for (const auto& sp : raw_special_pairs) {
        if (sp.second == tp.second) {
          is_special = true;
          break;
        }
      }
      if (!is_special) {
        bpe_token_pairs.push_back(tp);
      }
    }

    auto token_map_res = detail::build_token_map(std::move(bpe_token_pairs));
    auto special_token_map_res = detail::build_token_map(raw_special_pairs);

    if (!token_map_res.ok() || !special_token_map_res.ok()) {
      return nullptr;
    }

    auto token_map = std::move(*token_map_res);
    auto special_token_map = std::move(*special_token_map_res);

    // Build special token regex
    std::unique_ptr<IRegex> special_token_regex;
    auto special_token_regex_res =
        detail::build_special_token_regex(special_token_map);
    if (special_token_regex_res.ok()) {
      special_token_regex = std::move(*special_token_regex_res);
    }

    // Discovery Phase for sequence tokens
    std::string final_unk = unk_token.value_or("");
    std::string final_bos = bos_token.value_or("");
    std::string final_eos = eos_token.value_or("");

    auto check_extra = [&](const std::string& path) {
      if (path.empty())
        return;
      std::ifstream f(path);
      if (!f)
        return;
      try {
        nlohmann::json j = nlohmann::json::parse(f);
        if (final_bos.empty() && j.contains("bos_token"))
          final_bos = extract_token_string(j["bos_token"]);
        if (final_eos.empty() && j.contains("eos_token"))
          final_eos = extract_token_string(j["eos_token"]);
        if (final_unk.empty() && j.contains("unk_token")) {
          if (!j["unk_token"].is_null()) {
            final_unk = extract_token_string(j["unk_token"]);
          }
        }
      } catch (...) {
      }
    };

    check_extra(special_tokens_map_path.value_or(""));
    check_extra(model_config_path.value_or(""));

    // Fallback discovery: search by name
    if (final_bos.empty() || final_eos.empty()) {
      std::vector<std::string_view> bos_c, eos_c;
      for (size_t i = 0; i < special_token_map.size(); ++i) {
        const auto& [token, _] = special_token_map.getElement(i);
        if (final_bos.empty() &&
            (token.find("bos") != std::string::npos ||
             token.find("begin") != std::string::npos))
          bos_c.push_back(token);
        if (final_eos.empty() &&
            (token.find("eos") != std::string::npos ||
             token.find("end") != std::string::npos))
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
        if (special_token_map.tryGetInteger(name)) {
          final_unk = name;
          break;
        }
      }
    }

    // Resolve strings to IDs
    std::optional<uint64_t> unk_token_id;
    std::optional<uint64_t> bos_token_id;
    std::optional<uint64_t> eos_token_id;

    if (!final_unk.empty()) {
      auto id = special_token_map.tryGetInteger(final_unk);
      if (id) unk_token_id = *id;
    }
    if (!final_bos.empty()) {
      auto id = special_token_map.tryGetInteger(final_bos);
      if (id) bos_token_id = *id;
    }
    if (!final_eos.empty()) {
      auto id = special_token_map.tryGetInteger(final_eos);
      if (id) eos_token_id = *id;
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
        unk_token_id,
        bos_token_id,
        eos_token_id);
  }
  return nullptr;
}

} // namespace tokenizers