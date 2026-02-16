/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/hf_tokenizer.h>

// Standard
#include <algorithm>
#include <cinttypes>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Third Party
#include <nlohmann/json.hpp>

// Local
#include <pytorch/tokenizers/log.h>
#include <pytorch/tokenizers/model.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace tokenizers {

// -------------------------public method start-------------------------------

Error HFTokenizer::load(const std::string& path) {
  std::string model_json_path = path;
  std::string model_config_json_path = "";
  std::string special_tokens_map_json_path = "";

  if (fs::is_directory(path)) {
    const fs::path root(path);
    model_json_path = (root / "tokenizer.json").string();
    if (!fs::exists(model_json_path)) {
      TK_LOG(Info, "no tokenizer.json found in %s", path.c_str());
      return Error::LoadFailure;
    }
    const auto config_path = root / "tokenizer_config.json";
    if (fs::exists(config_path)) {
      model_config_json_path = config_path.string();
    }
    const auto map_path = root / "special_tokens_map.json";
    if (fs::exists(map_path)) {
      special_tokens_map_json_path = map_path.string();
    }
  }

  std::ifstream file(model_json_path);
  if (!file) {
    TK_LOG(Info, "failed to open encoder file: %s", path.c_str());
    return Error::LoadFailure;
  }
  std::string contents(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  json parsed_json;
  try {
    parsed_json = json::parse(contents);
  } catch (const std::exception& e) {
    TK_LOG(Error, "Error parsing json file: %s", e.what());
    return Error::LoadFailure;
  }

  // Setup components
  TK_CHECK_OK_OR_RETURN_ERROR(setup_normalizer(parsed_json));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_pretokenizer(parsed_json));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_postprocessor(parsed_json));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_decoder(parsed_json));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_truncation(parsed_json));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_padding(parsed_json));

  // Setup Model
  TK_CHECK_OK_OR_RETURN_ERROR(setup_model(
      parsed_json, model_config_json_path, special_tokens_map_json_path));

  initialized_ = true;
  return Error::Ok;
}

Result<std::vector<uint64_t>>
HFTokenizer::encode(const std::string& input, int8_t bos, int8_t eos) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }

  std::vector<uint64_t> tokens;
  std::string current_input = input;
  size_t offset = 0;

  while (offset < input.size()) {
    auto [special, sub_input] =
        _model->split_with_allowed_special_token(input, offset);

    // 1. Process regular text segment
    if (!sub_input.empty()) {
      // a. Normalization
      std::string normalized_segment = sub_input;
      if (_normalizer) {
        normalized_segment = _normalizer->normalize(sub_input);
      }

      // b. Pre-tokenization
      std::vector<std::string> pieces;
      if (_pretokenizer) {
        pieces = _pretokenizer->pre_tokenize(normalized_segment);
      } else {
        pieces.push_back(normalized_segment);
      }

      // c. Model Tokenize
      for (const auto& piece : pieces) {
        auto piece_tokens_result = _model->tokenize(piece);
        if (!piece_tokens_result.ok()) {
          return piece_tokens_result.error();
        }
        auto piece_tokens = std::move(*piece_tokens_result);
        tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
      }
    }
    offset += sub_input.size();

    // 2. Process special token
    if (special) {
      auto id_res = _model->piece_to_id(*special);
      if (!id_res.ok()) {
        return id_res.error();
      }
      tokens.push_back(*id_res);
      offset += special->size();
    } else {
      break;
    }
  }

  bool add_special = (bos > 0 || eos > 0);

  // 4. Truncation
  if (_truncation) {
    size_t added = 0;
    if (add_special) {
      if (_postprocessor) {
        added = _postprocessor->added_tokens(false); // is_pair=false
      } else {
        added += (bos > 0 ? 1 : 0);
        added += (eos > 0 ? 1 : 0);
      }
    }
    tokens = _truncation->truncate(std::move(tokens), added);
  }

  // 5. Post-Processing
  if (_postprocessor) {
    tokens = _postprocessor->process(tokens, add_special);
  } else {
    for (int i = 0; i < bos; ++i) {
      tokens.insert(tokens.begin(), bos_tok_);
    }
    for (int i = 0; i < eos; ++i) {
      tokens.push_back(eos_tok_);
    }
  }

  // 6. Padding
  if (_padding) {
    tokens = _padding->pad(std::move(tokens));
  }

  return tokens;
}

Result<std::string> HFTokenizer::id_to_piece(uint64_t token) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  return _model->id_to_piece(token);
}

Result<uint64_t> HFTokenizer::piece_to_id(const std::string& text) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  return _model->piece_to_id(text);
}

Result<std::string> HFTokenizer::decode(
    uint64_t prev_token,
    uint64_t token,
    bool skip_special_tokens) const {
  (void)prev_token;
  if (!initialized_) {
    return Error::Uninitialized;
  }

  if (skip_special_tokens && _model->is_special_token(token)) {
    return std::string("");
  }

  auto res = _model->id_to_piece(token);
  if (!res.ok()) {
    return res.error();
  }

  std::string piece = *res;
  std::string ret;
  if (_decoder) {
    auto decoded = _decoder->decode({piece});
    for (const auto& p : decoded) {
      ret += p;
    }
  } else {
    ret = piece;
  }
  return ret;
}

Result<std::string> HFTokenizer::decode(
    const std::vector<uint64_t>& tokens,
    bool skip_special_tokens) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  std::vector<std::string> pieces;
  for (uint64_t token : tokens) {
    if (skip_special_tokens && _model->is_special_token(token)) {
      continue;
    }
    auto res = _model->id_to_piece(token);
    if (!res.ok()) {
      TK_LOG(Error, "unknown token: %" PRIu64 "\n", token);
      return Error::DecodeFailure;
    }
    pieces.push_back(*res);
  }

  if (_decoder) {
    pieces = _decoder->decode(pieces);
  }

  std::string result_str;
  for (const auto& p : pieces) {
    result_str += p;
  }

  return result_str;
}

// -------------------------private method start--------------------------------

Error HFTokenizer::setup_normalizer(const json& parsed_json) {
  try {
    if (parsed_json.contains("normalizer") &&
        !parsed_json.at("normalizer").is_null()) {
      const auto& normalizer_json = parsed_json.at("normalizer");
      _normalizer = NormalizerConfig().parse_json(normalizer_json).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup normalizer: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_pretokenizer(const json& parsed_json) {
  try {
    if (parsed_json.contains("pre_tokenizer") &&
        !parsed_json.at("pre_tokenizer").is_null()) {
      const auto& pretokenizer_json = parsed_json.at("pre_tokenizer");
      _pretokenizer =
          PreTokenizerConfig().parse_json(pretokenizer_json).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup pretokenizer: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_postprocessor(const json& parsed_json) {
  try {
    if (parsed_json.contains("post_processor") &&
        !parsed_json.at("post_processor").is_null()) {
      const auto& post_processor_json = parsed_json.at("post_processor");
      _postprocessor =
          PostProcessorConfig().parse_json(post_processor_json).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup post_processor: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_decoder(const json& parsed_json) {
  try {
    if (parsed_json.contains("decoder") &&
        !parsed_json.at("decoder").is_null()) {
      _decoder =
          TokenDecoderConfig().parse_json(parsed_json.at("decoder")).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup decoder: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_truncation(const json& parsed_json) {
  try {
    if (parsed_json.contains("truncation") &&
        !parsed_json.at("truncation").is_null()) {
      const auto& t_json = parsed_json.at("truncation");
      _truncation = TruncationConfig().parse_json(t_json).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup truncation: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_padding(const json& parsed_json) {
  try {
    if (parsed_json.contains("padding") &&
        !parsed_json.at("padding").is_null()) {
      const auto& padding_json = parsed_json.at("padding");
      _padding = PaddingConfig().parse_json(padding_json).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup padding: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_model(
    const nlohmann::json& parsed_json,
    const std::string& model_config_path,
    const std::string& special_tokens_map_path) {
  ModelConfig model_config;
  model_config.set_model_config_path(model_config_path);
  model_config.set_special_tokens_map_path(special_tokens_map_path);
  model_config.parse_json(parsed_json);

  _model = model_config.create();
  if (!_model) {
    TK_LOG(Error, "Failed to create model");
    return Error::LoadFailure;
  }

  vocab_size_ = _model->vocab_size();
  bos_tok_ = _model->bos_token_id();
  eos_tok_ = _model->eos_token_id();

  return Error::Ok;
}

} // namespace tokenizers
