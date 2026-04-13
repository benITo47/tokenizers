/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/truncation.h>
#include <pytorch/tokenizers/log.h>
#include <algorithm>

namespace tokenizers {

Truncation::Truncation(const TruncationParams& params) : params_(params) {}

std::vector<uint64_t> Truncation::truncate(
    std::vector<uint64_t> tokens,
    size_t num_tokens_to_add) const {
  if (params_.max_length == 0) {
    return {};
  }

  size_t effective_max_length = params_.max_length;
  if (effective_max_length > num_tokens_to_add) {
    effective_max_length -= num_tokens_to_add;
  } else {
    // If we can't even fit special tokens, we truncate everything? 
    // Or we leave effective length as 0. 
    // HuggingFace typically errors or truncates to 0.
    effective_max_length = 0;
  }

  if (tokens.size() <= effective_max_length) {
    return tokens;
  }

  size_t to_remove = tokens.size() - effective_max_length;
  
  if (params_.direction == TruncationDirection::Right) {
    // Keep from the beginning, remove from the end
    tokens.resize(effective_max_length);
  } else {
    // Keep from the end, remove from the beginning
    tokens.erase(tokens.begin(), tokens.begin() + to_remove);
  }

  return tokens;
}

std::pair<std::vector<uint64_t>, std::vector<uint64_t>>
Truncation::truncate_pair(
    std::vector<uint64_t> a,
    std::vector<uint64_t> b,
    size_t num_tokens_to_add) const {
  if (params_.max_length == 0) {
    return {{}, {}};
  }

  size_t effective_max_length = params_.max_length;
  if (effective_max_length > num_tokens_to_add) {
    effective_max_length -= num_tokens_to_add;
  } else {
    effective_max_length = 0;
  }

  size_t total_len = a.size() + b.size();
  if (total_len <= effective_max_length) {
    return {a, b};
  }

  size_t to_remove = total_len - effective_max_length;

  if (params_.strategy == TruncationStrategy::LongestFirst) {
    size_t n1 = a.size();
    size_t n2 = b.size();

    while (n1 + n2 > effective_max_length) {
      if (n1 >= n2 && n1 > 0) {
        n1--;
      } else if (n2 > 0) {
        n2--;
      } else {
        break;
      }
    }

    if (params_.direction == TruncationDirection::Right) {
      a.resize(n1);
      b.resize(n2);
    } else {
      a.erase(a.begin(), a.begin() + (a.size() - n1));
      b.erase(b.begin(), b.begin() + (b.size() - n2));
    }
  } else if (params_.strategy == TruncationStrategy::OnlyFirst) {
    size_t target_a_len = (a.size() > to_remove) ? (a.size() - to_remove) : 0;
    if (params_.direction == TruncationDirection::Right) {
      a.resize(target_a_len);
    } else {
      a.erase(a.begin(), a.begin() + (a.size() - target_a_len));
    }
  } else if (params_.strategy == TruncationStrategy::OnlySecond) {
    size_t target_b_len = (b.size() > to_remove) ? (b.size() - to_remove) : 0;
    if (params_.direction == TruncationDirection::Right) {
      b.resize(target_b_len);
    } else {
      b.erase(b.begin(), b.begin() + (b.size() - target_b_len));
    }
  }

  return {a, b};
}

Truncation::Ptr TruncationConfig::create() const {
  return std::make_shared<Truncation>(params);
}

TruncationConfig& TruncationConfig::parse_json(const nlohmann::json& json_config) {
  std::string direction = json_config.value("direction", "Right");
  params.direction = (direction == "Left") ? TruncationDirection::Left : TruncationDirection::Right;

  params.max_length = json_config.value("max_length", 512);
  
  std::string strategy = json_config.value("strategy", "LongestFirst");
  if (strategy == "LongestFirst") {
    params.strategy = TruncationStrategy::LongestFirst;
  } else if (strategy == "OnlyFirst") {
    params.strategy = TruncationStrategy::OnlyFirst;
  } else if (strategy == "OnlySecond") {
    params.strategy = TruncationStrategy::OnlySecond;
  }

  params.stride = json_config.value("stride", 0);

  return *this;
}

} // namespace tokenizers
