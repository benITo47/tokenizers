/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/padding.h>
#include <pytorch/tokenizers/log.h>

namespace tokenizers {

Padding::Padding(const PaddingParams& params) : params_(params) {}

std::vector<uint64_t> Padding::pad(std::vector<uint64_t> tokens) const {
  size_t pad_length = tokens.size();

  if (params_.strategy == PaddingStrategy::Fixed && params_.fixed_size) {
    pad_length = *params_.fixed_size;
  } else if (params_.strategy == PaddingStrategy::BatchLongest) {
    // Since we process one by one, BatchLongest is effectively current length
    // unless pad_to_multiple_of is used.
    pad_length = tokens.size();
  }

  if (params_.pad_to_multiple_of) {
    size_t multiple = *params_.pad_to_multiple_of;
    if (multiple > 0 && pad_length % multiple > 0) {
      pad_length += multiple - (pad_length % multiple);
    }
  }

  if (tokens.size() >= pad_length) {
    return tokens;
  }

  size_t diff = pad_length - tokens.size();
  if (params_.direction == PaddingDirection::Right) {
    tokens.insert(tokens.end(), diff, params_.pad_id);
  } else {
    tokens.insert(tokens.begin(), diff, params_.pad_id);
  }

  return tokens;
}

std::vector<uint32_t> Padding::generate_mask(
    const std::vector<uint64_t>& tokens,
    size_t padded_size) const {
  std::vector<uint32_t> mask(padded_size, 0);
  size_t num_real = std::min(tokens.size(), padded_size);

  if (params_.direction == PaddingDirection::Right) {
    std::fill(mask.begin(), mask.begin() + num_real, 1);
  } else {
    std::fill(mask.end() - num_real, mask.end(), 1);
  }

  return mask;
}

Padding::Ptr PaddingConfig::create() const {
  return std::make_shared<Padding>(params);
}

PaddingConfig::PaddingConfig(std::string strategy) {
  if (strategy == "BatchLongest") {
    params.strategy = PaddingStrategy::BatchLongest;
  } else if (strategy == "Fixed") {
    params.strategy = PaddingStrategy::Fixed;
  }
}

PaddingConfig& PaddingConfig::parse_json(const nlohmann::json& json_config) {
  if (json_config.contains("strategy")) {
    if (json_config["strategy"].is_object()) {
      if (json_config["strategy"].contains("Fixed")) {
        params.strategy = PaddingStrategy::Fixed;
        params.fixed_size = json_config["strategy"]["Fixed"];
      }
    } else {
        // Default or specific string check if needed, but usually it's "BatchLongest"
        // if it's a string, or implied.
        // Rust code: pub enum PaddingStrategy { BatchLongest, Fixed(usize) }
        // JSON serialization of Rust enum usually is string for unit variants or object for tuple variants.
        std::string strat = json_config.value("strategy", "BatchLongest");
        if (strat == "BatchLongest") {
             params.strategy = PaddingStrategy::BatchLongest;
        }
    }
  } else {
    params.strategy = PaddingStrategy::BatchLongest;
  }

  std::string direction = json_config.value("direction", "Right");
  params.direction = (direction == "Left") ? PaddingDirection::Left : PaddingDirection::Right;

  if (json_config.contains("pad_to_multiple_of") &&
      !json_config["pad_to_multiple_of"].is_null()) {
    params.pad_to_multiple_of = json_config["pad_to_multiple_of"];
  }
  params.pad_id = json_config.value("pad_id", 0);
  params.pad_type_id = json_config.value("pad_type_id", 0);
  params.pad_token = json_config.value("pad_token", "[PAD]");

  // Handle implicit max_length logic if strategy implies it or if present
  // In some configs, max_length sits outside strategy but implies Fixed.
  if (json_config.contains("max_length") && !json_config["max_length"].is_null()) {
    params.fixed_size = json_config["max_length"];
    params.strategy = PaddingStrategy::Fixed;
  }

  return *this;
}

} // namespace tokenizers
