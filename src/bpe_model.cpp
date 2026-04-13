/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/bpe_model.h>

#include <algorithm>
#include <cinttypes>
#include <limits>

#include <pytorch/tokenizers/log.h>

namespace tokenizers {

namespace {
// Simple Word structure to mimic Rust's Word behavior, as used in HFTokenizer
struct HFWord {
  std::vector<uint64_t> tokens;
  std::vector<size_t> byte_lengths;

  void add(uint64_t token_id, size_t byte_len) {
    tokens.push_back(token_id);
    byte_lengths.push_back(byte_len);
  }

  size_t size() const {
    return tokens.size();
  }

  // Apply all possible merges using the merge ranks
  void merge_all(
      const detail::TokenMap& merge_ranks,
      const detail::TokenMap& token_map) {
    while (tokens.size() > 1) {
      std::optional<std::pair<size_t, uint32_t>> best_merge;

      // Find the best merge (lowest rank) among adjacent token pairs
      for (size_t i = 0; i < tokens.size() - 1; ++i) {
        // Create the merged token string to look up its rank
        auto first_token = token_map.tryGetString(tokens[i]);
        auto second_token = token_map.tryGetString(tokens[i + 1]);

        if (first_token && second_token) {
          std::string merged_token =
              std::string(*first_token) + std::string(*second_token);
          auto rank = merge_ranks.tryGetInteger(merged_token);

          if (rank && (!best_merge || *rank < best_merge->second)) {
            best_merge = std::make_pair(i, static_cast<uint32_t>(*rank));
          }
        }
      }

      if (!best_merge) {
        break; // No more merges possible
      }

      // Apply the best merge
      size_t merge_idx = best_merge->first;

      // Get the merged token ID
      auto first_token = token_map.tryGetString(tokens[merge_idx]);
      auto second_token = token_map.tryGetString(tokens[merge_idx + 1]);

      if (first_token && second_token) {
        std::string merged_token =
            std::string(*first_token) + std::string(*second_token);
        auto merged_id = token_map.tryGetInteger(merged_token);

        if (merged_id) {
          // Replace the two tokens with the merged token
          tokens[merge_idx] = *merged_id;
          byte_lengths[merge_idx] += byte_lengths[merge_idx + 1];

          // Remove the second token
          tokens.erase(tokens.begin() + merge_idx + 1);
          byte_lengths.erase(byte_lengths.begin() + merge_idx + 1);
        } else {
          break; // Merged token not found in vocabulary
        }
      } else {
        break; // Original tokens not found in vocabulary
      }
    }
  }
};
} // namespace

BPEModel::BPEModel(
    detail::TokenMap token_map,
    detail::TokenMap special_token_map,
    std::optional<detail::TokenMap> merge_ranks,
    std::unique_ptr<IRegex> special_token_regex,
    bool byte_fallback,
    std::optional<uint64_t> unk_token_id,
    std::optional<uint64_t> bos_token_id,
    std::optional<uint64_t> eos_token_id)
    : token_map_(std::move(token_map)),
      special_token_map_(std::move(special_token_map)),
      merge_ranks_(std::move(merge_ranks)),
      special_token_regex_(std::move(special_token_regex)),
      byte_fallback_(byte_fallback),
      unk_token_id_(unk_token_id),
      bos_token_id_(bos_token_id),
      eos_token_id_(eos_token_id) {
  vocab_size_ = token_map_.size() + special_token_map_.size();
  initialized_ = true;
}

Result<std::vector<uint64_t>> BPEModel::tokenize(
    const std::string& piece) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  auto encode_result = encode_with_special_token(piece);
  if (!encode_result.ok()) {
    return encode_result.error();
  }
  return (*encode_result).first;
}

Result<std::string> BPEModel::id_to_piece(uint64_t token) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  auto result = token_map_.tryGetString(token);
  if (!result) {
    result = special_token_map_.tryGetString(token);
  }
  if (!result) {
    return Error::OutOfRange;
  }
  return std::string(*result);
}

Result<uint64_t> BPEModel::piece_to_id(const std::string& token) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  auto result = token_map_.tryGetInteger(token);
  if (!result) {
    result = special_token_map_.tryGetInteger(token);
  }
  if (!result) {
    return Error::OutOfRange;
  }
  return *result;
}

bool BPEModel::is_special_token(uint64_t token) const {
  return special_token_map_.tryGetString(token).has_value();
}

Result<std::pair<std::vector<uint64_t>, uint64_t>>
BPEModel::encode_with_special_token(const std::string& text) const {
  std::vector<uint64_t> tokens;
  uint64_t last_piece_token_len = 0;
  size_t offset = 0;

  while (offset < text.size()) {
    auto [special, sub_input] =
        split_with_allowed_special_token(text, offset);

    if (!sub_input.empty()) {
      auto tokens_result = byte_pair_encode(sub_input);
      if (!tokens_result.ok()) {
        return tokens_result.error();
      }
      auto piece_tokens = std::move(*tokens_result);
      tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
      last_piece_token_len = piece_tokens.size();
    }
    offset += sub_input.size();

    if (special) {
      const auto result = special_token_map_.tryGetInteger(*special);
      if (!result) {
        TK_LOG(Error, "unknown special token: %s\n", special->c_str());
        return Error::EncodeFailure;
      }

      tokens.push_back(*result);
      last_piece_token_len = 0;
      offset += special->size(); // advance past the matched token
    } else {
      break;
    }
  }

  return std::make_pair(tokens, last_piece_token_len);
}

std::pair<std::optional<std::string>, std::string>
BPEModel::split_with_allowed_special_token(
    const std::string& input,
    size_t offset) const {
  if (!special_token_regex_) {
    return std::make_pair(std::nullopt, input.substr(offset));
  }

  auto matches = special_token_regex_->find_all(input.substr(offset));

  for (const auto& m : matches) {
    std::string matched_text = input.substr(offset + m.start, m.end - m.start);
    if (special_token_map_.tryGetInteger(matched_text).has_value()) {
      return {matched_text, input.substr(offset, m.start)};
    }
  }

  return {std::nullopt, input.substr(offset)};
}

Result<std::vector<uint64_t>> BPEModel::byte_pair_encode(
    const std::string& piece) const {
  if (piece.size() == 1) {
    const auto result = token_map_.tryGetInteger(piece);
    if (result) {
      return std::vector<uint64_t>(1, *result);
    }
    if (byte_fallback_) {
      char hex[7];
      snprintf(
          hex, sizeof(hex), "<0x%02X>", static_cast<unsigned char>(piece[0]));
      const auto byte_result = token_map_.tryGetInteger(std::string(hex));
      if (byte_result) {
        return std::vector<uint64_t>(1, *byte_result);
      }
      if (unk_token_id_) {
        return std::vector<uint64_t>(1, *unk_token_id_);
      }
      return Error::EncodeFailure;
    } else {
      if (unk_token_id_) {
        return std::vector<uint64_t>(1, *unk_token_id_);
      }
      return Error::EncodeFailure;
    }
  }

  const detail::TokenMap& merge_ranks =
      merge_ranks_ ? *merge_ranks_ : token_map_;

  return byte_pair_merge(
      piece,
      merge_ranks,
      [this, &piece](uint64_t start, uint64_t stop) {
        std::string key = piece.substr(start, stop - start);
        const auto result = token_map_.tryGetInteger(key);
        if (result) {
          return *result;
        }
        if (byte_fallback_) {
          return std::numeric_limits<uint64_t>::max();
        }
        if (unk_token_id_) {
          return *unk_token_id_;
        }
        return std::numeric_limits<uint64_t>::max() - 1;
      });
}

std::vector<uint64_t> BPEModel::byte_pair_merge(
    const std::string& piece,
    const detail::TokenMap& ranks,
    std::function<uint64_t(uint64_t, uint64_t)> func) const {
  HFWord word;
  size_t i = 0;
  while (i < piece.size()) {
    size_t char_start = i;
    size_t char_len = 1;
    unsigned char byte = static_cast<unsigned char>(piece[i]);
    if ((byte & 0x80) == 0)
      char_len = 1;
    else if ((byte & 0xE0) == 0xC0)
      char_len = 2;
    else if ((byte & 0xF0) == 0xE0)
      char_len = 3;
    else if ((byte & 0xF8) == 0xF0)
      char_len = 4;
    if (char_start + char_len > piece.size())
      char_len = piece.size() - char_start;

    uint64_t token_id = func(char_start, char_start + char_len);
    if (token_id == std::numeric_limits<uint64_t>::max()) { // byte_fallback
      for (size_t j = 0; j < char_len; ++j) {
        char hex[7];
        snprintf(
            hex,
            sizeof(hex),
            "<0x%02X>",
            static_cast<unsigned char>(piece[char_start + j]));
        const auto byte_result = token_map_.tryGetInteger(std::string(hex));
        if (byte_result) {
          word.add(*byte_result, 1);
        } else if (unk_token_id_) {
          word.add(*unk_token_id_, 1);
        } else {
          return {}; // Unhandled byte fallback
        }
      }
    } else if (token_id == (std::numeric_limits<uint64_t>::max() - 1)) { // error
      return {};
    } else {
      word.add(token_id, char_len);
    }
    i += char_len;
  }

  if (merge_ranks_) {
    word.merge_all(*merge_ranks_, token_map_);
  }
  return word.tokens;
}

} // namespace tokenizers
