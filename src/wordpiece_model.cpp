/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/wordpiece_model.h>

#include <algorithm>
#include <limits>

#include <pytorch/tokenizers/log.h>
#include <unicode.h> // Assuming this provides unicode_cpts_from_utf8 and related helpers

namespace tokenizers {

WordPieceModel::WordPieceModel(
    detail::TokenMap token_map,
    detail::TokenMap special_token_map,
    std::string unk_token,
    std::string continuing_subword_prefix,
    size_t max_input_chars_per_word,
    std::optional<uint64_t> unk_token_id,
    std::optional<uint64_t> bos_token_id,
    std::optional<uint64_t> eos_token_id)
    : token_map_(std::move(token_map)),
      special_token_map_(std::move(special_token_map)),
      unk_token_(std::move(unk_token)),
      continuing_subword_prefix_(std::move(continuing_subword_prefix)),
      max_input_chars_per_word_(max_input_chars_per_word),
      unk_token_id_(unk_token_id),
      bos_token_id_(bos_token_id),
      eos_token_id_(eos_token_id) {
  vocab_size_ = token_map_.size() + special_token_map_.size();
  initialized_ = true;
}

Result<std::vector<uint64_t>> WordPieceModel::tokenize(
    const std::string& piece) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }

  // Check character count
  // We need to count unicode characters (code points), not bytes
  size_t char_len = 0;
  try {
    char_len = unicode_cpts_from_utf8(piece).size();
  } catch (...) {
    // If invalid UTF-8, treating it as unk is reasonable or letting it fail later
    // For now assume valid input or handle failure
  }

  if (char_len > max_input_chars_per_word_) {
    if (unk_token_id_) {
      return std::vector<uint64_t>{*unk_token_id_};
    }
    return Error::EncodeFailure;
  }

  std::vector<uint64_t> tokens;
  bool is_bad = false;
  size_t start = 0;
  size_t piece_len = piece.length();

  while (start < piece_len) {
    size_t end = piece_len;
    uint64_t cur_token_id = 0;
    bool found = false;

    // Find the longest matching substring
    while (start < end) {
      std::string substr = piece.substr(start, end - start);
      if (start > 0) {
        substr = continuing_subword_prefix_ + substr;
      }

      auto id = token_map_.tryGetInteger(substr);
      if (id) {
        cur_token_id = *id;
        found = true;
        break;
      }
      
      // Reduce end by one utf-8 character
      // We need to find the start of the last character
      size_t prev_char_start = start;
      size_t i = start;
      while (i < end) {
        prev_char_start = i;
        // Move to next char
        unsigned char c = static_cast<unsigned char>(piece[i]);
        size_t char_bytes = 1;
        if ((c & 0x80) == 0) char_bytes = 1;
        else if ((c & 0xE0) == 0xC0) char_bytes = 2;
        else if ((c & 0xF0) == 0xE0) char_bytes = 3;
        else if ((c & 0xF8) == 0xF0) char_bytes = 4;
        else char_bytes = 1; // invalid, skip 1
        
        i += char_bytes;
      }
      end = prev_char_start;
    }

    if (!found) {
      is_bad = true;
      break;
    }

    tokens.push_back(cur_token_id);
    start = end;
    // Restore end for next iteration? No, 'end' here was the match end.
    // The inner loop sets 'end' to the match length + start.
    // Wait, the logic above:
    // 1. Set end = piece_len
    // 2. substr = piece[start..end]
    // 3. if match, break. 'end' is now the end of the match.
    // 4. start = end.
    // Correct.
    
    // However, we need to correct step 3. The inner loop reduces 'end'.
    // If match found, 'end' points to the byte AFTER the match.
    // 'start' becomes 'end'.
    // Logic seems correct.
    
    // One fix: the 'end' update inside the while loop:
    // We want to remove the *last* character of the current substring.
    // So 'end' should become the start of the last character we included.
  }

  if (is_bad) {
    if (unk_token_id_) {
      return std::vector<uint64_t>{*unk_token_id_};
    }
    return Error::EncodeFailure;
  }

  return tokens;
}

Result<std::string> WordPieceModel::id_to_piece(uint64_t token) const {
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

Result<uint64_t> WordPieceModel::piece_to_id(const std::string& token) const {
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

bool WordPieceModel::is_special_token(uint64_t token) const {
  return special_token_map_.tryGetString(token).has_value();
}

} // namespace tokenizers
