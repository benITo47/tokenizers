/*
 * Proper Unicode NFC/NFD normalization implementation
 */

#include <pytorch/tokenizers/unicode-nfc-data.h>
#include <pytorch/tokenizers/unicode-nfc.h>
#include <unicode.h>
#include <algorithm>

namespace tokenizers {

namespace {

uint8_t get_combining_class(uint32_t cpt) {
  auto it = combining_class_table.find(cpt);
  if (it != combining_class_table.end()) {
    return it->second;
  }
  return 0;
}

// Apply canonical ordering to a sequence of codepoints
// Uses stable_sort to maintain relative order of marks with same class
void canonical_order(std::vector<uint32_t>& cpts) {
  // Find sequences of combining marks (cc > 0) and sort each sequence
  // Never reorder across starters (cc == 0)
  size_t start = 0;
  while (start < cpts.size()) {
    // Skip starters
    if (get_combining_class(cpts[start]) == 0) {
      ++start;
      continue;
    }

    // Found a combining mark, find the end of this sequence
    size_t end = start + 1;
    while (end < cpts.size() && get_combining_class(cpts[end]) > 0) {
      ++end;
    }

    // Sort this sequence of combining marks by their combining class
    std::stable_sort(
        cpts.begin() + start, cpts.begin() + end, [](uint32_t a, uint32_t b) {
          return get_combining_class(a) < get_combining_class(b);
        });

    start = end;
  }
}

// Recursively decompose a single codepoint
void decompose_recursive(uint32_t cpt, std::vector<uint32_t>& result) {
  auto it = nfd_decomposition_table.find(cpt);

  if (it == nfd_decomposition_table.end()) {
    result.push_back(cpt);
  } else {
    for (uint32_t component : it->second) {
      decompose_recursive(component, result);
    }
  }
}

// Apply canonical composition
std::vector<uint32_t> canonical_compose(const std::vector<uint32_t>& cpts) {
  if (cpts.empty()) {
    return cpts;
  }

  std::vector<uint32_t> result;
  result.reserve(cpts.size());

  size_t i = 0;
  while (i < cpts.size()) {
    uint32_t starter = cpts[i];
    result.push_back(starter);

    if (get_combining_class(starter) == 0) {
      size_t last_starter_pos = result.size() - 1;

      // Look for composable combining marks after this starter
      size_t j = i + 1;
      while (j < cpts.size()) {
        uint32_t combining = cpts[j];
        uint8_t cc = get_combining_class(combining);

        if (cc == 0) {
          break;
        }

        auto key = std::make_pair(result[last_starter_pos], combining);
        auto it = nfc_composition_table.find(key);

        if (it != nfc_composition_table.end()) {
          result[last_starter_pos] = it->second;
          ++j;
        } else {
          result.push_back(combining);
          ++j;
        }
      }

      i = j;
    } else {
      ++i;
    }
  }

  return result;
}

} // anonymous namespace

std::vector<uint32_t> unicode_normalize_nfd(const std::vector<uint32_t>& cpts) {
  std::vector<uint32_t> result;
  result.reserve(cpts.size() * 2); // May expand due to decomposition

  for (uint32_t cpt : cpts) {
    decompose_recursive(cpt, result);
  }

  canonical_order(result);

  return result;
}

std::vector<uint32_t> unicode_normalize_nfc(const std::vector<uint32_t>& cpts) {
  auto nfd = unicode_normalize_nfd(cpts);

  return canonical_compose(nfd);
}

std::string unicode_normalize_nfd_utf8(const std::string& utf8) {
  auto cpts = unicode_cpts_from_utf8(utf8);

  auto normalized = unicode_normalize_nfd(cpts);

  std::string result;
  for (uint32_t cpt : normalized) {
    result += unicode_cpt_to_utf8(cpt);
  }

  return result;
}

std::string unicode_normalize_nfc_utf8(const std::string& utf8) {
  auto cpts = unicode_cpts_from_utf8(utf8);

  auto normalized = unicode_normalize_nfc(cpts);

  std::string result;
  for (uint32_t cpt : normalized) {
    result += unicode_cpt_to_utf8(cpt);
  }

  return result;
}

} // namespace tokenizers
