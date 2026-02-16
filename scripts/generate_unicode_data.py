#!/usr/bin/env python3
"""
Generate proper Unicode NFD decomposition data from UnicodeData.txt

Downloads UnicodeData.txt and generates C++ header file with correct
decomposition mappings that support multi-codepoint sequences.
"""

import urllib.request
import sys
from typing import Dict, List, Tuple

UNICODE_DATA_URL = "https://www.unicode.org/Public/UNIDATA/UnicodeData.txt"

def download_unicode_data() -> List[str]:
    """Download UnicodeData.txt from unicode.org"""
    print(f"Downloading {UNICODE_DATA_URL}...", file=sys.stderr)
    with urllib.request.urlopen(UNICODE_DATA_URL) as response:
        data = response.read().decode('utf-8')
    print(f"Downloaded {len(data)} bytes", file=sys.stderr)
    return data.splitlines()

def parse_decomposition(decomp_field: str) -> Tuple[bool, List[int]]:
    """
    Parse decomposition field from UnicodeData.txt

    Returns: (is_canonical, codepoints)
    - is_canonical: True if canonical decomposition (for NFD/NFC)
    - codepoints: List of codepoint values in the decomposition

    Examples:
    - "0041 0300" -> (True, [0x41, 0x300])  # canonical
    - "<compat> 0041" -> (False, [0x41])     # compatibility (skip for NFD)
    - "" -> (True, [])                       # no decomposition
    """
    if not decomp_field:
        return (True, [])

    # Check if it's a compatibility decomposition (starts with <tag>)
    is_canonical = not decomp_field.startswith('<')

    # Remove compatibility tag if present
    if not is_canonical:
        # Find the closing > and skip the tag
        close_idx = decomp_field.find('>')
        if close_idx != -1:
            decomp_field = decomp_field[close_idx + 1:].strip()

    # Parse hex codepoints
    if not decomp_field:
        return (is_canonical, [])

    codepoints = [int(cp, 16) for cp in decomp_field.split()]
    return (is_canonical, codepoints)

def build_combining_class_table(lines: List[str]) -> Dict[int, int]:
    """
    Build canonical combining class table from UnicodeData.txt

    Returns mapping: codepoint -> combining_class (0-254)
    Field 3 in UnicodeData.txt contains the canonical combining class
    """
    combining_classes = {}

    for line in lines:
        if not line or line.startswith('#'):
            continue

        fields = line.split(';')
        if len(fields) < 4:
            continue

        codepoint = int(fields[0], 16)
        combining_class_str = fields[3]  # Canonical combining class

        if combining_class_str:
            combining_class = int(combining_class_str)
            if combining_class > 0:  # Only store non-zero values
                combining_classes[codepoint] = combining_class

    print(f"Found {len(combining_classes)} codepoints with non-zero combining class", file=sys.stderr)
    return combining_classes

def build_decomposition_table(lines: List[str]) -> Dict[int, List[int]]:
    """
    Build canonical decomposition table from UnicodeData.txt

    Returns mapping: composed_codepoint -> [base, combining_marks...]
    Only includes canonical decompositions (for NFD/NFC)
    """
    decomp_table = {}

    for line in lines:
        if not line or line.startswith('#'):
            continue

        fields = line.split(';')
        if len(fields) < 6:
            continue

        codepoint = int(fields[0], 16)
        decomp_field = fields[5]  # Decomposition mapping

        is_canonical, decomp = parse_decomposition(decomp_field)

        # Only store canonical decompositions
        if is_canonical and decomp:
            decomp_table[codepoint] = decomp

    print(f"Found {len(decomp_table)} canonical decompositions", file=sys.stderr)
    return decomp_table

def recursive_decompose(cpt: int, decomp_table: Dict[int, List[int]]) -> List[int]:
    """
    Recursively decompose a codepoint to its fully decomposed form
    """
    if cpt not in decomp_table:
        return [cpt]

    result = []
    for cp in decomp_table[cpt]:
        result.extend(recursive_decompose(cp, decomp_table))
    return result

def build_full_decomposition_table(decomp_table: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """
    Build fully decomposed table by recursively applying decompositions
    """
    full_decomp = {}

    for cpt in decomp_table:
        full_decomp[cpt] = recursive_decompose(cpt, decomp_table)

    return full_decomp

def build_composition_table(base_decomp_table: Dict[int, List[int]]) -> Dict[Tuple[int, int], int]:
    """
    Build canonical composition table from IMMEDIATE decompositions

    Maps (base, combining_mark) -> composed_character
    Uses the base (non-recursive) decomposition to capture all composition levels.

    For example:
    - ồ → ô + grave (immediate decomposition)
    - This creates entry: (ô, grave) → ồ
    - Even though ô itself decomposes further to o + circumflex
    """
    comp_table = {}

    for composed, decomposed in base_decomp_table.items():
        # Only create composition pairs for 2-codepoint decompositions
        if len(decomposed) == 2:
            base, combining = decomposed[0], decomposed[1]
            # Store the composition mapping
            comp_table[(base, combining)] = composed

    print(f"Built {len(comp_table)} composition pairs from immediate decompositions", file=sys.stderr)
    return comp_table

def generate_cpp_header(decomp_table: Dict[int, List[int]],
                       comp_table: Dict[Tuple[int, int], int],
                       combining_classes: Dict[int, int]) -> str:
    """
    Generate C++ header file with decomposition and composition data
    """
    header = """/*
 * Auto-generated Unicode NFC/NFD normalization data
 * Generated from UnicodeData.txt
 *
 * This file contains proper canonical decomposition and composition mappings
 * that support multi-codepoint sequences, required for correct NFD/NFC normalization.
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <utility>

namespace tokenizers {

// Canonical decomposition table: composed -> [base, combining_marks...]
// Only includes canonical decompositions (not compatibility)
static const std::unordered_map<uint32_t, std::vector<uint32_t>> nfd_decomposition_table = {
"""

    # Decomposition table
    for cpt in sorted(decomp_table.keys()):
        decomp = decomp_table[cpt]
        decomp_str = ', '.join(f'0x{cp:04X}' for cp in decomp)

        # Add comment with character name if printable
        comment = ""
        try:
            char_repr = chr(cpt)
            if char_repr.isprintable() and not char_repr.isspace():
                comment = f"  // '{char_repr}'"
        except ValueError:
            pass

        header += f"    {{0x{cpt:04X}, {{{decomp_str}}}}},{comment}\n"

    header += """};

// Hash function for pair<uint32_t, uint32_t>
struct pair_hash {
    std::size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
        return std::hash<uint64_t>()((uint64_t(p.first) << 32) | p.second);
    }
};

// Canonical composition table: (base, combining_mark) -> composed
// Reverse mapping of decomposition for NFC normalization
static const std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, pair_hash> nfc_composition_table = {
"""

    # Composition table
    for (base, combining), composed in sorted(comp_table.items()):
        # Add comment with characters if printable
        comment = ""
        try:
            base_char = chr(base) if chr(base).isprintable() and not chr(base).isspace() else ""
            comb_char = chr(combining) if chr(combining).isprintable() and not chr(combining).isspace() else ""
            comp_char = chr(composed) if chr(composed).isprintable() and not chr(composed).isspace() else ""
            if base_char or comp_char:
                comment = f"  // '{base_char}' + combining -> '{comp_char}'"
        except ValueError:
            pass

        header += f"    {{{{0x{base:04X}, 0x{combining:04X}}}, 0x{composed:04X}}},{comment}\n"

    header += """};

// Canonical combining class table: codepoint -> combining_class (0-254)
// Only non-zero values are stored; missing entries have combining class 0 (starter)
static const std::unordered_map<uint32_t, uint8_t> combining_class_table = {
"""

    # Combining class table
    for cpt in sorted(combining_classes.keys()):
        cc = combining_classes[cpt]
        comment = ""
        try:
            char_repr = chr(cpt)
            if char_repr.isprintable() and not char_repr.isspace():
                comment = f"  // '{char_repr}' cc={cc}"
            else:
                comment = f"  // cc={cc}"
        except ValueError:
            comment = f"  // cc={cc}"

        header += f"    {{0x{cpt:04X}, {cc}}},{comment}\n"

    header += """};

} // namespace tokenizers
"""

    return header

def main():
    """Main entry point"""
    import os

    # Determine output file path
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        # Default: write to include/pytorch/tokenizers/unicode-nfc-data.h
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_file = os.path.join(project_root, "include/pytorch/tokenizers/unicode-nfc-data.h")

    print(f"Generating Unicode NFC/NFD normalization data...", file=sys.stderr)
    print(f"Output file: {output_file}", file=sys.stderr)

    # Download Unicode data once
    unicode_data = download_unicode_data()

    # Build all tables
    combining_classes = build_combining_class_table(unicode_data)
    base_decomp = build_decomposition_table(unicode_data)
    full_decomp = build_full_decomposition_table(base_decomp)

    # Build composition table from BASE decomposition (not full)
    # This is critical: composition uses immediate pairs, not fully decomposed forms
    comp_table = build_composition_table(base_decomp)

    # Generate C++ header
    header_content = generate_cpp_header(full_decomp, comp_table, combining_classes)

    # Write to file
    with open(output_file, 'w') as f:
        f.write(header_content)

    print(f"\n✓ Generated header with {len(full_decomp)} decompositions and {len(comp_table)} compositions", file=sys.stderr)
    print(f"✓ Written to: {output_file}", file=sys.stderr)

    print("\nExample decompositions:", file=sys.stderr)
    examples = [0x00E9, 0x00C0, 0x1ED3]  # é, À, ồ
    for cpt in examples:
        if cpt in full_decomp:
            decomp = full_decomp[cpt]
            chars = ''.join(chr(c) for c in decomp if c < 0x10000)
            print(f"  U+{cpt:04X} ('{chr(cpt)}') -> {[f'U+{c:04X}' for c in decomp]} ('{chars}')", file=sys.stderr)

    print("\nExample compositions:", file=sys.stderr)
    example_pairs = [
        (0x0065, 0x0301),  # e + acute -> é
        (0x0041, 0x0300),  # A + grave -> À
        (0x00F4, 0x0300),  # ô + grave -> ồ (multi-level!)
    ]
    for base, combining in example_pairs:
        key = (base, combining)
        if key in comp_table:
            composed = comp_table[key]
            print(f"  U+{base:04X} ('{chr(base)}') + U+{combining:04X} -> U+{composed:04X} ('{chr(composed)}')", file=sys.stderr)
        else:
            print(f"  U+{base:04X} ('{chr(base)}') + U+{combining:04X} -> [no composition]", file=sys.stderr)

if __name__ == '__main__':
    main()
