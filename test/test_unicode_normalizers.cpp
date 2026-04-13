/*
 * Test proper Unicode NFC/NFD normalization implementation
 */

#include <gtest/gtest.h>
#include <pytorch/tokenizers/unicode-nfc.h>
#include <unicode.h>
#include <string>
#include <vector>

namespace tokenizers {

// Helper to convert UTF-8 string to codepoints for inspection
std::vector<uint32_t> utf8_to_cpts(const std::string& utf8) {
    return unicode_cpts_from_utf8(utf8);
}

// Test NFD normalization
TEST(UnicodeNormalizerTest, NFD_BasicAccents) {
    // é (U+00E9) should decompose to e (U+0065) + combining acute (U+0301)
    std::string input = "café";
    std::string result = unicode_normalize_nfd_utf8(input);

    auto cpts = utf8_to_cpts(result);
    // Should be: c(63), a(61), f(66), e(65), combining_acute(301)
    ASSERT_EQ(cpts.size(), 5);
    EXPECT_EQ(cpts[0], 0x0063); // c
    EXPECT_EQ(cpts[1], 0x0061); // a
    EXPECT_EQ(cpts[2], 0x0066); // f
    EXPECT_EQ(cpts[3], 0x0065); // e
    EXPECT_EQ(cpts[4], 0x0301); // combining acute
}

TEST(UnicodeNormalizerTest, NFD_CapitalAccents) {
    // À (U+00C0) should decompose to A (U+0041) + combining grave (U+0300)
    std::string input = "À";
    std::string result = unicode_normalize_nfd_utf8(input);

    auto cpts = utf8_to_cpts(result);
    ASSERT_EQ(cpts.size(), 2);
    EXPECT_EQ(cpts[0], 0x0041); // A
    EXPECT_EQ(cpts[1], 0x0300); // combining grave
}

TEST(UnicodeNormalizerTest, NFD_MultipleCombiningMarks) {
    // ồ (U+1ED3) should decompose to o + circumflex + grave (3 codepoints)
    std::string input = "ồ";
    std::string result = unicode_normalize_nfd_utf8(input);

    auto cpts = utf8_to_cpts(result);
    ASSERT_EQ(cpts.size(), 3);
    EXPECT_EQ(cpts[0], 0x006F); // o
    EXPECT_EQ(cpts[1], 0x0302); // combining circumflex
    EXPECT_EQ(cpts[2], 0x0300); // combining grave
}

TEST(UnicodeNormalizerTest, NFD_PolishCharacters) {
    // ą (U+0105) should decompose to a + combining ogonek
    std::string input = "ą";
    std::string result = unicode_normalize_nfd_utf8(input);

    auto cpts = utf8_to_cpts(result);
    ASSERT_EQ(cpts.size(), 2);
    EXPECT_EQ(cpts[0], 0x0061); // a
    EXPECT_EQ(cpts[1], 0x0328); // combining ogonek
}

TEST(UnicodeNormalizerTest, NFD_MixedText) {
    // Mixed ASCII and accented characters
    std::string input = "naïve";
    std::string result = unicode_normalize_nfd_utf8(input);

    auto cpts = utf8_to_cpts(result);
    ASSERT_EQ(cpts.size(), 6); // n, a, i, combining_diaeresis, v, e
    EXPECT_EQ(cpts[0], 0x006E); // n
    EXPECT_EQ(cpts[1], 0x0061); // a
    EXPECT_EQ(cpts[2], 0x0069); // i
    EXPECT_EQ(cpts[3], 0x0308); // combining diaeresis
    EXPECT_EQ(cpts[4], 0x0076); // v
    EXPECT_EQ(cpts[5], 0x0065); // e
}

TEST(UnicodeNormalizerTest, NFD_ASCIIUnchanged) {
    // Plain ASCII should pass through unchanged
    std::string input = "Hello World!";
    std::string result = unicode_normalize_nfd_utf8(input);
    EXPECT_EQ(result, input);
}

TEST(UnicodeNormalizerTest, NFD_EmptyString) {
    std::string input = "";
    std::string result = unicode_normalize_nfd_utf8(input);
    EXPECT_EQ(result, "");
}

// Test NFC normalization
TEST(UnicodeNormalizerTest, NFC_RecomposesAccents) {
    // Start with decomposed: e + combining_acute
    std::vector<uint32_t> decomposed = {0x0065, 0x0301};
    auto composed = unicode_normalize_nfc(decomposed);

    ASSERT_EQ(composed.size(), 1);
    EXPECT_EQ(composed[0], 0x00E9); // é
}

TEST(UnicodeNormalizerTest, NFC_CapitalAccents) {
    // A + combining_grave should compose to À
    std::vector<uint32_t> decomposed = {0x0041, 0x0300};
    auto composed = unicode_normalize_nfc(decomposed);

    ASSERT_EQ(composed.size(), 1);
    EXPECT_EQ(composed[0], 0x00C0); // À
}

TEST(UnicodeNormalizerTest, NFC_FromUTF8) {
    // Decomposed café -> should recompose to café
    // Input: c, a, f, e, combining_acute
    std::string decomposed_input;
    decomposed_input += unicode_cpt_to_utf8(0x0063); // c
    decomposed_input += unicode_cpt_to_utf8(0x0061); // a
    decomposed_input += unicode_cpt_to_utf8(0x0066); // f
    decomposed_input += unicode_cpt_to_utf8(0x0065); // e
    decomposed_input += unicode_cpt_to_utf8(0x0301); // combining acute

    std::string result = unicode_normalize_nfc_utf8(decomposed_input);
    EXPECT_EQ(result, "café");
}

TEST(UnicodeNormalizerTest, NFC_RoundTrip) {
    // NFC(NFD(text)) should equal NFC(text)
    std::string original = "café naïve ồ ą";

    std::string nfd = unicode_normalize_nfd_utf8(original);
    std::string nfc_from_nfd = unicode_normalize_nfc_utf8(nfd);
    std::string nfc_direct = unicode_normalize_nfc_utf8(original);

    EXPECT_EQ(nfc_from_nfd, nfc_direct);
}

TEST(UnicodeNormalizerTest, NFC_AlreadyComposed) {
    // Text that's already in NFC should be unchanged
    std::string input = "café";
    std::string result = unicode_normalize_nfc_utf8(input);
    EXPECT_EQ(result, input);
}

TEST(UnicodeNormalizerTest, NFC_PolishText) {
    // Test with Polish diacritics
    // Input decomposed:ż (z + dot_above)
    std::string decomposed;
    decomposed += unicode_cpt_to_utf8(0x007A); // z
    decomposed += unicode_cpt_to_utf8(0x0307); // combining dot above

    std::string result = unicode_normalize_nfc_utf8(decomposed);
    EXPECT_EQ(result, "ż"); // Should compose to U+017C
}

// Test canonical ordering
TEST(UnicodeNormalizerTest, NFD_CanonicalOrdering) {
    // Multiple combining marks should be ordered by combining class
    // This is implicitly tested by the decomposition tests,
    // but let's be explicit

    // ồ has circumflex (cc=230) and grave (cc=230)
    // The order matters for proper normalization
    std::string input = "ồ";
    std::string nfd = unicode_normalize_nfd_utf8(input);
    std::string nfc = unicode_normalize_nfc_utf8(nfd);

    // Should round-trip correctly
    EXPECT_EQ(nfc, input);
}

// Test edge cases
TEST(UnicodeNormalizerTest, EdgeCase_OnlyCombiningMarks) {
    // Just a combining mark without a base
    std::vector<uint32_t> marks = {0x0301}; // combining acute
    auto result = unicode_normalize_nfc(marks);

    // Should remain as-is (can't compose without a base)
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 0x0301);
}

TEST(UnicodeNormalizerTest, EdgeCase_NoComposableSequence) {
    // Base + combining mark that don't have a precomposed form
    // (Most combinations should work, but some obscure ones might not)
    std::vector<uint32_t> input = {0x0061, 0x0320}; // a + combining minus below
    auto result = unicode_normalize_nfc(input);

    // Might compose or might not, depending on Unicode data
    // At minimum, shouldn't crash
    EXPECT_FALSE(result.empty());
}

// Regression tests for the bugs we found
TEST(UnicodeNormalizerTest, RegressionTest_HammerTokenizer_Cafe) {
    // This was failing: café should preserve the accent
    std::string input = "café";
    std::string nfc = unicode_normalize_nfc_utf8(input);
    EXPECT_EQ(nfc, "café");

    // Check the codepoints
    auto cpts = utf8_to_cpts(nfc);
    ASSERT_EQ(cpts.size(), 4);
    EXPECT_EQ(cpts[3], 0x00E9); // é (composed)
}

TEST(UnicodeNormalizerTest, RegressionTest_HammerTokenizer_Naive) {
    // This was failing: naïve should preserve the diaeresis
    std::string input = "naïve";
    std::string nfc = unicode_normalize_nfc_utf8(input);
    EXPECT_EQ(nfc, "naïve");

    // Check the codepoints
    auto cpts = utf8_to_cpts(nfc);
    ASSERT_EQ(cpts.size(), 5);
    EXPECT_EQ(cpts[2], 0x00EF); // ï (composed)
}

TEST(UnicodeNormalizerTest, RegressionTest_PolishDiacritics) {
    // This was failing: Polish characters should preserve diacritics
    std::string input = "cześć świecie";
    std::string nfc = unicode_normalize_nfc_utf8(input);

    // Should contain ś (s with acute) and ć (c with acute)
    EXPECT_NE(nfc.find("ś"), std::string::npos);
    EXPECT_NE(nfc.find("ć"), std::string::npos);

    // Should NOT be stripped to plain ASCII
    EXPECT_NE(nfc, "czesc swiecie");
}

} // namespace tokenizers
