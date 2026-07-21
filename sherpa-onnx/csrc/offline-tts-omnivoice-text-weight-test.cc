// sherpa-onnx/csrc/offline-tts-omnivoice-text-weight-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-omnivoice-text-weight.h"

#include <string>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(OmnivoiceTextWeight, EmptyString) {
  EXPECT_FLOAT_EQ(0.0f, OmnivoiceTextWeight(""));
}

TEST(OmnivoiceTextWeight, LatinAscii) {
  EXPECT_FLOAT_EQ(5.0f, OmnivoiceTextWeight("hello"));
  EXPECT_FLOAT_EQ(5.0f, OmnivoiceTextWeight("HELLO"));
}

TEST(OmnivoiceTextWeight, Digits) {
  EXPECT_FLOAT_EQ(3.5f, OmnivoiceTextWeight("0"));
  EXPECT_FLOAT_EQ(35.0f, OmnivoiceTextWeight("0123456789"));
}

TEST(OmnivoiceTextWeight, Whitespace) {
  EXPECT_FLOAT_EQ(0.2f, OmnivoiceTextWeight(" "));
  EXPECT_FLOAT_EQ(0.2f, OmnivoiceTextWeight("\t"));
  EXPECT_FLOAT_EQ(0.2f, OmnivoiceTextWeight("\n"));
  EXPECT_FLOAT_EQ(0.2f, OmnivoiceTextWeight("\r"));
  // 5 latin + 1 space + 5 latin
  EXPECT_FLOAT_EQ(10.2f, OmnivoiceTextWeight("hello world"));
}

TEST(OmnivoiceTextWeight, AsciiPunct) {
  EXPECT_FLOAT_EQ(0.5f, OmnivoiceTextWeight("."));
  EXPECT_FLOAT_EQ(0.5f, OmnivoiceTextWeight(","));
  EXPECT_FLOAT_EQ(0.5f, OmnivoiceTextWeight("!"));
  // 5 latin + 1 punct
  EXPECT_FLOAT_EQ(5.5f, OmnivoiceTextWeight("hello."));
}

TEST(OmnivoiceTextWeight, Cjk) {
  // "你好" — 2 CJK code points
  EXPECT_FLOAT_EQ(6.0f, OmnivoiceTextWeight("\xE4\xBD\xA0\xE5\xA5\xBD"));
  // "小米" — also 2 CJK
  EXPECT_FLOAT_EQ(6.0f, OmnivoiceTextWeight("\xE5\xB0\x8F\xE7\xB1\xB3"));
}

TEST(OmnivoiceTextWeight, Kana) {
  // "こんにちは" — 5 hiragana
  EXPECT_FLOAT_EQ(5 * 2.2f,
                  OmnivoiceTextWeight("\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB"
                                      "\xE3\x81\xA1\xE3\x81\xAF"));
}

TEST(OmnivoiceTextWeight, Hangul) {
  // "안녕" — 2 hangul syllables
  EXPECT_FLOAT_EQ(2 * 2.5f, OmnivoiceTextWeight("\xEC\x95\x88\xEB\x85\x95"));
}

TEST(OmnivoiceTextWeight, Arabic) {
  // "سلام" — 4 Arabic letters
  EXPECT_FLOAT_EQ(4 * 1.5f,
                  OmnivoiceTextWeight("\xD8\xB3\xD9\x84\xD8\xA7\xD9\x85"));
}

TEST(OmnivoiceTextWeight, ArabicTatweelIsZero) {
  // Tatweel U+0640 is a stretching character with no phonetic content.
  // "س" (1.5) + tatweel (0.0) + "لام" (3*1.5) = 6.0
  EXPECT_FLOAT_EQ(6.0f,
                  OmnivoiceTextWeight(
                      "\xD8\xB3\xD9\x80\xD9\x84\xD8\xA7\xD9\x85"));
}

TEST(OmnivoiceTextWeight, CombiningMarksAreZero) {
  // "e" (1.0) + U+0301 combining acute (0.0) = 1.0
  EXPECT_FLOAT_EQ(1.0f, OmnivoiceTextWeight("e\xCC\x81"));
}

TEST(OmnivoiceTextWeight, GreekAndCyrillic) {
  // Both fall through to Latin weight in the current port.
  // "αβγ" — 3 greek letters
  EXPECT_FLOAT_EQ(3.0f,
                  OmnivoiceTextWeight("\xCE\xB1\xCE\xB2\xCE\xB3"));
  // "абв" — 3 cyrillic letters
  EXPECT_FLOAT_EQ(3.0f,
                  OmnivoiceTextWeight("\xD0\xB0\xD0\xB1\xD0\xB2"));
}

TEST(OmnivoiceTextWeight, Indic) {
  // "नमस्ते" — Devanagari (all in the 0x0900-0x0DFF block)
  // 6 code points including the virama
  EXPECT_FLOAT_EQ(6 * 1.8f,
                  OmnivoiceTextWeight("\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8"
                                      "\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87"));
}

TEST(OmnivoiceTextWeight, Thai) {
  // "สวัสดี" — 6 Thai code points
  EXPECT_FLOAT_EQ(6 * 1.5f,
                  OmnivoiceTextWeight("\xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1"
                                      "\xE0\xB8\xAA\xE0\xB8\x94\xE0\xB8\xB5"));
}

TEST(OmnivoiceTextWeight, MixedLatinAndCjk) {
  // "hi 世界" -> 2 latin + 1 space + 2 CJK
  EXPECT_FLOAT_EQ(2.0f + 0.2f + 6.0f,
                  OmnivoiceTextWeight("hi \xE4\xB8\x96\xE7\x95\x8C"));
}

TEST(OmnivoiceTextWeight, Bmp4ByteEmoji) {
  // U+1F600 GRINNING FACE — falls in the >= 0x20000 range? No, it's 0x1F600.
  // Not covered by any explicit range -> falls to default kLatin = 1.0.
  // We only assert it does not crash and returns a finite non-negative value.
  float w = OmnivoiceTextWeight("\xF0\x9F\x98\x80");
  EXPECT_GE(w, 0.0f);
  EXPECT_LT(w, 10.0f);
}

TEST(OmnivoiceTextWeight, RefVsTargetRatioIsMonotonic) {
  // Sanity: adding more characters must never decrease the weight, so the
  // duration estimator's target/ref ratio behaves.
  float w1 = OmnivoiceTextWeight("hello");
  float w2 = OmnivoiceTextWeight("hello world");
  float w3 = OmnivoiceTextWeight("hello world, and welcome");
  EXPECT_LT(w1, w2);
  EXPECT_LT(w2, w3);
}

}  // namespace sherpa_onnx
