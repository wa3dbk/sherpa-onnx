// sherpa-onnx/csrc/offline-tts-omnivoice-text-weight.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_TEXT_WEIGHT_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_TEXT_WEIGHT_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace sherpa_onnx {

// Sum of per-character phonetic weights, mirroring the OmniVoice
// RuleDurationEstimator. Simplified port: precise weights for
// Latin / digits / punctuation / whitespace / marks; script-family
// approximations for common non-Latin ranges (CJK, kana, hangul, Arabic,
// Cyrillic, Greek, Indic, Thai/Lao). Unknown code points default to 1.0.
//
// The caller uses this to scale the reference-token count into a target
// token count for MaskGIT decoding, so the return value only needs to be
// consistent (same units for reference and target text), not calibrated
// to any absolute physical duration.
inline float OmnivoiceTextWeight(const std::string &s) {
  static constexpr float kLatin = 1.0f;
  static constexpr float kSpace = 0.2f;
  static constexpr float kPunct = 0.5f;
  static constexpr float kDigit = 3.5f;
  static constexpr float kCjk = 3.0f;
  static constexpr float kKana = 2.2f;
  static constexpr float kHangul = 2.5f;
  static constexpr float kArabic = 1.5f;
  static constexpr float kIndic = 1.8f;
  static constexpr float kThai = 1.5f;

  float total = 0.0f;
  for (size_t i = 0; i < s.size();) {
    unsigned char c0 = static_cast<unsigned char>(s[i]);
    uint32_t cp;
    int32_t adv;
    if (c0 < 0x80) {
      cp = c0;
      adv = 1;
    } else if ((c0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
      cp = ((c0 & 0x1F) << 6) |
           (static_cast<unsigned char>(s[i + 1]) & 0x3F);
      adv = 2;
    } else if ((c0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
      cp = ((c0 & 0x0F) << 12) |
           ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
           (static_cast<unsigned char>(s[i + 2]) & 0x3F);
      adv = 3;
    } else if ((c0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
      cp = ((c0 & 0x07) << 18) |
           ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
           ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
           (static_cast<unsigned char>(s[i + 3]) & 0x3F);
      adv = 4;
    } else {
      cp = c0;
      adv = 1;
    }
    i += adv;

    float w;
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) {
      w = kLatin;
    } else if (cp >= '0' && cp <= '9') {
      w = kDigit;
    } else if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') {
      w = kSpace;
    } else if (cp < 0x80) {
      // ASCII punctuation / symbols.
      w = kPunct;
    } else if (cp == 0x0640) {
      w = 0.0f;  // Arabic Tatweel
    } else if (cp >= 0x0300 && cp <= 0x036F) {
      w = 0.0f;  // Combining diacritical marks
    } else if (cp >= 0x0600 && cp <= 0x06FF) {
      w = kArabic;
    } else if (cp >= 0x0370 && cp <= 0x03FF) {
      w = kLatin;  // Greek
    } else if (cp >= 0x0400 && cp <= 0x04FF) {
      w = kLatin;  // Cyrillic
    } else if (cp >= 0x0900 && cp <= 0x0DFF) {
      w = kIndic;
    } else if (cp >= 0x0E00 && cp <= 0x0EFF) {
      w = kThai;
    } else if (cp >= 0x3040 && cp <= 0x30FF) {
      w = kKana;
    } else if (cp >= 0xAC00 && cp <= 0xD7AF) {
      w = kHangul;
    } else if ((cp >= 0x4E00 && cp <= 0x9FFF) ||
               (cp >= 0x3400 && cp <= 0x4DBF) || cp >= 0x20000) {
      w = kCjk;
    } else {
      w = kLatin;
    }
    total += w;
  }
  return total;
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_TEXT_WEIGHT_H_
