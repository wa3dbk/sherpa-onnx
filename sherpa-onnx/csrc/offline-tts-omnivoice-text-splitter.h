// sherpa-onnx/csrc/offline-tts-omnivoice-text-splitter.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_TEXT_SPLITTER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_TEXT_SPLITTER_H_

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace sherpa_onnx {

// Strategies for splitting an input string into chunks that each fit in one
// OmniVoice MaskGIT pass. Chunk boundaries are picked at natural pause
// points so the concatenated audio doesn't clip words in half.
//
//   kNone
//     Return the input unchanged. Use when the caller has already chunked
//     the text or when the input is known to be short. This is the default
//     to preserve backwards compatibility.
//
//   kSentence
//     Split on sentence-terminating punctuation ('.', '!', '?' and their
//     CJK counterparts '。', '！', '？'). If a single sentence still
//     exceeds `max_chars`, it falls back to kChars for that sentence only.
//     Best prosody for well-punctuated prose. Prefer this for narration.
//
//   kChars
//     Hard-limit chunks to at most `max_chars` bytes, preferring to break
//     at whitespace or ',' / '，' / ';' near the limit. Guarantees a bound
//     on per-chunk MaskGIT cost even for pathological inputs (URLs, IDs,
//     unpunctuated text). Prosody suffers because breaks may land mid-clause.
enum class OmnivoiceSplitStrategy {
  kNone = 0,
  kSentence = 1,
  kChars = 2,
};

inline OmnivoiceSplitStrategy ParseOmnivoiceSplitStrategy(
    const std::string &s) {
  if (s.empty() || s == "none") return OmnivoiceSplitStrategy::kNone;
  if (s == "sentence") return OmnivoiceSplitStrategy::kSentence;
  if (s == "chars" || s == "char") return OmnivoiceSplitStrategy::kChars;
  return OmnivoiceSplitStrategy::kNone;
}

namespace omnivoice_splitter_internal {

inline std::string Trim(const std::string &s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

// True if the byte sequence starting at `i` is one of the multi-byte CJK
// terminators we understand. Returns byte length of the match (0 if none).
inline int32_t CjkTerminatorLen(const std::string &s, size_t i) {
  // UTF-8: '。' = E3 80 82, '！' = EF BC 81, '？' = EF BC 9F,
  //        '，' = EF BC 8C, '；' = EF BC 9B
  if (i + 3 > s.size()) return 0;
  unsigned char a = s[i], b = s[i + 1], c = s[i + 2];
  if (a == 0xE3 && b == 0x80 && c == 0x82) return 3;   // 。
  if (a == 0xEF && b == 0xBC && (c == 0x81 || c == 0x9F)) return 3;  // ！？
  return 0;
}

inline int32_t CjkSoftBreakLen(const std::string &s, size_t i) {
  if (i + 3 > s.size()) return 0;
  unsigned char a = s[i], b = s[i + 1], c = s[i + 2];
  if (a == 0xEF && b == 0xBC && (c == 0x8C || c == 0x9B)) return 3;  // ，；
  return 0;
}

// Split into sentence-ish chunks. Each chunk ends right after its
// terminating punctuation so the boundary sits on a natural pause.
inline std::vector<std::string> SplitSentences(const std::string &text) {
  std::vector<std::string> out;
  size_t start = 0;
  for (size_t i = 0; i < text.size();) {
    int32_t cjk = CjkTerminatorLen(text, i);
    if (cjk) {
      i += cjk;
      std::string s = Trim(text.substr(start, i - start));
      if (!s.empty()) out.push_back(std::move(s));
      start = i;
      continue;
    }
    char c = text[i];
    if (c == '.' || c == '!' || c == '?') {
      // Skip runs of the same punctuation ("...", "!!") so we don't emit
      // empty chunks.
      size_t j = i + 1;
      while (j < text.size() && (text[j] == '.' || text[j] == '!' ||
                                 text[j] == '?')) {
        ++j;
      }
      std::string s = Trim(text.substr(start, j - start));
      if (!s.empty()) out.push_back(std::move(s));
      start = j;
      i = j;
      continue;
    }
    ++i;
  }
  if (start < text.size()) {
    std::string s = Trim(text.substr(start));
    if (!s.empty()) out.push_back(std::move(s));
  }
  return out;
}

// Hard chunker. Emits pieces of at most `max_chars` bytes, preferring to
// break at whitespace or a soft-break punct within the last 25% of the
// window. Never splits inside a multi-byte UTF-8 character.
inline std::vector<std::string> SplitByChars(const std::string &text,
                                             int32_t max_chars) {
  std::vector<std::string> out;
  if (max_chars < 16) max_chars = 16;
  size_t n = text.size();
  size_t start = 0;
  while (start < n) {
    size_t end = std::min<size_t>(start + max_chars, n);
    if (end < n) {
      size_t back_limit = start + (max_chars * 3 / 4);
      size_t best = 0;
      for (size_t i = end; i > back_limit; --i) {
        // Prefer whitespace break.
        if (std::isspace(static_cast<unsigned char>(text[i - 1]))) {
          best = i;
          break;
        }
        char c = text[i - 1];
        if (c == ',' || c == ';' || c == ':') {
          best = i;
          break;
        }
        int32_t cjk = CjkSoftBreakLen(text, i - 3);
        if (cjk && i >= 3) {
          best = i;
          break;
        }
      }
      if (best > start) {
        end = best;
      } else {
        // No good break inside the window. Walk back to a UTF-8 boundary
        // so we don't split a codepoint.
        while (end > start && (static_cast<unsigned char>(text[end]) & 0xC0) ==
                                 0x80) {
          --end;
        }
        if (end == start) end = std::min<size_t>(start + max_chars, n);
      }
    }
    std::string s = Trim(text.substr(start, end - start));
    if (!s.empty()) out.push_back(std::move(s));
    start = end;
  }
  return out;
}

}  // namespace omnivoice_splitter_internal

// Top-level entry point. Empty output means "synthesize nothing"; a
// single-element output means "no chunking necessary". Callers should just
// iterate over the returned vector, invoking Generate() per element and
// concatenating the resulting PCM.
inline std::vector<std::string> SplitOmnivoiceText(
    const std::string &text, OmnivoiceSplitStrategy strategy,
    int32_t max_chars) {
  namespace ns = omnivoice_splitter_internal;
  std::string trimmed = ns::Trim(text);
  if (trimmed.empty()) return {};
  if (strategy == OmnivoiceSplitStrategy::kNone) return {trimmed};
  if (max_chars < 16) max_chars = 200;

  if (strategy == OmnivoiceSplitStrategy::kSentence) {
    auto sents = ns::SplitSentences(trimmed);
    if (sents.empty()) return {trimmed};
    std::vector<std::string> out;
    for (auto &s : sents) {
      if (static_cast<int32_t>(s.size()) <= max_chars) {
        out.push_back(std::move(s));
      } else {
        // One sentence too long -> hard-split just that sentence.
        auto sub = ns::SplitByChars(s, max_chars);
        for (auto &t : sub) out.push_back(std::move(t));
      }
    }
    return out;
  }

  return ns::SplitByChars(trimmed, max_chars);
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_TEXT_SPLITTER_H_
