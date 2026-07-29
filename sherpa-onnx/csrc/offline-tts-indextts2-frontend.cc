// sherpa-onnx/csrc/offline-tts-indextts2-frontend.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-indextts2-frontend.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

// Decode a UTF-8 stream into a vector of Unicode codepoints. Malformed
// bytes fall through as U+FFFD.
std::vector<uint32_t> Utf8Decode(const std::string &s) {
  std::vector<uint32_t> out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    uint8_t b0 = static_cast<uint8_t>(s[i]);
    if (b0 < 0x80) {
      out.push_back(b0);
      i += 1;
    } else if ((b0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
      uint32_t cp = (b0 & 0x1F) << 6;
      cp |= (static_cast<uint8_t>(s[i + 1]) & 0x3F);
      out.push_back(cp);
      i += 2;
    } else if ((b0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
      uint32_t cp = (b0 & 0x0F) << 12;
      cp |= (static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6;
      cp |= (static_cast<uint8_t>(s[i + 2]) & 0x3F);
      out.push_back(cp);
      i += 3;
    } else if ((b0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
      uint32_t cp = (b0 & 0x07) << 18;
      cp |= (static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12;
      cp |= (static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6;
      cp |= (static_cast<uint8_t>(s[i + 3]) & 0x3F);
      out.push_back(cp);
      i += 4;
    } else {
      out.push_back(0xFFFD);
      i += 1;
    }
  }
  return out;
}

std::string Utf8Encode(uint32_t cp) {
  std::string s;
  if (cp < 0x80) {
    s.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return s;
}

}  // namespace

class OfflineTtsIndexTts2Frontend::Impl {
 public:
  Impl(const std::string &tokens_path, const std::string &merges_path,
       const std::string &pinyin_table_path, int32_t unk_token_id)
      : unk_token_id_(unk_token_id) {
    LoadTokens(tokens_path);
    LoadMerges(merges_path);
    LoadPinyin(pinyin_table_path);
  }

  template <typename Manager>
  Impl(Manager *mgr, const std::string &tokens_path,
       const std::string &merges_path,
       const std::string &pinyin_table_path, int32_t unk_token_id)
      : unk_token_id_(unk_token_id) {
    LoadTokensFromBuf(ReadFile(mgr, tokens_path));
    LoadMergesFromBuf(ReadFile(mgr, merges_path));
    LoadPinyinFromBuf(ReadFile(mgr, pinyin_table_path));
  }

  std::vector<int64_t> Encode(const std::string &text) const {
    return EncodeBpe(PinyinPreprocess(text));
  }

  std::vector<int64_t> EncodeRawBpe(const std::string &text) const {
    return EncodeBpe(text);
  }

 private:
  void LoadTokens(const std::string &path) {
    std::ifstream f(path);
    if (!f) {
      SHERPA_ONNX_LOGE("Cannot open tokens file: '%s'", path.c_str());
      return;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    LoadTokensFromBuf(ss.str());
  }

  void LoadTokensFromBuf(const std::string &buf) {
    std::stringstream ss(buf);
    std::string line;
    while (std::getline(ss, line)) {
      auto tab = line.find('\t');
      if (tab == std::string::npos) continue;
      std::string tok = line.substr(0, tab);
      int64_t id = std::stoll(line.substr(tab + 1));
      token_to_id_[tok] = id;
    }
  }

  void LoadTokensFromBuf(const std::vector<char> &buf) {
    LoadTokensFromBuf(std::string(buf.begin(), buf.end()));
  }

  void LoadMerges(const std::string &path) {
    std::ifstream f(path);
    if (!f) {
      SHERPA_ONNX_LOGE("Cannot open merges file: '%s'", path.c_str());
      return;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    LoadMergesFromBuf(ss.str());
  }

  void LoadMergesFromBuf(const std::string &buf) {
    std::stringstream ss(buf);
    std::string line;
    int32_t rank = 0;
    while (std::getline(ss, line)) {
      if (line.empty() || line[0] == '#') continue;
      auto sp = line.find(' ');
      if (sp == std::string::npos) continue;
      merges_[{line.substr(0, sp), line.substr(sp + 1)}] = rank++;
    }
  }

  void LoadMergesFromBuf(const std::vector<char> &buf) {
    LoadMergesFromBuf(std::string(buf.begin(), buf.end()));
  }

  void LoadPinyin(const std::string &path) {
    std::ifstream f(path);
    if (!f) {
      SHERPA_ONNX_LOGE("Cannot open pinyin table: '%s'", path.c_str());
      return;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    LoadPinyinFromBuf(ss.str());
  }

  void LoadPinyinFromBuf(const std::string &buf) {
    std::stringstream ss(buf);
    std::string line;
    while (std::getline(ss, line)) {
      if (line.size() < 3 || line[0] != 'U' || line[1] != '+') continue;
      auto tab = line.find('\t');
      if (tab == std::string::npos) continue;
      uint32_t cp = std::stoul(line.substr(2, tab - 2), nullptr, 16);
      std::string readings = line.substr(tab + 1);
      auto comma = readings.find(',');
      // First reading only.
      pinyin_[cp] =
          (comma == std::string::npos) ? readings : readings.substr(0, comma);
    }
  }

  void LoadPinyinFromBuf(const std::vector<char> &buf) {
    LoadPinyinFromBuf(std::string(buf.begin(), buf.end()));
  }

  std::string PinyinPreprocess(const std::string &text) const {
    auto cps = Utf8Decode(text);
    std::string out;
    out.reserve(text.size() * 2);
    for (uint32_t cp : cps) {
      auto it = pinyin_.find(cp);
      if (it != pinyin_.end()) {
        if (!out.empty() && out.back() != ' ') out.push_back(' ');
        out += it->second;
        out.push_back(' ');
      } else {
        out += Utf8Encode(cp);
      }
    }
    return out;
  }

  std::vector<int64_t> EncodeBpe(const std::string &text) const {
    // Simple GPT-2-style BPE: start with per-character tokens, then
    // iteratively apply the lowest-rank merge until no merges apply.
    std::vector<int64_t> out;
    if (text.empty()) return out;

    // Word-split on whitespace so we don't try to merge across word
    // boundaries (matches HF BPE behavior).
    std::vector<std::string> words;
    std::string current;
    for (char c : text) {
      if (c == ' ') {
        if (!current.empty()) {
          words.push_back(current);
          current.clear();
        }
      } else {
        current.push_back(c);
      }
    }
    if (!current.empty()) words.push_back(current);

    for (const auto &w : words) {
      auto cps = Utf8Decode(w);
      std::vector<std::string> parts;
      parts.reserve(cps.size());
      for (uint32_t cp : cps) parts.push_back(Utf8Encode(cp));
      ApplyMerges(&parts);
      for (const auto &p : parts) {
        auto it = token_to_id_.find(p);
        if (it != token_to_id_.end()) {
          out.push_back(it->second);
        } else {
          out.push_back(unk_token_id_);
        }
      }
    }
    return out;
  }

  void ApplyMerges(std::vector<std::string> *parts) const {
    while (parts->size() >= 2) {
      int32_t best_rank = -1;
      size_t best_i = 0;
      for (size_t i = 0; i + 1 < parts->size(); ++i) {
        auto it = merges_.find({(*parts)[i], (*parts)[i + 1]});
        if (it == merges_.end()) continue;
        if (best_rank < 0 || it->second < best_rank) {
          best_rank = it->second;
          best_i = i;
        }
      }
      if (best_rank < 0) break;
      (*parts)[best_i] = (*parts)[best_i] + (*parts)[best_i + 1];
      parts->erase(parts->begin() + best_i + 1);
    }
  }

  struct PairHash {
    size_t operator()(const std::pair<std::string, std::string> &p) const {
      return std::hash<std::string>()(p.first) ^
             (std::hash<std::string>()(p.second) << 1);
    }
  };

  std::unordered_map<std::string, int64_t> token_to_id_;
  std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash>
      merges_;
  std::unordered_map<uint32_t, std::string> pinyin_;
  int32_t unk_token_id_;
};

OfflineTtsIndexTts2Frontend::OfflineTtsIndexTts2Frontend(
    const std::string &tokens_path, const std::string &merges_path,
    const std::string &pinyin_table_path, int32_t unk_token_id)
    : impl_(std::make_unique<Impl>(tokens_path, merges_path,
                                   pinyin_table_path, unk_token_id)) {}

template <typename Manager>
OfflineTtsIndexTts2Frontend::OfflineTtsIndexTts2Frontend(
    Manager *mgr, const std::string &tokens_path,
    const std::string &merges_path, const std::string &pinyin_table_path,
    int32_t unk_token_id)
    : impl_(std::make_unique<Impl>(mgr, tokens_path, merges_path,
                                   pinyin_table_path, unk_token_id)) {}

OfflineTtsIndexTts2Frontend::~OfflineTtsIndexTts2Frontend() = default;

std::vector<int64_t> OfflineTtsIndexTts2Frontend::Encode(
    const std::string &text) const {
  return impl_->Encode(text);
}

std::vector<int64_t> OfflineTtsIndexTts2Frontend::EncodeRawBpe(
    const std::string &text) const {
  return impl_->EncodeRawBpe(text);
}

#if __ANDROID_API__ >= 9
template OfflineTtsIndexTts2Frontend::OfflineTtsIndexTts2Frontend(
    AAssetManager *mgr, const std::string &tokens_path,
    const std::string &merges_path, const std::string &pinyin_table_path,
    int32_t unk_token_id);
#endif

#if __OHOS__
template OfflineTtsIndexTts2Frontend::OfflineTtsIndexTts2Frontend(
    NativeResourceManager *mgr, const std::string &tokens_path,
    const std::string &merges_path, const std::string &pinyin_table_path,
    int32_t unk_token_id);
#endif

}  // namespace sherpa_onnx
