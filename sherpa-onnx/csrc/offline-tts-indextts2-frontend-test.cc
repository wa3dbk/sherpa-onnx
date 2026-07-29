// sherpa-onnx/csrc/offline-tts-indextts2-frontend-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-indextts2-frontend.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

// Test fixtures are written to a temp dir per test. Keeping them tiny
// so the parser paths are exercised without shipping a real bundle.
static std::string WriteTmp(const std::string &name,
                            const std::string &body) {
  std::string path = std::string(std::tmpnam(nullptr)) + "_" + name;
  std::ofstream f(path);
  f << body;
  return path;
}

static std::string BasicTokensTxt() {
  // Special tokens 0..3, then ASCII bytes as single-byte BPE tokens
  // (Ġ marks a leading space, GPT-2 style).
  std::string s =
      "<pad>\t0\n"
      "<unk>\t1\n"
      "<bos>\t2\n"
      "<eos>\t3\n";
  int id = 4;
  for (char c = 32; c < 127; ++c, ++id) {
    s += std::string(1, c) + "\t" + std::to_string(id) + "\n";
  }
  // A merged token so we can test BPE composition.
  s += "hi\t" + std::to_string(id++) + "\n";
  s += "hello\t" + std::to_string(id++) + "\n";
  return s;
}

static std::string BasicMergesTxt() {
  return
      "#version: 0.2\n"
      "h i\n"
      "h e\n"
      "he l\n"
      "hel l\n"
      "hell o\n";
}

static std::string BasicPinyinTable() {
  // U+4F60 = 你, U+597D = 好
  return
      "U+4F60\tni\n"
      "U+597D\thao,hao\n";
}

TEST(OfflineTtsIndexTts2Frontend, EncodeAsciiRoundTrip) {
  std::string tp = WriteTmp("tokens.txt", BasicTokensTxt());
  std::string mp = WriteTmp("merges.txt", BasicMergesTxt());
  std::string pp = WriteTmp("pinyin.txt", BasicPinyinTable());
  OfflineTtsIndexTts2Frontend fe(tp, mp, pp, /*unk_token_id=*/1);

  auto ids = fe.Encode("hi");
  ASSERT_FALSE(ids.empty());
  // "hi" should hit the merged BPE token.
  EXPECT_EQ(ids.size(), 1u);
}

TEST(OfflineTtsIndexTts2Frontend, EncodeUnknownProducesUnkNotCrash) {
  std::string tp = WriteTmp("tokens.txt", BasicTokensTxt());
  std::string mp = WriteTmp("merges.txt", BasicMergesTxt());
  std::string pp = WriteTmp("pinyin.txt", BasicPinyinTable());
  OfflineTtsIndexTts2Frontend fe(tp, mp, pp, /*unk_token_id=*/1);

  // The bell char (0x07) is guaranteed absent from our fixture vocab.
  auto ids = fe.Encode(std::string("\x07"));
  ASSERT_FALSE(ids.empty());
  EXPECT_EQ(ids.front(), 1);
}

TEST(OfflineTtsIndexTts2Frontend, EncodeCjkGoesThroughPinyinTable) {
  std::string tp = WriteTmp("tokens.txt", BasicTokensTxt());
  std::string mp = WriteTmp("merges.txt", BasicMergesTxt());
  std::string pp = WriteTmp("pinyin.txt", BasicPinyinTable());
  OfflineTtsIndexTts2Frontend fe(tp, mp, pp, /*unk_token_id=*/1);

  // "你好" -> "ni hao" (approximately, spaces optional).
  auto ids = fe.Encode(std::string("\xe4\xbd\xa0\xe5\xa5\xbd"));  // 你好
  ASSERT_FALSE(ids.empty());
  // No <unk>: pinyin substitution must have fired.
  for (int64_t id : ids) EXPECT_NE(id, 1);
}

TEST(OfflineTtsIndexTts2Frontend, EncodeRawBpeSkipsPinyinPath) {
  std::string tp = WriteTmp("tokens.txt", BasicTokensTxt());
  std::string mp = WriteTmp("merges.txt", BasicMergesTxt());
  std::string pp = WriteTmp("pinyin.txt", BasicPinyinTable());
  OfflineTtsIndexTts2Frontend fe(tp, mp, pp, /*unk_token_id=*/1);

  auto ids = fe.EncodeRawBpe("hi");
  EXPECT_EQ(ids.size(), 1u);
}

}  // namespace sherpa_onnx
