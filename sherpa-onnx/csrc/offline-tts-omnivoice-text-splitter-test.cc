// sherpa-onnx/csrc/offline-tts-omnivoice-text-splitter-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-omnivoice-text-splitter.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(OmnivoiceSplitter, EmptyInputYieldsEmpty) {
  auto out = SplitOmnivoiceText("", OmnivoiceSplitStrategy::kSentence, 200);
  EXPECT_TRUE(out.empty());
  out = SplitOmnivoiceText("   \t\n", OmnivoiceSplitStrategy::kNone, 200);
  EXPECT_TRUE(out.empty());
}

TEST(OmnivoiceSplitter, NoneStrategyPassesThrough) {
  auto out = SplitOmnivoiceText("hello. world?",
                                OmnivoiceSplitStrategy::kNone, 200);
  ASSERT_EQ(1u, out.size());
  EXPECT_EQ("hello. world?", out[0]);
}

TEST(OmnivoiceSplitter, NoneStrategyTrimsSurroundingWhitespace) {
  auto out = SplitOmnivoiceText("  hi  ", OmnivoiceSplitStrategy::kNone, 200);
  ASSERT_EQ(1u, out.size());
  EXPECT_EQ("hi", out[0]);
}

TEST(OmnivoiceSplitter, SentenceStrategySplitsOnAsciiTerminators) {
  auto out = SplitOmnivoiceText(
      "First sentence. Second one! Third? Trailing bit",
      OmnivoiceSplitStrategy::kSentence, 200);
  ASSERT_EQ(4u, out.size());
  EXPECT_EQ("First sentence.", out[0]);
  EXPECT_EQ("Second one!", out[1]);
  EXPECT_EQ("Third?", out[2]);
  EXPECT_EQ("Trailing bit", out[3]);
}

TEST(OmnivoiceSplitter, SentenceStrategyCollapsesEllipses) {
  auto out = SplitOmnivoiceText("Well... maybe. Yes!!",
                                OmnivoiceSplitStrategy::kSentence, 200);
  ASSERT_EQ(3u, out.size());
  EXPECT_EQ("Well...", out[0]);
  EXPECT_EQ("maybe.", out[1]);
  EXPECT_EQ("Yes!!", out[2]);
}

TEST(OmnivoiceSplitter, SentenceStrategySplitsCjkTerminators) {
  // 你好。世界！好吗？
  std::string in =
      "\xE4\xBD\xA0\xE5\xA5\xBD\xE3\x80\x82"       // 你好。
      "\xE4\xB8\x96\xE7\x95\x8C\xEF\xBC\x81"       // 世界！
      "\xE5\xA5\xBD\xE5\x90\x97\xEF\xBC\x9F";     // 好吗？
  auto out = SplitOmnivoiceText(in, OmnivoiceSplitStrategy::kSentence, 200);
  ASSERT_EQ(3u, out.size());
  EXPECT_EQ("\xE4\xBD\xA0\xE5\xA5\xBD\xE3\x80\x82", out[0]);
  EXPECT_EQ("\xE4\xB8\x96\xE7\x95\x8C\xEF\xBC\x81", out[1]);
  EXPECT_EQ("\xE5\xA5\xBD\xE5\x90\x97\xEF\xBC\x9F", out[2]);
}

TEST(OmnivoiceSplitter, SentenceStrategyFallsBackToCharsForOverlongSentence) {
  std::string huge(500, 'a');  // one giant "sentence" with no punct
  auto out = SplitOmnivoiceText(huge, OmnivoiceSplitStrategy::kSentence, 100);
  EXPECT_GE(out.size(), 5u);
  for (const auto &s : out) {
    EXPECT_LE(static_cast<int32_t>(s.size()), 100);
  }
}

TEST(OmnivoiceSplitter, CharsStrategyPrefersWhitespaceBreak) {
  std::string in = "one two three four five six seven eight nine ten eleven";
  auto out = SplitOmnivoiceText(in, OmnivoiceSplitStrategy::kChars, 24);
  ASSERT_GE(out.size(), 2u);
  for (const auto &s : out) {
    EXPECT_LE(static_cast<int32_t>(s.size()), 24);
    // No leading/trailing whitespace after trim.
    EXPECT_FALSE(s.empty());
    EXPECT_NE(' ', s.front());
    EXPECT_NE(' ', s.back());
  }
  // Round-tripping the chunks with single spaces recovers the input.
  std::string joined;
  for (size_t i = 0; i < out.size(); ++i) {
    if (i) joined += ' ';
    joined += out[i];
  }
  EXPECT_EQ(in, joined);
}

TEST(OmnivoiceSplitter, CharsStrategyRespectsUtf8Boundaries) {
  // Long CJK input; ensure no chunk ends inside a UTF-8 code point.
  std::string in;
  for (int i = 0; i < 50; ++i) {
    in += "\xE4\xBD\xA0\xE5\xA5\xBD";  // 你好
  }
  auto out = SplitOmnivoiceText(in, OmnivoiceSplitStrategy::kChars, 30);
  ASSERT_GE(out.size(), 2u);
  for (const auto &s : out) {
    // Every chunk must start with a UTF-8 lead byte (not a continuation).
    ASSERT_FALSE(s.empty());
    unsigned char first = static_cast<unsigned char>(s.front());
    EXPECT_NE(0x80, first & 0xC0);
  }
}

TEST(OmnivoiceSplitter, ParseStrategyKnownAndUnknown) {
  EXPECT_EQ(OmnivoiceSplitStrategy::kNone,
            ParseOmnivoiceSplitStrategy("none"));
  EXPECT_EQ(OmnivoiceSplitStrategy::kNone, ParseOmnivoiceSplitStrategy(""));
  EXPECT_EQ(OmnivoiceSplitStrategy::kSentence,
            ParseOmnivoiceSplitStrategy("sentence"));
  EXPECT_EQ(OmnivoiceSplitStrategy::kChars,
            ParseOmnivoiceSplitStrategy("chars"));
  EXPECT_EQ(OmnivoiceSplitStrategy::kChars,
            ParseOmnivoiceSplitStrategy("char"));
  EXPECT_EQ(OmnivoiceSplitStrategy::kNone,
            ParseOmnivoiceSplitStrategy("bogus"));
}

}  // namespace sherpa_onnx
