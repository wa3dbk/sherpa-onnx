// sherpa-onnx/csrc/offline-tts-indextts2-model-config-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-indextts2-model-config.h"

#include <string>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(OfflineTtsIndexTts2ModelConfig, Defaults) {
  OfflineTtsIndexTts2ModelConfig c;
  EXPECT_EQ(c.max_audio_tokens, 2000);
  EXPECT_EQ(c.top_k, 30);
  EXPECT_FLOAT_EQ(c.top_p, 0.8f);
  EXPECT_FLOAT_EQ(c.temperature, 0.8f);
  EXPECT_EQ(c.seed, -1);
}

TEST(OfflineTtsIndexTts2ModelConfig, ValidateRequiresFiles) {
  OfflineTtsIndexTts2ModelConfig c;
  EXPECT_FALSE(c.Validate());  // empty lm

  c.lm = "/nope/lm.onnx";
  EXPECT_FALSE(c.Validate());
}

TEST(OfflineTtsIndexTts2ModelConfig, ValidateRejectsBadKnobs) {
  OfflineTtsIndexTts2ModelConfig c;
  c.max_audio_tokens = 0;
  EXPECT_FALSE(c.Validate());

  c.max_audio_tokens = 100;
  c.top_k = -1;
  EXPECT_FALSE(c.Validate());

  c.top_k = 30;
  c.top_p = 1.5f;
  EXPECT_FALSE(c.Validate());

  c.top_p = 0.8f;
  c.temperature = -0.1f;
  EXPECT_FALSE(c.Validate());
}

TEST(OfflineTtsIndexTts2ModelConfig, ToStringMentionsAllFields) {
  OfflineTtsIndexTts2ModelConfig c;
  std::string s = c.ToString();
  EXPECT_NE(s.find("lm"), std::string::npos);
  EXPECT_NE(s.find("voice_encoder"), std::string::npos);
  EXPECT_NE(s.find("emotion_encoder"), std::string::npos);
  EXPECT_NE(s.find("emotion_text_encoder"), std::string::npos);
  EXPECT_NE(s.find("vocoder"), std::string::npos);
  EXPECT_NE(s.find("tokens"), std::string::npos);
  EXPECT_NE(s.find("merges"), std::string::npos);
  EXPECT_NE(s.find("pinyin_table"), std::string::npos);
  EXPECT_NE(s.find("max_audio_tokens"), std::string::npos);
  EXPECT_NE(s.find("top_k"), std::string::npos);
  EXPECT_NE(s.find("top_p"), std::string::npos);
  EXPECT_NE(s.find("temperature"), std::string::npos);
  EXPECT_NE(s.find("seed"), std::string::npos);
}

}  // namespace sherpa_onnx
