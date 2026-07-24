// sherpa-onnx/csrc/offline-tts-f5-model-config-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-f5-model-config.h"

#include <cmath>
#include <string>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(OfflineTtsF5ModelConfig, Defaults) {
  OfflineTtsF5ModelConfig c;
  EXPECT_EQ(c.num_steps, 32);
  EXPECT_FLOAT_EQ(c.guidance_scale, 2.0f);
  EXPECT_FLOAT_EQ(c.sway_coef, -1.0f);
  EXPECT_FLOAT_EQ(c.target_rms, 0.1f);
  EXPECT_EQ(c.seed, -1);
}

TEST(OfflineTtsF5ModelConfig, ValidateRequiresFiles) {
  OfflineTtsF5ModelConfig c;
  // Empty transformer path — Validate should reject.
  EXPECT_FALSE(c.Validate());

  c.transformer = "/definitely/not/a/real/path/transformer.onnx";
  EXPECT_FALSE(c.Validate());
}

TEST(OfflineTtsF5ModelConfig, ToStringMentionsAllFields) {
  OfflineTtsF5ModelConfig c;
  std::string s = c.ToString();
  EXPECT_NE(s.find("transformer"), std::string::npos);
  EXPECT_NE(s.find("vocoder"), std::string::npos);
  EXPECT_NE(s.find("tokens"), std::string::npos);
  EXPECT_NE(s.find("num_steps"), std::string::npos);
  EXPECT_NE(s.find("guidance_scale"), std::string::npos);
  EXPECT_NE(s.find("sway_coef"), std::string::npos);
  EXPECT_NE(s.find("target_rms"), std::string::npos);
  EXPECT_NE(s.find("seed"), std::string::npos);
}

// F5-TTS sway sampling: t_new = t + sway * (cos(pi/2 * t) - 1 + t)
// Endpoints stay pinned at 0 and 1 regardless of sway_coef.
TEST(OfflineTtsF5, SwaySamplingEndpoints) {
  const float pi_half = 1.57079632679489661923f;
  for (float sway : {-1.0f, -0.5f, 0.0f, 0.3f}) {
    auto sway_map = [&](float t) {
      return t + sway * (std::cos(pi_half * t) - 1.0f + t);
    };
    EXPECT_NEAR(sway_map(0.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(sway_map(1.0f), 1.0f, 1e-6f);
  }
}

// Negative sway_coef should front-load timesteps (t_new(mid) < mid).
TEST(OfflineTtsF5, SwaySamplingNegativeFrontLoads) {
  const float pi_half = 1.57079632679489661923f;
  float sway = -1.0f;
  float t = 0.5f;
  float mapped = t + sway * (std::cos(pi_half * t) - 1.0f + t);
  EXPECT_LT(mapped, t);
}

}  // namespace sherpa_onnx
