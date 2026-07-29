// sherpa-onnx/csrc/offline-tts-indextts2-sampler-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-indextts2-sampler.h"

#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(OfflineTtsIndexTts2Sampler, ArgmaxAtTemperatureZero) {
  std::vector<float> logits = {0.1f, 5.0f, 0.2f, 0.3f, 0.4f};
  OfflineTtsIndexTts2Sampler s(/*top_k=*/1, /*top_p=*/1.0f,
                               /*temperature=*/0.0f, /*seed=*/42);
  EXPECT_EQ(s.Sample(logits.data(), logits.size()), 1);
}

TEST(OfflineTtsIndexTts2Sampler, DeterministicWithSameSeed) {
  std::vector<float> logits = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  OfflineTtsIndexTts2Sampler a(5, 1.0f, 1.0f, 42);
  OfflineTtsIndexTts2Sampler b(5, 1.0f, 1.0f, 42);
  auto la = logits, lb = logits;
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(a.Sample(la.data(), la.size()),
              b.Sample(lb.data(), lb.size()));
    la = logits;
    lb = logits;
  }
}

TEST(OfflineTtsIndexTts2Sampler, TopKMasksOthers) {
  // Only index 0 and 4 have non-negligible logits; top-k=2 must never
  // sample from 1, 2, 3.
  std::vector<float> base = {10.0f, -10.0f, -10.0f, -10.0f, 9.0f};
  OfflineTtsIndexTts2Sampler s(2, 1.0f, 1.0f, 42);
  for (int i = 0; i < 50; ++i) {
    auto l = base;
    int64_t idx = s.Sample(l.data(), l.size());
    EXPECT_TRUE(idx == 0 || idx == 4);
  }
}

}  // namespace sherpa_onnx
