// sherpa-onnx/csrc/offline-tts-indextts2-sampler.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_SAMPLER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_SAMPLER_H_

#include <cstdint>
#include <random>

namespace sherpa_onnx {

class OfflineTtsIndexTts2Sampler {
 public:
  OfflineTtsIndexTts2Sampler(int32_t top_k, float top_p, float temperature,
                             int32_t seed);

  // Consumes a raw logits row of length vocab_size. Non-const so the
  // caller doesn't accidentally reuse the buffer for a second sample.
  int64_t Sample(float *logits, int64_t vocab_size);

 private:
  int32_t top_k_;
  float top_p_;
  float temperature_;
  std::mt19937 rng_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_SAMPLER_H_
