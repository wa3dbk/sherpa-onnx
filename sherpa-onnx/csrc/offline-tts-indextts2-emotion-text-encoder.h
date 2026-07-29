// sherpa-onnx/csrc/offline-tts-indextts2-emotion-text-encoder.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_EMOTION_TEXT_ENCODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_EMOTION_TEXT_ENCODER_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

// Wraps emotion_text_encoder.onnx — takes token ids + mask, returns
// (1, D) float32 embedding matching the audio emotion encoder's D.
//
// Uses the same BPE tokenizer we ship for the LM (tokens.txt +
// merges.txt); the caller tokenizes upstream and passes ids in.
class OfflineTtsIndexTts2EmotionTextEncoder {
 public:
  explicit OfflineTtsIndexTts2EmotionTextEncoder(
      const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsIndexTts2EmotionTextEncoder(Manager *mgr,
                                        const OfflineTtsModelConfig &config);

  ~OfflineTtsIndexTts2EmotionTextEncoder();

  Ort::Value Run(const std::vector<int64_t> &token_ids) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_EMOTION_TEXT_ENCODER_H_
