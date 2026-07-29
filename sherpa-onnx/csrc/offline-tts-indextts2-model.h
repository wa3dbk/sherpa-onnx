// sherpa-onnx/csrc/offline-tts-indextts2-model.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_MODEL_H_

#include <memory>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-indextts2-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

// Owns the dual-mode LM Ort::Session and drives the AR loop from C++.
//
// PrefixPass: [text_tokens, speaker_embed, emotion_embed]
//             -> (first-step logits, seeded KV cache)
// StepOne:    [prev_token, past_kv]
//             -> (next-step logits, updated KV cache)
class OfflineTtsIndexTts2Model {
 public:
  explicit OfflineTtsIndexTts2Model(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsIndexTts2Model(Manager *mgr,
                           const OfflineTtsModelConfig &config);

  ~OfflineTtsIndexTts2Model();

  const OfflineTtsIndexTts2ModelMetaData &GetMetaData() const;

  // Returns { logits, present_kv }. Caller owns them.
  struct StepOutput {
    Ort::Value logits;
    Ort::Value present_kv;
  };

  StepOutput PrefixPass(Ort::Value text_tokens, Ort::Value speaker_embed,
                        Ort::Value emotion_embed) const;

  StepOutput StepOne(Ort::Value prev_token, Ort::Value past_kv) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_MODEL_H_
