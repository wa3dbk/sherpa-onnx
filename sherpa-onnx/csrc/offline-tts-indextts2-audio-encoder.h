// sherpa-onnx/csrc/offline-tts-indextts2-audio-encoder.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_AUDIO_ENCODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_AUDIO_ENCODER_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

// Wraps voice_encoder.onnx OR emotion_encoder.onnx — both take
// (1, T_audio) float32 at 16 kHz and return (1, D) float32.
class OfflineTtsIndexTts2AudioEncoder {
 public:
  OfflineTtsIndexTts2AudioEncoder(const OfflineTtsModelConfig &config,
                                  const std::string &onnx_path);

  template <typename Manager>
  OfflineTtsIndexTts2AudioEncoder(Manager *mgr,
                                  const OfflineTtsModelConfig &config,
                                  const std::string &onnx_path);

  ~OfflineTtsIndexTts2AudioEncoder();

  // samples must be mono, 16 kHz. Caller resamples upstream.
  Ort::Value Run(const std::vector<float> &samples) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_AUDIO_ENCODER_H_
