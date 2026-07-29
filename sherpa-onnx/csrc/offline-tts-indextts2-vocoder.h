// sherpa-onnx/csrc/offline-tts-indextts2-vocoder.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_VOCODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_VOCODER_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

class OfflineTtsIndexTts2Vocoder {
 public:
  explicit OfflineTtsIndexTts2Vocoder(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsIndexTts2Vocoder(Manager *mgr,
                             const OfflineTtsModelConfig &config);

  ~OfflineTtsIndexTts2Vocoder();

  std::vector<float> Run(const std::vector<int64_t> &audio_tokens) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_VOCODER_H_
