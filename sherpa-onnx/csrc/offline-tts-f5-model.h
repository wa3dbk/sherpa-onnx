// sherpa-onnx/csrc/offline-tts-f5-model.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_MODEL_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-f5-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

// Wraps the F5-TTS DiT ONNX. The ONNX graph is one Euler step; this
// class runs the ODE loop externally so we can log per-step progress,
// interrupt, or reuse the KV-cache if a future export splits the graph.
class OfflineTtsF5Model {
 public:
  ~OfflineTtsF5Model();

  explicit OfflineTtsF5Model(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsF5Model(Manager *mgr, const OfflineTtsModelConfig &config);

  // Runs the flow-matching Euler ODE loop and returns the predicted
  // mel of shape (1, total_frames, mel_dim). The caller is expected
  // to slice out the target portion (frames after ref_frames) before
  // feeding the vocoder.
  //
  // - text_ids:      (1, total_frames) int64 (-1 for pad positions)
  // - cond:          (1, total_frames, mel_dim) float32 ref mel + zeros
  // - num_steps:     Euler steps (>=1)
  // - guidance_scale: CFG weight (0 disables)
  // - sway_coef:     F5-TTS sway-sampling coefficient (-1.0 = default)
  // - seed:          RNG seed for initial gaussian noise; <0 => nondet
  Ort::Value Run(Ort::Value text_ids, Ort::Value cond, int32_t num_steps,
                 float guidance_scale, float sway_coef, int32_t seed) const;

  const OfflineTtsF5ModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_MODEL_H_
