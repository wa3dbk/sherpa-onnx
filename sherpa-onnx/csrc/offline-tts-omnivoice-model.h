// sherpa-onnx/csrc/offline-tts-omnivoice-model.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_MODEL_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-omnivoice-model-meta-data.h"

namespace sherpa_onnx {

// Owns three ONNX sessions:
//   - OmniVoice LM             (text/style/audio-code tokens -> logits)
//   - Higgs-Audio-V2 encoder   (raw 24 kHz mono waveform    -> 8-codebook ids)
//   - Higgs-Audio-V2 decoder   (8-codebook ids              -> 24 kHz waveform)
//
// Pure plumbing: no algorithm. The MaskGIT-style generation loop lives in
// OfflineTtsOmnivoiceImpl.
class OfflineTtsOmnivoiceModel {
 public:
  ~OfflineTtsOmnivoiceModel();

  explicit OfflineTtsOmnivoiceModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsOmnivoiceModel(Manager *mgr, const OfflineTtsModelConfig &config);

  // input_ids     int64 [B, num_codebook, S]
  // audio_mask    bool  [B, S]
  // attention_mask int64 [B, S]  (1 = keep, 0 = pad)
  // position_ids  int64 [B, S]
  // returns logits float [B, num_codebook, S, audio_vocab_size]
  Ort::Value RunLM(Ort::Value input_ids, Ort::Value audio_mask,
                   Ort::Value attention_mask, Ort::Value position_ids) const;

  // input:  float [1, 1, num_samples] at 24 kHz mono
  // output: int64 [1, num_codebook, num_tokens]
  Ort::Value EncodeAudio(Ort::Value pcm) const;

  // input:  int64 [1, num_codebook, num_tokens]
  // output: float [1, 1, num_samples] at 24 kHz mono
  Ort::Value DecodeCodes(Ort::Value codes) const;

  const OfflineTtsOmnivoiceModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_MODEL_H_
