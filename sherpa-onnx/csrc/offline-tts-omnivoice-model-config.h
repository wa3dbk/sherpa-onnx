// sherpa-onnx/csrc/offline-tts-omnivoice-model-config.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_MODEL_CONFIG_H_

#include <cstdint>
#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsOmnivoiceModelConfig {
  // Path to the OmniVoice language model ONNX (with .onnx_data alongside).
  std::string model;

  // Higgs-Audio-V2 audio codec ONNX pair.
  std::string codec_encoder;
  std::string codec_decoder;

  // Directory containing vocab.json / merges.txt / tokenizer_config.json
  // for the Qwen3 tokenizer used by OmniVoice.
  std::string tokenizer_dir;

  // ---- generation defaults (overridable per-request via config.extra) ----

  // Number of MaskGIT-style denoising steps.
  int32_t num_steps = 32;

  // Time-step schedule shift (t_shift < 1.0 => more mass early).
  float t_shift = 0.1f;

  // Classifier-free guidance scale. 0 disables CFG (halves compute).
  float guidance_scale = 2.0f;

  // Penalty applied to later codebook layers when picking which positions to
  // unmask (later codebooks get delayed).
  float layer_penalty_factor = 5.0f;

  OfflineTtsOmnivoiceModelConfig() = default;

  OfflineTtsOmnivoiceModelConfig(const std::string &model,
                                 const std::string &codec_encoder,
                                 const std::string &codec_decoder,
                                 const std::string &tokenizer_dir,
                                 int32_t num_steps = 32, float t_shift = 0.1f,
                                 float guidance_scale = 2.0f,
                                 float layer_penalty_factor = 5.0f)
      : model(model),
        codec_encoder(codec_encoder),
        codec_decoder(codec_decoder),
        tokenizer_dir(tokenizer_dir),
        num_steps(num_steps),
        t_shift(t_shift),
        guidance_scale(guidance_scale),
        layer_penalty_factor(layer_penalty_factor) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_MODEL_CONFIG_H_
