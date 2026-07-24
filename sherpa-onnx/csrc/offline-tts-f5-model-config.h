// sherpa-onnx/csrc/offline-tts-f5-model-config.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_MODEL_CONFIG_H_

#include <cstdint>
#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsF5ModelConfig {
  // The DiT (flow-matching Diffusion Transformer). Called num_steps
  // times per generation.
  std::string transformer;

  // The Vocos mel-to-waveform vocoder. Called once at the end.
  std::string vocoder;

  // Vocab file (tab-separated "token<TAB>id" pairs) matching the
  // tokenizer used at export time.
  std::string tokens;

  // espeak-ng data dir; reused from the zipvoice / piper phonemizer
  // bundle. Contains phontab, phonindex, phondata, intonations.
  std::string data_dir;

  // Optional Chinese lexicon (mirrors ZipVoice's `--zipvoice-lexicon`).
  std::string lexicon;

  // Euler ODE steps for the flow-matching decoder. 32 is F5-TTS's
  // published default; smaller = faster + slightly lower quality.
  int32_t num_steps = 32;

  // Classifier-free guidance weight. 0 disables CFG (fastest,
  // lowest fidelity to reference); F5-TTS ships with 2.0.
  float guidance_scale = 2.0f;

  // "Sway sampling" coefficient — F5-TTS's time-schedule warp. See
  // https://arxiv.org/abs/2410.06885 §3.2. -1.0 = F5's default;
  // 0.0 = uniform (disabled).
  float sway_coef = -1.0f;

  // Target RMS for input reference; F5-TTS normalizes to this
  // before feeding the mel to the DiT. Matches ZipVoice's field.
  float target_rms = 0.1f;

  // RNG seed for the initial gaussian noise. <0 => nondeterministic.
  int32_t seed = -1;

  OfflineTtsF5ModelConfig() = default;

  OfflineTtsF5ModelConfig(const std::string &transformer,
                          const std::string &vocoder,
                          const std::string &tokens,
                          const std::string &data_dir,
                          const std::string &lexicon, int32_t num_steps = 32,
                          float guidance_scale = 2.0f,
                          float sway_coef = -1.0f, float target_rms = 0.1f,
                          int32_t seed = -1)
      : transformer(transformer),
        vocoder(vocoder),
        tokens(tokens),
        data_dir(data_dir),
        lexicon(lexicon),
        num_steps(num_steps),
        guidance_scale(guidance_scale),
        sway_coef(sway_coef),
        target_rms(target_rms),
        seed(seed) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_MODEL_CONFIG_H_
