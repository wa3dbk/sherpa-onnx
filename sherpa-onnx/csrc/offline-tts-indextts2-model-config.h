// sherpa-onnx/csrc/offline-tts-indextts2-model-config.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_MODEL_CONFIG_H_

#include <cstdint>
#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsIndexTts2ModelConfig {
  // GPT-style AR LM. Exported dual-mode (prefix + step); the C++ side
  // switches modes via a scalar input tensor. Called once for the
  // prefix pass + N times for the AR loop per sentence.
  std::string lm;

  // Reference wav -> speaker embedding.
  std::string voice_encoder;

  // Emotion reference wav -> emotion embedding.
  std::string emotion_encoder;

  // Natural-language emotion description -> emotion embedding.
  std::string emotion_text_encoder;

  // Audio-token sequence -> waveform (BigVGAN-family).
  std::string vocoder;

  // BPE vocab file (tab-separated "token<TAB>id" pairs).
  std::string tokens;

  // BPE merges file (HF format, "part_a part_b" per line + version
  // header).
  std::string merges;

  // Offline CJK-char -> pinyin lookup table dumped from `pypinyin` at
  // bundle-build time. One line: "U+HHHH<TAB>py1[,py2,...]".
  std::string pinyin_table;

  // Upper bound on generated audio tokens per sentence. If EOS is not
  // emitted before this cap, we vocode whatever we have.
  int32_t max_audio_tokens = 2000;

  // Sampling knobs for the AR loop.
  int32_t top_k = 30;
  float top_p = 0.8f;
  float temperature = 0.8f;

  // RNG seed for the AR sampler. <0 => nondeterministic.
  int32_t seed = -1;

  OfflineTtsIndexTts2ModelConfig() = default;

  OfflineTtsIndexTts2ModelConfig(
      const std::string &lm, const std::string &voice_encoder,
      const std::string &emotion_encoder,
      const std::string &emotion_text_encoder, const std::string &vocoder,
      const std::string &tokens, const std::string &merges,
      const std::string &pinyin_table, int32_t max_audio_tokens = 2000,
      int32_t top_k = 30, float top_p = 0.8f, float temperature = 0.8f,
      int32_t seed = -1)
      : lm(lm),
        voice_encoder(voice_encoder),
        emotion_encoder(emotion_encoder),
        emotion_text_encoder(emotion_text_encoder),
        vocoder(vocoder),
        tokens(tokens),
        merges(merges),
        pinyin_table(pinyin_table),
        max_audio_tokens(max_audio_tokens),
        top_k(top_k),
        top_p(top_p),
        temperature(temperature),
        seed(seed) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_MODEL_CONFIG_H_
