// sherpa-onnx/csrc/offline-tts-indextts2-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_IMPL_H_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-indextts2-audio-encoder.h"
#include "sherpa-onnx/csrc/offline-tts-indextts2-emotion-text-encoder.h"
#include "sherpa-onnx/csrc/offline-tts-indextts2-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-indextts2-model.h"
#include "sherpa-onnx/csrc/offline-tts-indextts2-sampler.h"
#include "sherpa-onnx/csrc/offline-tts-indextts2-vocoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineTtsIndexTts2Impl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsIndexTts2Impl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsIndexTts2Model>(config.model)),
        voice_enc_(std::make_unique<OfflineTtsIndexTts2AudioEncoder>(
            config.model, config.model.indextts2.voice_encoder)),
        emo_enc_(std::make_unique<OfflineTtsIndexTts2AudioEncoder>(
            config.model, config.model.indextts2.emotion_encoder)),
        emo_text_enc_(
            std::make_unique<OfflineTtsIndexTts2EmotionTextEncoder>(
                config.model)),
        vocoder_(std::make_unique<OfflineTtsIndexTts2Vocoder>(config.model)),
        frontend_(std::make_unique<OfflineTtsIndexTts2Frontend>(
            config.model.indextts2.tokens, config.model.indextts2.merges,
            config.model.indextts2.pinyin_table,
            model_->GetMetaData().pad_token_id)) {}

  template <typename Manager>
  OfflineTtsIndexTts2Impl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsIndexTts2Model>(mgr, config.model)),
        voice_enc_(std::make_unique<OfflineTtsIndexTts2AudioEncoder>(
            mgr, config.model, config.model.indextts2.voice_encoder)),
        emo_enc_(std::make_unique<OfflineTtsIndexTts2AudioEncoder>(
            mgr, config.model, config.model.indextts2.emotion_encoder)),
        emo_text_enc_(
            std::make_unique<OfflineTtsIndexTts2EmotionTextEncoder>(
                mgr, config.model)),
        vocoder_(std::make_unique<OfflineTtsIndexTts2Vocoder>(mgr,
                                                              config.model)),
        frontend_(std::make_unique<OfflineTtsIndexTts2Frontend>(
            mgr, config.model.indextts2.tokens,
            config.model.indextts2.merges,
            config.model.indextts2.pinyin_table,
            model_->GetMetaData().pad_token_id)) {}

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &config,
      GeneratedAudioCallback callback = nullptr) const override {
    // Extras understood by this backend:
    //   - "top_k", "top_p", "temperature", "seed", "max_audio_tokens"
    //   - "max_char_in_sentence" (default 200)
    //   - "min_char_in_sentence" (default 30)
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("%s", config.ToString().c_str());
    }
    if (config.reference_audio.empty()) {
      SHERPA_ONNX_LOGE("reference_audio is empty.");
      return {};
    }

    int32_t top_k =
        config.GetExtraInt("top_k", config_.model.indextts2.top_k);
    float top_p =
        config.GetExtraFloat("top_p", config_.model.indextts2.top_p);
    float temperature = config.GetExtraFloat(
        "temperature", config_.model.indextts2.temperature);
    int32_t seed = config.GetExtraInt("seed", config_.model.indextts2.seed);
    int32_t max_audio_tokens = config.GetExtraInt(
        "max_audio_tokens", config_.model.indextts2.max_audio_tokens);

    Ort::Value speaker_embed =
        EncodeVoice(config.reference_audio, config.reference_sample_rate);
    Ort::Value emo_embed =
        BuildEmoEmbed(config.emotion_audio, config.emotion_audio_sample_rate,
                      config.emotion_text);

    auto sentences = SplitByPunctuation(text);
    int32_t max_char = config.GetExtraInt("max_char_in_sentence", 200);
    int32_t min_char = config.GetExtraInt("min_char_in_sentence", 30);
    if (max_char <= 0) max_char = 200;
    if (min_char <= 0) min_char = 30;

    sentences = MergeShortSentences(sentences, min_char);
    std::vector<std::string> chunks;
    for (const auto &s : sentences) {
      auto pieces = SplitLongSentence(s, max_char);
      chunks.insert(chunks.end(), pieces.begin(), pieces.end());
    }
    if (chunks.empty()) return {};

    GeneratedAudio result;
    result.sample_rate = SampleRate();
    const int32_t total = static_cast<int32_t>(chunks.size());

    for (int32_t i = 0; i < total; ++i) {
      if (config_.model.debug) {
        SHERPA_ONNX_LOGE("IndexTTS-2 chunk %d/%d: %s", i + 1, total,
                         chunks[i].c_str());
      }
      GeneratedAudio cur = GenerateChunk(chunks[i], speaker_embed, emo_embed,
                                         top_k, top_p, temperature, seed,
                                         max_audio_tokens);
      if (cur.samples.empty()) continue;
      result.samples.insert(result.samples.end(), cur.samples.begin(),
                            cur.samples.end());
      if (callback) {
        if (!callback(cur.samples.data(),
                      static_cast<int32_t>(cur.samples.size()),
                      (i + 1) * 1.0f / total)) {
          break;
        }
      }
    }

    if (config.silence_scale != 1) {
      result = result.ScaleSilence(config.silence_scale);
    }
    return result;
  }

  GeneratedAudio Generate(
      const std::string &text, const std::string &prompt_text,
      const std::vector<float> &prompt_samples, int32_t sample_rate,
      float speed, int32_t num_steps,
      GeneratedAudioCallback callback = nullptr) const override {
    GenerationConfig config;
    config.speed = speed;
    config.num_steps = num_steps;
    config.reference_text = prompt_text;
    config.reference_audio = prompt_samples;
    config.reference_sample_rate = sample_rate;
    return Generate(text, config, std::move(callback));
  }

 private:
  Ort::Value EncodeVoice(const std::vector<float> &samples,
                         int32_t sample_rate) const {
    auto resampled = Resample16k(samples, sample_rate);
    return voice_enc_->Run(resampled);
  }

  Ort::Value BuildEmoEmbed(const std::vector<float> &emo_samples,
                           int32_t emo_sample_rate,
                           const std::string &emo_text) const {
    const auto &meta = model_->GetMetaData();
    int32_t D = meta.emotion_embed_dim;

    if (!emo_samples.empty()) {
      auto resampled = Resample16k(emo_samples, emo_sample_rate);
      return emo_enc_->Run(resampled);
    }
    if (!emo_text.empty()) {
      auto ids = frontend_->EncodeRawBpe(emo_text);
      if (ids.empty()) ids.push_back(0);
      return emo_text_enc_->Run(ids);
    }

    // Neutral zero-vector.
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> shape = {1, D};
    zero_emo_.assign(D, 0.0f);
    return Ort::Value::CreateTensor<float>(
        memory_info, zero_emo_.data(), zero_emo_.size(), shape.data(),
        shape.size());
  }

  std::vector<float> Resample16k(const std::vector<float> &samples,
                                 int32_t sample_rate) const {
    if (sample_rate == 16000) return samples;
    float min_freq = std::min<int32_t>(sample_rate, 16000);
    float lowpass_cutoff = 0.99f * 0.5f * min_freq;
    auto resampler = std::make_unique<LinearResample>(
        sample_rate, 16000, lowpass_cutoff, 6);
    std::vector<float> out;
    resampler->Resample(samples.data(), samples.size(), true, &out);
    return out;
  }

  GeneratedAudio GenerateChunk(const std::string &text,
                               const Ort::Value &speaker_embed,
                               const Ort::Value &emo_embed, int32_t top_k,
                               float top_p, float temperature, int32_t seed,
                               int32_t max_audio_tokens) const {
    auto ids = frontend_->Encode(text);
    if (ids.empty()) return {};

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> tt_shape = {1, static_cast<int64_t>(ids.size())};
    Ort::Value text_tokens = Ort::Value::CreateTensor<int64_t>(
        memory_info, ids.data(), ids.size(), tt_shape.data(), tt_shape.size());

    auto prefix =
        model_->PrefixPass(std::move(text_tokens), View(&speaker_embed),
                           View(&emo_embed));

    const auto &meta = model_->GetMetaData();
    OfflineTtsIndexTts2Sampler sampler(top_k, top_p, temperature, seed);

    std::vector<int64_t> audio_tokens;
    audio_tokens.reserve(max_audio_tokens);

    Ort::Value logits = std::move(prefix.logits);
    Ort::Value kv = std::move(prefix.present_kv);

    for (int32_t step = 0; step < max_audio_tokens; ++step) {
      float *lp = logits.GetTensorMutableData<float>();
      int64_t tok = sampler.Sample(lp, meta.vocab_size);
      if (tok == meta.eos_token_id) break;
      audio_tokens.push_back(tok);

      std::array<int64_t, 2> prev_shape = {1, 1};
      Ort::Value prev = Ort::Value::CreateTensor<int64_t>(
          memory_info, audio_tokens.data() + audio_tokens.size() - 1, 1,
          prev_shape.data(), prev_shape.size());

      auto stepped =
          model_->StepOne(std::move(prev), std::move(kv));
      logits = std::move(stepped.logits);
      kv = std::move(stepped.present_kv);
    }

    if (audio_tokens.empty()) return {};

    GeneratedAudio out;
    out.samples = vocoder_->Run(audio_tokens);
    out.sample_rate = meta.sample_rate;
    return out;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsIndexTts2Model> model_;
  std::unique_ptr<OfflineTtsIndexTts2AudioEncoder> voice_enc_;
  std::unique_ptr<OfflineTtsIndexTts2AudioEncoder> emo_enc_;
  std::unique_ptr<OfflineTtsIndexTts2EmotionTextEncoder> emo_text_enc_;
  std::unique_ptr<OfflineTtsIndexTts2Vocoder> vocoder_;
  std::unique_ptr<OfflineTtsIndexTts2Frontend> frontend_;
  mutable std::vector<float> zero_emo_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_IMPL_H_
