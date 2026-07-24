// sherpa-onnx/csrc/offline-tts-f5-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/stft.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/matcha-tts-lexicon.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-tts-f5-model.h"
#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/vocoder.h"

namespace sherpa_onnx {

class OfflineTtsF5Impl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsF5Impl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsF5Model>(config.model)),
        vocoder_(Vocoder::Create(config.model)) {
    InitFrontend();
    InitMelBanks();
  }

  template <typename Manager>
  OfflineTtsF5Impl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsF5Model>(mgr, config.model)),
        vocoder_(Vocoder::Create(mgr, config.model)) {
    InitFrontend(mgr);
    InitMelBanks();
  }

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &config,
      GeneratedAudioCallback callback = nullptr) const override {
    // Supported extra options in config.extra:
    //   - "num_steps" (int, default 32)
    //   - "guidance_scale" (float, default config.model.f5.guidance_scale)
    //   - "sway_coef" (float, default config.model.f5.sway_coef)
    //   - "target_rms" (float, default config.model.f5.target_rms)
    //   - "seed" (int, default config.model.f5.seed)
    //   - "max_char_in_sentence" (int, default 200)
    //   - "min_char_in_sentence" (int, default 30)
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("%s", config.ToString().c_str());
    }

    if (config.reference_audio.empty()) {
      SHERPA_ONNX_LOGE("reference_audio is empty.");
      return {};
    }
    if (config.reference_text.empty()) {
      SHERPA_ONNX_LOGE("reference_text is empty.");
      return {};
    }
    if (config.reference_sample_rate <= 0) {
      SHERPA_ONNX_LOGE("reference_sample_rate %d is invalid.",
                       config.reference_sample_rate);
      return {};
    }

    int32_t num_steps = config.GetExtraInt(
        "num_steps",
        config.num_steps > 0 ? config.num_steps : config_.model.f5.num_steps);
    if (num_steps <= 0) num_steps = 32;

    float guidance_scale = config.GetExtraFloat(
        "guidance_scale", config_.model.f5.guidance_scale);
    float sway_coef =
        config.GetExtraFloat("sway_coef", config_.model.f5.sway_coef);
    float target_rms =
        config.GetExtraFloat("target_rms", config_.model.f5.target_rms);
    if (target_rms <= 0) target_rms = 0.1f;

    int32_t seed = config.GetExtraInt("seed", config_.model.f5.seed);

    // RMS-normalize the reference clip; remember the ratio so we can restore
    // amplitude on the output.
    float ref_rms = ComputeRms(config.reference_audio);
    float rms_scale = 1.0f;
    std::vector<float> ref_samples = config.reference_audio;
    if (ref_rms > 0.0f && ref_rms < target_rms) {
      rms_scale = target_rms / ref_rms;
      for (auto &s : ref_samples) s *= rms_scale;
    }

    std::vector<float> ref_mel;
    ComputeMelSpectrogram(ref_samples, config.reference_sample_rate, &ref_mel);
    if (ref_mel.empty()) {
      SHERPA_ONNX_LOGE("No frames extracted from the reference audio");
      return {};
    }

    int32_t mel_dim = model_->GetMetaData().num_mels;
    int32_t ref_frames = static_cast<int32_t>(ref_mel.size() / mel_dim);

    // Tokenize the reference text; length is used to estimate target frames.
    auto ref_token_ids = frontend_->ConvertTextToTokenIds(config.reference_text);
    if (ref_token_ids.empty() ||
        (ref_token_ids.size() == 1 && ref_token_ids[0].tokens.empty())) {
      SHERPA_ONNX_LOGE("Failed to tokenize reference text");
      return {};
    }
    std::vector<int64_t> ref_tokens;
    for (const auto &t : ref_token_ids) {
      ref_tokens.insert(ref_tokens.end(), t.tokens.begin(), t.tokens.end());
    }

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
        SHERPA_ONNX_LOGE("F5 chunk %d/%d: %s", i + 1, total, chunks[i].c_str());
      }
      GeneratedAudio cur =
          GenerateChunk(chunks[i], ref_tokens, ref_mel, ref_frames,
                        config.reference_text, num_steps, guidance_scale,
                        sway_coef, seed, rms_scale);

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
  void InitFrontend() {
    frontend_ = std::make_unique<MatchaTtsLexicon>(
        config_.model.f5.lexicon, config_.model.f5.tokens,
        config_.model.f5.data_dir, config_.model.debug, true);
  }

  template <typename Manager>
  void InitFrontend(Manager *mgr) {
    frontend_ = std::make_unique<MatchaTtsLexicon>(
        mgr, config_.model.f5.lexicon, config_.model.f5.tokens,
        config_.model.f5.data_dir, config_.model.debug, true);
  }

  void InitMelBanks() {
    const auto &meta = model_->GetMetaData();
    knf::FrameExtractionOptions frame_opts;
    frame_opts.samp_freq = meta.sample_rate;
    frame_opts.frame_length_ms = meta.window_length * 1000 / meta.sample_rate;
    frame_opts.frame_shift_ms = meta.hop_length * 1000 / meta.sample_rate;
    frame_opts.window_type = "hanning";

    knf::MelBanksOptions mel_opts;
    mel_opts.num_bins = meta.num_mels;
    mel_opts.low_freq = 0;
    mel_opts.high_freq = meta.sample_rate / 2;
    mel_opts.is_librosa = true;
    mel_opts.use_slaney_mel_scale = false;
    mel_opts.norm = "";

    mel_banks_ = std::make_unique<knf::MelBanks>(mel_opts, frame_opts, 1.0f);
  }

  static float ComputeRms(const std::vector<float> &samples) {
    if (samples.empty()) return 0.0f;
    double sum_sq = 0.0;
    for (float s : samples) sum_sq += s * s;
    return static_cast<float>(std::sqrt(sum_sq / samples.size()));
  }

  void ComputeMelSpectrogram(const std::vector<float> &_samples,
                             int32_t sample_rate,
                             std::vector<float> *out) const {
    const auto &meta = model_->GetMetaData();
    if (sample_rate != meta.sample_rate) {
      float min_freq = std::min<int32_t>(sample_rate, meta.sample_rate);
      float lowpass_cutoff = 0.99f * 0.5f * min_freq;
      auto resampler = std::make_unique<LinearResample>(
          sample_rate, meta.sample_rate, lowpass_cutoff, 6);
      std::vector<float> resampled;
      resampler->Resample(_samples.data(), _samples.size(), true, &resampled);
      ComputeMelSpectrogramImpl(resampled, out);
      return;
    }
    ComputeMelSpectrogramImpl(_samples, out);
  }

  void ComputeMelSpectrogramImpl(const std::vector<float> &samples,
                                 std::vector<float> *out) const {
    const auto &meta = model_->GetMetaData();
    int32_t fft_bins = meta.n_fft / 2 + 1;

    knf::StftConfig stft_config;
    stft_config.n_fft = meta.n_fft;
    stft_config.hop_length = meta.hop_length;
    stft_config.win_length = meta.window_length;
    stft_config.window_type = "hann";
    stft_config.center = true;

    knf::Stft stft(stft_config);
    auto r = stft.Compute(samples.data(), samples.size());

    out->resize(r.num_frames * meta.num_mels);
    float *p = out->data();
    std::vector<float> magnitude(fft_bins);
    for (int32_t i = 0; i < r.num_frames; ++i, p += meta.num_mels) {
      for (int32_t k = 0; k < fft_bins; ++k) {
        float re = r.real[i * fft_bins + k];
        float im = r.imag[i * fft_bins + k];
        magnitude[k] = std::sqrt(re * re + im * im);
      }
      mel_banks_->Compute(magnitude.data(), p);
      // F5-TTS natural-log mel with 1e-5 floor (matches vocos-mel-24khz).
      for (int32_t j = 0; j < meta.num_mels; ++j) {
        p[j] = std::log(p[j] + 1e-5f);
      }
    }
  }

  GeneratedAudio GenerateChunk(const std::string &target_text,
                               const std::vector<int64_t> &ref_tokens,
                               const std::vector<float> &ref_mel,
                               int32_t ref_frames,
                               const std::string &reference_text,
                               int32_t num_steps, float guidance_scale,
                               float sway_coef, int32_t seed,
                               float rms_scale) const {
    auto target_token_ids = frontend_->ConvertTextToTokenIds(target_text);
    if (target_token_ids.empty() ||
        (target_token_ids.size() == 1 &&
         target_token_ids[0].tokens.empty())) {
      SHERPA_ONNX_LOGE("Failed to tokenize target text '%s'",
                       target_text.c_str());
      return {};
    }
    std::vector<int64_t> tgt_tokens;
    for (const auto &t : target_token_ids) {
      tgt_tokens.insert(tgt_tokens.end(), t.tokens.begin(), t.tokens.end());
    }

    // Estimate target duration from character-length ratio (F5-TTS heuristic).
    int64_t ref_char_len = static_cast<int64_t>(reference_text.size());
    int64_t tgt_char_len = static_cast<int64_t>(target_text.size());
    if (ref_char_len <= 0) ref_char_len = 1;
    int64_t target_frames =
        static_cast<int64_t>(ref_frames) * tgt_char_len / ref_char_len;
    if (target_frames < 1) target_frames = 1;
    int64_t total_frames = ref_frames + target_frames;
    int64_t mel_dim = model_->GetMetaData().num_mels;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // text_ids: concat(ref_tokens, tgt_tokens) right-padded with -1 to
    // total_frames.
    std::vector<int64_t> text_ids(total_frames, -1);
    size_t nfill = std::min<size_t>(
        ref_tokens.size() + tgt_tokens.size(),
        static_cast<size_t>(total_frames));
    size_t nref = std::min<size_t>(ref_tokens.size(), nfill);
    std::copy(ref_tokens.begin(), ref_tokens.begin() + nref, text_ids.begin());
    size_t remaining = nfill - nref;
    std::copy(tgt_tokens.begin(), tgt_tokens.begin() + remaining,
              text_ids.begin() + nref);

    std::array<int64_t, 2> text_shape = {1, total_frames};
    Ort::Value text_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, text_ids.data(), text_ids.size(), text_shape.data(),
        text_shape.size());

    // cond: [ref_mel; zeros(target_frames, mel_dim)]
    std::vector<float> cond_data(total_frames * mel_dim, 0.0f);
    std::copy(ref_mel.begin(), ref_mel.begin() + ref_frames * mel_dim,
              cond_data.begin());
    std::array<int64_t, 3> cond_shape = {1, total_frames, mel_dim};
    Ort::Value cond_tensor = Ort::Value::CreateTensor<float>(
        memory_info, cond_data.data(), cond_data.size(), cond_shape.data(),
        cond_shape.size());

    Ort::Value mel = model_->Run(std::move(text_tensor), std::move(cond_tensor),
                                 num_steps, guidance_scale, sway_coef, seed);

    // Slice out the target portion, transpose (T, C) -> (C, T) for Vocos.
    const float *mel_data = mel.GetTensorData<float>();
    const float *tgt_begin = mel_data + ref_frames * mel_dim;
    std::vector<float> mel_permuted = Transpose(tgt_begin, target_frames,
                                                mel_dim);

    std::array<int64_t, 3> new_shape = {1, mel_dim, target_frames};
    Ort::Value mel_new = Ort::Value::CreateTensor<float>(
        memory_info, mel_permuted.data(), mel_permuted.size(),
        new_shape.data(), new_shape.size());

    GeneratedAudio ans;
    ans.samples = vocoder_->Run(std::move(mel_new));
    ans.sample_rate = model_->GetMetaData().sample_rate;

    // Restore original amplitude if we scaled the reference up.
    if (rms_scale != 1.0f && rms_scale > 0.0f) {
      float inv = 1.0f / rms_scale;
      for (auto &s : ans.samples) s *= inv;
    }
    return ans;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsF5Model> model_;
  std::unique_ptr<Vocoder> vocoder_;
  std::unique_ptr<OfflineTtsFrontend> frontend_;
  std::unique_ptr<knf::MelBanks> mel_banks_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_IMPL_H_
