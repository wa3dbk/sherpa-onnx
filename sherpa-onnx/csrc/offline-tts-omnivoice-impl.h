// sherpa-onnx/csrc/offline-tts-omnivoice-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_IMPL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-omnivoice-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-omnivoice-model.h"
#include "sherpa-onnx/csrc/offline-tts-omnivoice-text-weight.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/qwen-asr-tokenizer.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

// End-to-end OmniVoice TTS backend.
//
// Pipeline per Generate() call:
//   1. Tokenize style prompt + text via QwenAsrTokenizer.
//   2. Resample reference audio to 24 kHz mono and encode with the
//      Higgs-Audio-V2 codec encoder -> 8-codebook token grid.
//   3. Build cond and uncond input_ids side by side (CFG batch of 2).
//   4. Iterative MaskGIT decoding: N full-graph forward passes, each pass
//      unmasks the k highest-confidence positions until the target span is
//      fully filled. No KV cache; the graph is non-autoregressive.
//   5. Decode the generated 8-codebook tokens with the codec decoder ->
//      24 kHz mono waveform.
class OfflineTtsOmnivoiceImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsOmnivoiceImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsOmnivoiceModel>(config.model)),
        tokenizer_(std::make_unique<QwenAsrTokenizer>(
            config.model.omnivoice.tokenizer_dir)) {}

  template <typename Manager>
  OfflineTtsOmnivoiceImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsOmnivoiceModel>(mgr, config.model)),
        tokenizer_(std::make_unique<QwenAsrTokenizer>(
            mgr, config.model.omnivoice.tokenizer_dir)) {}

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  // GenerationConfig.extra keys understood:
  //   "num_target_tokens" (int) - number of audio tokens to generate. If
  //       omitted, estimated from text length. 1 token = 1/25 s of audio.
  //   "duration_sec"      (float) - alternative to num_target_tokens.
  //   "language"          (string) - language tag (e.g. "en"). Default "None".
  //   "instruct"          (string) - free-form style instruction. Default
  //       "None".
  //   "denoise"           (int) - 1 to prepend <|denoise|> to the style
  //       prompt (recommended when using a noisy reference).  Default 1.
  //   "num_steps"         (int) - override model config.
  //   "guidance_scale"    (float) - override model config.
  //   "t_shift"           (float) - override model config.
  //   "layer_penalty"     (float) - override model config.
  //   "position_temperature" (float) - override model config.
  //   "seed"              (int) - override model config; <0 = nondeterministic.
  //   "cache_reference_audio" (int) - 1 (default) to reuse the cached codec
  //       encoding when the reference audio is unchanged.
  //   "chunk_ms"          (int) - streaming chunk size in ms for the
  //       GeneratedAudioCallback (default 500). Also emits progress-only
  //       ticks (n=0) after each MaskGIT step.
  GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &config,
      GeneratedAudioCallback callback = nullptr) const override {
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("omnivoice: %s", config.ToString().c_str());
    }

    if (config.reference_audio.empty() || config.reference_sample_rate <= 0) {
      SHERPA_ONNX_LOGE("omnivoice requires reference_audio + reference_sample_rate");
      return {};
    }
    if (config.reference_text.empty()) {
      SHERPA_ONNX_LOGE("omnivoice requires reference_text");
      return {};
    }
    if (text.empty()) {
      SHERPA_ONNX_LOGE("omnivoice: empty text");
      return {};
    }

    // Reference-audio duration sanity check. Below ~0.5s the codec cannot
    // extract a meaningful speaker embedding; above ~30s wastes compute and
    // usually indicates the wrong file was passed.
    float ref_sec =
        static_cast<float>(config.reference_audio.size()) /
        static_cast<float>(config.reference_sample_rate);
    if (ref_sec < 0.5f) {
      SHERPA_ONNX_LOGE(
          "omnivoice: reference_audio is only %.3f s; need at least 0.5 s",
          ref_sec);
      return {};
    }
    if (ref_sec > 30.0f) {
      SHERPA_ONNX_LOGE(
          "omnivoice: reference_audio is %.1f s; anything above ~30 s "
          "wastes compute. Trim it.", ref_sec);
    }

    const auto &m = model_->GetMetaData();
    const auto &mc = config_.model.omnivoice;

    int32_t num_steps = config.GetExtraInt("num_steps", mc.num_steps);
    float guidance_scale =
        config.GetExtraFloat("guidance_scale", mc.guidance_scale);
    float t_shift = config.GetExtraFloat("t_shift", mc.t_shift);
    float layer_penalty =
        config.GetExtraFloat("layer_penalty", mc.layer_penalty_factor);
    float position_temperature = config.GetExtraFloat(
        "position_temperature", mc.position_temperature);
    int32_t seed = config.GetExtraInt("seed", mc.seed);

    std::mt19937 rng(seed >= 0 ? static_cast<uint32_t>(seed)
                               : std::random_device{}());
    std::uniform_real_distribution<float> uni(1e-10f, 1.0f);
    std::string language = config.GetExtraString("language", "None");
    std::string instruct = config.GetExtraString("instruct", "None");
    bool denoise = config.GetExtraInt("denoise", 1) != 0;

    // --- 1. Encode reference audio into codec tokens ------------------------
    // The encoder output depends only on the reference clip, so cache the
    // result by content hash and skip the codec forward on repeat calls with
    // the same voice. Disable per-request with extra "cache_reference_audio=0".
    std::vector<int64_t> ref_codes;  // flattened [C, T_ref] row-major
    int32_t ref_tok_len = 0;
    bool cache_enabled =
        config.GetExtraInt("cache_reference_audio", 1) != 0;
    uint64_t ref_hash = HashPcm(config.reference_audio.data(),
                                static_cast<int32_t>(
                                    config.reference_audio.size()),
                                config.reference_sample_rate);
    {
      std::lock_guard<std::mutex> lock(ref_cache_mutex_);
      if (cache_enabled && ref_cache_valid_ && ref_cache_key_ == ref_hash) {
        ref_codes = ref_cache_codes_;
        ref_tok_len = ref_cache_tok_len_;
        if (config_.model.debug) {
          SHERPA_ONNX_LOGE("omnivoice: reference audio cache hit (%d tokens)",
                           ref_tok_len);
        }
      }
    }
    if (ref_tok_len == 0) {
      std::vector<float> pcm =
          Resample(config.reference_audio.data(),
                   static_cast<int32_t>(config.reference_audio.size()),
                   config.reference_sample_rate, m.sample_rate);
      Ort::Value codes = model_->EncodeAudio(MakePcmTensor(pcm));
      auto shape = codes.GetTensorTypeAndShapeInfo().GetShape();
      // expect [1, C, T]
      if (shape.size() != 3 || shape[1] != m.num_codebook) {
        SHERPA_ONNX_LOGE("codec encoder returned bad shape");
        return {};
      }
      ref_tok_len = static_cast<int32_t>(shape[2]);
      if (ref_tok_len < 1) {
        SHERPA_ONNX_LOGE("codec encoder produced 0 reference tokens");
        return {};
      }
      const int64_t *src = codes.GetTensorData<int64_t>();
      ref_codes.assign(src, src + m.num_codebook * ref_tok_len);

      if (cache_enabled) {
        std::lock_guard<std::mutex> lock(ref_cache_mutex_);
        ref_cache_key_ = ref_hash;
        ref_cache_codes_ = ref_codes;
        ref_cache_tok_len_ = ref_tok_len;
        ref_cache_valid_ = true;
      }
    }

    // --- 2. Tokenize style + text prompts ----------------------------------
    std::string style_text;
    if (denoise) style_text += "<|denoise|>";
    style_text += "<|lang_start|>" + language + "<|lang_end|>";
    style_text += "<|instruct_start|>" + instruct + "<|instruct_end|>";
    std::vector<int64_t> style_ids = tokenizer_->Encode(style_text);

    std::string full_text = Strip(config.reference_text) + " " + Strip(text);
    std::string wrapped_text = "<|text_start|>" + full_text + "<|text_end|>";
    std::vector<int64_t> text_ids = tokenizer_->Encode(wrapped_text);

    // --- 3. Decide target token length -------------------------------------
    // Matches OmniVoice's RuleDurationEstimator: scale target/ref character
    // weights by the actual ref-audio token count. Weight is char-count for
    // Latin scripts; heavier for CJK/Indic/etc. Falls back to the
    // low-threshold boost for very short targets.
    int32_t num_target_tokens = config.GetExtraInt("num_target_tokens", 0);
    if (num_target_tokens <= 0) {
      float dur = config.GetExtraFloat("duration_sec", 0.0f);
      if (dur > 0) {
        num_target_tokens = static_cast<int32_t>(dur * m.frame_rate + 0.5f);
      } else {
        float ref_w = OmnivoiceTextWeight(config.reference_text);
        float tgt_w = OmnivoiceTextWeight(text);
        float est_tokens;
        if (ref_w > 0 && ref_tok_len > 0) {
          float speed = ref_w / static_cast<float>(ref_tok_len);
          est_tokens = tgt_w / std::max(1e-6f, speed);
        } else {
          // Fallback: 3.0 chars per token (~7.5 chars/sec at 25 tok/s).
          est_tokens = std::max(1.0f, tgt_w / 3.0f);
        }
        // low_threshold boost from RuleDurationEstimator (short-text safety).
        const float low = 50.0f;
        const float boost = 3.0f;
        if (est_tokens < low) {
          float alpha = 1.0f / boost;
          est_tokens = low * std::pow(est_tokens / low, alpha);
        }
        num_target_tokens = static_cast<int32_t>(est_tokens + 0.5f);
      }
    }
    if (num_target_tokens < 1) num_target_tokens = 1;
    // Hard cap: 60 s of audio at 25 tok/s. Above this the LM forward passes
    // will blow memory and quality drops anyway. Callers who need long-form
    // TTS should chunk on the sentence level.
    const int32_t kMaxTargetTokens = 60 * m.frame_rate;
    if (num_target_tokens > kMaxTargetTokens) {
      SHERPA_ONNX_LOGE(
          "omnivoice: target length %d tokens (%.1f s) exceeds cap %d; "
          "clipping. Split the input text into shorter sentences.",
          num_target_tokens,
          num_target_tokens / static_cast<float>(m.frame_rate),
          kMaxTargetTokens);
      num_target_tokens = kMaxTargetTokens;
    }
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE(
          "omnivoice: ref_tok=%d style=%d text=%d target=%d",
          ref_tok_len, static_cast<int32_t>(style_ids.size()),
          static_cast<int32_t>(text_ids.size()), num_target_tokens);
    }

    // --- 4. Assemble cond / uncond input_ids etc. --------------------------
    const int32_t C = m.num_codebook;
    const int64_t mask_id = m.audio_mask_id;

    int32_t n_style = static_cast<int32_t>(style_ids.size());
    int32_t n_text = static_cast<int32_t>(text_ids.size());
    int32_t c_len = n_style + n_text + ref_tok_len + num_target_tokens;
    int32_t u_len = num_target_tokens;  // uncond = target region only
    int32_t max_len = std::max(c_len, u_len);

    // Rolling buffer for the currently-generated target tokens.
    // Shape [C, num_target_tokens]. Initially all mask.
    std::vector<int64_t> tokens(C * num_target_tokens, mask_id);

    // Build cond & uncond input_ids [2, C, max_len], initialized to mask_id.
    std::vector<int64_t> input_ids(2 * C * max_len, mask_id);
    // audio_mask [2, max_len] as int64 for easy tensor build (still 0/1)
    std::vector<uint8_t> audio_mask(2 * max_len, 0);
    // attention_mask [2, max_len] int64, position_ids [2, max_len] int64.
    std::vector<int64_t> attn_mask(2 * max_len, 0);
    std::vector<int64_t> pos_ids(2 * max_len, 0);

    auto row_ids = [&](int32_t b, int32_t c) -> int64_t * {
      return input_ids.data() + (b * C + c) * max_len;
    };

    // ---- cond (batch 0): [style | text | ref_codes | target_mask] --------
    int32_t cond_audio_start = n_style + n_text;  // start of ref_codes
    int32_t cond_target_start = cond_audio_start + ref_tok_len;
    for (int32_t c = 0; c < C; ++c) {
      int64_t *dst = row_ids(0, c);
      std::copy(style_ids.begin(), style_ids.end(), dst);
      std::copy(text_ids.begin(), text_ids.end(), dst + n_style);
      // ref codes: [C, T_ref] row c into positions [cond_audio_start ..)
      std::copy(ref_codes.begin() + c * ref_tok_len,
                ref_codes.begin() + (c + 1) * ref_tok_len,
                dst + cond_audio_start);
      // target positions already mask_id.
    }
    for (int32_t i = 0; i < c_len; ++i) {
      audio_mask[i] = (i >= cond_audio_start) ? 1 : 0;
      attn_mask[i] = 1;
      pos_ids[i] = i;
    }

    // ---- uncond (batch 1): only the target mask region -------------------
    int32_t uncond_off = max_len;  // stride to batch-1 row in flat arrays
    for (int32_t c = 0; c < C; ++c) {
      int64_t *dst = row_ids(1, c);
      // First u_len positions already mask_id; rest already mask_id (unused).
      (void)dst;
    }
    for (int32_t i = 0; i < u_len; ++i) {
      audio_mask[uncond_off + i] = 1;
      attn_mask[uncond_off + i] = 1;
      pos_ids[uncond_off + i] = i;
    }

    // --- 5. Iterative decoding schedule ------------------------------------
    std::vector<float> ts(num_steps + 1);
    for (int32_t i = 0; i <= num_steps; ++i) {
      float t = static_cast<float>(i) / num_steps;
      ts[i] = t_shift * t / (1.0f + (t_shift - 1.0f) * t);
    }
    int32_t total_mask = num_target_tokens * C;
    std::vector<int32_t> schedule(num_steps);
    {
      int32_t rem = total_mask;
      for (int32_t s = 0; s < num_steps; ++s) {
        int32_t k;
        if (s == num_steps - 1) {
          k = rem;
        } else {
          k = static_cast<int32_t>(
              std::ceil(total_mask * (ts[s + 1] - ts[s])));
          k = std::min(k, rem);
        }
        schedule[s] = k;
        rem -= k;
      }
    }

    // --- 6. MaskGIT loop ---------------------------------------------------
    const int32_t V = m.audio_vocab_size;
    for (int32_t step = 0; step < num_steps; ++step) {
      int32_t k = schedule[step];
      if (k <= 0) continue;

      // Refresh cond target region and uncond region from current tokens.
      for (int32_t c = 0; c < C; ++c) {
        std::copy(tokens.begin() + c * num_target_tokens,
                  tokens.begin() + (c + 1) * num_target_tokens,
                  row_ids(0, c) + cond_target_start);
        std::copy(tokens.begin() + c * num_target_tokens,
                  tokens.begin() + (c + 1) * num_target_tokens,
                  row_ids(1, c));
      }

      Ort::Value logits = model_->RunLM(
          MakeI64Tensor(input_ids, {2, C, max_len}),
          MakeBoolTensor(audio_mask, {2, max_len}),
          MakeI64Tensor(attn_mask, {2, max_len}),
          MakeI64Tensor(pos_ids, {2, max_len}));

      const float *lp = logits.GetTensorData<float>();
      auto lshape = logits.GetTensorTypeAndShapeInfo().GetShape();
      // Expect [2, C, max_len, V]
      int64_t stride_b = C * max_len * V;
      int64_t stride_c = max_len * V;
      int64_t stride_s = V;

      // For each still-masked position (c, t) compute confidence + best id
      // via classifier-free guidance. Stash predictions in a flat vector.
      std::vector<float> conf(C * num_target_tokens,
                              -std::numeric_limits<float>::infinity());
      std::vector<int64_t> pred(C * num_target_tokens, 0);

      for (int32_t c = 0; c < C; ++c) {
        for (int32_t t = 0; t < num_target_tokens; ++t) {
          int32_t idx = c * num_target_tokens + t;
          if (tokens[idx] != mask_id) continue;

          const float *c_row =
              lp + 0 * stride_b + c * stride_c +
              (cond_target_start + t) * stride_s;
          const float *u_row =
              lp + 1 * stride_b + c * stride_c + t * stride_s;

          std::vector<float> log_probs =
              GuidedLogProbs(c_row, u_row, V, guidance_scale);
          log_probs[mask_id] = -std::numeric_limits<float>::infinity();

          int64_t best = 0;
          float best_lp = -std::numeric_limits<float>::infinity();
          for (int32_t v = 0; v < V; ++v) {
            if (log_probs[v] > best_lp) {
              best_lp = log_probs[v];
              best = v;
            }
          }
          pred[idx] = best;
          conf[idx] = best_lp - layer_penalty * static_cast<float>(c);
        }
      }

      // Perturb confidences with annealed Gumbel noise for stochastic
      // position selection (MaskGIT-style). The noise scale decays linearly
      // to zero at the final step, so early passes explore and late passes
      // commit deterministically to the highest-confidence positions.
      if (position_temperature > 0.0f) {
        float anneal = 1.0f - static_cast<float>(step) / num_steps;
        float scale = position_temperature * anneal;
        if (scale > 0.0f) {
          for (int32_t idx = 0; idx < C * num_target_tokens; ++idx) {
            if (tokens[idx] != mask_id) continue;
            // Gumbel(0,1) = -log(-log(U)), U in (0,1).
            float u = uni(rng);
            float g = -std::log(-std::log(u));
            conf[idx] += scale * g;
          }
        }
      }

      // Top-k unmask by (possibly perturbed) confidence.
      std::vector<int32_t> order(C * num_target_tokens);
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(
          order.begin(), order.begin() + std::min<int32_t>(k, order.size()),
          order.end(),
          [&](int32_t a, int32_t b) { return conf[a] > conf[b]; });
      for (int32_t i = 0; i < k && i < static_cast<int32_t>(order.size());
           ++i) {
        int32_t idx = order[i];
        if (tokens[idx] == mask_id) tokens[idx] = pred[idx];
      }

      // MaskGIT progress tick: no audio yet, but let UI callers show a
      // progress bar and give them a chance to abort. Cap at
      // kMaskGitProgressFrac so the chunked-decode phase can fill the
      // remaining fraction as real samples arrive.
      if (callback) {
        constexpr float kMaskGitProgressFrac = 0.9f;
        float p = kMaskGitProgressFrac *
                  (static_cast<float>(step + 1) / num_steps);
        if (!callback(nullptr, 0, p)) {
          if (config_.model.debug) {
            SHERPA_ONNX_LOGE(
                "omnivoice: callback returned 0 during MaskGIT step %d; "
                "aborting",
                step);
          }
          return {};
        }
      }
    }

    // --- 7. Decode 8-codebook tokens back into a waveform ------------------
    Ort::Value pcm_out;
    {
      std::vector<int64_t> codes_flat(1 * C * num_target_tokens);
      std::copy(tokens.begin(), tokens.end(), codes_flat.begin());
      Ort::Value codes = MakeI64Tensor(codes_flat, {1, C, num_target_tokens});
      pcm_out = model_->DecodeCodes(std::move(codes));
    }

    auto pshape = pcm_out.GetTensorTypeAndShapeInfo().GetShape();
    // Expect [1, 1, N] or [1, N].
    int64_t n_samples = pshape.back();
    const float *psrc = pcm_out.GetTensorData<float>();

    GeneratedAudio ans;
    ans.sample_rate = m.sample_rate;
    ans.samples.assign(psrc, psrc + n_samples);

    // Chunked delivery. The Higgs codec is a single ONNX graph so we cannot
    // start emitting audio *before* it finishes, but streaming the finished
    // PCM in ~500 ms chunks still lets the caller start playback ~one chunk
    // after decode ends instead of buffering the entire clip. Override via
    // extra "chunk_ms".
    if (callback && n_samples > 0) {
      int32_t chunk_ms = config.GetExtraInt("chunk_ms", 500);
      if (chunk_ms < 20) chunk_ms = 20;  // sub-20ms just wastes callbacks
      int64_t chunk_samples =
          static_cast<int64_t>(m.sample_rate) * chunk_ms / 1000;
      if (chunk_samples < 1) chunk_samples = 1;

      constexpr float kMaskGitProgressFrac = 0.9f;
      constexpr float kDecodeFrac = 1.0f - kMaskGitProgressFrac;

      for (int64_t off = 0; off < n_samples; off += chunk_samples) {
        int64_t end = std::min(off + chunk_samples, n_samples);
        float p = kMaskGitProgressFrac +
                  kDecodeFrac *
                      (static_cast<float>(end) / static_cast<float>(n_samples));
        if (!callback(ans.samples.data() + off,
                      static_cast<int32_t>(end - off), p)) {
          // Caller asked to stop mid-stream. Return what we already produced
          // (up to the end of the last delivered chunk).
          if (config_.model.debug) {
            SHERPA_ONNX_LOGE(
                "omnivoice: callback returned 0 during chunked decode at "
                "%.1f%%; truncating output",
                p * 100.0f);
          }
          ans.samples.resize(end);
          return ans;
        }
      }
    }
    return ans;
  }

 private:
  // FNV-1a 64-bit over sample rate + count + raw PCM bytes. Fast enough
  // (~1 GB/s) that hashing 30 s of 24 kHz float is <3 ms, negligible next
  // to the codec forward it protects.
  static uint64_t HashPcm(const float *data, int32_t n, int32_t sr) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&](const void *p, size_t bytes) {
      const uint8_t *b = static_cast<const uint8_t *>(p);
      for (size_t i = 0; i < bytes; ++i) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
      }
    };
    mix(&sr, sizeof(sr));
    mix(&n, sizeof(n));
    if (n > 0) mix(data, sizeof(float) * static_cast<size_t>(n));
    return h;
  }

  static std::string Strip(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
  }

  static std::vector<float> Resample(const float *in, int32_t n, int32_t sr_in,
                                     int32_t sr_out) {
    if (sr_in == sr_out) return std::vector<float>(in, in + n);
    float cutoff = std::min(sr_in, sr_out) * 0.475f;
    LinearResample r(sr_in, sr_out, cutoff, 6);
    std::vector<float> out;
    r.Resample(in, n, true, &out);
    return out;
  }

  static Ort::Value MakePcmTensor(std::vector<float> &pcm) {
    auto mem =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, 1, static_cast<int64_t>(pcm.size())};
    return Ort::Value::CreateTensor<float>(mem, pcm.data(), pcm.size(),
                                           shape.data(), shape.size());
  }

  static Ort::Value MakeI64Tensor(std::vector<int64_t> &v,
                                  std::vector<int64_t> shape) {
    auto mem =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<int64_t>(mem, v.data(), v.size(),
                                             shape.data(), shape.size());
  }

  static Ort::Value MakeBoolTensor(std::vector<uint8_t> &v,
                                   std::vector<int64_t> shape) {
    auto mem =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    // ORT bool tensors are laid out as bytes (0/1).
    return Ort::Value::CreateTensor(
        mem, v.data(), v.size(), shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
  }

  // Compute log-softmax of (c + s * (c - u)) row-wise. Returns V-long vector.
  static std::vector<float> GuidedLogProbs(const float *c_logits,
                                           const float *u_logits, int32_t V,
                                           float scale) {
    std::vector<float> out(V);
    if (scale == 0.0f) {
      // Simple log-softmax of c.
      float mx = c_logits[0];
      for (int32_t v = 1; v < V; ++v) mx = std::max(mx, c_logits[v]);
      double sum = 0.0;
      for (int32_t v = 0; v < V; ++v) {
        out[v] = c_logits[v] - mx;
        sum += std::exp(out[v]);
      }
      float lse = static_cast<float>(std::log(sum));
      for (int32_t v = 0; v < V; ++v) out[v] -= lse;
      return out;
    }

    // Log-softmax of c and of u independently.
    auto log_softmax_into = [&](const float *src, std::vector<float> &dst) {
      float mx = src[0];
      for (int32_t v = 1; v < V; ++v) mx = std::max(mx, src[v]);
      double sum = 0.0;
      for (int32_t v = 0; v < V; ++v) {
        dst[v] = src[v] - mx;
        sum += std::exp(dst[v]);
      }
      float lse = static_cast<float>(std::log(sum));
      for (int32_t v = 0; v < V; ++v) dst[v] -= lse;
    };

    std::vector<float> lc(V), lu(V);
    log_softmax_into(c_logits, lc);
    log_softmax_into(u_logits, lu);

    // combined = lc + scale * (lc - lu); then log-softmax again.
    float mx = -std::numeric_limits<float>::infinity();
    for (int32_t v = 0; v < V; ++v) {
      out[v] = lc[v] + scale * (lc[v] - lu[v]);
      if (out[v] > mx) mx = out[v];
    }
    double sum = 0.0;
    for (int32_t v = 0; v < V; ++v) {
      out[v] -= mx;
      sum += std::exp(out[v]);
    }
    float lse = static_cast<float>(std::log(sum));
    for (int32_t v = 0; v < V; ++v) out[v] -= lse;
    return out;
  }

  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsOmnivoiceModel> model_;
  std::unique_ptr<QwenAsrTokenizer> tokenizer_;

  // Single-entry reference-audio cache. Voice-cloning workloads typically
  // reuse the same reference clip for many synthesis calls, so a one-slot
  // last-value cache eliminates the codec-encoder pass on all repeats while
  // costing effectively no memory (~a few KB per entry).
  mutable std::mutex ref_cache_mutex_;
  mutable uint64_t ref_cache_key_ = 0;
  mutable bool ref_cache_valid_ = false;
  mutable std::vector<int64_t> ref_cache_codes_;
  mutable int32_t ref_cache_tok_len_ = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_IMPL_H_
