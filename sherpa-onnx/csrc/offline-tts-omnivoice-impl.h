// sherpa-onnx/csrc/offline-tts-omnivoice-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_IMPL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-omnivoice-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-omnivoice-model.h"
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

    const auto &m = model_->GetMetaData();
    const auto &mc = config_.model.omnivoice;

    int32_t num_steps = config.GetExtraInt("num_steps", mc.num_steps);
    float guidance_scale =
        config.GetExtraFloat("guidance_scale", mc.guidance_scale);
    float t_shift = config.GetExtraFloat("t_shift", mc.t_shift);
    float layer_penalty =
        config.GetExtraFloat("layer_penalty", mc.layer_penalty_factor);
    std::string language = config.GetExtraString("language", "None");
    std::string instruct = config.GetExtraString("instruct", "None");
    bool denoise = config.GetExtraInt("denoise", 1) != 0;

    // --- 1. Encode reference audio into codec tokens ------------------------
    std::vector<int64_t> ref_codes;  // flattened [C, T_ref] row-major
    int32_t ref_tok_len = 0;
    {
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
      const int64_t *src = codes.GetTensorData<int64_t>();
      ref_codes.assign(src, src + m.num_codebook * ref_tok_len);
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
        float ref_w = TextWeight(config.reference_text);
        float tgt_w = TextWeight(text);
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

      // Top-k unmask by confidence.
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

    if (callback) callback(ans.samples.data(), n_samples, 1.0f);
    return ans;
  }

 private:
  // Sum of per-character phonetic weights, mirroring the OmniVoice
  // RuleDurationEstimator. Simplified port: precise weights for
  // Latin / digits / punctuation / whitespace / marks; script-family
  // approximations for common non-Latin ranges (CJK, kana, hangul, Arabic,
  // Cyrillic, Greek, Indic, Thai/Lao). Unknown code points default to 1.0.
  static float TextWeight(const std::string &s) {
    static constexpr float kLatin = 1.0f;
    static constexpr float kSpace = 0.2f;
    static constexpr float kPunct = 0.5f;
    static constexpr float kDigit = 3.5f;
    static constexpr float kCjk = 3.0f;
    static constexpr float kKana = 2.2f;
    static constexpr float kHangul = 2.5f;
    static constexpr float kArabic = 1.5f;
    static constexpr float kIndic = 1.8f;
    static constexpr float kThai = 1.5f;

    float total = 0.0f;
    for (size_t i = 0; i < s.size();) {
      unsigned char c0 = static_cast<unsigned char>(s[i]);
      uint32_t cp;
      int32_t adv;
      if (c0 < 0x80) {
        cp = c0;
        adv = 1;
      } else if ((c0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
        cp = ((c0 & 0x1F) << 6) |
             (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        adv = 2;
      } else if ((c0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
        cp = ((c0 & 0x0F) << 12) |
             ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        adv = 3;
      } else if ((c0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
        cp = ((c0 & 0x07) << 18) |
             ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
             ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        adv = 4;
      } else {
        cp = c0;
        adv = 1;
      }
      i += adv;

      float w;
      if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) {
        w = kLatin;
      } else if (cp >= '0' && cp <= '9') {
        w = kDigit;
      } else if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') {
        w = kSpace;
      } else if (cp < 0x80) {
        // ASCII punctuation / symbols.
        w = kPunct;
      } else if (cp == 0x0640) {
        w = 0.0f;  // Arabic Tatweel
      } else if (cp >= 0x0300 && cp <= 0x036F) {
        w = 0.0f;  // Combining diacritical marks
      } else if (cp >= 0x0600 && cp <= 0x06FF) {
        w = kArabic;
      } else if (cp >= 0x0370 && cp <= 0x03FF) {
        w = kLatin;  // Greek
      } else if (cp >= 0x0400 && cp <= 0x04FF) {
        w = kLatin;  // Cyrillic
      } else if (cp >= 0x0900 && cp <= 0x0DFF) {
        w = kIndic;
      } else if (cp >= 0x0E00 && cp <= 0x0EFF) {
        w = kThai;
      } else if (cp >= 0x3040 && cp <= 0x30FF) {
        w = kKana;
      } else if (cp >= 0xAC00 && cp <= 0xD7AF) {
        w = kHangul;
      } else if ((cp >= 0x4E00 && cp <= 0x9FFF) ||
                 (cp >= 0x3400 && cp <= 0x4DBF) || cp >= 0x20000) {
        w = kCjk;
      } else {
        w = kLatin;
      }
      total += w;
    }
    return total;
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
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_IMPL_H_
