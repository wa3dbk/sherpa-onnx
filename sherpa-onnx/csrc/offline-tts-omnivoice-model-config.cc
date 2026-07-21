// sherpa-onnx/csrc/offline-tts-omnivoice-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-omnivoice-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsOmnivoiceModelConfig::Register(ParseOptions *po) {
  po->Register("omnivoice-model", &model,
               "Path to OmniVoice language model ONNX "
               "(with .onnx_data alongside)");
  po->Register("omnivoice-codec-encoder", &codec_encoder,
               "Path to Higgs-Audio-V2 codec encoder ONNX");
  po->Register("omnivoice-codec-decoder", &codec_decoder,
               "Path to Higgs-Audio-V2 codec decoder ONNX");
  po->Register("omnivoice-tokenizer-dir", &tokenizer_dir,
               "Directory with vocab.json/merges.txt/tokenizer_config.json "
               "for the Qwen3 tokenizer used by OmniVoice");
  po->Register("omnivoice-prefix-model", &prefix_model,
               "Optional: KV-cache prefix graph "
               "(omnivoice_prefix.onnx). Enables ~1.5-2x speedup when set "
               "together with --omnivoice-target-model.");
  po->Register("omnivoice-target-model", &target_model,
               "Optional: KV-cache target graph "
               "(omnivoice_target.onnx). Enables ~1.5-2x speedup when set "
               "together with --omnivoice-prefix-model.");
  po->Register("omnivoice-num-steps", &num_steps,
               "MaskGIT-style denoising steps (default: 32)");
  po->Register("omnivoice-t-shift", &t_shift,
               "Time-step schedule shift (default: 0.1)");
  po->Register("omnivoice-guidance-scale", &guidance_scale,
               "Classifier-free guidance scale; 0 disables CFG (default: 2.0)");
  po->Register("omnivoice-layer-penalty-factor", &layer_penalty_factor,
               "Confidence penalty for later codebook layers (default: 5.0)");
  po->Register("omnivoice-position-temperature", &position_temperature,
               "Gumbel-noise temperature for MaskGIT position sampling; "
               "annealed linearly to 0 by the last step. 0 = deterministic "
               "(default: 5.0)");
  po->Register("omnivoice-seed", &seed,
               "RNG seed for Gumbel sampling. <0 = nondeterministic "
               "(default: -1)");
}

bool OfflineTtsOmnivoiceModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --omnivoice-model");
    return false;
  }
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("--omnivoice-model: '%s' does not exist", model.c_str());
    return false;
  }

  if (codec_encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --omnivoice-codec-encoder");
    return false;
  }
  if (!FileExists(codec_encoder)) {
    SHERPA_ONNX_LOGE("--omnivoice-codec-encoder: '%s' does not exist",
                     codec_encoder.c_str());
    return false;
  }

  if (codec_decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --omnivoice-codec-decoder");
    return false;
  }
  if (!FileExists(codec_decoder)) {
    SHERPA_ONNX_LOGE("--omnivoice-codec-decoder: '%s' does not exist",
                     codec_decoder.c_str());
    return false;
  }

  if (tokenizer_dir.empty()) {
    SHERPA_ONNX_LOGE("Please provide --omnivoice-tokenizer-dir");
    return false;
  }
  for (const auto &f : {"vocab.json", "merges.txt", "tokenizer_config.json"}) {
    if (!FileExists(tokenizer_dir + "/" + f)) {
      SHERPA_ONNX_LOGE("'%s/%s' does not exist", tokenizer_dir.c_str(), f);
      return false;
    }
  }

  // Cached-LM pair is optional but must be provided as a pair.
  if (prefix_model.empty() != target_model.empty()) {
    SHERPA_ONNX_LOGE(
        "--omnivoice-prefix-model and --omnivoice-target-model must be "
        "provided together (or neither).");
    return false;
  }
  if (!prefix_model.empty() && !FileExists(prefix_model)) {
    SHERPA_ONNX_LOGE("--omnivoice-prefix-model: '%s' does not exist",
                     prefix_model.c_str());
    return false;
  }
  if (!target_model.empty() && !FileExists(target_model)) {
    SHERPA_ONNX_LOGE("--omnivoice-target-model: '%s' does not exist",
                     target_model.c_str());
    return false;
  }

  if (num_steps < 1) {
    SHERPA_ONNX_LOGE("--omnivoice-num-steps must be >= 1. Given: %d",
                     num_steps);
    return false;
  }
  if (t_shift <= 0) {
    SHERPA_ONNX_LOGE("--omnivoice-t-shift must be > 0. Given: %f", t_shift);
    return false;
  }
  if (guidance_scale < 0) {
    SHERPA_ONNX_LOGE("--omnivoice-guidance-scale must be >= 0. Given: %f",
                     guidance_scale);
    return false;
  }
  if (position_temperature < 0) {
    SHERPA_ONNX_LOGE(
        "--omnivoice-position-temperature must be >= 0. Given: %f",
        position_temperature);
    return false;
  }

  return true;
}

std::string OfflineTtsOmnivoiceModelConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineTtsOmnivoiceModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "codec_encoder=\"" << codec_encoder << "\", ";
  os << "codec_decoder=\"" << codec_decoder << "\", ";
  os << "tokenizer_dir=\"" << tokenizer_dir << "\", ";
  os << "prefix_model=\"" << prefix_model << "\", ";
  os << "target_model=\"" << target_model << "\", ";
  os << "num_steps=" << num_steps << ", ";
  os << "t_shift=" << t_shift << ", ";
  os << "guidance_scale=" << guidance_scale << ", ";
  os << "layer_penalty_factor=" << layer_penalty_factor << ", ";
  os << "position_temperature=" << position_temperature << ", ";
  os << "seed=" << seed << ")";
  return os.str();
}

}  // namespace sherpa_onnx
