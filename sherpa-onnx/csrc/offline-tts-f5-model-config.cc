// sherpa-onnx/csrc/offline-tts-f5-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-f5-model-config.h"

#include <sstream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsF5ModelConfig::Register(ParseOptions *po) {
  po->Register("f5-transformer", &transformer,
               "Path to the F5-TTS DiT (flow-matching Diffusion Transformer) "
               "ONNX file.");
  po->Register("f5-vocoder", &vocoder,
               "Path to the Vocos mel-to-waveform vocoder ONNX.");
  po->Register("f5-tokens", &tokens,
               "Path to tokens.txt matching the F5-TTS tokenizer.");
  po->Register("f5-data-dir", &data_dir,
               "Path to the espeak-ng data dir "
               "(phontab, phonindex, phondata, intonations).");
  po->Register("f5-lexicon", &lexicon,
               "Optional path to lexicon.txt for Chinese phonemization.");
  po->Register("f5-num-steps", &num_steps,
               "Euler ODE steps for the flow-matching decoder. "
               "Default 32.");
  po->Register("f5-guidance-scale", &guidance_scale,
               "Classifier-free guidance weight. 0 disables CFG. "
               "Default 2.0.");
  po->Register("f5-sway-coef", &sway_coef,
               "Sway-sampling coefficient (F5-TTS time-schedule warp). "
               "-1.0 = F5 default, 0.0 = uniform.");
  po->Register("f5-target-rms", &target_rms,
               "Target RMS for reference audio normalization. "
               "Default 0.1.");
  po->Register("f5-seed", &seed,
               "RNG seed for initial gaussian noise. "
               "<0 => nondeterministic.");
}

bool OfflineTtsF5ModelConfig::Validate() const {
  if (transformer.empty()) {
    SHERPA_ONNX_LOGE("Please provide --f5-transformer");
    return false;
  }
  if (!FileExists(transformer)) {
    SHERPA_ONNX_LOGE("--f5-transformer: '%s' does not exist",
                     transformer.c_str());
    return false;
  }

  if (vocoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --f5-vocoder");
    return false;
  }
  if (!FileExists(vocoder)) {
    SHERPA_ONNX_LOGE("--f5-vocoder: '%s' does not exist", vocoder.c_str());
    return false;
  }

  if (tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --f5-tokens");
    return false;
  }
  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("--f5-tokens: '%s' does not exist", tokens.c_str());
    return false;
  }

  if (!data_dir.empty()) {
    std::vector<std::string> required_files = {
        "phontab",
        "phonindex",
        "phondata",
        "intonations",
    };
    for (const auto &f : required_files) {
      if (!FileExists(data_dir + "/" + f)) {
        SHERPA_ONNX_LOGE("'%s/%s' does not exist. Please check --f5-data-dir",
                         data_dir.c_str(), f.c_str());
        return false;
      }
    }
  }

  if (!lexicon.empty() && !FileExists(lexicon)) {
    SHERPA_ONNX_LOGE("--f5-lexicon: '%s' does not exist", lexicon.c_str());
    return false;
  }

  if (num_steps < 1) {
    SHERPA_ONNX_LOGE("--f5-num-steps must be >= 1. Given: %d", num_steps);
    return false;
  }

  if (guidance_scale < 0) {
    SHERPA_ONNX_LOGE("--f5-guidance-scale must be >= 0. Given: %f",
                     guidance_scale);
    return false;
  }

  if (target_rms <= 0) {
    SHERPA_ONNX_LOGE("--f5-target-rms must be positive. Given: %f",
                     target_rms);
    return false;
  }

  return true;
}

std::string OfflineTtsF5ModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsF5ModelConfig(";
  os << "transformer=\"" << transformer << "\", ";
  os << "vocoder=\"" << vocoder << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "data_dir=\"" << data_dir << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "num_steps=" << num_steps << ", ";
  os << "guidance_scale=" << guidance_scale << ", ";
  os << "sway_coef=" << sway_coef << ", ";
  os << "target_rms=" << target_rms << ", ";
  os << "seed=" << seed << ")";

  return os.str();
}

}  // namespace sherpa_onnx
