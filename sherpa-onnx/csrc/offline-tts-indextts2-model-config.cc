// sherpa-onnx/csrc/offline-tts-indextts2-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-indextts2-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsIndexTts2ModelConfig::Register(ParseOptions *po) {
  po->Register("indextts2-lm", &lm,
               "Path to the IndexTTS-2 LM (dual-mode prefix+step) ONNX.");
  po->Register("indextts2-voice-encoder", &voice_encoder,
               "Path to the IndexTTS-2 voice encoder ONNX.");
  po->Register("indextts2-emotion-encoder", &emotion_encoder,
               "Path to the IndexTTS-2 emotion encoder ONNX.");
  po->Register("indextts2-emotion-text-encoder", &emotion_text_encoder,
               "Path to the IndexTTS-2 emotion-text encoder ONNX.");
  po->Register("indextts2-vocoder", &vocoder,
               "Path to the IndexTTS-2 neural vocoder ONNX.");
  po->Register("indextts2-tokens", &tokens,
               "Path to tokens.txt (BPE vocab).");
  po->Register("indextts2-merges", &merges,
               "Path to merges.txt (BPE merges).");
  po->Register("indextts2-pinyin-table", &pinyin_table,
               "Path to pinyin_table.txt (offline CJK -> pinyin lookup).");
  po->Register("indextts2-max-audio-tokens", &max_audio_tokens,
               "Hard cap on generated audio tokens per sentence. "
               "Default 2000.");
  po->Register("indextts2-top-k", &top_k,
               "Top-k sampling parameter for the AR loop. Default 30.");
  po->Register("indextts2-top-p", &top_p,
               "Top-p (nucleus) sampling parameter. Default 0.8. "
               "Range (0, 1].");
  po->Register("indextts2-temperature", &temperature,
               "Sampling temperature. Default 0.8. Must be >= 0. "
               "0 = argmax.");
  po->Register("indextts2-seed", &seed,
               "RNG seed for the AR sampler. <0 => nondeterministic.");
}

bool OfflineTtsIndexTts2ModelConfig::Validate() const {
  auto require = [](const std::string &path, const char *flag) {
    if (path.empty()) {
      SHERPA_ONNX_LOGE("Please provide --%s", flag);
      return false;
    }
    if (!FileExists(path)) {
      SHERPA_ONNX_LOGE("--%s: '%s' does not exist", flag, path.c_str());
      return false;
    }
    return true;
  };

  if (!require(lm, "indextts2-lm")) return false;
  if (!require(voice_encoder, "indextts2-voice-encoder")) return false;
  if (!require(emotion_encoder, "indextts2-emotion-encoder")) return false;
  if (!require(emotion_text_encoder, "indextts2-emotion-text-encoder"))
    return false;
  if (!require(vocoder, "indextts2-vocoder")) return false;
  if (!require(tokens, "indextts2-tokens")) return false;
  if (!require(merges, "indextts2-merges")) return false;
  if (!require(pinyin_table, "indextts2-pinyin-table")) return false;

  if (max_audio_tokens <= 0) {
    SHERPA_ONNX_LOGE("--indextts2-max-audio-tokens must be > 0. Given: %d",
                     max_audio_tokens);
    return false;
  }
  if (top_k < 0) {
    SHERPA_ONNX_LOGE("--indextts2-top-k must be >= 0. Given: %d", top_k);
    return false;
  }
  if (top_p <= 0.0f || top_p > 1.0f) {
    SHERPA_ONNX_LOGE("--indextts2-top-p must be in (0, 1]. Given: %f", top_p);
    return false;
  }
  if (temperature < 0.0f) {
    SHERPA_ONNX_LOGE("--indextts2-temperature must be >= 0. Given: %f",
                     temperature);
    return false;
  }
  return true;
}

std::string OfflineTtsIndexTts2ModelConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineTtsIndexTts2ModelConfig(";
  os << "lm=\"" << lm << "\", ";
  os << "voice_encoder=\"" << voice_encoder << "\", ";
  os << "emotion_encoder=\"" << emotion_encoder << "\", ";
  os << "emotion_text_encoder=\"" << emotion_text_encoder << "\", ";
  os << "vocoder=\"" << vocoder << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "merges=\"" << merges << "\", ";
  os << "pinyin_table=\"" << pinyin_table << "\", ";
  os << "max_audio_tokens=" << max_audio_tokens << ", ";
  os << "top_k=" << top_k << ", ";
  os << "top_p=" << top_p << ", ";
  os << "temperature=" << temperature << ", ";
  os << "seed=" << seed << ")";
  return os.str();
}

}  // namespace sherpa_onnx
