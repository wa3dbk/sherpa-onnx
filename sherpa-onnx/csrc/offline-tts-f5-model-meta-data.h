// sherpa-onnx/csrc/offline-tts-f5-model-meta-data.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_MODEL_META_DATA_H_

#include <cstdint>
#include <string>

namespace sherpa_onnx {

// Fixed constants for the F5-TTS + Vocos pair used by this fork.
// If a future bundle diverges (e.g. a 22 kHz variant), extend
// export_f5_onnx.py + export_vocos_onnx.py to embed these as ONNX
// metadata and read them at load time instead.
struct OfflineTtsF5ModelMetaData {
  int32_t version = 1;
  int32_t feat_dim = 100;      // number of mel bins
  int32_t sample_rate = 24000;
  int32_t n_fft = 1024;
  int32_t hop_length = 256;
  int32_t window_length = 1024;
  int32_t num_mels = 100;

  // Rank-2 vs rank-3 text ids. F5-TTS's DiT expects (B, T) longs.
  int32_t use_espeak = 1;

  // Vocab size at export time (informational; not used to reshape).
  int32_t vocab_size = 2545;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_F5_MODEL_META_DATA_H_
