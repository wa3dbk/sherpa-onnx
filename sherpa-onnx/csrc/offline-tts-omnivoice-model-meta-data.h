// sherpa-onnx/csrc/offline-tts-omnivoice-model-meta-data.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_MODEL_META_DATA_H_

#include <cstdint>

namespace sherpa_onnx {

struct OfflineTtsOmnivoiceModelMetaData {
  // Constants baked by the OmniVoice / Higgs-Audio-V2 pair.
  int32_t num_codebook = 8;
  int32_t audio_vocab_size = 1025;  // 1024 codes + 1 mask
  int32_t audio_mask_id = 1024;

  // Higgs-Audio-V2 codec I/O.
  int32_t sample_rate = 24000;
  int32_t frame_rate = 25;   // audio tokens per second
  int32_t hop_length = 960;  // sample_rate / frame_rate
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_MODEL_META_DATA_H_
