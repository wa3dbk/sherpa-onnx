// sherpa-onnx/csrc/offline-tts-indextts2-model-meta-data.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_MODEL_META_DATA_H_

#include <cstdint>

namespace sherpa_onnx {

struct OfflineTtsIndexTts2ModelMetaData {
  int32_t version = 1;
  int32_t sample_rate = 24000;
  int32_t vocab_size = 32000;

  int32_t bos_token_id = 1;
  int32_t eos_token_id = 2;
  int32_t pad_token_id = 0;

  int32_t speaker_embed_dim = 512;
  int32_t emotion_embed_dim = 512;

  int32_t kv_num_layers = 16;
  int32_t kv_num_heads = 16;
  int32_t kv_head_dim = 64;

  int32_t max_audio_tokens = 2000;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_MODEL_META_DATA_H_
