// sherpa-onnx/csrc/offline-tts-indextts2-frontend.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_FRONTEND_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_FRONTEND_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sherpa_onnx {

// IndexTTS-2 text frontend:
//   1. For each char, if it's in the CJK range and present in the
//      pinyin lookup table, replace with its first pinyin reading and
//      pad with spaces so the BPE sees a word boundary.
//   2. BPE-encode the resulting mixed string via the merges table.
//   3. Return int64 token ids.
//
// Unknown chars pass through as raw UTF-8 into the BPE; anything the
// BPE can't split emits `<unk>` (id from tokens.txt).
class OfflineTtsIndexTts2Frontend {
 public:
  OfflineTtsIndexTts2Frontend(const std::string &tokens_path,
                              const std::string &merges_path,
                              const std::string &pinyin_table_path,
                              int32_t unk_token_id = 0);

  template <typename Manager>
  OfflineTtsIndexTts2Frontend(Manager *mgr, const std::string &tokens_path,
                              const std::string &merges_path,
                              const std::string &pinyin_table_path,
                              int32_t unk_token_id = 0);

  ~OfflineTtsIndexTts2Frontend();

  std::vector<int64_t> Encode(const std::string &text) const;

  // Exposed for the emotion-text encoder path.
  std::vector<int64_t> EncodeRawBpe(const std::string &text) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_INDEXTTS2_FRONTEND_H_
