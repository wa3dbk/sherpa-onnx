// sherpa-onnx/csrc/offline-tts-omnivoice-ref-cache.h
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder
//
// Cross-process cache for OmniVoice reference-audio codec encodings.
//
// Encoding a reference wav through the Higgs codec encoder is the same
// order of magnitude as one MaskGIT step (~50-200 ms on CPU). For
// voice-cloning workloads that reuse the same reference across many
// synthesis calls -- or across many short-lived processes, e.g. a serving
// worker restart, a CLI invoked from a script loop -- avoiding that
// encoder forward is meaningful.
//
// Layout on disk (default ~/.cache/sherpa-onnx-omnivoice/, override via
// $SHERPA_ONNX_OMNIVOICE_CACHE_DIR, disable by setting that env var to
// "off"):
//
//   ref-<hex>.v1
//     [8-byte magic "SXOMNIV1"]
//     [4-byte int32 num_codebook]
//     [4-byte int32 num_frames]
//     [num_codebook * num_frames * 8 bytes int64 codes, row-major]
//
// The <hex> component encodes (sample_rate, num_samples,
// FNV-1a-hash-of-PCM, num_codebook). Keying on all four makes the cache
// self-invalidating when the codec ONNX changes shape.
//
// This is a header-only helper so the impl.h can include it directly
// without adding a new .cc to the build.

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_REF_CACHE_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_REF_CACHE_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace sherpa_onnx {

namespace omnivoice_ref_cache_internal {

inline bool MkdirP(const std::string &path) {
  if (path.empty()) return false;
#if defined(_WIN32)
  // Windows: attempt to create; ignore "already exists".
  int rc = _mkdir(path.c_str());
  return rc == 0 || errno == EEXIST;
#else
  // POSIX: create parents one at a time.
  std::string acc;
  for (size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/') {
      if (!acc.empty()) {
        int rc = mkdir(acc.c_str(), 0755);
        if (rc != 0 && errno != EEXIST) return false;
      }
    }
    if (i < path.size()) acc.push_back(path[i]);
  }
  return true;
#endif
}

inline std::string DefaultCacheDir() {
  const char *env = std::getenv("SHERPA_ONNX_OMNIVOICE_CACHE_DIR");
  if (env && *env) return env;
  const char *xdg = std::getenv("XDG_CACHE_HOME");
  if (xdg && *xdg) {
    return std::string(xdg) + "/sherpa-onnx-omnivoice";
  }
  const char *home = std::getenv("HOME");
  if (home && *home) {
    return std::string(home) + "/.cache/sherpa-onnx-omnivoice";
  }
  return "";
}

inline std::string HexHash(uint64_t pcm_hash, int32_t sr, int32_t n_samples,
                           int32_t num_codebook) {
  // Compose a 24-hex-char id: 16 for the PCM hash + 8 for the
  // (sr, n_samples, num_codebook) tuple. Enough to avoid collisions in
  // practice while keeping the filename short.
  uint64_t meta = 0xcbf29ce484222325ULL;
  auto mix = [&](uint32_t v) {
    for (int b = 0; b < 4; ++b) {
      meta ^= static_cast<uint8_t>(v >> (b * 8));
      meta *= 0x100000001b3ULL;
    }
  };
  mix(static_cast<uint32_t>(sr));
  mix(static_cast<uint32_t>(n_samples));
  mix(static_cast<uint32_t>(num_codebook));

  static const char *kHex = "0123456789abcdef";
  std::string out(24, '0');
  for (int i = 0; i < 16; ++i) {
    out[i] = kHex[(pcm_hash >> ((15 - i) * 4)) & 0xF];
  }
  for (int i = 0; i < 8; ++i) {
    out[16 + i] = kHex[(meta >> ((7 - i) * 4)) & 0xF];
  }
  return out;
}

}  // namespace omnivoice_ref_cache_internal

class OmnivoiceRefDiskCache {
 public:
  // dir=="off" disables. dir=="" auto-picks $SHERPA_ONNX_OMNIVOICE_CACHE_DIR,
  // then $XDG_CACHE_HOME/sherpa-onnx-omnivoice, then
  // $HOME/.cache/sherpa-onnx-omnivoice.
  explicit OmnivoiceRefDiskCache(const std::string &dir = "") {
    namespace ns = omnivoice_ref_cache_internal;
    std::string chosen = dir.empty() ? ns::DefaultCacheDir() : dir;
    if (chosen == "off" || chosen.empty()) {
      enabled_ = false;
      return;
    }
    if (!ns::MkdirP(chosen)) {
      // Silently disable if the target isn't writable. Not being able to
      // cache is never a hard failure.
      enabled_ = false;
      return;
    }
    dir_ = chosen;
    enabled_ = true;
  }

  bool enabled() const { return enabled_; }
  const std::string &dir() const { return dir_; }

  // Look up and load into `out_codes` (flattened row-major [C, T]).
  // Returns the frame count on hit, or -1 on miss / IO error.
  int32_t TryLoad(uint64_t pcm_hash, int32_t sr, int32_t n_samples,
                  int32_t num_codebook,
                  std::vector<int64_t> *out_codes) const {
    if (!enabled_ || !out_codes) return -1;
    std::string path = PathFor(pcm_hash, sr, n_samples, num_codebook);
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return -1;

    char magic[8] = {0};
    int32_t nc = 0, nt = 0;
    if (std::fread(magic, 1, 8, f) != 8 ||
        std::memcmp(magic, "SXOMNIV1", 8) != 0 ||
        std::fread(&nc, sizeof(nc), 1, f) != 1 ||
        std::fread(&nt, sizeof(nt), 1, f) != 1 || nc != num_codebook ||
        nt <= 0) {
      std::fclose(f);
      return -1;
    }
    out_codes->assign(static_cast<size_t>(nc) * nt, 0);
    size_t got =
        std::fread(out_codes->data(), sizeof(int64_t), out_codes->size(), f);
    std::fclose(f);
    if (got != out_codes->size()) {
      out_codes->clear();
      return -1;
    }
    return nt;
  }

  // Write. Best-effort: failures do not raise. Uses a rename dance so a
  // partially-written file never satisfies a lookup.
  void TryStore(uint64_t pcm_hash, int32_t sr, int32_t n_samples,
                int32_t num_codebook, const std::vector<int64_t> &codes,
                int32_t num_frames) const {
    if (!enabled_ || codes.empty() || num_frames <= 0) return;
    std::string final_path = PathFor(pcm_hash, sr, n_samples, num_codebook);
    std::string tmp = final_path + ".tmp";
    FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) return;

    bool ok = std::fwrite("SXOMNIV1", 1, 8, f) == 8 &&
              std::fwrite(&num_codebook, sizeof(num_codebook), 1, f) == 1 &&
              std::fwrite(&num_frames, sizeof(num_frames), 1, f) == 1 &&
              std::fwrite(codes.data(), sizeof(int64_t), codes.size(), f) ==
                  codes.size();
    std::fclose(f);
    if (!ok) {
      std::remove(tmp.c_str());
      return;
    }
    // On POSIX rename is atomic within a filesystem. On Windows we
    // remove-then-rename because rename over an existing file fails.
#if defined(_WIN32)
    std::remove(final_path.c_str());
#endif
    if (std::rename(tmp.c_str(), final_path.c_str()) != 0) {
      std::remove(tmp.c_str());
    }
  }

 private:
  std::string PathFor(uint64_t pcm_hash, int32_t sr, int32_t n_samples,
                      int32_t num_codebook) const {
    return dir_ + "/ref-" +
           omnivoice_ref_cache_internal::HexHash(pcm_hash, sr, n_samples,
                                                 num_codebook) +
           ".v1";
  }

  bool enabled_ = false;
  std::string dir_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_OMNIVOICE_REF_CACHE_H_
