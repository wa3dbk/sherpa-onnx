// cxx-api-examples/omnivoice-tts-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

// Example: zero-shot voice cloning with OmniVoice via the sherpa-onnx CXX API.
//
// Requires an OmniVoice bundle (see scripts/omnivoice/build_bundle.sh) and a
// short (3-10 s) reference clip of the target speaker together with its
// transcript.
//
// clang-format off
/*
Usage
  ./omnivoice-tts-cxx-api \
      /path/to/sherpa-onnx-omnivoice-YYYY-MM-DD \
      /path/to/reference.wav \
      "Verbatim transcript of the reference clip." \
      "Text to speak in the reference speaker's voice."

Optional 5th argument: path to omnivoice_prefix.onnx (must sit next to
omnivoice_target.onnx). When present, the KV-cache fast path is enabled.
*/
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#include "sherpa-onnx/c-api/cxx-api.h"

static int32_t ProgressCallback(const float *samples, int32_t num_samples,
                                float progress, void *arg) {
  (void)samples;
  (void)arg;
  if (num_samples == 0) {
    fprintf(stderr, "MaskGIT progress: %.1f%%\r", progress * 100.0f);
  } else {
    fprintf(stderr, "Decoded %d samples (%.1f%%)\n", num_samples,
            progress * 100.0f);
  }
  // Return 1 to continue, 0 to abort.
  return 1;
}

int32_t main(int32_t argc, char *argv[]) {
  if (argc < 5) {
    fprintf(stderr,
            "Usage: %s <bundle-dir> <reference.wav> <reference-text> "
            "<text-to-synthesize> [prefix.onnx]\n",
            argv[0]);
    return 1;
  }

  const std::string bundle = argv[1];
  const std::string ref_wav = argv[2];
  const std::string ref_text = argv[3];
  const std::string text = argv[4];
  const std::string prefix_model = (argc >= 6) ? argv[5] : "";

  using namespace sherpa_onnx::cxx;  // NOLINT

  OfflineTtsConfig config;
  config.model.omnivoice.model = bundle + "/omnivoice.onnx";
  config.model.omnivoice.codec_encoder = bundle + "/higgs_codec_encoder.onnx";
  config.model.omnivoice.codec_decoder = bundle + "/higgs_codec_decoder.onnx";
  config.model.omnivoice.tokenizer_dir = bundle + "/tokenizer";

  if (!prefix_model.empty()) {
    config.model.omnivoice.prefix_model = prefix_model;
    // Sibling target graph, by convention.
    config.model.omnivoice.target_model =
        prefix_model.substr(0, prefix_model.find_last_of('/') + 1) +
        "omnivoice_target.onnx";
  }

  config.model.num_threads = 2;
  config.model.debug = 0;
  config.model.provider = "cpu";

  auto tts = OfflineTts::Create(config);

  GenerationConfig gen_config;
  gen_config.reference_text = ref_text;

  Wave wave = ReadWave(ref_wav);
  if (wave.samples.empty()) {
    fprintf(stderr, "Failed to read reference wav: %s\n", ref_wav.c_str());
    return 1;
  }
  gen_config.reference_audio = std::move(wave.samples);
  gen_config.reference_sample_rate = wave.sample_rate;

  // Optional: per-request overrides via the `extra` map. See OMNIVOICE.md for
  // the full list. Uncomment to try:
  // gen_config.extra["language"] = "en";
  // gen_config.extra["denoise"] = "1";
  // gen_config.extra["seed"] = "42";

  GeneratedAudio audio = tts.Generate(text, gen_config, ProgressCallback);
  if (audio.samples.empty()) {
    fprintf(stderr, "\nGeneration failed.\n");
    return 1;
  }

  const std::string out = "./generated-omnivoice-cxx.wav";
  WriteWave(out, {audio.samples, audio.sample_rate});

  fprintf(stderr, "\nInput text: %s\n", text.c_str());
  fprintf(stderr, "Saved to: %s (%.2f s at %d Hz)\n", out.c_str(),
          static_cast<float>(audio.samples.size()) / audio.sample_rate,
          audio.sample_rate);
  return 0;
}
