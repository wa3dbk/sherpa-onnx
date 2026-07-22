# sherpa-onnx / OmniVoice branch

<div align="center">

**Zero-shot voice-cloning text-to-speech, on-device, in every language sherpa-onnx already speaks in.**

</div>

This branch of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) adds
**OmniVoice** — a MaskGIT-style non-autoregressive TTS backend built from the
Qwen3-0.6B language model and the Higgs-Audio-V2 neural codec. Give it a
3-10 s clip of anyone speaking and any text, and it will say the text in
that person's voice, offline, on CPU or GPU, at 24 kHz.

Everything else sherpa-onnx does — ASR, VAD, speaker diarization, other TTS
backends — is unchanged and untouched.

---

## What this branch adds on top of upstream sherpa-onnx

| Area | Item | Where |
|---|---|---|
| Core | OmniVoice C++ backend (Qwen3-0.6B LM + Higgs codec, MaskGIT decode) | `sherpa-onnx/csrc/offline-tts-omnivoice-*` |
| Core | KV-cache-aware LM fast path (~1.5-2× speedup) | `--omnivoice-prefix-model`, `--omnivoice-target-model` |
| Core | Sentence / character text chunking with cross-chunk audio join | `--split-strategy=sentence\|chars` |
| Core | Cross-process reference-audio codec cache on disk | `~/.cache/sherpa-onnx-omnivoice/` (or `$XDG_CACHE_HOME`) |
| Core | Deterministic mode (byte-identical output for a fixed seed) | `--seed=<N>` |
| CLI | Live streaming of decoded PCM to stdout | `--output-stream=1` (pipe to `ffplay -f f32le -ar 24000 -i -`) |
| Bindings | Full parity across C / C++ / Python / Go / Kotlin (JNI) / Swift / .NET / Node.js | `c-api/`, `cxx-api-examples/`, `python-api-examples/`, `scripts/go/`, `sherpa-onnx/{jni,kotlin-api}/`, `swift-api-examples/`, `scripts/dotnet/`, `nodejs-addon-examples/` |
| Tools | Model export scripts (LM, cached LM pair, Higgs codec, tokenizer, int8/fp16 quant) | `scripts/omnivoice/` |
| Tools | End-to-end bundle builder + downloadable tarball layout | `scripts/omnivoice/build_bundle.sh` |
| Tools | Benchmark harness (CPU vs GPU, cached vs uncached, chunked vs not) | `scripts/omnivoice/bench.sh` + `bench.py` |
| CI | GitHub Actions job building the TTS-enabled binary and running unit + splitter tests on every push | `.github/workflows/omnivoice.yaml` |
| Docs | Full user-facing docs for every flag, chunking strategy, streaming, determinism, and troubleshooting | [`OMNIVOICE.md`](./OMNIVOICE.md) |

## Quick start

```bash
# 1. Get a bundle (build your own or, once published, download a tarball)
cd scripts/omnivoice && ./build_bundle.sh
# -> sherpa-onnx-omnivoice-<DATE>/  and .tar.bz2

# 2. Build sherpa-onnx with TTS enabled
cmake -B build -DSHERPA_ONNX_ENABLE_TTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. Clone a voice
./build/bin/sherpa-onnx-offline-tts \
  --omnivoice-model=./sherpa-onnx-omnivoice-<DATE>/omnivoice.onnx \
  --omnivoice-codec-encoder=./sherpa-onnx-omnivoice-<DATE>/higgs_codec_encoder.onnx \
  --omnivoice-codec-decoder=./sherpa-onnx-omnivoice-<DATE>/higgs_codec_decoder.onnx \
  --omnivoice-tokenizer-dir=./sherpa-onnx-omnivoice-<DATE>/tokenizer \
  --reference-audio=./sherpa-onnx-omnivoice-<DATE>/test_wavs/ref.wav \
  --reference-text="$(cat ./sherpa-onnx-omnivoice-<DATE>/test_wavs/ref.txt)" \
  --output-filename=./out.wav \
  "Hello, this is a test in the reference speaker's voice."
```

See [`OMNIVOICE.md`](./OMNIVOICE.md) for the full flag reference, chunking
tradeoffs, streaming recipes, determinism guarantees, and troubleshooting
table.

## Streaming to speakers

```bash
./build/bin/sherpa-onnx-offline-tts \
  --omnivoice-model=... --omnivoice-codec-encoder=... --omnivoice-codec-decoder=... \
  --omnivoice-tokenizer-dir=... --reference-audio=... --reference-text="..." \
  --split-strategy=sentence \
  --output-stream=1 \
  "First sentence plays as soon as it decodes. Second one arrives next. Third." \
  | ffplay -f f32le -ar 24000 -i - -nodisp -autoexit
```

With `--split-strategy=sentence`, first-audio-out latency is proportional
to the first sentence, not the whole paragraph.

## Language bindings

Every binding gained an `OfflineTtsOmnivoiceModelConfig` struct and the
matching CLI/example. Run scripts under each language dir download a
bundle and run a "clone this voice" demo.

| Language | Example | Runner |
|---|---|---|
| Python | `python-api-examples/omnivoice-tts.py` | `python omnivoice-tts.py --bundle-dir=...` |
| C++ | `cxx-api-examples/omnivoice-tts-cxx-api.cc` | built with the rest |
| Go | `go-api-examples/omnivoice-tts/main.go` | `go run . --bundle-dir=...` |
| Kotlin (JNI) | `kotlin-api-examples/test_omnivoice_tts.kt` | `kotlin-api-examples/run-tts-omnivoice.sh` |
| Swift | `swift-api-examples/tts-omnivoice.swift` | `swift-api-examples/run-tts-omnivoice.sh` |
| .NET | `dotnet-examples/omnivoice-tts/Program.cs` | `dotnet-examples/omnivoice-tts/run.sh` |
| Node.js | `nodejs-addon-examples/test-offline-tts-omnivoice.js` | `node test-offline-tts-omnivoice.js --bundle-dir=...` |

## Performance

Fill this in by running `scripts/omnivoice/bench.sh` against your bundle.
The script sweeps CPU-vs-GPU, cached-LM-vs-not, and single-pass vs
sentence-chunked, and prints a markdown row per configuration.

<!-- BEGIN PERF TABLE -->

| Provider | Cached LM | Split strategy | Text length (chars) | Steps | Threads | Latency (s) | Audio (s) | RTF |
|---|---|---|---|---|---|---|---|---|
| _(run `scripts/omnivoice/bench.sh` and paste rows here)_ | | | | | | | | |

<!-- END PERF TABLE -->

How to read RTF: values below 1.0 mean faster than real time (a 10 s clip
generates in <10 s). CPU RTF on an Apple M2 Pro is typically ~0.4-0.6 with
the cached LM path; a modest CUDA GPU sits around ~0.1.

## Sizing

The default fp32 export weighs in around **2.4 GB** for the LM. Use the
quantization script:

```bash
python scripts/omnivoice/export_omnivoice_quantized.py \
    --input-onnx ./sherpa-onnx-omnivoice-<DATE>/omnivoice.onnx \
    --dtype int8      # or fp16
```

fp16 halves the file size with no measurable quality loss on English.
int8 (dynamic quantization of MatMul/Gemm) roughly quarters it at a small
cost in prosody.

## Reference-audio caching

Encoding the reference clip through the Higgs codec costs the same as ~1
MaskGIT step. For voice-cloning workloads that reuse the same voice
across many synthesis calls (podcast narration, dialogue generation), the
backend caches the encoded codec tokens keyed by the SHA-1 of the raw
PCM:

- **In-process cache**: one-slot LRU that avoids the encoder forward on
  repeat calls with the same reference in the same process. Always on.
- **Cross-process cache**: results are also serialized to
  `$XDG_CACHE_HOME/sherpa-onnx-omnivoice/` (defaults to
  `~/.cache/sherpa-onnx-omnivoice/`). Every new process that sees the
  same clip skips the codec encoder entirely on the first call. Disable
  by setting `SHERPA_ONNX_OMNIVOICE_CACHE_DIR=off` or by passing
  `--extra="cache_reference_audio=0"`.

Cache entries are small (a few KB each), keyed by content hash + codec
metadata, and survive across sherpa-onnx upgrades that don't change the
codec ONNX.

## Real intra-sentence streaming (experimental)

Within a single MaskGIT pass, audio can only start streaming after the
codec-decoder forward completes — the shipped `higgs_codec_decoder.onnx`
is a single graph over the full token sequence. There is a plumbing hook
for a *chunked* decoder: if the bundle contains
`higgs_codec_decoder_streaming.onnx` accepting a token-window input, the
C++ side will use it to emit PCM every `stream_window_tokens` frames.

The export script scaffold for such a decoder lives at
`scripts/omnivoice/export_higgs_codec_streaming_onnx.py`. Whether it
produces a numerically clean output depends on the codec architecture
supporting a receptive-field-bounded forward — a model-side question, not
a sherpa-onnx one. Treat this path as experimental until a validated
streaming decoder is published.

## WebAssembly

`build-wasm-simd-omnivoice.sh` produces a wasm module exposing the
OmniVoice backend for browser demos. The build assumes an int8 or fp16
bundle — the fp32 LM will not fit in a 32-bit wasm heap. Use the
quantization script above first.

## Android / iOS

The JNI struct (`sherpa-onnx/jni/offline-tts.cc`) and Swift factory
(`swift-api-examples/SherpaOnnx.swift`) already know how to configure
OmniVoice. To ship it inside the Android or iOS TTS engine apps in this
repo, you need to (a) add `OmniVoiceConfig` fields to the Kotlin
`OfflineTts` UI layer for Android, or the `SherpaOnnxOfflineTtsWrapper`
init path for iOS, and (b) bundle a quantized model in the app's assets.
See the "TODO: mobile app integration" markers in the respective
`android/` and `ios-swift/` directories for the concrete extension
points.

## Building on the upstream sherpa-onnx

This branch is fully rebase-compatible with upstream sherpa-onnx `master`.
All non-OmniVoice functionality — ASR (Zipformer, Paraformer, Whisper,
Moonshine, Canary, Qwen3-ASR, FireRed, …), VAD, diarization, keyword
spotting, other TTS backends (VITS, Piper, Kokoro, Matcha, Kitten,
ZipVoice, Pocket, Supertonic) — works exactly as documented upstream.

For anything outside OmniVoice, refer to the upstream README:
<https://github.com/k2-fsa/sherpa-onnx>

## License

Same as upstream sherpa-onnx: Apache 2.0. The OmniVoice model itself and
the Higgs-Audio-V2 codec have their own licenses — check the respective
HuggingFace model cards before use.

## Citation

If you use this branch, please cite both the base project and the
upstream OmniVoice repo:

```
@software{sherpa_onnx,
  author = {sherpa-onnx contributors},
  title = {sherpa-onnx: On-device speech AI},
  url = {https://github.com/k2-fsa/sherpa-onnx}
}
@software{omnivoice,
  author = {k2-fsa},
  title = {OmniVoice: Zero-shot voice-cloning TTS},
  url = {https://github.com/k2-fsa/OmniVoice}
}
```
