<!--
Copyright (c)  2026  Xiaomi Corporation
                2026  Waad Ben Kheder
-->

# sherpa-onnx-zero-shot-tts

<div align="center">

**A fork of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) focused on modern zero-shot voice-cloning TTS, with first-class C++, Python, and Node.js APIs.**

</div>

Upstream sherpa-onnx is the reference on-device audio-AI runtime — ASR, VAD,
diarization, KWS, and a wide TTS surface. That project ships and supports
bindings for a dozen languages and every platform under the sun.

**This fork narrows the scope.** It focuses on one job — running the current
generation of *zero-shot voice-cloning* TTS models on CPU and GPU — and one
API surface: **C++, Python, and Node.js**. Give it a 3-10 s reference clip and
any text; get back that text spoken in that voice, offline.

Everything upstream provides is still in the tree. If you need Go, Kotlin,
Swift, .NET, Dart, HarmonyOS, Android/iOS apps, WebAssembly, or any of the
non-TTS pipelines, those work — but the docs and examples we actively maintain
here cover C++, Python, and Node.js. For anything else, follow the upstream
docs at <https://k2-fsa.github.io/sherpa/onnx/>.

---

## Model status

| Model | Architecture | License | Status | Docs |
|---|---|---|---|---|
| **OmniVoice** | Qwen3-0.6B LM + Higgs-Audio-V2 codec, MaskGIT NAR | Apache 2.0 | ✅ Merged | [OMNIVOICE.md](OMNIVOICE.md) |
| **F5-TTS** | Flow-matching DiT + Vocos vocoder | MIT | 🚧 In progress | — |
| **IndexTTS-2** | GPT-style LM + BigVGAN | Apache 2.0 | 📋 Planned | — |
| **ChatterBox / Turbo** | LLaMA-backbone + S3 codec + HiFi-GAN | MIT | 📋 Planned | — |
| everything from upstream (Piper VITS, Matcha, Kokoro, Kitten, Zipvoice, Pocket, Supertonic) | various | various | ✅ Inherited, unchanged | upstream docs |

Roadmap and selection rationale live in [tests.md](tests.md) and the model
comparison notes at the bottom of this file.

---

## Quick start

Prereqs: a C++17 compiler, CMake ≥ 3.13, Python ≥ 3.8. GPU is optional
(CUDA 11.x or 12.x for the ONNX Runtime CUDA EP).

### Build the C++ core

```bash
git clone https://github.com/wa3dbk/sherpa-onnx.git
cd sherpa-onnx
mkdir build && cd build
cmake -DSHERPA_ONNX_ENABLE_TTS=ON ..
make -j4
```

Binaries land in `build/bin/`. The one you'll use most is
`sherpa-onnx-offline-tts`.

### Get a model bundle

For OmniVoice today:

```bash
scripts/omnivoice/build_bundle.sh              # downloads + assembles ~2 GB
export BUNDLE_DIR=./sherpa-onnx-omnivoice-en
```

For F5-TTS / IndexTTS-2 / ChatterBox: bundle scripts land as those models are
integrated (see [Model status](#model-status)).

### Try it — C++ (CLI)

```bash
./build/bin/sherpa-onnx-offline-tts \
  --omnivoice-model=$BUNDLE_DIR/omnivoice.onnx \
  --omnivoice-codec-encoder=$BUNDLE_DIR/higgs_codec_encoder.onnx \
  --omnivoice-codec-decoder=$BUNDLE_DIR/higgs_codec_decoder.onnx \
  --omnivoice-tokenizer-dir=$BUNDLE_DIR/tokenizer \
  --reference-audio=$BUNDLE_DIR/test_wavs/ref.wav \
  --reference-text="$(cat $BUNDLE_DIR/test_wavs/ref.txt)" \
  --output-filename=hello.wav \
  "Hello world, this is a cloned voice."
```

### Try it — Python

```bash
pip install -e .   # from repo root, builds and installs the C++ core as a wheel
python3 python-api-examples/omnivoice-mic-loopback.py \
  --bundle-dir $BUNDLE_DIR \
  "Hello world, this is a cloned voice."
```

The `omnivoice-mic-loopback.py` example records 5 s from your mic and
synthesizes in that captured voice — the fastest way to hear the model.

### Try it — Node.js

```bash
cd scripts/node-addon-api && npm install && npm run build
cd ../../nodejs-addon-examples
BUNDLE_DIR=../$BUNDLE_DIR node test_tts_non_streaming_omnivoice.js
```

### Streaming to your speakers

```bash
SHERPA_ONNX_OMNIVOICE_BUNDLE_DIR=$BUNDLE_DIR \
  scripts/omnivoice/stream_to_ffplay.sh \
  "Long text. Multiple sentences. Ffplay starts playing before decoding finishes."
```

---

## Feature matrix

| Feature | C++ | Python | Node.js |
|---|---|---|---|
| Basic synthesis | ✅ | ✅ | ✅ |
| Zero-shot cloning from reference wav + text | ✅ | ✅ | ✅ |
| Reference-audio resampling (16 / 22 / 44 / 48 kHz → 24 kHz) | ✅ | ✅ | ✅ |
| In-process reference codec cache | ✅ | ✅ | ✅ |
| Cross-process disk cache (`~/.cache/sherpa-onnx-omnivoice/`) | ✅ | ✅ | ✅ |
| Sentence / char / no-op text chunking | ✅ | ✅ | ✅ |
| Live PCM streaming to stdout (`--output-stream`) | ✅ | via CLI | via CLI |
| Deterministic mode (`--seed`) | ✅ | ✅ | ✅ |
| KV-cache LM fast path (prefix / target ONNX) | ✅ | ✅ | ✅ |
| CPU EP | ✅ | ✅ | ✅ |
| CUDA EP | ✅ | ✅ | ✅ |
| CoreML / DirectML | untested | untested | untested |

---

## Performance

Run once on your hardware, then paste the tables between the markers.

```bash
BENCH_PROVIDERS=cpu scripts/omnivoice/bench.sh | tee bench-cpu.md
python3 scripts/omnivoice/bench.py --providers cpu --num-threads 4
```

<!-- BEGIN PERF TABLE -->
<!-- END PERF TABLE -->

Fp32 bundles are large. For deployment on constrained hardware or in
WebAssembly, quantize first:

```bash
python3 scripts/omnivoice/export_omnivoice_quantized.py \
  --in ./sherpa-onnx-omnivoice-en --out ./sherpa-onnx-omnivoice-en-int8 --dtype int8
```

Typical size ratios: fp32 → fp16 ≈ 50 %, fp32 → int8 (MatMul/Gemm only) ≈ 30 %.

---

## Building from source (details)

### C++ core

```bash
mkdir build && cd build
cmake \
  -DSHERPA_ONNX_ENABLE_TTS=ON \
  -DSHERPA_ONNX_ENABLE_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  ..
make -j$(nproc)
ctest -R offline-tts       # unit tests for the TTS backends
```

Optional flags: `-DSHERPA_ONNX_ENABLE_GPU=ON` for CUDA,
`-DBUILD_SHARED_LIBS=OFF` for a self-contained binary, `-DSHERPA_ONNX_ENABLE_JNI=ON`
if you also want to build the Kotlin bindings (see upstream docs).

### Python

```bash
pip install -e .
```

This shells out to CMake with the TTS backend enabled by default in this
fork. Editable install so `import sherpa_onnx` picks up your local changes.

### Node.js addon

```bash
cd scripts/node-addon-api
npm install
npm run build
npm link
# then, in your app:  npm link sherpa-onnx-node
```

### WebAssembly (OmniVoice)

```bash
source /path/to/emsdk_env.sh
./build-wasm-simd-omnivoice.sh
```

Requires an int8 bundle — fp32 does not fit in the wasm32 heap. Output lands
in `build-wasm-simd-omnivoice/install/bin/wasm/tts/`.

---

## Relationship to upstream

This fork is **not** a hard fork. The `upstream` branch tracks
`k2-fsa/sherpa-onnx:master` unmodified; new upstream commits are pulled into
`upstream` and merged into `main` on a rolling basis.

```
upstream ──▶ (periodic ff-only pull from k2-fsa/sherpa-onnx)
                   │
                   ▼ (merged as needed)
main ────────────────────────────────────────▶  ← this fork's default
```

**Where to file issues:**
- Zero-shot TTS backends (OmniVoice, F5-TTS, IndexTTS-2, ChatterBox), the
  C++ / Python / Node.js APIs added by this fork, or any of the scripts
  under `scripts/omnivoice/` → open an issue **here**.
- Everything else (upstream ASR, VAD, TTS backends inherited unchanged,
  mobile / Go / Swift / .NET / Dart / HarmonyOS bindings, upstream CI) →
  open an issue at [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/issues).

**Contributing:** PRs against `main`, please. Rebase on top of `upstream` if
you're touching files that upstream also modifies frequently — makes the
next sync cheaper.

---

## Design notes

Longer notes and background live next to the code:

- [OMNIVOICE.md](OMNIVOICE.md) — MaskGIT decode loop, CFG, reference caching,
  chunking, streaming, determinism, sample-rate handling.
- [tests.md](tests.md) — 12-stage validation checklist to run on your cluster
  after each build.
- `sherpa-onnx/csrc/offline-tts-*-model.cc` — per-backend implementations.
  New backends should mirror these.

## Model selection — why these three

Full ranking of the 15 candidates evaluated is in the internal notes; the
short version:

- **F5-TTS** — MIT, best-in-class quality on public benchmarks, flow-matching
  DiT + Vocos vocoder. Architecturally close to the existing Zipvoice
  integration; low-medium C++ port effort.
- **IndexTTS-2** — Apache, big in the CN community, GPT-style LM + BigVGAN.
  Direct comparison point against OmniVoice's Qwen3-based LM.
- **ChatterBox (+Turbo)** — MIT, ~500 M, currently trending on HuggingFace,
  adds emotion-exaggeration axis. LM+codec architecture reuses OmniVoice
  scaffolding heavily.

Dropped: XTTS-v2 (CPML), MARS5 (AGPL), Qwen3-TTS (API-only),
MOSS / Echo / VoxCPM / NeuTTS-Nano (small user base), Higgs v3 (codec-only
overlap), CosyVoice 3 (deferred — bigger port), Sesame CSM-1B (deferred —
conversational, different API shape).

---

## License

Apache 2.0 — same as upstream sherpa-onnx.

## Citations

If you use this fork in research, please cite both upstream sherpa-onnx and
the specific TTS model you used. For OmniVoice:

```bibtex
@misc{sherpa_onnx,
  title  = {sherpa-onnx},
  author = {The k2-fsa team},
  year   = {2023},
  url    = {https://github.com/k2-fsa/sherpa-onnx}
}

@misc{omnivoice,
  title  = {OmniVoice: Qwen3 + Higgs-Audio-V2 MaskGIT TTS},
  year   = {2025},
  url    = {https://huggingface.co/BosonAI/higgs-audio-v2}
}
```
