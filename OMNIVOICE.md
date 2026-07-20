# OmniVoice TTS support for sherpa-onnx

This branch adds a new TTS backend to sherpa-onnx: **OmniVoice**, a
voice-cloning text-to-speech system built from:

- **OmniVoice LM** (Qwen3-0.6B) — <https://huggingface.co/k2-fsa/OmniVoice>
- **Higgs-Audio-V2 codec** — <https://huggingface.co/eustlb/higgs-audio-v2-tokenizer>

Decoding is MaskGIT-style non-autoregressive across 8 codebooks. Output is
24 kHz mono. A reference audio clip and its transcript are required at
inference time to clone the speaker's timbre.

The backend plugs into the existing `sherpa-onnx-offline-tts` binary; no new
binary is added. When `--omnivoice-model` is set, dispatch picks OmniVoice
automatically.

## 1. Prepare the model bundle

You need three ONNX files, a tokenizer directory, and (optionally) a
reference wav + transcript.

### Option A — pre-built bundle (recommended)

Download the tarball from the sherpa-onnx releases page and extract it.
The bundle already contains everything below. Skip to §2.

### Option B — build the bundle yourself

Requires a Python env with the OmniVoice training repo installed
(<https://github.com/k2-fsa/OmniVoice>) plus `onnx`, `onnxruntime`, and
`transformers>=4.44`.

```bash
cd scripts/omnivoice
./build_bundle.sh
# produces sherpa-onnx-omnivoice-<DATE>/ and sherpa-onnx-omnivoice-<DATE>.tar.bz2
```

Environment overrides: `HF_REPO`, `HIGGS_REPO`, `DATE`, `OUT`, `DEVICE`.

The script runs three exporters:

| Script | Produces |
| --- | --- |
| `export_omnivoice_onnx.py` | `omnivoice.onnx` (+ `omnivoice.onnx_data`) — the LM |
| `export_higgs_codec_onnx.py` | `higgs_codec_encoder.onnx`, `higgs_codec_decoder.onnx` |
| `export_tokenizer.py` | `tokenizer/vocab.json`, `tokenizer/merges.txt`, configs |

### Note on the tokenizer

The upstream `k2-fsa/OmniVoice` HF repo ships **only** the fast
`tokenizer.json`. sherpa-onnx's C++ Qwen tokenizer needs the slow
`vocab.json` + `merges.txt` format. `export_tokenizer.py` handles this
via `AutoTokenizer.from_pretrained(...).save_pretrained(..., legacy_format=True)`.

## 2. Build sherpa-onnx

Same as the standard build. No new dependencies.

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSHERPA_ONNX_ENABLE_TTS=ON ..
make -j
```

The `sherpa-onnx-offline-tts` binary now accepts OmniVoice flags.

## 3. Run

```bash
./bin/sherpa-onnx-offline-tts \
  --omnivoice-model=./bundle/omnivoice.onnx \
  --omnivoice-codec-encoder=./bundle/higgs_codec_encoder.onnx \
  --omnivoice-codec-decoder=./bundle/higgs_codec_decoder.onnx \
  --omnivoice-tokenizer-dir=./bundle/tokenizer \
  --reference-audio=./bundle/test_wavs/ref.wav \
  --reference-text="$(cat ./bundle/test_wavs/ref.txt)" \
  --output-filename=./out.wav \
  "Text to synthesize in the reference speaker's voice."
```

### Generation knobs

Defaults match the OmniVoice Python reference and generally do not need
tuning:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--omnivoice-num-steps` | 32 | MaskGIT denoising steps |
| `--omnivoice-t-shift` | 0.1 | Time-schedule shift (smaller = more mass early) |
| `--omnivoice-guidance-scale` | 2.0 | Classifier-free guidance; 0 disables CFG (halves compute) |
| `--omnivoice-layer-penalty-factor` | 5.0 | Delays later codebook layers when picking unmask positions |

## 4. What was added / modified

### New files

```
sherpa-onnx/csrc/offline-tts-omnivoice-model-config.{h,cc}   CLI flags
sherpa-onnx/csrc/offline-tts-omnivoice-model-meta-data.h     shape/vocab constants
sherpa-onnx/csrc/offline-tts-omnivoice-model.{h,cc}          three-session PIMPL wrapper
                                                             (RunLM, EncodeAudio, DecodeCodes)
sherpa-onnx/csrc/offline-tts-omnivoice-impl.h                MaskGIT generation loop with
                                                             optional classifier-free guidance
                                                             and rule-based duration estimator
scripts/omnivoice/                                           export scripts + build_bundle.sh
```

### Modified files

```
sherpa-onnx/csrc/CMakeLists.txt                new sources
sherpa-onnx/csrc/offline-tts-model-config.{h,cc}   embeds OmniVoice sub-config + Register/Validate
sherpa-onnx/csrc/offline-tts-impl.cc           dispatch: pick OmniVoice when --omnivoice-model is set
sherpa-onnx/csrc/sherpa-onnx-offline-tts.cc    treat OmniVoice like ZipVoice for
                                               --reference-audio / --reference-text
```

### Notes on the port

- Duration estimation mirrors OmniVoice's `RuleDurationEstimator`: UTF-8-aware
  per-script character weights (Latin 1.0, CJK 3.0, kana 2.2, hangul 2.5,
  Arabic 1.5, Indic 1.8, digits 3.5, punct 0.5, …), scaled by the ratio of
  reference-text weight to reference-token count, with a low-count boost
  below 50 tokens.
- CFG is implemented by batching cond + uncond into one LM forward and
  combining logits with `guidance_scale`.
- The HF-style 2-D padding `attention_mask` is expanded to the 4-D block
  mask inside the exported ONNX wrapper — the C++ side only ships the 2-D
  version.
