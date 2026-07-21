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
| `--omnivoice-position-temperature` | 5.0 | Gumbel-noise temperature for stochastic position sampling; annealed to 0 by the last step. Set 0 for deterministic decoding |
| `--omnivoice-seed` | -1 | RNG seed for Gumbel sampling. Use a fixed non-negative value for reproducible output |
| `--omnivoice-prefix-model` | *(empty)* | Optional path to `omnivoice_prefix.onnx`. When set together with `--omnivoice-target-model`, the cond branch of the MaskGIT loop reuses cached prefix K/V across steps for ~1.5-2x speedup. The uncond branch still uses `--omnivoice-model`, which remains required |
| `--omnivoice-target-model` | *(empty)* | Optional path to `omnivoice_target.onnx`; must be provided together with `--omnivoice-prefix-model` |

Build the cached ONNX pair with `WITH_CACHED=1 ./scripts/omnivoice/build_bundle.sh`
(or run `scripts/omnivoice/export_omnivoice_cached_onnx.py` directly).

## 4. Troubleshooting

Enable verbose logging on any run to see the estimated target length, the
tokenized reference / text lengths, and the LM I/O shapes:

```bash
./bin/sherpa-onnx-offline-tts --debug=1 ...
```

The line you want is:

```
omnivoice: ref_tok=125 style=7 text=32 target=124
```

`ref_tok` should be roughly `reference-audio-duration × 25`. `target`
should be roughly `expected-output-duration × 25`. If either is
wildly off, the rest of the output will be too.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Garbled audio after the first ~1 s (`"here is the tex ssssss"`) | Generation defaults out of sync with the training-time config | Confirm defaults: `num_steps=32`, `t_shift=0.1`, `guidance_scale=2.0`, `layer_penalty_factor=5.0`. Do **not** lower `num_steps` below ~16 |
| Output has correct words but 3-4× longer than expected, padded with silence / hisses | `--reference-text` does not match the actual content of `--reference-audio` (duration estimator is scaled by the ref-text:ref-tok ratio) | Transcribe the reference clip accurately, or supply `--extra="duration_sec=3.5"` |
| `"omnivoice: reference_audio is only 0.320 s"` | Reference clip too short for the codec to build a speaker embedding | Use a 3-10 s clip of clean speech |
| `"codec encoder produced 0 reference tokens"` | Reference clip is silent or all-zero | Sanity-check the wav |
| `"target length N tokens exceeds cap"` | Text too long for a single MaskGIT pass | Split input at sentence boundaries and concatenate outputs |
| `"'.../tokenizer/vocab.json' does not exist"` | The upstream HF repo ships only the fast `tokenizer.json`; you skipped the slow-format conversion | Run `scripts/omnivoice/export_tokenizer.py` (or the full `build_bundle.sh`) |
| Cloned voice sounds nothing like the reference | Reference is too noisy, or `--extra="denoise=0"` was passed on a noisy clip | Try `denoise=1` (default) or a cleaner clip |
| Non-English text produces wrong prosody | Language tag defaulting to `"None"` | Pass `--extra="language=zh"` (or `en`, `ja`, …) as expected by the training data |
| Very slow (RTF > 1) | CPU-only, or `num_steps` set high | Build with the CUDA / CoreML EP and/or lower `num_steps` to 16-24 for a small quality hit |

If you file a bug report, please include the full `--debug=1` log and the
output of `sha256sum` on each ONNX file in your bundle so mismatched
exports can be ruled out quickly.

## 5. What was added / modified

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
