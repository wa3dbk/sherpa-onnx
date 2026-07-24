# F5-TTS support for sherpa-onnx

This adds a new zero-shot TTS backend to sherpa-onnx: **F5-TTS**, a
flow-matching Diffusion Transformer with a Vocos vocoder.

- **F5-TTS DiT** — <https://github.com/SWivid/F5-TTS>
- **Vocos (mel → audio, 24 kHz)** — <https://huggingface.co/charactr/vocos-mel-24khz>

Decoding is an Euler ODE loop over the flow-matching field, run
externally in C++ so we can log per-step progress, interrupt, and later
reuse a KV-cache if the export is split. Output is 24 kHz mono. A
reference audio clip and its transcript are required at inference time
to clone the speaker's timbre.

The backend plugs into the existing `sherpa-onnx-offline-tts` binary; no
new binary is added. When `--f5-transformer` is set, dispatch picks
F5-TTS automatically.

## 1. Prepare the model bundle

You need two ONNX files (transformer + vocoder), a tokens list, and an
espeak-ng data directory.

### Option A — pre-built bundle (planned)

A tarball will be published on the sherpa-onnx releases page under the
`tts-models` tag. Extract and skip to §2.

### Option B — build the bundle yourself

Requires a Python env with the F5-TTS training repo installed
(<https://github.com/SWivid/F5-TTS>) plus `onnx`, `onnxruntime`,
`safetensors`, `transformers`, and `vocos`.

```bash
cd scripts/f5-tts
OUT_DIR=./sherpa-onnx-f5-tts-base-24khz bash build_bundle.sh
# produces sherpa-onnx-f5-tts-base-24khz/ with transformer.onnx, vocos.onnx,
# tokens.txt, and espeak-ng-data/
```

Environment overrides: `F5_TTS_CKPT`, `F5_TTS_VARIANT`, `VOCOS_CKPT`,
`OUT_DIR`, `REFERENCE_HF_PATH`, `DTYPE`. See
`scripts/f5-tts/README.md` for the full bundle contract.

The script runs three exporters:

| Script | Produces |
| --- | --- |
| `export_f5_onnx.py` | `transformer.onnx` — the DiT with CFG folded into a batch-of-2 wrapper |
| `export_vocos_onnx.py` | `vocos.onnx` — mel `(1, 100, T)` → audio `(1, T*256)` |
| — (in `build_bundle.sh`) | `tokens.txt` from `f5_tts.model.utils.get_tokenizer(...)` |

### Note on the tokenizer

The reference F5-TTS checkpoints use a pinyin+char tokenizer
(`Emilia_ZH_EN`). The bundle writes `tokens.txt` in the same form and
the C++ side reuses the `MatchaTtsLexicon` scaffold to map text →
token IDs. Espeak-ng data is included so the lookup path works
end-to-end for multilingual inputs; if you retrain with an espeak
phoneme vocabulary, re-export `tokens.txt` accordingly.

## 2. Build sherpa-onnx

Same as the standard build. No new dependencies.

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSHERPA_ONNX_ENABLE_TTS=ON ..
make -j
```

The `sherpa-onnx-offline-tts` binary now accepts F5-TTS flags.

## 3. Run

```bash
./bin/sherpa-onnx-offline-tts \
  --f5-transformer=./bundle/transformer.onnx \
  --f5-vocoder=./bundle/vocos.onnx \
  --f5-tokens=./bundle/tokens.txt \
  --f5-data-dir=./bundle/espeak-ng-data \
  --reference-audio=./bundle/test_wavs/ref.wav \
  --reference-text="$(cat ./bundle/test_wavs/ref.txt)" \
  --output-filename=./out.wav \
  "Text to synthesize in the reference speaker's voice."
```

### Generation knobs

Defaults match the F5-TTS reference implementation and generally do not
need tuning:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--f5-num-steps` | 32 | Euler ODE steps. Lower to 16 for a ~2x speedup with a small quality hit; below ~12 the pronunciation starts to slur. |
| `--f5-guidance-scale` | 2.0 | Classifier-free guidance weight. 0 disables CFG (halves compute); higher values push harder toward the conditioned text but can distort prosody. |
| `--f5-sway-coef` | -1.0 | Timestep-schedule warp coefficient. Negative front-loads (matches F5-TTS paper); 0 gives uniform steps. Endpoints stay pinned at 0/1. |
| `--f5-target-rms` | 0.1 | Reference-clip RMS normalization target. The impl scales the reference up if quieter than this, and de-normalizes the output by the inverse ratio so amplitudes match the reference. |
| `--f5-seed` | -1 | RNG seed for the initial Gaussian noise. Use a fixed non-negative value for reproducible output. |
| `--f5-lexicon` | *(empty)* | Optional custom lexicon file passed to `MatchaTtsLexicon`; usually not needed since the tokens file is authoritative. |

## 3a. Long-form input: chunking

The impl splits input text on punctuation, then merges short pieces and
splits over-long ones. Extras:

- `--extra="max_char_in_sentence=200"` — hard upper bound per chunk.
  Raise for fewer joins (each chunk costs more); lower if the ODE loop
  runs out of memory or you want faster first-audio-out.
- `--extra="min_char_in_sentence=30"` — merge shorter fragments until
  this size, to avoid short-and-choppy generation.

Each chunk gets its own duration estimate from the reference/target
character-length ratio (`target_frames = ref_frames * len(target_text) /
len(reference_text)`). Reference tokenization is cached across chunks,
so per-chunk cost is dominated by the ODE loop and the Vocos forward.

## 3b. Determinism

Initial noise for the ODE loop is drawn from a `std::mt19937`. Passing
`--f5-seed=<N>` (any non-negative integer) makes the whole pipeline
reproducible: two runs with the same bundle, the same text, the same
reference clip, and the same seed produce **byte-identical** WAVs.

Notes:

- `--f5-seed=-1` (the default) reseeds from `std::random_device` each
  request, so consecutive runs vary as intended.
- Determinism also requires `--num-threads` and the ONNX Runtime
  execution provider to be stable across runs. On CPU with a fixed
  thread count this holds. On GPU (`--provider=cuda`) atomic reductions
  in some kernels may perturb the last few bits — determinism is
  best-effort there.
- The RNG feeds *only* the initial noise; guidance, sway, and step size
  are deterministic functions of the config flags.

## 3c. Reference audio sample rate

Anything readable by `sherpa-onnx-offline-tts` works: mono or stereo, 16
kHz, 22.05 kHz, 44.1 kHz, 48 kHz. The impl calls the sherpa-onnx
`LinearResample` to convert to Vocos's native 24 kHz before extracting
the log-mel, so **you do not need to resample yourself**. Keep the clip
in the 3-10 s range for clean speaker capture.

## 4. Python and Node

Both bindings expose the same knobs:

```python
# python-api-examples/f5-tts.py
import sherpa_onnx
cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        f5=sherpa_onnx.OfflineTtsF5ModelConfig(
            transformer="./bundle/transformer.onnx",
            vocoder="./bundle/vocos.onnx",
            tokens="./bundle/tokens.txt",
            data_dir="./bundle/espeak-ng-data",
            num_steps=32,
            guidance_scale=2.0,
            sway_coef=-1.0,
        ),
    )
)
```

```js
// nodejs-addon-examples/test_tts_non_streaming_f5.js
const config = {
  model: {
    f5: {
      transformer: './bundle/transformer.onnx',
      vocoder: './bundle/vocos.onnx',
      tokens: './bundle/tokens.txt',
      dataDir: './bundle/espeak-ng-data',
      numSteps: 32,
      guidanceScale: 2.0,
      swayCoef: -1.0,
    },
  },
};
```

## 5. Troubleshooting

Enable verbose logging on any run to see the mel dims, tokenized
reference / target lengths, and per-chunk progress:

```bash
./bin/sherpa-onnx-offline-tts --debug=1 ...
```

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Output is roughly the right length but sounds noisy / underdriven | `num_steps` too low | Raise `--f5-num-steps` to 32+ |
| Output length is wildly wrong (too short or padded silence) | `--reference-text` does not match the actual content of `--reference-audio` (duration estimator uses the length ratio) | Transcribe the reference clip accurately |
| Cloned voice sounds nothing like the reference | Reference clip is too short, too noisy, or contains music/effects | Use 3-10 s of clean speech |
| `"reference_audio is empty"` / `"reference_text is empty"` | Missing required flag | Both `--reference-audio` and `--reference-text` are mandatory |
| First run is slow, subsequent runs are fast | Cold ONNX Runtime session (graph optimization + weight load) | Expected — the model impl is created once per `OfflineTts` instance; reuse it across requests |
| Amplitude of output is much quieter than reference | Reference clip has an RMS above `target_rms=0.1` — no scaling was applied on input, but output also isn't boosted | Increase `--f5-target-rms` or post-normalize the WAV |
| Very slow (RTF > 1) on CPU | 32 ODE steps × transformer forward × batch-of-2 (CFG) is expensive | Build with the CUDA / CoreML EP, lower `--f5-num-steps` to 16-24, or set `--f5-guidance-scale=0` to disable CFG (halves per-step cost) |

## 6. What was added

### New files

```
scripts/f5-tts/                                       export scripts + build_bundle.sh
sherpa-onnx/csrc/offline-tts-f5-model-config.{h,cc}   CLI flags
sherpa-onnx/csrc/offline-tts-f5-model-meta-data.h     mel/FFT/vocab constants
sherpa-onnx/csrc/offline-tts-f5-model.{h,cc}          PIMPL transformer wrapper
                                                      with external Euler ODE loop
sherpa-onnx/csrc/offline-tts-f5-impl.h                OfflineTtsImpl subclass:
                                                      mel extraction, tokenization,
                                                      duration estimate, vocoding
sherpa-onnx/csrc/offline-tts-f5-model-config-test.cc  gtest (defaults, validation,
                                                      sway-sampling math)
sherpa-onnx/python/csrc/offline-tts-f5-model-config.{h,cc}   Python bindings
python-api-examples/f5-tts.py                         Python usage example
nodejs-addon-examples/test_tts_non_streaming_f5.js    Node addon usage example
```

### Modified files

```
sherpa-onnx/csrc/CMakeLists.txt                    new sources + test
sherpa-onnx/csrc/offline-tts-model-config.{h,cc}   embeds F5 sub-config + Register/Validate
sherpa-onnx/csrc/offline-tts-impl.cc               dispatch: pick F5 when --f5-transformer is set
sherpa-onnx/c-api/c-api.{h,cc}                     C surface + SHERPA_ONNX_OR defaults
sherpa-onnx/python/csrc/offline-tts-model-config.cc            registers F5 sub-config
sherpa-onnx/python/csrc/CMakeLists.txt                          new source
scripts/node-addon-api/src/non-streaming-tts.cc    GetOfflineTtsF5ModelConfig,
                                                    factory hookup, DELETE macro
```

### Notes on the port

- CFG is folded into the exported ONNX wrapper (batch of 2:
  cond + uncond, combined with `guidance_scale` before returning). The
  C++ side runs the Euler loop as a plain sequence of transformer
  forwards.
- Sway sampling matches the F5-TTS paper:
  `t_new = t + sway_coef * (cos(pi/2 * t) - 1 + t)`, with `sway_coef=-1`
  the F5-TTS default. Endpoints are pinned at 0 and 1.
- Target duration comes from the character-length ratio between the
  reference text and the target text; this mirrors the F5-TTS reference
  script and is deliberately simple. If your reference and target
  languages differ in average syllables-per-character, adjust the target
  text length or pre-chunk your input.
- Log-mel is computed with a 1e-5 floor to match `charactr/vocos-mel-24khz`.
