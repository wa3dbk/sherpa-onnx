# IndexTTS-2 support for sherpa-onnx

This adds a new zero-shot TTS backend to sherpa-onnx: **IndexTTS-2**, a
GPT-style autoregressive language model over discrete audio tokens paired
with a BigVGAN-family neural vocoder.

- **IndexTTS-2** — <https://github.com/index-tts/index-tts>
- **BigVGAN vocoder (mel-like features → audio, 24 kHz)** — bundled ONNX export.

What makes it different from the other backends: IndexTTS-2 takes **two
independent conditioning inputs** — a *reference audio* clip (for speaker
timbre) and an *emotion* signal. The emotion signal can be supplied as
either an audio clip (`--emotion-audio`) or as free-form text
(`--emotion-text`). When both are supplied, **audio wins**: the emotion
text encoder is bypassed and only the emotion-audio embedding is fed to
the LM.

Decoding is a GPT-style token loop over discrete audio codes, run
externally in C++ so we can log per-step progress, cap output length,
and drive top-k / top-p / temperature sampling from CLI flags. Output is
24 kHz mono.

The backend plugs into the existing `sherpa-onnx-offline-tts` binary; no
new binary is added. When `--indextts2-lm` is set, dispatch picks
IndexTTS-2 automatically.

## 1. Prepare the model bundle

You need five ONNX files (LM + voice encoder + emotion audio encoder +
emotion text encoder + vocoder), a `tokens.txt` / `merges.txt` BPE pair,
and a pinyin lookup table.

### Option A — pre-built bundle

A tarball will be published on the sherpa-onnx releases page under the
`tts-models` tag. Extract and skip to §2.

### Option B — build the bundle yourself

Requires a Python env with the IndexTTS-2 reference repo installed
(<https://github.com/index-tts/index-tts>) plus `onnx`, `onnxruntime`,
`safetensors`, `transformers`, and `huggingface_hub`.

```bash
OUT_DIR=./sherpa-onnx-indextts2-base bash scripts/indextts2/build_bundle.sh
# produces sherpa-onnx-indextts2-base/ with lm.onnx, voice_encoder.onnx,
# emotion_encoder.onnx, emotion_text_encoder.onnx, vocoder.onnx,
# tokens.txt, merges.txt, pinyin_table.txt
```

Environment overrides: `INDEXTTS2_CKPT`, `INDEXTTS2_VARIANT`, `OUT_DIR`,
`REFERENCE_HF_PATH`, `DTYPE`. See `scripts/indextts2/README.md` for the
full bundle contract.

The script runs five exporters:

| Script | Produces |
| --- | --- |
| `export_indextts2_lm_onnx.py` | `lm.onnx` — GPT-style LM over audio tokens, with KV cache inputs/outputs |
| `export_indextts2_voice_encoder_onnx.py` | `voice_encoder.onnx` — reference mel → speaker embedding |
| `export_indextts2_emotion_encoder_onnx.py` | `emotion_encoder.onnx` — emotion mel → emotion embedding |
| `export_indextts2_emotion_text_encoder_onnx.py` | `emotion_text_encoder.onnx` — text ids → emotion embedding |
| `export_indextts2_vocoder_onnx.py` | `vocoder.onnx` — features `(1, F, T)` → audio `(1, T*hop)` |

### Note on the tokenizer

IndexTTS-2 ships a byte-level BPE tokenizer over UTF-8 text plus a
pinyin lookup table for CN normalization. The bundle writes
`tokens.txt` + `merges.txt` in the standard HF form and
`pinyin_table.txt` for the CN fallback. The C++ side uses a small BPE
implementation (`BpeTokenizer`) that reads the same pair of files.

## 2. Build sherpa-onnx

Same as the standard build. No new dependencies.

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSHERPA_ONNX_ENABLE_TTS=ON ..
make -j
```

The `sherpa-onnx-offline-tts` binary now accepts IndexTTS-2 flags.

## 3. Run

```bash
./bin/sherpa-onnx-offline-tts \
  --indextts2-lm=./bundle/lm.onnx \
  --indextts2-voice-encoder=./bundle/voice_encoder.onnx \
  --indextts2-emotion-encoder=./bundle/emotion_encoder.onnx \
  --indextts2-emotion-text-encoder=./bundle/emotion_text_encoder.onnx \
  --indextts2-vocoder=./bundle/vocoder.onnx \
  --indextts2-tokens=./bundle/tokens.txt \
  --indextts2-merges=./bundle/merges.txt \
  --indextts2-pinyin-table=./bundle/pinyin_table.txt \
  --reference-audio=./bundle/test_wavs/ref.wav \
  --reference-text="$(cat ./bundle/test_wavs/ref.txt)" \
  --emotion-audio=./bundle/test_wavs/happy.wav \
  --output-filename=./out.wav \
  "Text to synthesize in the reference speaker's voice."
```

To drive emotion from text instead:

```bash
  --emotion-text="cheerful and excited" \
```

If both `--emotion-audio` and `--emotion-text` are set, the audio path
wins and the text encoder is skipped.

### Generation knobs

Defaults match the IndexTTS-2 reference implementation and generally do
not need tuning:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--indextts2-max-audio-tokens` | 2000 | Hard cap on the LM's autoregressive audio-token loop. At ~50 tokens/sec of output audio, 2000 tokens ≈ 40 s. Raise for very long single utterances; lower to fail-fast on runaway generations. |
| `--indextts2-top-k` | 30 | Top-K sampling cutoff. 0 disables (falls back to top-p only); very small values (≤5) get robotic. |
| `--indextts2-top-p` | 0.8 | Nucleus sampling cumulative-probability threshold. 1.0 disables. |
| `--indextts2-temperature` | 0.8 | Logit temperature before sampling. <1.0 sharpens (more consistent, less varied); >1.0 flattens. |
| `--indextts2-seed` | -1 | RNG seed for the sampler. Use a fixed non-negative value for reproducible output. |

**Emotion-source policy**: when both `--emotion-audio` and
`--emotion-text` are provided, the audio embedding is used and the text
embedding is discarded. This mirrors the reference implementation and
avoids silently blending two conflicting emotion signals.

## 3a. Long-form input: chunking

The impl splits input text on punctuation, then merges short pieces and
splits over-long ones. Extras:

- `--extra="max_char_in_sentence=200"` — hard upper bound per chunk.
  Raise for fewer joins (each chunk costs more); lower if the LM loop
  runs out of memory or you want faster first-audio-out.
- `--extra="min_char_in_sentence=30"` — merge shorter fragments until
  this size, to avoid short-and-choppy generation.

Each chunk is generated independently with its own `max_audio_tokens`
budget, then concatenated. Reference / emotion embeddings are computed
once and cached across chunks, so per-chunk cost is dominated by the LM
autoregressive loop and the vocoder forward.

## 3b. Determinism

Sampling uses a `std::mt19937`. Passing `--indextts2-seed=<N>` (any
non-negative integer) makes the whole pipeline reproducible: two runs
with the same bundle, the same text, the same reference clip, the same
emotion input, and the same seed produce **byte-identical** WAVs.

Notes:

- `--indextts2-seed=-1` (the default) reseeds from `std::random_device`
  each request, so consecutive runs vary as intended.
- Determinism also requires `--num-threads` and the ONNX Runtime
  execution provider to be stable across runs. On CPU with a fixed
  thread count this holds. On GPU (`--provider=cuda`) atomic reductions
  in some kernels may perturb the last few bits — determinism is
  best-effort there.
- The RNG feeds *only* the token sampler; the LM forwards, the voice /
  emotion encoders, and the vocoder are deterministic functions of the
  config flags.

## 3c. Reference / emotion audio sample rate

Anything readable by `sherpa-onnx-offline-tts` works: mono or stereo, 16
kHz, 22.05 kHz, 44.1 kHz, 48 kHz. The impl calls the sherpa-onnx
`LinearResample` to convert to the encoders' native rate before
extracting features, so **you do not need to resample yourself**. Keep
each clip in the 3-10 s range for clean speaker / emotion capture.

## 4. Python and Node

Both bindings expose the same knobs:

```python
# python-api-examples/indextts2.py
import sherpa_onnx
cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        indextts2=sherpa_onnx.OfflineTtsIndexTts2ModelConfig(
            lm="./bundle/lm.onnx",
            voice_encoder="./bundle/voice_encoder.onnx",
            emotion_encoder="./bundle/emotion_encoder.onnx",
            emotion_text_encoder="./bundle/emotion_text_encoder.onnx",
            vocoder="./bundle/vocoder.onnx",
            tokens="./bundle/tokens.txt",
            merges="./bundle/merges.txt",
            pinyin_table="./bundle/pinyin_table.txt",
            max_audio_tokens=2000,
            top_k=30,
            top_p=0.8,
            temperature=0.8,
        ),
    )
)
```

```js
// nodejs-addon-examples/test_tts_non_streaming_indextts2.js
const config = {
  model: {
    indextts2: {
      lm: './bundle/lm.onnx',
      voiceEncoder: './bundle/voice_encoder.onnx',
      emotionEncoder: './bundle/emotion_encoder.onnx',
      emotionTextEncoder: './bundle/emotion_text_encoder.onnx',
      vocoder: './bundle/vocoder.onnx',
      tokens: './bundle/tokens.txt',
      merges: './bundle/merges.txt',
      pinyinTable: './bundle/pinyin_table.txt',
      maxAudioTokens: 2000,
      topK: 30,
      topP: 0.8,
      temperature: 0.8,
    },
  },
};
```

## 5. Troubleshooting

Enable verbose logging on any run to see the tokenized reference /
target lengths, the emotion path chosen, and per-chunk progress:

```bash
./bin/sherpa-onnx-offline-tts --debug=1 ...
```

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Output is roughly the right length but sounds robotic / monotone | `top_k` or `top_p` too aggressive (very peaked distribution) | Raise `--indextts2-top-k` to 30+ or `--indextts2-top-p` to 0.8+ |
| LM hits the `max_audio_tokens` cap and output is truncated | Chunk is too long, or the LM is looping | Split input via `--extra="max_char_in_sentence=200"`, or raise `--indextts2-max-audio-tokens` |
| Cloned voice sounds nothing like the reference | Reference clip is too short, too noisy, or contains music/effects | Use 3-10 s of clean speech |
| Emotion doesn't match the `--emotion-text` prompt | You also passed `--emotion-audio`, which wins | Drop `--emotion-audio` or clear it to `""` |
| `"reference_audio is empty"` / `"reference_text is empty"` | Missing required flag | Both `--reference-audio` and `--reference-text` are mandatory |
| Output is high-pitched noise / garbled | Vocoder / LM bundle mismatch (mixed exports) | Rebuild from a single `scripts/indextts2/build_bundle.sh` run |
| First run is slow, subsequent runs are fast | Cold ONNX Runtime session (graph optimization + weight load) | Expected — the model impl is created once per `OfflineTts` instance; reuse it across requests |
| Very slow (RTF > 1) on CPU | 2000 token autoregressive loop × LM forward is expensive | Build with the CUDA / CoreML EP, or lower `--indextts2-max-audio-tokens` |

## 6. What was added

### New files

```
scripts/indextts2/                                        export scripts + build_bundle.sh
sherpa-onnx/csrc/offline-tts-omnivoice-model-config.{h,cc}    (kept for reference)
sherpa-onnx/csrc/offline-tts-indextts2-model-config.{h,cc}    CLI flags
sherpa-onnx/csrc/offline-tts-indextts2-model-meta-data.h      feature / vocab constants
sherpa-onnx/csrc/offline-tts-indextts2-model.{h,cc}           PIMPL LM + encoders + vocoder
                                                              wrapper with external token loop
sherpa-onnx/csrc/offline-tts-indextts2-impl.h                 OfflineTtsImpl subclass:
                                                              feature extraction, tokenization,
                                                              emotion selection, vocoding
sherpa-onnx/csrc/offline-tts-indextts2-model-config-test.cc   gtest (defaults, validation)
sherpa-onnx/python/csrc/offline-tts-indextts2-model-config.{h,cc}   Python bindings
python-api-examples/indextts2.py                              Python usage example
nodejs-addon-examples/test_tts_non_streaming_indextts2.js     Node addon usage example
INDEXTTS2.md                                                  this document
```

### Modified files

```
sherpa-onnx/csrc/CMakeLists.txt                    new sources + test
sherpa-onnx/csrc/offline-tts-model-config.{h,cc}   embeds IndexTTS-2 sub-config + Register/Validate
sherpa-onnx/csrc/offline-tts-impl.cc               dispatch: pick IndexTTS-2 when --indextts2-lm is set
sherpa-onnx/csrc/sherpa-onnx-offline-tts.cc        threads emotion / reference flags into the request
sherpa-onnx/c-api/c-api.{h,cc}                     C surface + SHERPA_ONNX_OR defaults
sherpa-onnx/python/csrc/offline-tts-model-config.cc            registers IndexTTS-2 sub-config
sherpa-onnx/python/csrc/CMakeLists.txt                          new source
scripts/node-addon-api/src/non-streaming-tts.cc    GetOfflineTtsIndexTts2ModelConfig,
                                                    factory hookup, DELETE macro
README.md                                          flip status row to Merged
```

### Notes on the port

- The LM is exported with KV-cache inputs/outputs so the autoregressive
  loop is O(T) rather than O(T^2). The C++ side owns the cache tensors
  and swaps them between steps.
- Sampling is a plain top-k + top-p + temperature pipeline on the final
  logit tensor. Temperature is applied first, then top-k truncates, then
  top-p re-normalizes.
- Emotion-source policy: if `emotion_audio` is non-empty, only
  `emotion_encoder` is invoked. If it is empty and `emotion_text` is
  non-empty, only `emotion_text_encoder` is invoked. If both are empty,
  a zero emotion embedding is fed to the LM (neutral).
- Feature extraction uses the same log-mel path as F5-TTS with the mel
  parameters set from `offline-tts-indextts2-model-meta-data.h`.
