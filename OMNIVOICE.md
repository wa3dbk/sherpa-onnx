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

## 3a. Long-form input: chunking strategies

A single MaskGIT pass caps at ~60 s of output (25 tokens/s × 1500-token
LM window). Beyond that, either the LM refuses the input or per-pass
memory blows up. For paragraph-scale narration, opt into a splitter that
breaks the text into pieces, runs one MaskGIT pass per piece, and
concatenates the outputs.

The reference-audio encoding is cached across pieces (see
`cache_reference_audio`), so per-chunk cost is dominated by the LM and
codec-decoder forwards, not the codec encoder.

| `--split-strategy` | When to use | Tradeoffs |
| --- | --- | --- |
| `none` *(default)* | Short single-sentence inputs, or when you're managing chunks yourself in the caller. | Fastest — one pass. Fails hard once you cross the 60 s cap. |
| `sentence` | Well-punctuated prose (novels, articles, transcripts). | Best prosody: breaks land on `.`/`!`/`?` (and `。`/`！`/`？`), which is where humans pause anyway. Overlong single sentences fall back to `chars` for just that sentence. |
| `chars` | Unpunctuated or adversarial input: URLs, IDs, phone numbers, generated text with no full stops. | Guarantees a bound on per-chunk cost. Prefers whitespace / `,` / `;` breaks near the limit and never splits a UTF-8 code point. Prosody joins may be audible mid-clause. |

Extra flags:

- `--split-max-chars=200` — byte limit per chunk. 200 is a safe default for
  a 32-step MaskGIT run; raise it for longer chunks (fewer joins, more
  cost per chunk) or lower it if you see the "target length exceeds cap"
  warning even after chunking.
- `--extra="chunk_gap_ms=60"` — silence inserted between chunks. 60 ms is
  about a comma-length pause. Set 0 for no gap, or 150-200 for a fuller
  breath between sentences.

Chunking is per-request: chunk 1 finishes MaskGIT + decode, its audio
streams to the callback / stdout, then chunk 2 starts. With
`--output-stream=1` this gives you first-audio-out latency proportional
to the *first chunk*, not the whole paragraph. That is the main reason
to prefer `sentence` over `none` for interactive use even when the
paragraph would technically fit in one pass.

## 3b. Streaming output

`--output-stream=1` writes raw **float32 little-endian** PCM to stdout as
each decoded chunk arrives. The output sample rate (always 24 000 Hz for
the current OmniVoice bundle) is printed to stderr on startup. In this
mode, `--output-filename` is ignored — no WAV is written.

```bash
./bin/sherpa-onnx-offline-tts \
  --omnivoice-model=./bundle/omnivoice.onnx \
  ...
  --split-strategy=sentence \
  --output-stream=1 \
  "First sentence to play. Second sentence, also long. Third." \
  | ffplay -f f32le -ar 24000 -i - -nodisp -autoexit
```

Or with `aplay`:

```bash
... --output-stream=1 "Hello world" | aplay -f FLOAT_LE -c 1 -r 24000
```

Two things to know:

1. The Higgs codec decoder is a single ONNX graph, so within one MaskGIT
   pass audio can only start streaming *after* that pass finishes. The
   PCM then arrives in `chunk_ms`-sized slices (default 500 ms). Combined
   with `--split-strategy=sentence`, this means you hear the first
   sentence roughly one MaskGIT+decode later — not after the whole
   paragraph.
2. If the downstream consumer closes the pipe (e.g. you hit `q` in
   `ffplay`), the write fails, the callback returns 0, and the backend
   aborts the current chunk cleanly.

## 3c. Determinism

MaskGIT unmask-order sampling uses Gumbel noise driven by a
`std::mt19937`. Passing `--seed=<N>` (any non-negative integer) makes the
whole pipeline reproducible: two runs with the same bundle, the same
text, the same reference clip, and the same seed produce **byte-identical**
WAVs. This is what the `scripts/omnivoice/test_e2e.sh` determinism check
asserts.

Notes:

- `--seed=-1` (the default) reseeds from `std::random_device` each
  request, so consecutive runs vary as intended.
- Determinism also requires `--num-threads` and the ONNX Runtime
  execution provider to be stable across runs. On CPU with a fixed
  thread count this holds. On GPU (`--provider=cuda`) atomic reductions
  in some kernels may perturb the last few bits — determinism is
  best-effort there.
- The RNG feeds *position sampling only*, not the CFG logit combination,
  so `--seed` does not interact with `--omnivoice-guidance-scale` or the
  layer penalty.

## 3d. Reference audio sample rate

Anything readable by `sherpa-onnx-offline-tts` works: mono or stereo, 16
kHz, 22.05 kHz, 44.1 kHz, 48 kHz. The impl calls the sherpa-onnx
`LinearResample` to convert to the codec's native 24 kHz before
encoding, so **you do not need to resample yourself**. The one thing
that still matters is duration: keep the clip in the 3-10 s range for
clean speaker capture (hard error below 0.5 s, warning above 30 s).

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
| `"target length N tokens exceeds cap"` | Text too long for a single MaskGIT pass | Add `--split-strategy=sentence` (or `chars` for unpunctuated input); see §3a |
| Chunked output has audible clicks / breath pops at sentence joins | `chunk_gap_ms` too short or reference clip clipped at its ends | Try `--extra="chunk_gap_ms=120"`; trim the reference wav so it starts and ends on silence |
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
                                                             optional classifier-free guidance,
                                                             rule-based duration estimator,
                                                             sentence/char chunking
sherpa-onnx/csrc/offline-tts-omnivoice-text-splitter.h       kNone / kSentence / kChars
sherpa-onnx/csrc/offline-tts-omnivoice-text-splitter-test.cc gtest
sherpa-onnx/csrc/offline-tts-omnivoice-text-weight.h         per-script char weights
scripts/omnivoice/                                           export scripts + build_bundle.sh
scripts/omnivoice/test_e2e.sh                                bundle-gated end-to-end test
                                                             (basic run, determinism, chunking,
                                                             --output-stream)
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
