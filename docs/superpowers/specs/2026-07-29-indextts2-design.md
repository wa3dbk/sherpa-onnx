# IndexTTS-2 support for sherpa-onnx-zero-shot-tts

**Status:** Design approved 2026-07-29. Ready for implementation planning.

**Owners:** Waad Ben Kheder

**Related:** [F5-TTS.md](../../../F5-TTS.md), [OMNIVOICE.md](../../../OMNIVOICE.md)

---

## 1. Goal

Add IndexTTS-2 as a third zero-shot voice-cloning TTS backend in this fork,
alongside OmniVoice (Qwen3 LM + Higgs codec, MaskGIT) and F5-TTS
(flow-matching DiT + Vocos). IndexTTS-2's differentiator is dual-conditioning
on **emotion**: alongside the voice reference clip, the caller can supply an
emotion reference audio clip **and/or** a natural-language emotion text
description ("sad and tired", "excited").

Full parity with the reference implementation:
[IndexTeam/IndexTTS-2](https://github.com/index-tts/index-tts) (~1.5B base
checkpoint), including voice cloning, emotion audio conditioning, and emotion
text conditioning.

## 2. Architecture

GPT-style autoregressive LM + neural vocoder (BigVGAN family, confirmed at
export time). Five ONNX subnets, kept separate for cache reuse and
smaller-graph optimization:

| File | Purpose |
| --- | --- |
| `voice_encoder.onnx` | Reference wav → speaker embedding tensor |
| `emotion_encoder.onnx` | Emotion reference wav → emotion embedding tensor |
| `emotion_text_encoder.onnx` | Emotion text → emotion embedding tensor (same shape as audio one) |
| `lm.onnx` | GPT step: one input token + KV cache → next-token logits + updated KV cache. Also runs in "prefix mode" for the initial pass over `[speaker_embed, emo_embed, text_tokens]` |
| `vocoder.onnx` | Sequence of audio tokens → PCM float samples |

The LM runs autoregressively in C++ with an external KV cache: a one-shot
prefix forward seeds the cache, then a loop of single-token steps until EOS
or a hard cap.

### Tokenizer

Pinyin + BPE for Chinese, char + BPE for English/mixed. Rather than link
`pypinyin` (Python-only, hard to reproduce byte-exact in C++), we dump the
pinyin table **at bundle-build time** and ship it as `pinyin_table.txt`.
Runtime: char loop → CJK chars go through the lookup table → the resulting
mixed string is BPE-encoded via `merges.txt`. Same byte-exact output as the
reference implementation, no Python at runtime.

### Streaming

Chunk-boundary streaming only (no intra-sentence token streaming this pass).
Reuses `SplitByPunctuation` + `MergeShortSentences` from the existing text
frontend. Each sentence: prefix pass → AR loop → vocode → fire
`GeneratedAudioCallback(samples, sample_rate, progress)`. Matches F5 and
OmniVoice behavior; users get first audio out within one sentence's latency.

## 3. Components

### New C++ files under `sherpa-onnx/csrc/`

```
offline-tts-indextts2-model-config.{h,cc}     CLI flags + Validate + ToString
offline-tts-indextts2-model-meta-data.h       vocab size, sample rate, EOS id, etc.
offline-tts-indextts2-voice-encoder.{h,cc}    voice_encoder.onnx wrapper (PIMPL)
offline-tts-indextts2-emotion-encoder.{h,cc}  emotion_encoder.onnx wrapper
offline-tts-indextts2-emotion-text-encoder.{h,cc}  emotion_text_encoder.onnx wrapper
offline-tts-indextts2-model.{h,cc}            LM wrapper: PrefixPass + StepOne + KV cache mgmt
offline-tts-indextts2-frontend.{h,cc}         pinyin lookup + BPE tokenizer
offline-tts-indextts2-impl.h                  OfflineTtsImpl subclass: orchestrates the pipeline

offline-tts-indextts2-model-config-test.cc    gtest for config
offline-tts-indextts2-frontend-test.cc        gtest for pinyin + BPE
offline-tts-indextts2-sampler-test.cc         gtest for top-k / top-p / temperature sampling
```

### Bundle layout

```
sherpa-onnx-indextts2-base/
  lm.onnx
  voice_encoder.onnx
  emotion_encoder.onnx
  emotion_text_encoder.onnx
  vocoder.onnx
  tokens.txt                 BPE vocab
  merges.txt                 BPE merges
  pinyin_table.txt           CJK char → pinyin lookup (dumped from pypinyin)
  meta.json                  sample rate, EOS id, max audio tokens, etc.
  test_wavs/
    voice.wav
    voice.txt
    emo_happy.wav
```

### C API surface

New struct `SherpaOnnxOfflineTtsIndexTts2ModelConfig` embedded in
`SherpaOnnxOfflineTtsModelConfig` alongside the existing OmniVoice / F5
sub-configs.

New optional field on `SherpaOnnxOfflineTtsGenerationConfig`:
- `const char *emotion_audio` — path to emo reference wav
- `const char *emotion_text` — natural-language emo description

Both may be `nullptr` / empty. This is admittedly a wart — an IndexTTS-2
specific field on the shared generation config — but the alternative
(overloaded string extras, or a second Generate entry point) is worse for the
Python/Node bindings.

### Python + Node bindings

New `sherpa_onnx.OfflineTtsIndexTts2ModelConfig` (Python) /
`{ model: { indextts2: {...} } }` (Node), plus `emotion_audio` and
`emotion_text` on `GenerationConfig`. Same pattern as F5.

### Reused unchanged

- `Vocoder::Create` (wraps `vocoder.onnx`, produces float samples)
- `LinearResample`
- `SplitByPunctuation`, `MergeShortSentences`
- Extras plumbing (`--extra="max_char_in_sentence=..."`)
- `OfflineTts` public API — no shape changes
- Reference caching pattern from OmniVoice (per-path + mtime key, in-process
  LRU + on-disk under `~/.cache/sherpa-onnx-indextts2/`)

### Not reused

`MatchaTtsLexicon` is espeak-shaped and doesn't fit pinyin + BPE. New
frontend class.

## 4. Data flow

Per request (one text string + `voice_ref` path + optional `emo_ref_audio`
path + optional `emo_text`):

1. **Cache lookup.** Three independent caches, keyed by absolute path + mtime:
   - `voice_ref` → speaker embedding tensor (voice_encoder.onnx)
   - `emo_ref_audio` → emotion embedding tensor (emotion_encoder.onnx)
   - `emo_text` → emotion embedding tensor (emotion_text_encoder.onnx)

   Miss: resample wav to encoder's native rate via `LinearResample`, run the
   encoder, store. Hit: use tensor directly.

2. **Emotion fusion.** Only one emotion embedding feeds the LM:
   - Both provided → **audio wins** (higher-fidelity signal; text is a
     fallback UX). Debug-log the choice.
   - One provided → use it.
   - Neither → zero-vector of the emo embedding shape (neutral).

3. **Text → tokens.**
   - `SplitByPunctuation` → sentence list; `MergeShortSentences` up to
     `max_char_in_sentence`.
   - Per sentence: char loop → CJK chars via `pinyin_table.txt` → BPE-encode
     the mixed string via `merges.txt` → `int64` token IDs.

4. **Per-sentence LM pass.**
   - **Prefix pass** (one forward): LM input =
     `[speaker_embed, emo_embed, text_tokens]` concatenated per the export's
     expected layout; output = seeded KV cache + first audio token logits.
   - **AR loop:** sample audio token (top-k / top-p / temperature) → append →
     `lm.onnx` in step mode (previous token + KV cache → new KV cache +
     next-token logits). Stop on EOS token or `max_audio_tokens` cap
     (default 2000). The same `lm.onnx` handles both prefix and step modes;
     export-side branch is selected by input signature.

5. **Vocode + callback.**
   - Feed accumulated audio-token sequence to `vocoder.onnx` → PCM float
     samples at the vocoder's native rate.
   - Fire `GeneratedAudioCallback(samples, sample_rate, progress)` — same
     signature as F5 / OmniVoice.
   - Reset KV cache, move to next sentence.

## 5. Error handling

Boundary validation only. No defensive code for scenarios that can't happen.

### Config-time (`Validate()`)
- Any of the 5 ONNX paths missing / unreadable → log + return false.
- `tokens.txt`, `merges.txt`, `pinyin_table.txt` missing → false.
- `max_audio_tokens <= 0` → false.

### Request-time
- `voice_ref` empty or unreadable → log + return empty `GeneratedAudio{}`.
- `emo_ref_audio` set but unreadable → warn, fall back to text emo (or zero).
- `emo_text` > 500 chars → truncate + log; don't reject.
- Text empty after `SplitByPunctuation` → return empty audio, debug-log.
- Sample-rate mismatch → not an error; `LinearResample` handles it.

### Runtime
- EOS never reached before `max_audio_tokens` cap → warn, vocode what we
  have. Don't throw.
- ONNX Runtime exception → propagate (same as every other backend).

### Explicitly NOT handled (YAGNI)
- No retry logic on ONNX inference.
- No fallback tokenizers.
- No "did you mean" for unknown BPE tokens — the tokenizer emits `<unk>`.

## 6. Testing

### Unit (gtest, CI without bundles)

1. `offline-tts-indextts2-model-config-test.cc`
   - Defaults: `max_audio_tokens`, `top_k`, `top_p`, `temperature`, `seed`.
   - `Validate()` false for empty / bogus paths.
   - `ToString()` mentions every field.

2. `offline-tts-indextts2-frontend-test.cc`
   - Pinyin lookup: known CJK char → known pinyin (small fixture table).
   - Pass-through for ASCII, digits, punctuation.
   - BPE round-trip on fixed merges fixture: "hello world" → known IDs.
   - Unknown char → `<unk>` token, not a crash.

3. `offline-tts-indextts2-sampler-test.cc`
   - Argmax at `temperature=0` matches expected index.
   - Fixed seed + `top_k=5` + `temperature=1.0` → deterministic index.

### Integration (bundle-gated, skipped in CI)

If `INDEXTTS2_BUNDLE_DIR` env is set, run the CLI end-to-end on a canned
reference clip + short target text; assert non-empty WAV with plausible RMS.

### Manual (documented in INDEXTTS2.md)

- Voice-only request.
- Voice + `emo_audio`.
- Voice + `emo_text`.
- Voice + both (verify audio wins via debug log).
- Long text → callback fires per sentence.
- `--seed=42` twice → byte-identical WAV.

### Not tested
- Model quality (subjective — human eval only).
- CUDA numerical equivalence to CPU (best-effort, same caveat as F5).

## 7. Non-goals

- Intra-sentence token streaming (chunk-boundary only this pass).
- Multi-speaker cross-fade / speaker interpolation.
- Fine-tuning support (inference only).
- WebAssembly build (fp32 too large; deferred to post-int8-export).
- Mobile bindings (Kotlin / Swift). Upstream sherpa-onnx handles those for
  the shared surface; per-backend bindings for the fork's zero-shot TTS
  stack are deferred.

## 8. Open questions (for implementation phase, not design)

- Confirm the exact vocoder architecture (BigVGAN vs. HiFi-GAN vs. something
  else) once export runs; adjust `vocoder.onnx` I/O shape docs accordingly.
- Confirm KV cache tensor layout from the export — may need a small adapter
  if the export produces a nested structure vs. flat `past_key`/`past_value`.
- Whether `pinyin_table.txt` should be one entry per char or per multi-char
  token (traditional vs. simplified handling). Decide during export scripting.
