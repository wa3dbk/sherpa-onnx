<!--
Copyright (c)  2026  Xiaomi Corporation
                2026  Waad Ben Kheder
-->

# F5-TTS bundle builder

Builds a sherpa-onnx-consumable bundle from the upstream
[SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) PyTorch checkpoints.
F5-TTS is a flow-matching DiT (Diffusion Transformer) zero-shot TTS
model. Bundle layout:

```
sherpa-onnx-f5-tts-<lang>/
├── f5_transformer.onnx        # the DiT (called N times per generation)
├── vocos.onnx                 # mel-spectrogram → 24 kHz audio vocoder
├── tokens.txt                 # vocab (utf-8 chars + phonemes)
├── espeak-ng-data/            # reused from zipvoice's phonemizer bundle
│   ├── phontab
│   ├── phonindex
│   ├── phondata
│   └── intonations
└── test_wavs/
    ├── ref.wav                # ~3-10 s 24 kHz mono reference clip
    └── ref.txt                # transcript of ref.wav
```

## Quick build

```bash
# 1. one-time: fetch F5-TTS source + a reference clip
./build_bundle.sh
```

By default this pulls `F5-TTS_Base` (multilingual, 335 M params, MIT
license) into `./sherpa-onnx-f5-tts-multi/`. Override with env vars:

| Env | Default | Meaning |
|---|---|---|
| `F5_TTS_CKPT` | HF hub `SWivid/F5-TTS` | Path or HF repo of the PyTorch checkpoint |
| `F5_TTS_VARIANT` | `F5TTS_Base` | Either `F5TTS_Base` or `F5TTS_v1_Base` |
| `VOCOS_CKPT` | HF hub `charactr/vocos-mel-24khz` | Vocos vocoder checkpoint |
| `OUT_DIR` | `./sherpa-onnx-f5-tts-multi` | Bundle output dir |
| `REFERENCE_HF_PATH` | (none) | Optional HF path to a reference wav to copy in |
| `REFERENCE_HF_TEXT_PATH` | (none) | Companion transcript for the reference |
| `DTYPE` | `fp32` | Set `fp16` or `int8` to quantize the DiT after export |

## Requirements

```bash
pip install -U torch onnx onnxruntime f5-tts vocos huggingface_hub
# For quantization:
pip install onnxconverter_common
```

CUDA is not required for export, but a machine with ≥ 16 GB RAM is —
the F5 DiT has ~335 M parameters and the fp32 ONNX serialization
peaks around 2× that during export.

## What each script does

### `export_f5_onnx.py`

Loads the F5-TTS PyTorch checkpoint, wraps the DiT forward in a
`nn.Module` that takes the exact tensor signature sherpa-onnx expects
(see docstring in the script), and exports with dynamic axes on
sequence lengths. External-data storage is used because the fp32
graph exceeds ONNX's 2 GB single-file limit for `F5TTS_v1_Base`.

Exported I/O shapes (fp32):

| Name | Shape | Notes |
|---|---|---|
| `x` (input) | `(1, T, 100)` | noisy mel spectrogram at step k |
| `cond` (input) | `(1, T, 100)` | reference mel (zero-padded to full T) |
| `text_ids` (input) | `(1, T)` | text tokens (reference + target concat) |
| `t` (input) | `(1,)` | flow-matching timestep in [0, 1] |
| `guidance_scale` (input) | `(1,)` | classifier-free guidance weight |
| `pred` (output) | `(1, T, 100)` | predicted velocity for the ODE step |

The ODE solver (Euler, ~32 steps by default) runs *in C++* — the
transformer graph only does one step per call so it stays small and
KV-cache-friendly if we ever add caching.

### `export_vocos_onnx.py`

Loads the Charactr `vocos-mel-24khz` PyTorch model and exports a
single-graph mel→waveform ONNX. I/O:

| Name | Shape | Notes |
|---|---|---|
| `mel` (input) | `(1, 100, T)` | note the transposed layout vs the DiT |
| `audio` (output) | `(1, T * 256)` | 24 kHz mono, hop = 256 |

### `build_bundle.sh`

Orchestrates: create out dir → run both export scripts → copy the
espeak-ng data dir from a nearby zipvoice bundle if present (or
prompt to fetch it separately) → drop a `tokens.txt` derived from
the F5-TTS text tokenizer → optionally fetch a reference clip.

## Runtime knobs (once the C++ side lands)

| CLI flag | Default | Meaning |
|---|---|---|
| `--f5-transformer` | (required) | path to `f5_transformer.onnx` |
| `--f5-vocoder` | (required) | path to `vocos.onnx` |
| `--f5-tokens` | (required) | path to `tokens.txt` |
| `--f5-data-dir` | (required) | espeak-ng data dir |
| `--f5-num-steps` | 32 | Euler ODE steps |
| `--f5-guidance-scale` | 2.0 | CFG weight |
| `--f5-sway-coef` | -1.0 | sway-sampling coefficient (F5's t-schedule warp) |

Reference audio + transcript come through the shared
`GenerationConfig` interface (same shape as OmniVoice / Zipvoice).

## License

Bundle-builder code: Apache 2.0 (matches sherpa-onnx).
F5-TTS weights: MIT (SWivid).
Vocos weights: MIT (Charactr).
