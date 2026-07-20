# OmniVoice for sherpa-onnx

Voice-cloning TTS built from:

- **OmniVoice LM** (Qwen3-0.6B) — <https://huggingface.co/k2/OmniVoice>
- **Higgs-Audio-V2 codec** — <https://huggingface.co/eustlb/higgs-audio-v2-tokenizer>

Non-autoregressive MaskGIT-style decoding across 8 codebooks. Output is
24 kHz mono. A reference audio clip + its transcript are required at inference
to clone the speaker's timbre.

## End-user path (no Python)

Download the pre-built bundle from the sherpa-onnx releases page and run
`sherpa-onnx-offline-tts` — see the bundle's own `README.md`.

## Rebuilding the bundle

You only need this if you retrain OmniVoice or bump the codec.

Requires a Python env with the OmniVoice training repo
(<https://github.com/k2-fsa/OmniVoice>) plus `onnx`, `onnxruntime`, and
`transformers>=4.44`.

```bash
./build_bundle.sh
# produces sherpa-onnx-omnivoice-<DATE>.tar.bz2
```

Environment overrides: `HF_REPO`, `HIGGS_REPO`, `DATE`, `OUT`, `DEVICE`.

## Individual scripts

| Script | Purpose |
| --- | --- |
| `export_omnivoice_onnx.py` | Wrap OmniVoice LM (HF PyTorch) as a single ONNX with 4 inputs: `input_ids`, `audio_mask`, `attention_mask`, `position_ids`. |
| `export_higgs_codec_onnx.py` | Export Higgs-Audio-V2 encoder (waveform → 8 codebook ids) and decoder (ids → waveform) as separate ONNX graphs. |
| `export_tokenizer.py` | Save the Qwen3 BPE tokenizer in slow (`vocab.json` + `merges.txt`) format so the C++ `QwenAsrTokenizer` can load it. |
| `inspect_onnx.py` | Dump I/O signatures + ONNX metadata for any of the produced graphs. |
| `roundtrip_onnx_smoketest.py` | Load a wav, run encoder → decoder via the ONNX codec, report RMSE + SI-SDR. |

## C++ backend

- `sherpa-onnx/csrc/offline-tts-omnivoice-model-config.{h,cc}` — CLI flags
  (`--omnivoice-model`, `--omnivoice-codec-encoder`, `--omnivoice-codec-decoder`,
  `--omnivoice-tokenizer-dir`, plus generation knobs).
- `sherpa-onnx/csrc/offline-tts-omnivoice-model.{h,cc}` — three-session
  PIMPL wrapper (`RunLM`, `EncodeAudio`, `DecodeCodes`).
- `sherpa-onnx/csrc/offline-tts-omnivoice-impl.h` — MaskGIT generation loop
  with optional classifier-free guidance.
- Dispatch is wired in `offline-tts-impl.cc`; the `sherpa-onnx-offline-tts`
  binary picks OmniVoice automatically when `--omnivoice-model` is set.
