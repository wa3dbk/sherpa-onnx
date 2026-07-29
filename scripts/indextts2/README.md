# IndexTTS-2 bundle builder

Produces a `sherpa-onnx-consumable IndexTTS-2 bundle from the
IndexTeam/IndexTTS-2 base checkpoint. Requires a Python env with:

    pip install torch onnx onnxruntime safetensors transformers \
                huggingface_hub pypinyin sentencepiece

## Environment overrides

| Var | Default | Meaning |
| --- | --- | --- |
| `INDEXTTS2_CKPT` | `IndexTeam/IndexTTS-2` | HF repo id or local path |
| `INDEXTTS2_VARIANT` | `base` | Sub-directory / variant name |
| `OUT_DIR` | `./sherpa-onnx-indextts2-base` | Bundle output dir |
| `REFERENCE_HF_PATH` | *(empty)* | Optional voice reference wav |
| `REFERENCE_HF_TEXT_PATH` | *(empty)* | Optional voice reference transcript |
| `EMO_REF_HF_PATH` | *(empty)* | Optional emotion reference wav |
| `DTYPE` | `fp32` | `fp32`, `fp16`, or `int8` |

## Output

    OUT_DIR/
      lm.onnx
      voice_encoder.onnx
      emotion_encoder.onnx
      emotion_text_encoder.onnx
      vocoder.onnx
      tokens.txt
      merges.txt
      pinyin_table.txt
      meta.json
      test_wavs/{voice.wav, voice.txt, emo_happy.wav}

## Bundle contract

Each ONNX file's input / output names and shapes are documented in the
corresponding `export_*.py` script. `meta.json` carries the runtime
knobs the C++ side reads via `SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT`:
`sample_rate`, `vocab_size`, `bos_token_id`, `eos_token_id`,
`speaker_embed_dim`, `emotion_embed_dim`, `kv_num_layers`,
`kv_num_heads`, `kv_head_dim`, `max_audio_tokens`.
