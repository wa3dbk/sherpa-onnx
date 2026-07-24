<!--
Copyright (c)  2026  Xiaomi Corporation
                2026  Waad Ben Kheder
-->

# OmniVoice in the Android TTS engine

This app exposes sherpa-onnx TTS as an Android `TextToSpeechService`.
To route it through OmniVoice zero-shot voice cloning, the changes are
localized to the config builder — the JNI surface already exposes the
OmniVoice fields via `OfflineTtsOmnivoiceModelConfig` (see
`android/SherpaOnnxTts/app/src/main/java/com/k2fsa/sherpa/onnx/Tts.kt`).

## 1. Bundle the model in assets

Quantize first — the fp32 bundle will not fit in an APK/AAB:

```
python3 scripts/omnivoice/export_omnivoice_quantized.py \
  --in ./sherpa-onnx-omnivoice-en --out ./bundle-int8 --dtype int8
```

Copy the int8 bundle into
`android/SherpaOnnxTtsEngine/app/src/main/assets/omnivoice/` (or use
Play Asset Delivery for anything over 150 MB — recommended).

Bundle layout inside assets:

```
omnivoice/
  omnivoice.onnx
  higgs_codec_encoder.onnx
  higgs_codec_decoder.onnx
  tokenizer/
  test_wavs/ref.wav
  test_wavs/ref.txt
```

## 2. Build the config

In `OfflineTtsEngine.kt` (or the config helper you use), replace the
existing model config block with:

```kotlin
val omni = OfflineTtsOmnivoiceModelConfig(
    model = "omnivoice/omnivoice.onnx",
    codecEncoder = "omnivoice/higgs_codec_encoder.onnx",
    codecDecoder = "omnivoice/higgs_codec_decoder.onnx",
    tokenizerDir = "omnivoice/tokenizer",
)
val modelConfig = OfflineTtsModelConfig(
    omnivoice = omni,
    numThreads = 2,
    provider = "cpu",
)
val config = OfflineTtsConfig(model = modelConfig, maxNumSentences = 1)
val tts = OfflineTts(assetManager = assets, config = config)
```

## 3. Pass reference audio at generate time

Reference audio + transcript come in via `GenerationConfig`:

```kotlin
val ref = WaveReader.read(assets, "omnivoice/test_wavs/ref.wav")
val refText = assets.open("omnivoice/test_wavs/ref.txt")
    .bufferedReader().use { it.readText() }.trim()

val gen = GenerationConfig(
    referenceAudio = ref.samples,
    referenceSampleRate = ref.sampleRate,
    referenceText = refText,
    extra = mapOf("split_strategy" to "sentence", "split_max_chars" to "200"),
)
val audio = tts.generate(text, generationConfig = gen)
```

## 4. Notes

- First synthesis warms up the graph; expect a 2-4 s latency the first
  time. Subsequent calls hit the in-process reference cache.
- For a system-wide TTS engine, do *not* let arbitrary apps swap the
  reference audio — pin one voice per voice ID.
- Runtime threading: `provider = "nnapi"` is not yet validated with
  OmniVoice's Qwen3 graph; stick to CPU on Android for now.
