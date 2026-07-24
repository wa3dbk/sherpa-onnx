<!--
Copyright (c)  2026  Xiaomi Corporation
                2026  Waad Ben Kheder
-->

# OmniVoice in the iOS Swift sample app

The C API surface already exposes OmniVoice; wiring it into an iOS
Swift app is a change of the model config only. Use the same shims
that `swift-api-examples/tts-omnivoice.swift` demonstrates.

## 1. Bundle the model

Quantize the bundle (fp32 will crash a device under memory pressure):

```
python3 scripts/omnivoice/export_omnivoice_quantized.py \
  --in ./sherpa-onnx-omnivoice-en --out ./bundle-int8 --dtype int8
```

Drag `bundle-int8/` into the Xcode project as a **folder reference**
(blue folder) — not a group — so the tokenizer subdir is preserved in
the app bundle at runtime as `Bundle.main.resourceURL/omnivoice/`.

## 2. Build the config in Swift

```swift
let root = Bundle.main.resourceURL!.appendingPathComponent("omnivoice")
func p(_ n: String) -> String { root.appendingPathComponent(n).path }

var omni = sherpaOnnxOfflineTtsOmnivoiceModelConfig(
    model: p("omnivoice.onnx"),
    codecEncoder: p("higgs_codec_encoder.onnx"),
    codecDecoder: p("higgs_codec_decoder.onnx"),
    tokenizerDir: p("tokenizer"))

var model = sherpaOnnxOfflineTtsModelConfig(
    omnivoice: omni, numThreads: 2, provider: "cpu")

var cfg = sherpaOnnxOfflineTtsConfig(model: model, maxNumSentences: 1)
let tts = SherpaOnnxOfflineTtsWrapper(config: &cfg)
```

## 3. Reference audio

Read `test_wavs/ref.wav` via `AVAudioFile`, pass samples + sample
rate + transcript through `SherpaOnnxGenerationConfig`. See the
demo in `swift-api-examples/tts-omnivoice.swift` for the exact call
sequence including the streaming progress callback.

## 4. Notes

- CoreML EP is not validated with the Qwen3 LM graph. Use `"cpu"`.
- If you ship a system extension (SiriKit / AVSpeechSynthesizer
  provider), pin the reference audio to a specific voice identifier
  rather than accepting arbitrary reference input from clients.
