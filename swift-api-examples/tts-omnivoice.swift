// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

import Foundation

class OmnivoiceTtsProgressHandler {
  func progress(samples: [Float], progress: Float) {
    print(String(format: "Received %d samples, Progress: %.2f%%", samples.count, progress * 100))
  }
}

func runOmnivoiceTtsDemo() {
  let bundleDir = "./sherpa-onnx-omnivoice-en"

  let omnivoice = sherpaOnnxOfflineTtsOmnivoiceModelConfig(
    model: "\(bundleDir)/omnivoice.onnx",
    codecEncoder: "\(bundleDir)/higgs_codec_encoder.onnx",
    codecDecoder: "\(bundleDir)/higgs_codec_decoder.onnx",
    tokenizerDir: "\(bundleDir)/tokenizer"
    // prefixModel: "\(bundleDir)/omnivoice_prefix.onnx",
    // targetModel: "\(bundleDir)/omnivoice_target.onnx"
  )

  let modelConfig = sherpaOnnxOfflineTtsModelConfig(numThreads: 2, omnivoice: omnivoice)
  var ttsConfig = sherpaOnnxOfflineTtsConfig(model: modelConfig)
  ttsConfig.model.debug = 1

  let tts = SherpaOnnxOfflineTtsWrapper(config: &ttsConfig)

  let referenceAudioFile = "\(bundleDir)/reference.wav"
  let referenceWave = SherpaOnnxWaveWrapper.readWave(filename: referenceAudioFile)

  var genConfig = SherpaOnnxGenerationConfigSwift()
  genConfig.referenceAudio = referenceWave.samples
  genConfig.referenceSampleRate = referenceWave.sampleRate
  genConfig.referenceText = "This is the verbatim transcript of the reference audio clip."

  let text = "Hello world, this is a test of OmniVoice zero-shot text to speech."

  let progressHandler = OmnivoiceTtsProgressHandler()
  let arg = Unmanaged.passUnretained(progressHandler).toOpaque()

  let callback: TtsProgressCallbackWithArg = { samples, n, progress, arg in
    let handler = Unmanaged<OmnivoiceTtsProgressHandler>.fromOpaque(arg!).takeUnretainedValue()
    let buffer: [Float] =
      samples != nil ? Array(UnsafeBufferPointer(start: samples, count: Int(n))) : []
    handler.progress(samples: buffer, progress: progress)
    return 1
  }

  let audio = tts.generateWithConfig(
    text: text, config: genConfig, callback: callback, arg: arg)

  let outputFile = "generated-omnivoice.wav"
  if audio.save(filename: outputFile) == 1 {
    print("Saved to: \(outputFile)")
  } else {
    print("Failed to save to \(outputFile)")
  }
}

@main
struct App {
  static func main() {
    runOmnivoiceTtsDemo()
  }
}
