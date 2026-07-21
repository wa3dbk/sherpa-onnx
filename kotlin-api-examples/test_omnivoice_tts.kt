// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

package com.k2fsa.sherpa.onnx

fun main() {
  testOmnivoiceTts()
}

fun testOmnivoiceTts() {
  val bundleDir = "./sherpa-onnx-omnivoice-en"
  val referenceAudioFilename = "$bundleDir/reference.wav"
  val wave = WaveReader.readWave(filename = referenceAudioFilename)

  val config = OfflineTtsConfig(
    model = OfflineTtsModelConfig(
      omnivoice = OfflineTtsOmnivoiceModelConfig(
        model = "$bundleDir/omnivoice.onnx",
        codecEncoder = "$bundleDir/higgs_codec_encoder.onnx",
        codecDecoder = "$bundleDir/higgs_codec_decoder.onnx",
        tokenizerDir = "$bundleDir/tokenizer",
        // Set prefixModel and targetModel for the KV-cache fast path
        // prefixModel = "$bundleDir/omnivoice_prefix.onnx",
        // targetModel = "$bundleDir/omnivoice_target.onnx",
      ),
      numThreads = 2,
      debug = false,
    ),
  )

  val tts = OfflineTts(config = config)
  val text = "Hello world, this is a test of OmniVoice zero-shot text to speech."
  val referenceText = "This is the verbatim transcript of the reference audio clip."
  val genConfig = GenerationConfig(
    referenceAudio = wave.samples,
    referenceSampleRate = wave.sampleRate,
    referenceText = referenceText,
  )

  val audio = tts.generateWithConfigAndCallback(
    text = text, config = genConfig, callback = ::omnivoiceCallback)
  audio.save(filename = "test-omnivoice.wav")
  tts.release()
  println("Saved to test-omnivoice.wav")
}

fun omnivoiceCallback(samples: FloatArray): Int {
  println("callback got called with ${samples.size} samples")

  // 1 means to continue
  // 0 means to stop
  return 1
}
