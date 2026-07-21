// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder
//
// This file shows how to use a non-streaming OmniVoice model
// (Qwen3-0.6B LM + Higgs-Audio-V2 codec) for zero-shot text-to-speech.
using SherpaOnnx;
using System.Runtime.InteropServices;

class OmnivoiceTtsDemo
{
  static void Main(string[] args)
  {
    TestEn();
  }

  static void TestEn()
  {
    var bundleDir = "./sherpa-onnx-omnivoice-en";

    var config = new OfflineTtsConfig();
    config.Model.Omnivoice.Model = $"{bundleDir}/omnivoice.onnx";
    config.Model.Omnivoice.CodecEncoder = $"{bundleDir}/higgs_codec_encoder.onnx";
    config.Model.Omnivoice.CodecDecoder = $"{bundleDir}/higgs_codec_decoder.onnx";
    config.Model.Omnivoice.TokenizerDir = $"{bundleDir}/tokenizer";
    // Uncomment to enable the KV-cache fast path:
    // config.Model.Omnivoice.PrefixModel = $"{bundleDir}/omnivoice_prefix.onnx";
    // config.Model.Omnivoice.TargetModel = $"{bundleDir}/omnivoice_target.onnx";

    config.Model.NumThreads = 2;
    config.Model.Debug = 1;
    config.Model.Provider = "cpu";

    var referenceWaveFilename = $"{bundleDir}/reference.wav";
    var reader = new WaveReader(referenceWaveFilename);

    OfflineTtsGenerationConfig genConfig = new OfflineTtsGenerationConfig();
    genConfig.ReferenceAudio = reader.Samples;
    genConfig.ReferenceSampleRate = reader.SampleRate;
    genConfig.ReferenceText = "This is the verbatim transcript of the reference audio clip.";

    var tts = new OfflineTts(config);
    var text = "Hello world, this is a test of OmniVoice zero-shot text to speech.";

    var myCallback = (IntPtr samples, int n, float progress, IntPtr arg) =>
    {
      if (n > 0)
      {
        float[] data = new float[n];
        Marshal.Copy(samples, data, 0, n);
      }
      Console.WriteLine($"Progress {progress * 100}%");
      return 1;
    };

    var callback = new OfflineTtsCallbackProgressWithArg(myCallback);

    var audio = tts.GenerateWithConfig(text, genConfig, callback);

    var outputFilename = "./generated-omnivoice.wav";
    var ok = audio.SaveToWaveFile(outputFilename);

    if (ok)
    {
      Console.WriteLine($"Wrote to {outputFilename} succeeded!");
    }
    else
    {
      Console.WriteLine($"Failed to write {outputFilename}");
    }
  }
}
