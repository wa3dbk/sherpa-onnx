// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder
//
// OmniVoice zero-shot voice cloning via sherpa-onnx-node. Requires the
// Node addon to have been built against a sherpa-onnx that includes the
// omnivoice config field. See README section on Node.js bindings.
//
// Bundle: build via scripts/omnivoice/build_bundle.sh and point BUNDLE_DIR
// at the resulting sherpa-onnx-omnivoice-<DATE>/ directory.

const sherpa_onnx = require('sherpa-onnx-node');

const bundleDir = process.env.BUNDLE_DIR || './sherpa-onnx-omnivoice-en';

function createOfflineTts() {
  const config = {
    model: {
      omnivoice: {
        model: `${bundleDir}/omnivoice.onnx`,
        codecEncoder: `${bundleDir}/higgs_codec_encoder.onnx`,
        codecDecoder: `${bundleDir}/higgs_codec_decoder.onnx`,
        tokenizerDir: `${bundleDir}/tokenizer`,
        // Uncomment for the KV-cache fast path (build with WITH_CACHED=1):
        // prefixModel: `${bundleDir}/omnivoice_prefix.onnx`,
        // targetModel: `${bundleDir}/omnivoice_target.onnx`,
      },
      debug: false,
      numThreads: 2,
      provider: 'cpu',
    },
    maxNumSentences: 1,
  };
  return new sherpa_onnx.OfflineTts(config);
}

const tts = createOfflineTts();

const referenceAudioFilename = `${bundleDir}/test_wavs/ref.wav`;
const referenceText = require('fs')
  .readFileSync(`${bundleDir}/test_wavs/ref.txt`, 'utf-8')
  .trim();
const referenceWave = sherpa_onnx.readWave(referenceAudioFilename);

const text = 'Hello world, this is a test of OmniVoice zero-shot text to speech.';

const generationConfig = new sherpa_onnx.GenerationConfig({
  referenceAudio: referenceWave.samples,
  referenceSampleRate: referenceWave.sampleRate,
  referenceText,
  // Try 'sentence' for paragraph-length inputs (see OMNIVOICE.md §3a):
  extra: {
    split_strategy: 'none',       // 'none' | 'sentence' | 'chars'
    split_max_chars: '200',
    // seed: '42',                // set for byte-identical output
  },
});

const start = Date.now();
const audio = tts.generate({text, generationConfig});
const stop = Date.now();

const elapsedSeconds = (stop - start) / 1000;
const duration = audio.samples.length / audio.sampleRate;
const rtf = elapsedSeconds / duration;
console.log('Wave duration', duration.toFixed(3), 's');
console.log('Elapsed', elapsedSeconds.toFixed(3), 's');
console.log(`RTF = ${elapsedSeconds.toFixed(3)}/${duration.toFixed(3)} =`,
            rtf.toFixed(3));

const out = 'test-omnivoice.wav';
sherpa_onnx.writeWave(out, {samples: audio.samples,
                            sampleRate: audio.sampleRate});
console.log(`Saved to ${out}`);
