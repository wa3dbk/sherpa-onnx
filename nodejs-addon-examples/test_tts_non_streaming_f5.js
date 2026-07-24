// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder
//
// Zero-shot voice cloning with F5-TTS.
// Build a bundle first (see scripts/f5-tts/README.md), then run:
//   node ./nodejs-addon-examples/test_tts_non_streaming_f5.js
const sherpa_onnx = require('sherpa-onnx-node');

const BUNDLE = './sherpa-onnx-f5-tts-base-24khz';

function createOfflineTts() {
  const config = {
    model: {
      f5: {
        transformer: `${BUNDLE}/transformer.onnx`,
        vocoder: `${BUNDLE}/vocos.onnx`,
        tokens: `${BUNDLE}/tokens.txt`,
        dataDir: `${BUNDLE}/espeak-ng-data`,
        lexicon: '',
        numSteps: 32,
        guidanceScale: 2.0,
        swayCoef: -1.0,
        targetRms: 0.1,
        seed: -1,
      },
      debug: true,
      numThreads: 2,
      provider: 'cpu',
    },
    maxNumSentences: 1,
  };
  return new sherpa_onnx.OfflineTts(config);
}

const tts = createOfflineTts();
const text = 'Hello world, this is F5-TTS running under sherpa-onnx.';
const referenceText = 'This is the reference text spoken in the clip.';
const referenceAudioFilename = `${BUNDLE}/test_wavs/ref.wav`;
const referenceWave = sherpa_onnx.readWave(referenceAudioFilename);

const generationConfig = new sherpa_onnx.GenerationConfig({
  referenceAudio: referenceWave.samples,
  referenceSampleRate: referenceWave.sampleRate,
  referenceText,
  numSteps: 32,
  extra: {guidance_scale: '2.0', sway_coef: '-1.0'},
});

const start = Date.now();
const audio = tts.generate({text, generationConfig});
const stop = Date.now();

const elapsed = (stop - start) / 1000;
const duration = audio.samples.length / audio.sampleRate;
console.log('Wave duration', duration.toFixed(3), 's');
console.log('Elapsed', elapsed.toFixed(3), 's');
console.log('RTF', (elapsed / duration).toFixed(3));

const filename = 'test-f5-tts.wav';
sherpa_onnx.writeWave(
    filename, {samples: audio.samples, sampleRate: audio.sampleRate});
console.log(`Saved to ${filename}`);
