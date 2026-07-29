// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

const path = require('path');
const sherpa_onnx = require('sherpa-onnx-node');

const BUNDLE = process.env.BUNDLE_DIR || './sherpa-onnx-indextts2-base';
const p = (name) => path.join(BUNDLE, name);

const config = {
  model: {
    indextts2: {
      lm: p('lm.onnx'),
      voiceEncoder: p('voice_encoder.onnx'),
      emotionEncoder: p('emotion_encoder.onnx'),
      emotionTextEncoder: p('emotion_text_encoder.onnx'),
      vocoder: p('vocoder.onnx'),
      tokens: p('tokens.txt'),
      merges: p('merges.txt'),
      pinyinTable: p('pinyin_table.txt'),
    },
  },
};

const tts = new sherpa_onnx.OfflineTts(config);

const ref = sherpa_onnx.readWave(p('test_wavs/voice.wav'));
const emo = process.env.EMO_WAV
  ? sherpa_onnx.readWave(process.env.EMO_WAV)
  : { samples: new Float32Array(), sampleRate: 0 };

const audio = tts.generate({
  text: 'Hello world, this is a cloned voice with an emotion knob.',
  referenceAudio: ref.samples,
  referenceSampleRate: ref.sampleRate,
  emotionAudio: emo.samples,
  emotionAudioSampleRate: emo.sampleRate,
  emotionText: process.env.EMO_TEXT || '',
});

sherpa_onnx.writeWave('out.wav', audio.samples, audio.sampleRate);
console.log('Wrote out.wav');
