// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

package main

import (
	"log"
	"path/filepath"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	var bundleDir string
	var referenceAudio string
	var referenceText string
	var text string
	var outputFilename string = "./generated-omnivoice-go.wav"
	var cached bool

	flag.StringVar(&bundleDir, "bundle-dir", "", "Path to the sherpa-onnx-omnivoice-* bundle")
	flag.StringVar(&referenceAudio, "reference-audio", "", "Path to the reference wav (3-10 s, 24 kHz-ish)")
	flag.StringVar(&referenceText, "reference-text", "", "Verbatim transcript of the reference clip")
	flag.StringVar(&text, "text", "", "Text to synthesize")
	flag.StringVar(&outputFilename, "output-filename", outputFilename, "File to save the generated audio")
	flag.BoolVar(&cached, "cached", false, "Enable KV-cache fast path (needs omnivoice_prefix.onnx and omnivoice_target.onnx in the bundle)")
	flag.Parse()

	if bundleDir == "" || referenceAudio == "" || referenceText == "" || text == "" {
		log.Fatal("--bundle-dir, --reference-audio, --reference-text, --text are all required")
	}

	var config sherpa.OfflineTtsConfig
	config.Model.Omnivoice.Model = filepath.Join(bundleDir, "omnivoice.onnx")
	config.Model.Omnivoice.CodecEncoder = filepath.Join(bundleDir, "higgs_codec_encoder.onnx")
	config.Model.Omnivoice.CodecDecoder = filepath.Join(bundleDir, "higgs_codec_decoder.onnx")
	config.Model.Omnivoice.TokenizerDir = filepath.Join(bundleDir, "tokenizer")
	if cached {
		config.Model.Omnivoice.PrefixModel = filepath.Join(bundleDir, "omnivoice_prefix.onnx")
		config.Model.Omnivoice.TargetModel = filepath.Join(bundleDir, "omnivoice_target.onnx")
	}
	config.Model.NumThreads = 2
	config.Model.Debug = 0
	config.Model.Provider = "cpu"

	log.Println("Creating Offline TTS")
	tts := sherpa.NewOfflineTts(&config)
	if tts == nil {
		log.Fatal("Failed to create OfflineTts")
	}
	defer sherpa.DeleteOfflineTts(tts)

	wave := sherpa.ReadWave(referenceAudio)
	if wave == nil {
		log.Fatal("Failed to read reference wav: ", referenceAudio)
	}

	var cfg sherpa.GenerationConfig
	cfg.ReferenceAudio = wave.Samples
	cfg.ReferenceSampleRate = wave.SampleRate
	cfg.ReferenceText = referenceText

	log.Println("Start generating")
	audio := tts.GenerateWithConfig(
		text, &cfg,
		func(samples []float32, progress float32) bool {
			if len(samples) == 0 {
				log.Printf("MaskGIT progress: %.1f%%", progress*100)
			} else {
				log.Printf("Decoded %d samples (%.1f%%)", len(samples), progress*100)
			}
			return true
		},
	)
	if audio == nil {
		log.Fatal("Generation failed")
	}
	if !audio.Save(outputFilename) {
		log.Fatal("Failed to save wav")
	}
	log.Println("Saved to:", outputFilename)
}
