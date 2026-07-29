// sherpa-onnx/python/csrc/offline-tts-indextts2-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/python/csrc/offline-tts-indextts2-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-indextts2-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsIndexTts2ModelConfig(py::module *m) {
  using PyClass = OfflineTtsIndexTts2ModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsIndexTts2ModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &, int32_t, int32_t,
                    float, float, int32_t>(),
           py::arg("lm"), py::arg("voice_encoder"),
           py::arg("emotion_encoder"), py::arg("emotion_text_encoder"),
           py::arg("vocoder"), py::arg("tokens"), py::arg("merges"),
           py::arg("pinyin_table"), py::arg("max_audio_tokens") = 2000,
           py::arg("top_k") = 30, py::arg("top_p") = 0.8f,
           py::arg("temperature") = 0.8f, py::arg("seed") = -1)
      .def_readwrite("lm", &PyClass::lm)
      .def_readwrite("voice_encoder", &PyClass::voice_encoder)
      .def_readwrite("emotion_encoder", &PyClass::emotion_encoder)
      .def_readwrite("emotion_text_encoder", &PyClass::emotion_text_encoder)
      .def_readwrite("vocoder", &PyClass::vocoder)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("merges", &PyClass::merges)
      .def_readwrite("pinyin_table", &PyClass::pinyin_table)
      .def_readwrite("max_audio_tokens", &PyClass::max_audio_tokens)
      .def_readwrite("top_k", &PyClass::top_k)
      .def_readwrite("top_p", &PyClass::top_p)
      .def_readwrite("temperature", &PyClass::temperature)
      .def_readwrite("seed", &PyClass::seed)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_onnx
