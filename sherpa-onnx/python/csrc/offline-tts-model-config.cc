// sherpa-onnx/python/csrc/offline-tts-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/python/csrc/offline-tts-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tts-f5-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tts-indextts2-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tts-kitten-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tts-kokoro-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tts-matcha-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tts-omnivoice-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tts-pocket-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tts-supertonic-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tts-vits-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tts-zipvoice-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsModelConfig(py::module *m) {
  PybindOfflineTtsVitsModelConfig(m);
  PybindOfflineTtsMatchaModelConfig(m);
  PybindOfflineTtsKokoroModelConfig(m);
  PybindOfflineTtsZipvoiceModelConfig(m);
  PybindOfflineTtsKittenModelConfig(m);
  PybindOfflineTtsPocketModelConfig(m);
  PybindOfflineTtsSupertonicModelConfig(m);
  PybindOfflineTtsOmnivoiceModelConfig(m);
  PybindOfflineTtsF5ModelConfig(m);
  PybindOfflineTtsIndexTts2ModelConfig(m);

  using PyClass = OfflineTtsModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsModelConfig")
      .def(py::init<>())
      .def(py::init([](const OfflineTtsVitsModelConfig &vits,
                       const OfflineTtsMatchaModelConfig &matcha,
                       const OfflineTtsKokoroModelConfig &kokoro,
                       const OfflineTtsZipvoiceModelConfig &zipvoice,
                       const OfflineTtsKittenModelConfig &kitten,
                       const OfflineTtsPocketModelConfig &pocket,
                       const OfflineTtsSupertonicModelConfig &supertonic,
                       const OfflineTtsOmnivoiceModelConfig &omnivoice,
                       const OfflineTtsF5ModelConfig &f5,
                       const OfflineTtsIndexTts2ModelConfig &indextts2,
                       int32_t num_threads, bool debug,
                       const std::string &provider) {
             OfflineTtsModelConfig c(vits, matcha, kokoro, zipvoice, kitten,
                                     pocket, supertonic, omnivoice, f5,
                                     num_threads, debug, provider);
             c.indextts2 = indextts2;
             return c;
           }),
           py::arg("vits") = OfflineTtsVitsModelConfig{},
           py::arg("matcha") = OfflineTtsMatchaModelConfig{},
           py::arg("kokoro") = OfflineTtsKokoroModelConfig{},
           py::arg("zipvoice") = OfflineTtsZipvoiceModelConfig{},
           py::arg("kitten") = OfflineTtsKittenModelConfig{},
           py::arg("pocket") = OfflineTtsPocketModelConfig{},
           py::arg("supertonic") = OfflineTtsSupertonicModelConfig{},
           py::arg("omnivoice") = OfflineTtsOmnivoiceModelConfig{},
           py::arg("f5") = OfflineTtsF5ModelConfig{},
           py::arg("indextts2") = OfflineTtsIndexTts2ModelConfig{},
           py::arg("num_threads") = 1, py::arg("debug") = false,
           py::arg("provider") = "cpu")
      .def_readwrite("vits", &PyClass::vits)
      .def_readwrite("matcha", &PyClass::matcha)
      .def_readwrite("kokoro", &PyClass::kokoro)
      .def_readwrite("zipvoice", &PyClass::zipvoice)
      .def_readwrite("kitten", &PyClass::kitten)
      .def_readwrite("pocket", &PyClass::pocket)
      .def_readwrite("supertonic", &PyClass::supertonic)
      .def_readwrite("omnivoice", &PyClass::omnivoice)
      .def_readwrite("f5", &PyClass::f5)
      .def_readwrite("indextts2", &PyClass::indextts2)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
