// sherpa-onnx/python/csrc/offline-tts-f5-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/python/csrc/offline-tts-f5-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-f5-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsF5ModelConfig(py::module *m) {
  using PyClass = OfflineTtsF5ModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsF5ModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, int32_t, float, float, float,
                    int32_t>(),
           py::arg("transformer"), py::arg("vocoder"), py::arg("tokens"),
           py::arg("data_dir") = "", py::arg("lexicon") = "",
           py::arg("num_steps") = 32, py::arg("guidance_scale") = 2.0f,
           py::arg("sway_coef") = -1.0f, py::arg("target_rms") = 0.1f,
           py::arg("seed") = -1)
      .def_readwrite("transformer", &PyClass::transformer)
      .def_readwrite("vocoder", &PyClass::vocoder)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("data_dir", &PyClass::data_dir)
      .def_readwrite("lexicon", &PyClass::lexicon)
      .def_readwrite("num_steps", &PyClass::num_steps)
      .def_readwrite("guidance_scale", &PyClass::guidance_scale)
      .def_readwrite("sway_coef", &PyClass::sway_coef)
      .def_readwrite("target_rms", &PyClass::target_rms)
      .def_readwrite("seed", &PyClass::seed)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_onnx
