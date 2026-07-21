// sherpa-onnx/python/csrc/offline-tts-omnivoice-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/python/csrc/offline-tts-omnivoice-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-omnivoice-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsOmnivoiceModelConfig(py::module *m) {
  using PyClass = OfflineTtsOmnivoiceModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsOmnivoiceModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &, int32_t, float,
                    float, float, float, int32_t>(),
           py::arg("model"), py::arg("codec_encoder"),
           py::arg("codec_decoder"), py::arg("tokenizer_dir"),
           py::arg("num_steps") = 32, py::arg("t_shift") = 0.1f,
           py::arg("guidance_scale") = 2.0f,
           py::arg("layer_penalty_factor") = 5.0f,
           py::arg("position_temperature") = 5.0f, py::arg("seed") = -1)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("codec_encoder", &PyClass::codec_encoder)
      .def_readwrite("codec_decoder", &PyClass::codec_decoder)
      .def_readwrite("tokenizer_dir", &PyClass::tokenizer_dir)
      .def_readwrite("prefix_model", &PyClass::prefix_model)
      .def_readwrite("target_model", &PyClass::target_model)
      .def_readwrite("num_steps", &PyClass::num_steps)
      .def_readwrite("t_shift", &PyClass::t_shift)
      .def_readwrite("guidance_scale", &PyClass::guidance_scale)
      .def_readwrite("layer_penalty_factor", &PyClass::layer_penalty_factor)
      .def_readwrite("position_temperature", &PyClass::position_temperature)
      .def_readwrite("seed", &PyClass::seed)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_onnx
