// sherpa-onnx/csrc/offline-tts-indextts2-audio-encoder.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-indextts2-audio-encoder.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/ort-env.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineTtsIndexTts2AudioEncoder::Impl {
 public:
  Impl(const OfflineTtsModelConfig &config, const std::string &onnx_path)
      : env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(onnx_path), sess_opts_);
    Init();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config,
       const std::string &onnx_path)
      : env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, onnx_path);
    sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                           sess_opts_);
    Init();
  }

  Ort::Value Run(const std::vector<float> &samples) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> shape = {1, static_cast<int64_t>(samples.size())};
    Ort::Value audio = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(samples.data()), samples.size(),
        shape.data(), shape.size());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(audio));

    auto out = sess_->Run({}, input_names_ptr_.data(), inputs.data(),
                          inputs.size(), output_names_ptr_.data(),
                          output_names_ptr_.size());
    return std::move(out[0]);
  }

 private:
  void Init() {
    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
  }

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;
  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

OfflineTtsIndexTts2AudioEncoder::OfflineTtsIndexTts2AudioEncoder(
    const OfflineTtsModelConfig &config, const std::string &onnx_path)
    : impl_(std::make_unique<Impl>(config, onnx_path)) {}

template <typename Manager>
OfflineTtsIndexTts2AudioEncoder::OfflineTtsIndexTts2AudioEncoder(
    Manager *mgr, const OfflineTtsModelConfig &config,
    const std::string &onnx_path)
    : impl_(std::make_unique<Impl>(mgr, config, onnx_path)) {}

OfflineTtsIndexTts2AudioEncoder::~OfflineTtsIndexTts2AudioEncoder() = default;

Ort::Value OfflineTtsIndexTts2AudioEncoder::Run(
    const std::vector<float> &samples) const {
  return impl_->Run(samples);
}

#if __ANDROID_API__ >= 9
template OfflineTtsIndexTts2AudioEncoder::OfflineTtsIndexTts2AudioEncoder(
    AAssetManager *mgr, const OfflineTtsModelConfig &config,
    const std::string &onnx_path);
#endif

#if __OHOS__
template OfflineTtsIndexTts2AudioEncoder::OfflineTtsIndexTts2AudioEncoder(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config,
    const std::string &onnx_path);
#endif

}  // namespace sherpa_onnx
