// sherpa-onnx/csrc/offline-tts-indextts2-vocoder.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-indextts2-vocoder.h"

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

class OfflineTtsIndexTts2Vocoder::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.indextts2.vocoder), sess_opts_);
    Init();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.indextts2.vocoder);
    sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                           sess_opts_);
    Init();
  }

  std::vector<float> Run(const std::vector<int64_t> &audio_tokens) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> shape = {
        1, static_cast<int64_t>(audio_tokens.size())};
    Ort::Value in = Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t *>(audio_tokens.data()),
        audio_tokens.size(), shape.data(), shape.size());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(in));

    auto out = sess_->Run({}, input_names_ptr_.data(), inputs.data(),
                          inputs.size(), output_names_ptr_.data(),
                          output_names_ptr_.size());

    const float *p = out[0].GetTensorData<float>();
    auto shape_out = out[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t n = 1;
    for (auto d : shape_out) n *= d;
    return std::vector<float>(p, p + n);
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

OfflineTtsIndexTts2Vocoder::OfflineTtsIndexTts2Vocoder(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsIndexTts2Vocoder::OfflineTtsIndexTts2Vocoder(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsIndexTts2Vocoder::~OfflineTtsIndexTts2Vocoder() = default;

std::vector<float> OfflineTtsIndexTts2Vocoder::Run(
    const std::vector<int64_t> &audio_tokens) const {
  return impl_->Run(audio_tokens);
}

#if __ANDROID_API__ >= 9
template OfflineTtsIndexTts2Vocoder::OfflineTtsIndexTts2Vocoder(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsIndexTts2Vocoder::OfflineTtsIndexTts2Vocoder(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
