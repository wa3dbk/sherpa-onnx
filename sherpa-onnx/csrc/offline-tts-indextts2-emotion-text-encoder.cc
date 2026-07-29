// sherpa-onnx/csrc/offline-tts-indextts2-emotion-text-encoder.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-indextts2-emotion-text-encoder.h"

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

class OfflineTtsIndexTts2EmotionTextEncoder::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    sess_ = std::make_unique<Ort::Session>(
        env_,
        SHERPA_ONNX_TO_ORT_PATH(config.indextts2.emotion_text_encoder),
        sess_opts_);
    Init();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.indextts2.emotion_text_encoder);
    sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                           sess_opts_);
    Init();
  }

  Ort::Value Run(const std::vector<int64_t> &token_ids) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> shape = {1,
                                    static_cast<int64_t>(token_ids.size())};
    Ort::Value ids = Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t *>(token_ids.data()),
        token_ids.size(), shape.data(), shape.size());

    std::vector<int64_t> mask_data(token_ids.size(), 1);
    Ort::Value mask = Ort::Value::CreateTensor<int64_t>(
        memory_info, mask_data.data(), mask_data.size(), shape.data(),
        shape.size());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(ids));
    inputs.push_back(std::move(mask));

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

OfflineTtsIndexTts2EmotionTextEncoder::OfflineTtsIndexTts2EmotionTextEncoder(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsIndexTts2EmotionTextEncoder::OfflineTtsIndexTts2EmotionTextEncoder(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsIndexTts2EmotionTextEncoder::
    ~OfflineTtsIndexTts2EmotionTextEncoder() = default;

Ort::Value OfflineTtsIndexTts2EmotionTextEncoder::Run(
    const std::vector<int64_t> &token_ids) const {
  return impl_->Run(token_ids);
}

#if __ANDROID_API__ >= 9
template OfflineTtsIndexTts2EmotionTextEncoder::
    OfflineTtsIndexTts2EmotionTextEncoder(
        AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsIndexTts2EmotionTextEncoder::
    OfflineTtsIndexTts2EmotionTextEncoder(
        NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
