// sherpa-onnx/csrc/offline-tts-indextts2-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-indextts2-model.h"

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

class OfflineTtsIndexTts2Model::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.indextts2.lm), sess_opts_);
    Init(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.indextts2.lm);
    Init(buf.data(), buf.size());
  }

  const OfflineTtsIndexTts2ModelMetaData &GetMetaData() const {
    return meta_data_;
  }

  StepOutput PrefixPass(Ort::Value text_tokens, Ort::Value speaker_embed,
                        Ort::Value emotion_embed) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t mode_val = 0;
    std::array<int64_t, 0> mode_shape{};
    Ort::Value mode = Ort::Value::CreateTensor<int32_t>(
        memory_info, &mode_val, 1, mode_shape.data(), 0);

    int64_t dummy_prev = 0;
    std::array<int64_t, 2> prev_shape = {1, 1};
    Ort::Value prev_token = Ort::Value::CreateTensor<int64_t>(
        memory_info, &dummy_prev, 1, prev_shape.data(), prev_shape.size());

    std::array<int64_t, 6> past_kv_shape = {
        meta_data_.kv_num_layers, 2, 1, meta_data_.kv_num_heads, 1,
        meta_data_.kv_head_dim,
    };
    int64_t past_kv_n = past_kv_shape[0] * past_kv_shape[1] * past_kv_shape[2] *
                        past_kv_shape[3] * past_kv_shape[4] * past_kv_shape[5];
    std::vector<float> past_kv_data(past_kv_n, 0.0f);
    Ort::Value past_kv = Ort::Value::CreateTensor<float>(
        memory_info, past_kv_data.data(), past_kv_data.size(),
        past_kv_shape.data(), past_kv_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.reserve(6);
    inputs.push_back(std::move(mode));
    inputs.push_back(std::move(text_tokens));
    inputs.push_back(std::move(speaker_embed));
    inputs.push_back(std::move(emotion_embed));
    inputs.push_back(std::move(prev_token));
    inputs.push_back(std::move(past_kv));

    auto out = sess_->Run({}, input_names_ptr_.data(), inputs.data(),
                          inputs.size(), output_names_ptr_.data(),
                          output_names_ptr_.size());
    return {std::move(out[0]), std::move(out[1])};
  }

  StepOutput StepOne(Ort::Value prev_token, Ort::Value past_kv) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t mode_val = 1;
    std::array<int64_t, 0> mode_shape{};
    Ort::Value mode = Ort::Value::CreateTensor<int32_t>(
        memory_info, &mode_val, 1, mode_shape.data(), 0);

    std::array<int64_t, 2> tt_shape = {1, 0};
    Ort::Value text_tokens = Ort::Value::CreateTensor<int64_t>(
        allocator_, tt_shape.data(), tt_shape.size());

    std::array<int64_t, 2> se_shape = {1, meta_data_.speaker_embed_dim};
    std::vector<float> se_zero(meta_data_.speaker_embed_dim, 0.0f);
    Ort::Value speaker_embed = Ort::Value::CreateTensor<float>(
        memory_info, se_zero.data(), se_zero.size(), se_shape.data(),
        se_shape.size());

    std::array<int64_t, 2> ee_shape = {1, meta_data_.emotion_embed_dim};
    std::vector<float> ee_zero(meta_data_.emotion_embed_dim, 0.0f);
    Ort::Value emotion_embed = Ort::Value::CreateTensor<float>(
        memory_info, ee_zero.data(), ee_zero.size(), ee_shape.data(),
        ee_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.reserve(6);
    inputs.push_back(std::move(mode));
    inputs.push_back(std::move(text_tokens));
    inputs.push_back(std::move(speaker_embed));
    inputs.push_back(std::move(emotion_embed));
    inputs.push_back(std::move(prev_token));
    inputs.push_back(std::move(past_kv));

    auto out = sess_->Run({}, input_names_ptr_.data(), inputs.data(),
                          inputs.size(), output_names_ptr_.data(),
                          output_names_ptr_.size());
    return {std::move(out[0]), std::move(out[1])};
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    if (model_data) {
      sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                             model_data_length, sess_opts_);
    } else if (!sess_) {
      SHERPA_ONNX_LOGE("IndexTTS-2 LM session not initialized");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();

    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.version, "version", 1);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.sample_rate,
                                            "sample_rate", 24000);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.vocab_size,
                                            "vocab_size", 32000);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.bos_token_id,
                                            "bos_token_id", 1);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.eos_token_id,
                                            "eos_token_id", 2);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.pad_token_id,
                                            "pad_token_id", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.speaker_embed_dim,
                                            "speaker_embed_dim", 512);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.emotion_embed_dim,
                                            "emotion_embed_dim", 512);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.kv_num_layers,
                                            "kv_num_layers", 16);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.kv_num_heads,
                                            "kv_num_heads", 16);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.kv_head_dim,
                                            "kv_head_dim", 64);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.max_audio_tokens,
                                            "max_audio_tokens", 2000);

    if (config_.debug) {
      std::ostringstream os;
      os << "---indextts2 lm---\n";
      PrintModelMetadata(os, meta_data);
      os << "----------input names----------\n";
      for (size_t i = 0; i < input_names_.size(); ++i) {
        os << i << " " << input_names_[i] << "\n";
      }
      os << "----------output names----------\n";
      for (size_t i = 0; i < output_names_.size(); ++i) {
        os << i << " " << output_names_[i] << "\n";
      }
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }
  }

 private:
  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;
  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  OfflineTtsIndexTts2ModelMetaData meta_data_;
};

OfflineTtsIndexTts2Model::OfflineTtsIndexTts2Model(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsIndexTts2Model::OfflineTtsIndexTts2Model(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsIndexTts2Model::~OfflineTtsIndexTts2Model() = default;

const OfflineTtsIndexTts2ModelMetaData &
OfflineTtsIndexTts2Model::GetMetaData() const {
  return impl_->GetMetaData();
}

OfflineTtsIndexTts2Model::StepOutput OfflineTtsIndexTts2Model::PrefixPass(
    Ort::Value text_tokens, Ort::Value speaker_embed,
    Ort::Value emotion_embed) const {
  return impl_->PrefixPass(std::move(text_tokens), std::move(speaker_embed),
                           std::move(emotion_embed));
}

OfflineTtsIndexTts2Model::StepOutput OfflineTtsIndexTts2Model::StepOne(
    Ort::Value prev_token, Ort::Value past_kv) const {
  return impl_->StepOne(std::move(prev_token), std::move(past_kv));
}

#if __ANDROID_API__ >= 9
template OfflineTtsIndexTts2Model::OfflineTtsIndexTts2Model(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsIndexTts2Model::OfflineTtsIndexTts2Model(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
