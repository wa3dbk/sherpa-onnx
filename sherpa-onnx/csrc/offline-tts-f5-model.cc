// sherpa-onnx/csrc/offline-tts-f5-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-f5-model.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
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

namespace sherpa_onnx {

class OfflineTtsF5Model::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.f5.transformer), sess_opts_);
    Init(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.f5.transformer);
    Init(buf.data(), buf.size());
  }

  const OfflineTtsF5ModelMetaData &GetMetaData() const { return meta_data_; }

  Ort::Value Run(Ort::Value text_ids, Ort::Value cond, int32_t num_steps,
                 float guidance_scale, float sway_coef, int32_t seed) const {
    auto cond_shape = cond.GetTensorTypeAndShapeInfo().GetShape();
    int64_t batch = cond_shape[0];
    int64_t total_frames = cond_shape[1];
    int64_t mel_dim = cond_shape[2];
    int64_t N = batch * total_frames * mel_dim;

    std::mt19937 rng;
    if (seed < 0) {
      std::random_device rd;
      rng.seed(rd());
    } else {
      rng.seed(static_cast<uint32_t>(seed));
    }
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> x_data(N);
    for (int64_t i = 0; i < N; ++i) x_data[i] = nd(rng);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> x_shape = {batch, total_frames, mel_dim};
    Ort::Value x = Ort::Value::CreateTensor<float>(
        memory_info, x_data.data(), x_data.size(), x_shape.data(),
        x_shape.size());

    // Build time schedule with F5-TTS sway sampling:
    //   t_new = t + sway_coef * (cos(pi/2 * t) - 1 + t)
    std::vector<float> timesteps(num_steps + 1);
    const float kPiHalf = 1.57079632679489661923f;
    for (int32_t i = 0; i <= num_steps; ++i) {
      float t = static_cast<float>(i) / static_cast<float>(num_steps);
      timesteps[i] =
          t + sway_coef * (std::cos(kPiHalf * t) - 1.0f + t);
    }

    int64_t scalar_shape = 1;
    Ort::Value gs_tensor = Ort::Value::CreateTensor<float>(
        memory_info, &guidance_scale, 1, &scalar_shape, 1);

    float *x_ptr = x.GetTensorMutableData<float>();

    for (int32_t step = 0; step < num_steps; ++step) {
      float t = timesteps[step];
      float dt = timesteps[step + 1] - t;

      Ort::Value t_tensor = Ort::Value::CreateTensor<float>(
          memory_info, &t, 1, &scalar_shape, 1);

      std::vector<Ort::Value> inputs;
      inputs.reserve(5);
      inputs.push_back(View(&x));
      inputs.push_back(View(&cond));
      inputs.push_back(View(&text_ids));
      inputs.push_back(std::move(t_tensor));
      inputs.push_back(View(&gs_tensor));

      auto out = sess_->Run({}, input_names_ptr_.data(), inputs.data(),
                            inputs.size(), output_names_ptr_.data(),
                            output_names_ptr_.size());

      const float *v_ptr = out[0].GetTensorData<float>();
      for (int64_t i = 0; i < N; ++i) {
        x_ptr[i] += v_ptr[i] * dt;
      }
    }

    return x;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    if (model_data) {
      sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                             model_data_length, sess_opts_);
    } else if (!sess_) {
      SHERPA_ONNX_LOGE("F5 transformer session not initialized");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    Ort::AllocatorWithDefaultOptions allocator;  // used in macro
    Ort::ModelMetadata meta = sess_->GetModelMetadata();

    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.version, "version", 1);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.feat_dim, "feat_dim",
                                            100);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.sample_rate,
                                            "sample_rate", 24000);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.n_fft, "n_fft", 1024);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.hop_length, "hop_length",
                                            256);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.window_length,
                                            "window_length", 1024);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.num_mels, "num_mels",
                                            100);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.use_espeak, "use_espeak",
                                            1);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.vocab_size, "vocab_size",
                                            2545);

    if (config_.debug) {
      std::ostringstream os;
      os << "---f5 transformer---\n";
      PrintModelMetadata(os, meta);
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

  OfflineTtsF5ModelMetaData meta_data_;
};

OfflineTtsF5Model::OfflineTtsF5Model(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsF5Model::OfflineTtsF5Model(Manager *mgr,
                                     const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsF5Model::~OfflineTtsF5Model() = default;

const OfflineTtsF5ModelMetaData &OfflineTtsF5Model::GetMetaData() const {
  return impl_->GetMetaData();
}

Ort::Value OfflineTtsF5Model::Run(Ort::Value text_ids, Ort::Value cond,
                                  int32_t num_steps, float guidance_scale,
                                  float sway_coef, int32_t seed) const {
  return impl_->Run(std::move(text_ids), std::move(cond), num_steps,
                    guidance_scale, sway_coef, seed);
}

#if __ANDROID_API__ >= 9
template OfflineTtsF5Model::OfflineTtsF5Model(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsF5Model::OfflineTtsF5Model(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
