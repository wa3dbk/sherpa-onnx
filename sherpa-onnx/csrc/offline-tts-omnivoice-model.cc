// sherpa-onnx/csrc/offline-tts-omnivoice-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-omnivoice-model.h"

#include <memory>
#include <sstream>
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

class OfflineTtsOmnivoiceModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    lm_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.omnivoice.model), sess_opts_);
    InitLm(nullptr, 0);

    enc_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.omnivoice.codec_encoder),
        sess_opts_);
    InitEnc(nullptr, 0);

    dec_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.omnivoice.codec_decoder),
        sess_opts_);
    InitDec(nullptr, 0);

    if (!config.omnivoice.prefix_model.empty() &&
        !config.omnivoice.target_model.empty()) {
      pref_sess_ = std::make_unique<Ort::Session>(
          env_, SHERPA_ONNX_TO_ORT_PATH(config.omnivoice.prefix_model),
          sess_opts_);
      InitPref(nullptr, 0);
      tgt_sess_ = std::make_unique<Ort::Session>(
          env_, SHERPA_ONNX_TO_ORT_PATH(config.omnivoice.target_model),
          sess_opts_);
      InitTgt(nullptr, 0);
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.omnivoice.model);
    InitLm(buf.data(), buf.size());

    buf = ReadFile(mgr, config.omnivoice.codec_encoder);
    InitEnc(buf.data(), buf.size());

    buf = ReadFile(mgr, config.omnivoice.codec_decoder);
    InitDec(buf.data(), buf.size());

    if (!config.omnivoice.prefix_model.empty() &&
        !config.omnivoice.target_model.empty()) {
      buf = ReadFile(mgr, config.omnivoice.prefix_model);
      InitPref(buf.data(), buf.size());
      buf = ReadFile(mgr, config.omnivoice.target_model);
      InitTgt(buf.data(), buf.size());
    }
  }

  bool HasCachedLM() const {
    return pref_sess_ && tgt_sess_;
  }

  const OfflineTtsOmnivoiceModelMetaData &GetMetaData() const {
    return meta_data_;
  }

  Ort::Value RunLM(Ort::Value input_ids, Ort::Value audio_mask,
                   Ort::Value attention_mask, Ort::Value position_ids) {
    std::vector<Ort::Value> inputs;
    inputs.reserve(4);
    inputs.push_back(std::move(input_ids));
    inputs.push_back(std::move(audio_mask));
    inputs.push_back(std::move(attention_mask));
    inputs.push_back(std::move(position_ids));

    auto out = lm_sess_->Run({}, lm_input_names_ptr_.data(), inputs.data(),
                             inputs.size(), lm_output_names_ptr_.data(),
                             lm_output_names_ptr_.size());
    return std::move(out[0]);
  }

  std::vector<Ort::Value> RunLMPrefix(Ort::Value input_ids,
                                      Ort::Value audio_mask,
                                      Ort::Value attention_mask,
                                      Ort::Value position_ids) {
    std::vector<Ort::Value> inputs;
    inputs.reserve(4);
    inputs.push_back(std::move(input_ids));
    inputs.push_back(std::move(audio_mask));
    inputs.push_back(std::move(attention_mask));
    inputs.push_back(std::move(position_ids));

    return pref_sess_->Run({}, pref_input_names_ptr_.data(), inputs.data(),
                           inputs.size(), pref_output_names_ptr_.data(),
                           pref_output_names_ptr_.size());
  }

  Ort::Value RunLMTarget(Ort::Value input_ids, Ort::Value audio_mask,
                         Ort::Value attention_mask, Ort::Value position_ids,
                         std::vector<Ort::Value> past) {
    std::vector<Ort::Value> inputs;
    inputs.reserve(4 + past.size());
    inputs.push_back(std::move(input_ids));
    inputs.push_back(std::move(audio_mask));
    inputs.push_back(std::move(attention_mask));
    inputs.push_back(std::move(position_ids));
    for (auto &p : past) inputs.push_back(std::move(p));

    auto out = tgt_sess_->Run({}, tgt_input_names_ptr_.data(), inputs.data(),
                              inputs.size(), tgt_output_names_ptr_.data(),
                              tgt_output_names_ptr_.size());
    return std::move(out[0]);
  }

  Ort::Value EncodeAudio(Ort::Value pcm) {
    std::vector<Ort::Value> inputs;
    inputs.reserve(1);
    inputs.push_back(std::move(pcm));

    auto out = enc_sess_->Run({}, enc_input_names_ptr_.data(), inputs.data(),
                              inputs.size(), enc_output_names_ptr_.data(),
                              enc_output_names_ptr_.size());
    return std::move(out[0]);
  }

  Ort::Value DecodeCodes(Ort::Value codes) {
    std::vector<Ort::Value> inputs;
    inputs.reserve(1);
    inputs.push_back(std::move(codes));

    auto out = dec_sess_->Run({}, dec_input_names_ptr_.data(), inputs.data(),
                              inputs.size(), dec_output_names_ptr_.data(),
                              dec_output_names_ptr_.size());
    return std::move(out[0]);
  }

 private:
  void InitLm(void *data, size_t data_len) {
    if (data) {
      lm_sess_ =
          std::make_unique<Ort::Session>(env_, data, data_len, sess_opts_);
    } else if (!lm_sess_) {
      SHERPA_ONNX_LOGE("LM session not initialized");
      SHERPA_ONNX_EXIT(-1);
    }
    GetInputNames(lm_sess_.get(), &lm_input_names_, &lm_input_names_ptr_);
    GetOutputNames(lm_sess_.get(), &lm_output_names_, &lm_output_names_ptr_);
    DebugPrint("lm", lm_sess_.get(), lm_input_names_, lm_output_names_);
  }

  void InitEnc(void *data, size_t data_len) {
    if (data) {
      enc_sess_ =
          std::make_unique<Ort::Session>(env_, data, data_len, sess_opts_);
    } else if (!enc_sess_) {
      SHERPA_ONNX_LOGE("codec encoder session not initialized");
      SHERPA_ONNX_EXIT(-1);
    }
    GetInputNames(enc_sess_.get(), &enc_input_names_, &enc_input_names_ptr_);
    GetOutputNames(enc_sess_.get(), &enc_output_names_,
                   &enc_output_names_ptr_);
    DebugPrint("codec_encoder", enc_sess_.get(), enc_input_names_,
               enc_output_names_);
  }

  void InitDec(void *data, size_t data_len) {
    if (data) {
      dec_sess_ =
          std::make_unique<Ort::Session>(env_, data, data_len, sess_opts_);
    } else if (!dec_sess_) {
      SHERPA_ONNX_LOGE("codec decoder session not initialized");
      SHERPA_ONNX_EXIT(-1);
    }
    GetInputNames(dec_sess_.get(), &dec_input_names_, &dec_input_names_ptr_);
    GetOutputNames(dec_sess_.get(), &dec_output_names_,
                   &dec_output_names_ptr_);
    DebugPrint("codec_decoder", dec_sess_.get(), dec_input_names_,
               dec_output_names_);
  }

  void InitPref(void *data, size_t data_len) {
    if (data) {
      pref_sess_ =
          std::make_unique<Ort::Session>(env_, data, data_len, sess_opts_);
    } else if (!pref_sess_) {
      SHERPA_ONNX_LOGE("prefix session not initialized");
      SHERPA_ONNX_EXIT(-1);
    }
    GetInputNames(pref_sess_.get(), &pref_input_names_,
                  &pref_input_names_ptr_);
    GetOutputNames(pref_sess_.get(), &pref_output_names_,
                   &pref_output_names_ptr_);
    DebugPrint("prefix", pref_sess_.get(), pref_input_names_,
               pref_output_names_);
  }

  void InitTgt(void *data, size_t data_len) {
    if (data) {
      tgt_sess_ =
          std::make_unique<Ort::Session>(env_, data, data_len, sess_opts_);
    } else if (!tgt_sess_) {
      SHERPA_ONNX_LOGE("target session not initialized");
      SHERPA_ONNX_EXIT(-1);
    }
    GetInputNames(tgt_sess_.get(), &tgt_input_names_, &tgt_input_names_ptr_);
    GetOutputNames(tgt_sess_.get(), &tgt_output_names_,
                   &tgt_output_names_ptr_);
    DebugPrint("target", tgt_sess_.get(), tgt_input_names_, tgt_output_names_);
  }

  void DebugPrint(const char *tag, Ort::Session *sess,
                  const std::vector<std::string> &ins,
                  const std::vector<std::string> &outs) {
    if (!config_.debug) return;
    std::ostringstream os;
    os << "---" << tag << "---\n";
    auto md = sess->GetModelMetadata();
    PrintModelMetadata(os, md);
    os << "inputs:\n";
    for (size_t i = 0; i < ins.size(); ++i) os << "  " << i << " " << ins[i] << "\n";
    os << "outputs:\n";
    for (size_t i = 0; i < outs.size(); ++i) os << "  " << i << " " << outs[i] << "\n";
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
  }

  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> lm_sess_;
  std::unique_ptr<Ort::Session> enc_sess_;
  std::unique_ptr<Ort::Session> dec_sess_;

  std::vector<std::string> lm_input_names_;
  std::vector<const char *> lm_input_names_ptr_;
  std::vector<std::string> lm_output_names_;
  std::vector<const char *> lm_output_names_ptr_;

  std::vector<std::string> enc_input_names_;
  std::vector<const char *> enc_input_names_ptr_;
  std::vector<std::string> enc_output_names_;
  std::vector<const char *> enc_output_names_ptr_;

  std::vector<std::string> dec_input_names_;
  std::vector<const char *> dec_input_names_ptr_;
  std::vector<std::string> dec_output_names_;
  std::vector<const char *> dec_output_names_ptr_;

  std::unique_ptr<Ort::Session> pref_sess_;
  std::vector<std::string> pref_input_names_;
  std::vector<const char *> pref_input_names_ptr_;
  std::vector<std::string> pref_output_names_;
  std::vector<const char *> pref_output_names_ptr_;

  std::unique_ptr<Ort::Session> tgt_sess_;
  std::vector<std::string> tgt_input_names_;
  std::vector<const char *> tgt_input_names_ptr_;
  std::vector<std::string> tgt_output_names_;
  std::vector<const char *> tgt_output_names_ptr_;

  OfflineTtsOmnivoiceModelMetaData meta_data_;
};

OfflineTtsOmnivoiceModel::OfflineTtsOmnivoiceModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsOmnivoiceModel::OfflineTtsOmnivoiceModel(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsOmnivoiceModel::~OfflineTtsOmnivoiceModel() = default;

const OfflineTtsOmnivoiceModelMetaData &
OfflineTtsOmnivoiceModel::GetMetaData() const {
  return impl_->GetMetaData();
}

Ort::Value OfflineTtsOmnivoiceModel::RunLM(Ort::Value input_ids,
                                           Ort::Value audio_mask,
                                           Ort::Value attention_mask,
                                           Ort::Value position_ids) const {
  return impl_->RunLM(std::move(input_ids), std::move(audio_mask),
                      std::move(attention_mask), std::move(position_ids));
}

bool OfflineTtsOmnivoiceModel::HasCachedLM() const {
  return impl_->HasCachedLM();
}

std::vector<Ort::Value> OfflineTtsOmnivoiceModel::RunLMPrefix(
    Ort::Value input_ids, Ort::Value audio_mask, Ort::Value attention_mask,
    Ort::Value position_ids) const {
  return impl_->RunLMPrefix(std::move(input_ids), std::move(audio_mask),
                            std::move(attention_mask),
                            std::move(position_ids));
}

Ort::Value OfflineTtsOmnivoiceModel::RunLMTarget(
    Ort::Value input_ids, Ort::Value audio_mask, Ort::Value attention_mask,
    Ort::Value position_ids, std::vector<Ort::Value> past) const {
  return impl_->RunLMTarget(std::move(input_ids), std::move(audio_mask),
                            std::move(attention_mask),
                            std::move(position_ids), std::move(past));
}

Ort::Value OfflineTtsOmnivoiceModel::EncodeAudio(Ort::Value pcm) const {
  return impl_->EncodeAudio(std::move(pcm));
}

Ort::Value OfflineTtsOmnivoiceModel::DecodeCodes(Ort::Value codes) const {
  return impl_->DecodeCodes(std::move(codes));
}

#if __ANDROID_API__ >= 9
template OfflineTtsOmnivoiceModel::OfflineTtsOmnivoiceModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsOmnivoiceModel::OfflineTtsOmnivoiceModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
