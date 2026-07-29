// sherpa-onnx/csrc/offline-tts-indextts2-sampler.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//                2026  Waad Ben Kheder

#include "sherpa-onnx/csrc/offline-tts-indextts2-sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace sherpa_onnx {

OfflineTtsIndexTts2Sampler::OfflineTtsIndexTts2Sampler(int32_t top_k,
                                                      float top_p,
                                                      float temperature,
                                                      int32_t seed)
    : top_k_(top_k), top_p_(top_p), temperature_(temperature) {
  if (seed < 0) {
    std::random_device rd;
    rng_.seed(rd());
  } else {
    rng_.seed(static_cast<uint32_t>(seed));
  }
}

int64_t OfflineTtsIndexTts2Sampler::Sample(float *logits,
                                           int64_t vocab_size) {
  // Argmax fast path.
  if (temperature_ <= 0.0f || top_k_ == 1) {
    int64_t best = 0;
    float best_v = logits[0];
    for (int64_t i = 1; i < vocab_size; ++i) {
      if (logits[i] > best_v) {
        best_v = logits[i];
        best = i;
      }
    }
    return best;
  }

  // Scale by temperature.
  float inv_t = 1.0f / temperature_;
  for (int64_t i = 0; i < vocab_size; ++i) logits[i] *= inv_t;

  // Top-k mask.
  if (top_k_ > 0 && top_k_ < vocab_size) {
    std::vector<int64_t> idx(vocab_size);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + top_k_, idx.end(),
                      [&](int64_t a, int64_t b) {
                        return logits[a] > logits[b];
                      });
    float threshold = logits[idx[top_k_ - 1]];
    for (int64_t i = 0; i < vocab_size; ++i) {
      if (logits[i] < threshold) logits[i] = -std::numeric_limits<float>::infinity();
    }
  }

  // Softmax.
  float max_l = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < vocab_size; ++i) max_l = std::max(max_l, logits[i]);
  double sum = 0.0;
  std::vector<double> probs(vocab_size);
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[i] = std::exp(static_cast<double>(logits[i] - max_l));
    sum += probs[i];
  }
  if (sum <= 0.0) return 0;
  for (int64_t i = 0; i < vocab_size; ++i) probs[i] /= sum;

  // Top-p mask.
  if (top_p_ < 1.0f) {
    std::vector<int64_t> order(vocab_size);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int64_t a, int64_t b) { return probs[a] > probs[b]; });
    double cum = 0.0;
    size_t cutoff = order.size();
    for (size_t i = 0; i < order.size(); ++i) {
      cum += probs[order[i]];
      if (cum >= top_p_) {
        cutoff = i + 1;
        break;
      }
    }
    std::vector<bool> keep(vocab_size, false);
    for (size_t i = 0; i < cutoff; ++i) keep[order[i]] = true;
    double kept_sum = 0.0;
    for (int64_t i = 0; i < vocab_size; ++i) {
      if (!keep[i]) probs[i] = 0.0;
      kept_sum += probs[i];
    }
    if (kept_sum > 0.0) {
      for (int64_t i = 0; i < vocab_size; ++i) probs[i] /= kept_sum;
    }
  }

  // Draw.
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double r = u(rng_);
  double acc = 0.0;
  for (int64_t i = 0; i < vocab_size; ++i) {
    acc += probs[i];
    if (r <= acc) return i;
  }
  return vocab_size - 1;
}

}  // namespace sherpa_onnx
