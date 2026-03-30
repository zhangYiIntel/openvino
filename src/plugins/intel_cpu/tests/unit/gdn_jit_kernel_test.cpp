// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include "kernels/x64/gdn_jit_kernel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace {

using GDNJitParams = std::tuple<ov::element::Type, size_t, size_t, bool>;  // precision, seq_len(T), head_size(K), fuse_qk_l2norm

class GDNJitKernelTest : public testing::Test, public testing::WithParamInterface<GDNJitParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GDNJitParams>& obj) {
        const auto& [precision, seq_len, head_size, fuse_qk_l2norm] = obj.param;
        std::ostringstream result;
        result << "Prec=" << precision.to_string() << "_T=" << seq_len << "_K=" << head_size
               << "_Fuse=" << (fuse_qk_l2norm ? "1" : "0");
        return result.str();
    }
};

inline uint16_t f32_to_bf16_trunc(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

inline float bf16_trunc_to_f32(uint16_t x) {
    uint32_t u = static_cast<uint32_t>(x) << 16;
    float out;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

inline float quantize_like_kernel(float x, ov::element::Type prc) {
    if (prc == ov::element::f32) {
        return x;
    }
    if (prc == ov::element::f16) {
        return static_cast<float>(ov::float16(x));
    }
    if (prc == ov::element::bf16) {
        return bf16_trunc_to_f32(f32_to_bf16_trunc(x));
    }
    return x;
}

inline size_t data_size(ov::element::Type prc) {
    return prc == ov::element::f32 ? sizeof(float) : sizeof(uint16_t);
}

void gdn_step_ref(std::vector<float>& state,
                  const std::vector<float>& key,
                  const std::vector<float>& query,
                  float value,
                  float gate,
                  float beta,
                  ov::element::Type prc,
                  float& output) {
    float hk = 0.0F;
    for (size_t i = 0; i < state.size(); i++) {
        state[i] = quantize_like_kernel(state[i] * gate, prc);
        hk += state[i] * key[i];
    }

    const float delta = (value - hk) * beta;

    output = 0.0F;
    for (size_t i = 0; i < state.size(); i++) {
        state[i] = quantize_like_kernel(state[i] + key[i] * delta, prc);
        output += state[i] * query[i];
    }
}

void l2norm_ref(std::vector<float>& x, float eps) {
    float sum = 0.0F;
    for (const auto& v : x) {
        sum += v * v;
    }
    const float inv = 1.0F / std::sqrt(sum + eps);
    for (auto& v : x) {
        v *= inv;
    }
}

float load_raw_as_f32(const std::vector<uint8_t>& raw, size_t idx, ov::element::Type prc) {
    if (prc == ov::element::f32) {
        const auto* p = reinterpret_cast<const float*>(raw.data());
        return p[idx];
    }
    if (prc == ov::element::f16) {
        const auto* p = reinterpret_cast<const ov::float16*>(raw.data());
        return static_cast<float>(p[idx]);
    }
    const auto* p = reinterpret_cast<const uint16_t*>(raw.data());
    return bf16_trunc_to_f32(p[idx]);
}

template <typename T>
std::vector<uint8_t> to_raw_bytes(const std::vector<float>& src) {
    std::vector<T> tmp(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        tmp[i] = static_cast<T>(src[i]);
    }
    std::vector<uint8_t> raw(tmp.size() * sizeof(T));
    std::memcpy(raw.data(), tmp.data(), raw.size());
    return raw;
}

std::vector<uint8_t> to_raw_bytes_bf16_trunc(const std::vector<float>& src) {
    std::vector<uint16_t> tmp(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        tmp[i] = f32_to_bf16_trunc(src[i]);
    }
    std::vector<uint8_t> raw(tmp.size() * sizeof(uint16_t));
    std::memcpy(raw.data(), tmp.data(), raw.size());
    return raw;
}

std::vector<float> from_raw_bytes_to_float(const std::vector<uint8_t>& raw, ov::element::Type prc) {
    if (prc == ov::element::f32) {
        std::vector<float> out(raw.size() / sizeof(float));
        std::memcpy(out.data(), raw.data(), raw.size());
        return out;
    }
    if (prc == ov::element::f16) {
        const auto n = raw.size() / sizeof(ov::float16);
        std::vector<float> out(n);
        const auto* p = reinterpret_cast<const ov::float16*>(raw.data());
        for (size_t i = 0; i < n; i++) {
            out[i] = static_cast<float>(p[i]);
        }
        return out;
    }

    const auto n = raw.size() / sizeof(uint16_t);
    std::vector<float> out(n);
    const auto* p = reinterpret_cast<const uint16_t*>(raw.data());
    for (size_t i = 0; i < n; i++) {
        out[i] = bf16_trunc_to_f32(p[i]);
    }
    return out;
}

void gdn_sequence_ref_same_inputs(std::vector<float>& state,
                                  const std::vector<uint8_t>& key_seq_raw,
                                  const std::vector<uint8_t>& query_seq_raw,
                                  const std::vector<float>& value_seq,
                                  const std::vector<float>& gate_seq,
                                  const std::vector<float>& beta_seq,
                                  size_t T,
                                  size_t K,
                                  bool fuse_qk_l2norm,
                                  float q_l2_norm_eps,
                                  float k_l2_norm_eps,
                                  ov::element::Type prc,
                                  std::vector<float>& out_ref) {
    const float q_scale = 1.0F / std::sqrt(static_cast<float>(K));
    for (size_t t = 0; t < T; t++) {
        std::vector<float> k_t(K);
        std::vector<float> q_t(K);
        for (size_t j = 0; j < K; j++) {
            k_t[j] = load_raw_as_f32(key_seq_raw, t * K + j, prc);
            q_t[j] = load_raw_as_f32(query_seq_raw, t * K + j, prc);
        }

        if (fuse_qk_l2norm) {
            l2norm_ref(k_t, k_l2_norm_eps);
            l2norm_ref(q_t, q_l2_norm_eps);
        }
        for (auto& qv : q_t) {
            qv *= q_scale;
        }

        const float gate_exp = std::exp(gate_seq[t]);
        gdn_step_ref(state, k_t, q_t, value_seq[t], gate_exp, beta_seq[t], prc, out_ref[t]);
    }
}

TEST_P(GDNJitKernelTest, CompareFinalResultsWithSameInputs) {
    const auto& [prc, T, K, fuse_qk_l2norm] = GetParam();
    constexpr float q_l2_norm_eps = 1e-6F;
    constexpr float k_l2_norm_eps = 1e-6F;

    std::mt19937 gen(static_cast<uint32_t>(20260329 + T * 131 + K * 17 + (fuse_qk_l2norm ? 1 : 0)));
    std::uniform_real_distribution<float> dist_state(-1.0F, 1.0F);
    std::uniform_real_distribution<float> dist_kq(-0.5F, 0.5F);
    std::uniform_real_distribution<float> dist_value(-0.8F, 0.8F);
    std::uniform_real_distribution<float> dist_gate_log(-1.0F, 1.0F);
    std::uniform_real_distribution<float> dist_beta(0.0F, 1.0F);

    auto jit = ov::intel_cpu::kernel::create_gdn_jit_kernel(prc, K, fuse_qk_l2norm, q_l2_norm_eps, k_l2_norm_eps);
    if (!jit) {
        GTEST_SKIP() << "GDN JIT kernel is unavailable for precision " << prc << ", T=" << T << ", K=" << K;
    }

    // Prepare one shared input set for both reference and JIT.
    std::vector<float> state_init(K);
    for (auto& v : state_init) {
        v = dist_state(gen);
    }

    std::vector<uint8_t> state_jit_raw;
    if (prc == ov::element::f32) {
        state_jit_raw = to_raw_bytes<float>(state_init);
    } else if (prc == ov::element::f16) {
        state_jit_raw = to_raw_bytes<ov::float16>(state_init);
    } else {
        state_jit_raw = to_raw_bytes_bf16_trunc(state_init);
    }

    std::vector<float> key_src(T * K);
    std::vector<float> query_src(T * K);
    for (auto& v : key_src) {
        v = dist_kq(gen);
    }
    for (auto& v : query_src) {
        v = dist_kq(gen);
    }

    std::vector<uint8_t> key_seq_raw;
    std::vector<uint8_t> query_seq_raw;
    if (prc == ov::element::f32) {
        key_seq_raw = to_raw_bytes<float>(key_src);
        query_seq_raw = to_raw_bytes<float>(query_src);
    } else if (prc == ov::element::f16) {
        key_seq_raw = to_raw_bytes<ov::float16>(key_src);
        query_seq_raw = to_raw_bytes<ov::float16>(query_src);
    } else {
        key_seq_raw = to_raw_bytes_bf16_trunc(key_src);
        query_seq_raw = to_raw_bytes_bf16_trunc(query_src);
    }

    std::vector<float> value_seq(T);
    std::vector<float> gate_seq(T);
    std::vector<float> beta_seq(T);
    for (size_t t = 0; t < T; t++) {
        value_seq[t] = dist_value(gen);
        gate_seq[t] = dist_gate_log(gen);
        beta_seq[t] = dist_beta(gen);
    }

    // Reference path (entire sequence first).
    std::vector<float> out_ref(T, 0.0F);
    std::vector<float> state_ref = from_raw_bytes_to_float(state_jit_raw, prc);
    gdn_sequence_ref_same_inputs(state_ref,
                                 key_seq_raw,
                                 query_seq_raw,
                                 value_seq,
                                 gate_seq,
                                 beta_seq,
                                 T,
                                 K,
                                 fuse_qk_l2norm,
                                 q_l2_norm_eps,
                                 k_l2_norm_eps,
                                 prc,
                                 out_ref);

    // JIT path (same exact inputs, one call for full sequence).
    std::vector<uint8_t> key_tmp_raw(K * data_size(prc), 0);
    std::vector<uint8_t> query_tmp_raw(K * data_size(prc), 0);
    std::vector<float> out_jit(T, 0.0F);

    ov::intel_cpu::kernel::jit_gdn_call_args args{};
    args.state = state_jit_raw.data();
    args.key_seq = key_seq_raw.data();
    args.query_seq = query_seq_raw.data();
    args.value_seq = value_seq.data();
    args.gate_seq = gate_seq.data();
    args.beta_seq = beta_seq.data();
    args.t_size = T;
    args.key_query_stride = K;
    args.gate_beta_stride = 1;
    args.value_stride = 1;
    args.output_stride = 1;
    args.key_tmp = key_tmp_raw.data();
    args.query_tmp = query_tmp_raw.data();
    args.output_seq = out_jit.data();
    (*jit)(&args);

    // Final results comparison.
    for (size_t t = 0; t < T; t++) {
        ASSERT_NEAR(out_jit[t], out_ref[t], 2e-3F)
            << "Output mismatch at precision=" << prc << ", T=" << T << ", K=" << K << ", step=" << t;
    }

    const auto state_jit = from_raw_bytes_to_float(state_jit_raw, prc);
    for (size_t i = 0; i < K; i++) {
        ASSERT_NEAR(state_jit[i], state_ref[i], 2e-3F)
            << "State mismatch at precision=" << prc << ", T=" << T << ", K=" << K << ", idx=" << i;
    }
}

const std::vector<GDNJitParams> gdn_jit_params = {
    // f32
    {ov::element::f32, 1, 16, false},
    {ov::element::f32, 1, 16, true},
    {ov::element::f32, 2, 32, false},
    {ov::element::f32, 5, 64, true},
    {ov::element::f32, 11, 96, false},
    // bf16 (will skip if ISA unavailable)
    {ov::element::bf16, 1, 16, false},
    {ov::element::bf16, 5, 64, true},
    // f16 (will skip if ISA unavailable)
    {ov::element::f16, 1, 16, false},
    {ov::element::f16, 2, 32, true},
};

INSTANTIATE_TEST_SUITE_P(GDNJitKernelUnitTest,
                         GDNJitKernelTest,
                         ::testing::ValuesIn(gdn_jit_params),
                         GDNJitKernelTest::getTestCaseName);

}  // namespace

