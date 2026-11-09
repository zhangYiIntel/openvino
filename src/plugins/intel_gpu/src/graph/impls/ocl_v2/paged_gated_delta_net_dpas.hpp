// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <string_view>
#include <vector>

#include "gated_delta_net_dpas_common.hpp"
#include "intel_gpu/primitives/paged_gated_delta_net.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

// Host side of paged_gated_delta_net_dpas.cl: the chunked DPAS path for long prefill calls of paged_gated_delta_net.
namespace ov::intel_gpu::ocl::paged_gdn_dpas {

using namespace gdn_dpas;

// The chunked path is used for calls with few, long sequences (at least min_tokens_per_sequence tokens each on average),
// and only where it beats the per-token kernel the call would otherwise run. The per-token kernels get more parallelism
// from every extra sequence, the chunked path does not. Measured on a 28 Xe core Xe2 dGPU (32 value heads, K = V = 128,
// f16, 16k tokens), speedup of the chunked path over the short / general kernel:
//   1 sequence: 1.4x / 1.2x      2 sequences: 1.4x / 0.6x      3+ sequences: 1.0x / 0.6x
// so it runs for one sequence, and for two when the short kernel is the alternative. Break-even is at about 128 tokens.
// TODO: tune on other devices and head configurations.
inline constexpr size_t min_tokens_per_sequence = 256;
inline constexpr size_t max_sequences_vs_general = 1;  // the alternative is the general kernel
inline constexpr size_t max_sequences_vs_short = 2;    // the alternative is the short kernel

// Internal scratch buffers, in the order of the INTERNAL_BUFFER arguments.
enum Scratch : uint32_t { QN = 0, KN, XINV, W, UT, HBUF, VNEWT };

struct Dims {
    size_t tokens = 0;
    size_t sequences = 0;
    size_t hk = 0;
    size_t hv = 0;
    size_t dk = 0;
    size_t dv = 0;
    size_t max_chunks = 0;

    // Head sizes are always known; tokens / sequences are only static for runtime params.
    explicit Dims(const RuntimeParams& params) {
        const auto& q_shape = params.get_input_layout(cldnn::paged_gated_delta_net::QUERY).get_partial_shape();
        const auto& v_shape = params.get_input_layout(cldnn::paged_gated_delta_net::VALUE).get_partial_shape();
        const auto& seq_shape = params.get_input_layout(cldnn::paged_gated_delta_net::SUBSEQUENCE_BEGINS).get_partial_shape();
        hk = q_shape[1].get_length();
        dk = q_shape[2].get_length();
        hv = v_shape[1].get_length();
        dv = v_shape[2].get_length();
        if (q_shape[0].is_static() && seq_shape[0].is_static()) {
            tokens = q_shape[0].get_length();
            sequences = seq_shape[0].get_length() > 0 ? seq_shape[0].get_length() - 1 : 0;
            max_chunks = gdn_dpas::max_chunks(tokens, sequences);
        }
    }
};

// Device, precision and head layout allow the chunked path (independent of the call). The GEMMs run in f16, so an f32
// execution keeps the full-precision per-token kernels.
inline bool supported(const RuntimeParams& params) {
    const Dims d(params);
    return params.get_input_layout(cldnn::paged_gated_delta_net::QUERY).data_type == ov::element::f16 && device_supported(params.get_device_info()) &&
           head_layout_supported(static_cast<int64_t>(d.hk), static_cast<int64_t>(d.dk), static_cast<int64_t>(d.hv), static_cast<int64_t>(d.dv));
}

// This call runs the chunked path: static shapes, few long sequences. short_alternative: the per-token kernel the call
// would run otherwise is the short one (not the general one).
inline bool use_for_call(const RuntimeParams& params, bool short_alternative) {
    if (params.is_dynamic() || !supported(params)) {
        return false;
    }
    const Dims d(params);
    const size_t max_sequences = short_alternative ? max_sequences_vs_short : max_sequences_vs_general;
    return d.sequences > 0 && d.sequences <= max_sequences && d.tokens >= min_tokens_per_sequence * d.sequences;
}

inline std::vector<cldnn::BufferDescriptor> buffer_descs(const RuntimeParams& params) {
    const Dims d(params);
    const auto f16 = ov::element::f16;
    const size_t chunk_rows = d.max_chunks * chunk_size;
    return {cldnn::BufferDescriptor{d.hk * chunk_rows * d.dk, f16},           // QN
            cldnn::BufferDescriptor{d.hk * chunk_rows * d.dk, f16},           // KN
            cldnn::BufferDescriptor{d.hv * chunk_rows * chunk_size, f16},     // XINV
            cldnn::BufferDescriptor{d.hv * chunk_rows * d.dk, f16},           // W
            cldnn::BufferDescriptor{d.hv * chunk_rows * d.dv, f16},           // UT
            cldnn::BufferDescriptor{d.hv * d.max_chunks * d.dv * d.dk, f16},  // HBUF
            cldnn::BufferDescriptor{d.hv * chunk_rows * d.dv, f16}};          // VNEWT
}

class Generator : public KernelGenerator {
public:
    Generator(size_t stage, std::string_view suffix) : KernelGenerator("paged_gated_delta_net_dpas", suffix), m_stage(stage) {}

protected:
    size_t m_stage;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<cldnn::paged_gated_delta_net>();
        const Dims d(params);

        make_jit_constants(jit, m_stage);
        jit.make("K_HEAD_NUM", d.hk);
        jit.make("V_HEAD_NUM", d.hv);
        jit.make("K_HEAD_DIM", d.dk);
        jit.make("V_HEAD_DIM", d.dv);
        jit.make("SUBGROUP_SIZE", subgroup_size);
        jit.make("FUSE_QK_L2NORM", desc->use_qk_l2norm ? 1 : 0);
        jit.make("Q_L2_NORM_EPS", desc->q_l2_norm_eps);
        jit.make("K_L2_NORM_EPS", desc->k_l2_norm_eps);
        jit.make("SCALE_FACTOR", 1.0f / std::sqrt(static_cast<float>(d.dk)));
        return jit;
    }

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return add_build_options(KernelGenerator::get_build_options(params), m_stage);
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        using T = ArgumentDescriptor::Types;
        using In = cldnn::paged_gated_delta_net;
        Arguments args;
        auto in = [&](uint32_t i) {
            args.push_back({T::INPUT, i});
        };
        auto buf = [&](uint32_t i) {
            args.push_back({T::INTERNAL_BUFFER, i});
        };
        auto scalars = [&](uint32_t n) {
            for (uint32_t i = 0; i < n; i++) {
                args.push_back({T::SCALAR, i});
            }
        };

        switch (m_stage) {
        case PREP:  // q, k, qn, kn, begins | num_sequences, max_chunks, q/k head offsets and strides
            in(In::QUERY);
            in(In::KEY);
            buf(QN);
            buf(KN);
            in(In::SUBSEQUENCE_BEGINS);
            scalars(8);
            break;
        case KKT_SOLVE:  // kn, gate, beta, xinv, begins | num_sequences, max_chunks
            buf(KN);
            in(In::GATE);
            in(In::BETA);
            buf(XINV);
            in(In::SUBSEQUENCE_BEGINS);
            scalars(2);
            break;
        case WY:  // kn, value, gate, beta, xinv, w, uT, begins | num_sequences, max_chunks, v head offset and strides
            buf(KN);
            in(In::VALUE);
            in(In::GATE);
            in(In::BETA);
            buf(XINV);
            buf(W);
            buf(UT);
            in(In::SUBSEQUENCE_BEGINS);
            scalars(5);
            break;
        case FWD_H:  // kn, w, uT, gate, state table, hbuf, vnewT, begins, block table | num_sequences, max_chunks
            buf(KN);
            buf(W);
            buf(UT);
            in(In::GATE);
            in(In::RECURRENT_STATE_TABLE);
            buf(HBUF);
            buf(VNEWT);
            in(In::SUBSEQUENCE_BEGINS);
            in(In::BLOCK_INDICES);
            in(In::BLOCK_INDICES_BEGINS);
            in(In::PAST_LENS);
            in(In::CACHE_INTERVAL);
            scalars(2);
            break;
        case FWD_O:  // qn, kn, gate, hbuf, vnewT, output, begins | num_sequences, max_chunks
            buf(QN);
            buf(KN);
            in(In::GATE);
            buf(HBUF);
            buf(VNEWT);
            args.push_back({T::OUTPUT, 0});
            in(In::SUBSEQUENCE_BEGINS);
            scalars(2);
            break;
        default:
            OPENVINO_THROW("Unknown paged_gated_delta_net_dpas stage ", m_stage);
        }
        (void)params;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        const size_t stage = m_stage;
        return DispatchDataFunc{[stage](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            using In = cldnn::paged_gated_delta_net;
            const Dims d(params);
            auto& wgs = kd.params.workGroups;
            wgs.local = {wg_lanes, 1, 1};

            auto head_offset = [](const cldnn::layout& layout) {
                const auto& lower_pads = layout.data_padding._lower_size;
                return lower_pads.size() > 1 ? static_cast<int32_t>(lower_pads[1]) : 0;
            };
            auto pitch = [](const cldnn::layout& layout, size_t idx) {
                const auto& pitches = layout.get_pitches();
                return idx < pitches.size() ? static_cast<int32_t>(pitches[idx]) : 1;
            };
            const auto& q_layout = params.input_layouts[In::QUERY];
            const auto& k_layout = params.input_layouts[In::KEY];
            const auto& v_layout = params.input_layouts[In::VALUE];

            std::vector<int32_t> scalars{static_cast<int32_t>(d.sequences), static_cast<int32_t>(d.max_chunks)};
            switch (stage) {
            case PREP:
                wgs.global = {d.max_chunks * wg_lanes, d.hk, 1};
                scalars.insert(scalars.end(),
                               {head_offset(q_layout), head_offset(k_layout), pitch(q_layout, 0), pitch(q_layout, 1), pitch(k_layout, 0), pitch(k_layout, 1)});
                break;
            case KKT_SOLVE:
            case FWD_O:
                wgs.global = {d.max_chunks * wg_lanes, d.hv, 1};
                break;
            case WY:
                wgs.global = {d.max_chunks * wg_lanes, d.hv, 1};
                scalars.insert(scalars.end(), {head_offset(v_layout), pitch(v_layout, 0), pitch(v_layout, 1)});
                break;
            case FWD_H:
                wgs.global = {d.sequences * d.hv * (d.dv / v_block) * wg_lanes, 1, 1};
                break;
            default:
                OPENVINO_THROW("Unknown paged_gated_delta_net_dpas stage ", stage);
            }

            kd.params.scalars.clear();
            for (auto v : scalars) {
                scalar_desc desc;
                desc.t = scalar_desc::Types::INT32;
                desc.v.s32 = v;
                kd.params.scalars.push_back(desc);
            }
        }};
    }
};

}  // namespace ov::intel_gpu::ocl::paged_gdn_dpas
