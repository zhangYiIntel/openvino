// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net_opt.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <string_view>

#include "gated_delta_net_dpas_common.hpp"
#include "gated_delta_net_ref_generator.hpp"
#include "intel_gpu/primitives/gated_delta_net.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

using cldnn::BufferDescriptor;

using namespace gdn_dpas;

// The chunked path pads every sequence to whole chunks, so it only pays off from a few chunks on; shorter calls (decode
// included) run the per-token ref kernel. The threshold is the point where the chunked kernels overtake it.
// TODO: tune on other devices and head configurations.
constexpr size_t min_opt_seq_len = 256;

// Internal scratch buffers.
enum GdnScratch : uint32_t { QN = 0, KN, XINV, W, UT, HBUF, VNEWT };

struct Dims {
    size_t batch, seq_len, hk, hv, dk, dv, nt;

    explicit Dims(const RuntimeParams& params) {
        const auto& q_shape = params.get_input_layout(0).get_partial_shape();
        const auto& v_shape = params.get_input_layout(2).get_partial_shape();
        // Batch and sequence length may be dynamic when the kernels are created; they are only used with static params at runtime.
        batch = q_shape[0].is_static() ? q_shape[0].get_length() : 0;
        seq_len = q_shape[1].is_static() ? q_shape[1].get_length() : 0;
        hk = q_shape[2].get_length();
        dk = q_shape[3].get_length();
        hv = v_shape[2].get_length();
        dv = v_shape[3].get_length();
        nt = (seq_len + chunk_size - 1) / chunk_size;
    }
};

// True if the device and the head layout allow the chunked DPAS kernels.
bool dpas_eligible(const RuntimeParams& params) {
    const auto& q = params.get_input_layout(0);
    const auto& q_shape = q.get_partial_shape();
    const auto& v_shape = params.get_input_layout(2).get_partial_shape();
    return q.data_type == ov::element::f16 &&
           head_layout_supported(q_shape[2].get_length(), q_shape[3].get_length(), v_shape[2].get_length(), v_shape[3].get_length()) &&
           device_supported(params.get_device_info());
}

// True if the device and the head layout allow the per-token kernel (gated_delta_net_tok.cl).
bool tok_eligible(const RuntimeParams& params) {
    static const bool tmp_off = std::getenv("GDN_DISABLE_TOK") != nullptr;  // TEMP(bench)
    const auto& q_shape = params.get_input_layout(0).get_partial_shape();
    const auto& v_shape = params.get_input_layout(2).get_partial_shape();
    return !tmp_off && gdn_tok::head_layout_supported(q_shape[2].get_length(), q_shape[3].get_length(), v_shape[2].get_length(), v_shape[3].get_length()) &&
           gdn_tok::device_supported(params.get_device_info());
}

// True if this call runs the chunked kernels (static shape and a long enough sequence).
bool use_opt(const RuntimeParams& params) {
    if (params.is_dynamic() || !dpas_eligible(params)) {
        return false;
    }
    return params.get_input_layout(0).get_partial_shape()[1].get_length() >= static_cast<int64_t>(min_opt_seq_len);
}

// True if this call runs the per-token kernel: it is eligible, the chunked path is not taken, and there is at least one token (the kernel
// reads the first token's operands ahead).
bool use_tok(const RuntimeParams& params) {
    if (params.is_dynamic() || !tok_eligible(params) || use_opt(params)) {
        return false;
    }
    return params.get_input_layout(0).get_partial_shape()[1].get_length() > 0;
}

void set_scalars(KernelData& kd, const std::vector<int32_t>& scalars) {
    kd.params.scalars.clear();
    for (auto v : scalars) {
        scalar_desc desc;
        desc.t = scalar_desc::Types::INT32;
        desc.v.s32 = v;
        kd.params.scalars.push_back(desc);
    }
}

int32_t get_head_offset(const cldnn::layout& layout) {
    const auto& lower_pads = layout.data_padding._lower_size;
    return lower_pads.size() > 2 ? static_cast<int32_t>(lower_pads[2]) : 0;
}

int32_t get_token_stride(const cldnn::layout& layout) {
    const auto& pitches = layout.get_pitches();
    if (pitches.size() > 1) {
        return static_cast<int32_t>(pitches[1]);
    }
    const auto& shape = layout.get_partial_shape();
    return static_cast<int32_t>(shape[2].get_length() * shape[3].get_length());
}

class GatedDeltaNetOptGenerator : public KernelGenerator {
public:
    GatedDeltaNetOptGenerator(size_t stage, std::string_view suffix) : KernelGenerator("gated_delta_net_opt", suffix), m_stage(stage) {}

protected:
    size_t m_stage;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<gated_delta_net>();
        const Dims d(params);

        make_jit_constants(jit, m_stage);
        jit.make("K_HEAD_NUM", d.hk);
        jit.make("V_HEAD_NUM", d.hv);
        jit.make("K_HEAD_DIM", d.dk);
        jit.make("V_HEAD_DIM", d.dv);
        jit.make("SUBGROUP_SIZE", subgroup_size);
        jit.make("FUSE_QK_L2NORM", desc->fuse_qk_l2norm ? 1 : 0);
        jit.make("Q_L2_NORM_EPS", desc->q_l2_norm_eps);
        jit.make("K_L2_NORM_EPS", desc->k_l2_norm_eps);
        jit.make("SCALE_FACTOR", 1.0f / std::sqrt(static_cast<float>(d.dk)));
        jit.make("OUTPUT_STATE", params.output_layouts.size() > 1 ? 1 : 0);
        return jit;
    }

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return add_build_options(KernelGenerator::get_build_options(params), m_stage);
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        using T = ArgumentDescriptor::Types;
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
        case PREP:  // q, k, qn, kn, seq_len, q_off, k_off, q_t_stride, k_t_stride
            in(0);
            in(1);
            buf(QN);
            buf(KN);
            scalars(5);
            break;
        case KKT_SOLVE:  // kn, g, beta, xinv, seq_len
            buf(KN);
            in(4);
            in(5);
            buf(XINV);
            scalars(1);
            break;
        case WY:  // kn, v, g, beta, xinv, w, uT, seq_len, v_off, v_t_stride
            buf(KN);
            in(2);
            in(4);
            in(5);
            buf(XINV);
            buf(W);
            buf(UT);
            scalars(3);
            break;
        case FWD_H:  // kn, w, uT, g, state, hbuf, vnewT, [output_state], seq_len
            buf(KN);
            buf(W);
            buf(UT);
            in(4);
            in(3);
            buf(HBUF);
            buf(VNEWT);
            if (params.output_layouts.size() > 1) {
                args.push_back({T::OUTPUT, 1});
            }
            scalars(1);
            break;
        case FWD_HO:  // kn, w, uT, g, state, qn, output, [output_state], seq_len
            buf(KN);
            buf(W);
            buf(UT);
            in(4);
            in(3);
            buf(QN);
            args.push_back({T::OUTPUT, 0});
            if (params.output_layouts.size() > 1) {
                args.push_back({T::OUTPUT, 1});
            }
            scalars(1);
            break;
        case FWD_O:  // qn, kn, g, hbuf, vnewT, output, seq_len, out_row_stride
            buf(QN);
            buf(KN);
            in(4);
            buf(HBUF);
            buf(VNEWT);
            args.push_back({T::OUTPUT, 0});
            scalars(2);
            break;
        default:
            OPENVINO_THROW("Unknown gated_delta_net_opt stage ", m_stage);
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        const size_t stage = m_stage;
        return DispatchDataFunc{[stage](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const Dims d(params);
            const auto seq_len = static_cast<int32_t>(d.seq_len);
            const size_t nv = d.dv / v_block;

            // One workgroup (8 subgroups x 16 lanes) per point of the logical grid; the kernels read it via get_group_id().
            wgs.local = {wg_lanes, 1, 1};
            switch (stage) {
            case PREP: {
                wgs.global = {d.nt * wg_lanes, d.batch * d.hk, 1};
                const auto& q_layout = params.input_layouts[0];
                const auto& k_layout = params.input_layouts[1];
                set_scalars(kd, {seq_len, get_head_offset(q_layout), get_head_offset(k_layout), get_token_stride(q_layout), get_token_stride(k_layout)});
                break;
            }
            case KKT_SOLVE:
                wgs.global = {d.nt * wg_lanes, d.batch * d.hv, 1};
                set_scalars(kd, {seq_len});
                break;
            case WY: {
                wgs.global = {d.nt * wg_lanes, d.batch * d.hv, 1};
                const auto& v_layout = params.input_layouts[2];
                set_scalars(kd, {seq_len, get_head_offset(v_layout), get_token_stride(v_layout)});
                break;
            }
            case FWD_HO:
            case FWD_H:
                wgs.global = {d.batch * d.hv * nv * wg_lanes, 1, 1};
                set_scalars(kd, {seq_len});
                break;
            case FWD_O:
                // one workgroup covers all v-blocks of a (chunk, head)
                wgs.global = {d.nt * wg_lanes, d.batch * d.hv, 1};
                set_scalars(kd, {seq_len, static_cast<int32_t>(d.hv * d.dv)});
                break;
            default:
                OPENVINO_THROW("Unknown gated_delta_net_opt stage ", stage);
            }
        }};
    }
};

// The per-token kernel; arguments: q, k, v, state, g, beta, output, [output_state], seq_len, q / k / v head offsets and token strides.
class GatedDeltaNetTokGenerator : public KernelGenerator {
public:
    GatedDeltaNetTokGenerator() : KernelGenerator("gated_delta_net_tok") {}

protected:
    static constexpr size_t max_k_vec_size = 8;  // elements per lane of one q / k block read, at most
    static constexpr size_t acc_ways = 4;        // independent accumulator chains of the dot products over the k axis
    static constexpr size_t prefetch_ahead = 8;  // tokens
    static constexpr size_t prefetch_bytes = 8;  // request size

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<gated_delta_net>();
        const Dims d(params);

        jit.make("K_HEAD_NUM", d.hk);
        jit.make("V_HEAD_NUM", d.hv);
        jit.make("K_HEAD_DIM", d.dk);
        jit.make("V_HEAD_DIM", d.dv);
        jit.make("SUBGROUP_SIZE", gdn_tok::v_rows_per_subgroup);
        jit.make("V_BLOCK_SIZE", gdn_tok::v_rows_per_subgroup);
        // The largest power of two (up to max_k_vec_size) that divides a lane's share of the k axis.
        size_t k_vec_size = max_k_vec_size;
        while (k_vec_size > 1 && (d.dk / gdn_tok::v_rows_per_subgroup) % k_vec_size != 0) {
            k_vec_size /= 2;
        }
        jit.make("K_VEC_SIZE", k_vec_size);
        jit.make("ACC_WAYS", acc_ways);
        jit.make("PF_AHEAD", prefetch_ahead);
        jit.make("PF_BYTES", prefetch_bytes);
        jit.make("FUSE_QK_L2NORM", desc->fuse_qk_l2norm ? 1 : 0);
        jit.make("Q_L2_NORM_EPS", desc->q_l2_norm_eps);
        jit.make("K_L2_NORM_EPS", desc->k_l2_norm_eps);
        jit.make("SCALE_FACTOR", 1.0f / std::sqrt(static_cast<float>(d.dk)));
        jit.make("OUTPUT_STATE", params.output_layouts.size() > 1 ? 1 : 0);
        return jit;
    }

    // The state lives in registers: built for the large register file.
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + " -cl-intel-256-GRF-per-thread";
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        using T = ArgumentDescriptor::Types;
        Arguments args;
        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({T::INPUT, i});
        }
        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({T::OUTPUT, i});
        }
        for (uint32_t i = 0; i < 7; i++) {
            args.push_back({T::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const Dims d(params);
            const size_t sg = gdn_tok::v_rows_per_subgroup;
            // The v-tile axis goes first so that the tiles of a head, which read the same q and k rows, have consecutive group ids.
            wgs.global = {(d.dv / sg) * sg, d.hv, d.batch};
            wgs.local = {sg, 1, 1};
            const auto& q_layout = params.input_layouts[0];
            const auto& k_layout = params.input_layouts[1];
            const auto& v_layout = params.input_layouts[2];
            set_scalars(kd,
                        {static_cast<int32_t>(d.seq_len),
                         get_head_offset(q_layout),
                         get_head_offset(k_layout),
                         get_head_offset(v_layout),
                         get_token_stride(q_layout),
                         get_token_stride(k_layout),
                         get_token_stride(v_layout)});
        }};
    }
};

class GatedDeltaNetOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GatedDeltaNetOptImpl)

    // The ref stage is declared (and added) first, so the opt stages added after it keep their order.
    // Opt stage declaration order must match the GdnStage enum and GDN_STAGE values.
    Stage::Ptr ref = make_stage<GatedDeltaNetRefGenerator>();
    Stage::Ptr prep = make_stage<GatedDeltaNetOptGenerator>(PREP, "prep");
    Stage::Ptr kkt_solve = make_stage<GatedDeltaNetOptGenerator>(KKT_SOLVE, "kkt_solve");
    Stage::Ptr wy = make_stage<GatedDeltaNetOptGenerator>(WY, "wy");
    Stage::Ptr fwd_h = make_stage<GatedDeltaNetOptGenerator>(FWD_H, "fwd_h");
    Stage::Ptr fwd_o = make_stage<GatedDeltaNetOptGenerator>(FWD_O, "fwd_o");
    Stage::Ptr fwd_ho = make_stage<GatedDeltaNetOptGenerator>(FWD_HO, "fwd_ho");
    Stage::Ptr tok = make_stage<GatedDeltaNetTokGenerator>();

    GatedDeltaNetOptImpl() : PrimitiveImplOCL(GatedDeltaNetOpt::get_type_info_static()) {}
    GatedDeltaNetOptImpl(const program_node& node, const RuntimeParams& params) : GatedDeltaNetOptImpl() {
        add_stage(ref, params);
        // Dynamic: both paths are compiled and chosen per call. Static: only the one the shape needs.
        if (dpas_eligible(params) && (params.is_dynamic() || use_opt(params))) {
            add_stage(prep, params);
            add_stage(kkt_solve, params);
            add_stage(wy, params);
            add_stage(fwd_h, params);
            add_stage(fwd_o, params);
            add_stage(fwd_ho, params);
        }
        if (tok_eligible(params)) {
            add_stage(tok, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatedDeltaNetOptImpl>(this);
    }

    // Indices in the order of make_stage(): 0 is ref, 1..5 are the opt stages, 6 the fused fwd_ho (experiment), 7 the per-token kernel.
    [[nodiscard]] std::vector<size_t> get_stages_execution_order(const RuntimeParams& params) const override {
        if (has_stage(prep) && use_opt(params)) {
            static const bool tmp_ho = std::getenv("GDN_FUSE_FWD_HO") != nullptr;  // TEMP(bench)
            return tmp_ho ? std::vector<size_t>{1, 2, 3, 6} : std::vector<size_t>{1, 2, 3, 4, 5};
        }
        if (has_stage(tok) && use_tok(params)) {
            return {7};
        }
        return {0};
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        if (!use_opt(params)) {
            return {};  // dynamic compile-time params or a short call: only the ref kernel runs, it needs no scratch
        }
        const Dims d(params);
        const size_t padded_tokens = d.nt * chunk_size;
        const auto f16 = ov::element::f16;
        // Order must match GdnScratch.
        return {BufferDescriptor{d.batch * d.hk * padded_tokens * d.dk, f16},        // QN
                BufferDescriptor{d.batch * d.hk * padded_tokens * d.dk, f16},        // KN
                BufferDescriptor{d.batch * d.hv * padded_tokens * chunk_size, f16},  // XINV
                BufferDescriptor{d.batch * d.hv * padded_tokens * d.dk, f16},        // W
                BufferDescriptor{d.batch * d.hv * padded_tokens * d.dv, f16},        // UT
                BufferDescriptor{d.batch * d.hv * d.nt * d.dv * d.dk, f16},          // HBUF
                BufferDescriptor{d.batch * d.hv * padded_tokens * d.dv, f16}};       // VNEWT
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GatedDeltaNetOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<gated_delta_net>());
    return std::make_unique<GatedDeltaNetOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GatedDeltaNetOptImpl)
