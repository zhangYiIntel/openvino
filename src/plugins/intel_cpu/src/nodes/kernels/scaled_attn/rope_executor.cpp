#include "rope_executor.hpp"

#include "common/bfloat16.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "nodes/kernels/x64/rope_kernel.hpp"
#include "openvino/core/parallel.hpp"
using namespace ov::intel_cpu::kernel;
static void execJitKernel([[maybe_unused]] const std::shared_ptr<JitKernelBase>& ker,
                          [[maybe_unused]] const void* src,
                          [[maybe_unused]] void* dst,
                          [[maybe_unused]] const float* cos,
                          [[maybe_unused]] const float* sin) {
#if defined(OPENVINO_ARCH_X86_64)

    jit_rotary_call_args call_args;
    call_args.src = src;
    call_args.cos = cos;
    call_args.sin = sin;
    call_args.dst = dst;
    (*ker)(&call_args);

#endif  // OPENVINO_ARCH_X86_64
}

static std::shared_ptr<JitKernelBase> createJitKernel([[maybe_unused]] const jit_rotary_compile_params& param,
                                                      [[maybe_unused]] bool check_vec_size2 = false) {
    std::shared_ptr<JitKernelBase> res;

#if defined(OPENVINO_ARCH_X86_64)

    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        bool flag = true;
        if (check_vec_size2) {
            auto vec_size = jit_rotary_kernel<dnnl::impl::cpu::x64::avx512_core>::vec_size;
            if (param.rotary_ndims % (vec_size * 2) != 0) {
                flag = false;
            }
        }
        if (flag) {
            res = std::make_shared<jit_rotary_kernel<dnnl::impl::cpu::x64::avx512_core>>(param);
        }
    } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        bool flag = true;
        if (check_vec_size2) {
            auto vec_size = jit_rotary_kernel<dnnl::impl::cpu::x64::avx2>::vec_size;
            if (param.rotary_ndims % (vec_size * 2) != 0) {
                flag = false;
            }
        }
        if (flag) {
            res = std::make_shared<jit_rotary_kernel<dnnl::impl::cpu::x64::avx2>>(param);
        }
    }

    if (res) {
        res->create_kernel();
    }

#endif  // OPENVINO_ARCH_X86_64

    return res;
}

template <typename T>
struct ov::intel_cpu::paged_attn::RoPEExecutorRotateHalf : public ov::intel_cpu::paged_attn::RopeExecutor {
    const op::internal::RoPE::Config& m_config;
    std::shared_ptr<kernel::JitKernelBase> m_rotaryKernel;
    RoPEExecutorRotateHalf(const op::internal::RoPE::Config& config) : m_config(config) {
        ov::intel_cpu::kernel::jit_rotary_compile_params jcp;
        jcp.src_prc = precision_of<T>::value;
        jcp.dst_prc = precision_of<T>::value;
        jcp.rotary_ndims = config.rotary_ndims;
        jcp.interleave = false;
        m_rotaryKernel = createJitKernel(jcp);
    }

    virtual void execute(std::vector<PlainTensor>& inputs, std::vector<PlainTensor>& outputs) override {
        ov::intel_cpu::PlainTensor& t_src = inputs[0];
        ov::intel_cpu::PlainTensor& t_cos = inputs[1];
        ov::intel_cpu::PlainTensor& t_sin = inputs[2];
        ov::intel_cpu::PlainTensor& t_dst = outputs[0];
        ov::intel_cpu::PlainTensor gather;
        auto rotary_dims = m_config.rotary_ndims;
        bool can_inplace = true;
        if (m_config.slice_stop - m_config.slice_start > 0) {
            std::cout << "transpose slice|" << std::endl;
            t_src = t_src.slice(3, m_config.slice_start, m_config.slice_stop);
            can_inplace = false;
        }
        if (m_config.input_trans0213) {
            std::cout << "transpose rope|" << t_src.m_rank << std::endl;
            t_src = t_src.permute({0, 2, 1, 3});
            std::cout << "finish transpose" << std::endl;
            can_inplace = false;
        }
        if (m_config.gather_position_arg_id > 0) {
            gather = inputs[m_config.gather_position_arg_id];
        }

        if (t_cos.m_rank == 2) {
            t_cos = t_cos.reshape({1, 1, t_cos.size(0), t_cos.size(1)});
        }
        if (t_sin.m_rank == 2) {
            t_sin = t_sin.reshape({1, 1, t_sin.size(0), t_sin.size(1)});
        }

        auto batch_size = t_src.size(0);
        auto head_cnt = t_src.size(1);
        auto seq_len = t_src.size(2);
        auto feature_size = t_src.size(3);
        printf("batch_size %ld head_cnt %ld seq_len %ld feature_size %ld\n",
               batch_size,
               head_cnt,
               seq_len,
               feature_size);
        parallel_for3d(batch_size, head_cnt, seq_len, [&](size_t b, size_t h, size_t p) {
            auto cos_pos = p;
            if (gather) {
                if (gather.m_rank == 4) {
                    cos_pos = gather.at<int32_t>({b, h, p, 0}, true);
                } else {
                    cos_pos = gather.at<int32_t>({b, p}, true);
                }
            }
            auto* src = t_src.ptr<T>(b, h, p);
            auto* cos = &t_cos.at<float>({b, h, cos_pos, 0}, true);
            auto* sin = &t_sin.at<float>({b, h, cos_pos, 0}, true);
            auto* dst = t_dst.ptr<T>(b, h, p, 0);

            if (m_rotaryKernel) {
                execJitKernel(m_rotaryKernel, src, dst, cos, sin);
            } else {
                auto half_rotary_dims = rotary_dims / 2;
                size_t i = 0;
                for (; i < half_rotary_dims; i++) {
                    auto src0 = src[i];
                    auto src1 = src[i + half_rotary_dims];
                    dst[i] = cos[i] * src0 - sin[i] * src1;
                    dst[i + half_rotary_dims] = cos[i + half_rotary_dims] * src1 + sin[i + half_rotary_dims] * src0;
                }
            }
            if (!can_inplace) {
                memcpy(dst + rotary_dims, src + rotary_dims, (feature_size - rotary_dims) * sizeof(T));
            }
        });
    }
};

template <typename T>
struct ov::intel_cpu::paged_attn::RoPEExecutorQwen : public ov::intel_cpu::paged_attn::RopeExecutor {
    const op::internal::RoPE::Config& m_config;
    std::shared_ptr<kernel::JitKernelBase> m_rotaryKernel;

    RoPEExecutorQwen(const op::internal::RoPE::Config& config) : m_config(config) {
        jit_rotary_compile_params jcp;
        jcp.src_prc = precision_of<T>::value;
        jcp.dst_prc = precision_of<T>::value;
        jcp.rotary_ndims = config.rotary_ndims;
        jcp.interleave = false;
        m_rotaryKernel = createJitKernel(jcp);
    }

    virtual void execute(std::vector<PlainTensor>& inputs, std::vector<PlainTensor>& outputs) override {
        ov::intel_cpu::PlainTensor& t_src = inputs[0];   // [batch, length, head_cnt*head_size * 3]
        ov::intel_cpu::PlainTensor& t_cos = inputs[1];  // [1, present-kv-length, 1, rotary_dims]
        ov::intel_cpu::PlainTensor& t_sin = inputs[2];   // [1, present-kv-length, 1, rotary_dims]
        ov::intel_cpu::PlainTensor& t_dst = outputs[0];  // [batch, length, head_cnt, head_size]>
        ov::intel_cpu::PlainTensor gather;

        auto rotary_dims = t_cos.size(3);

        if (m_config.slice_stop - m_config.slice_start > 0) {
            t_src = t_src.slice(2, m_config.slice_start, m_config.slice_stop);
        }
        if (m_config.gather_position_arg_id > 0) {
            gather = inputs[m_config.gather_position_arg_id];
        }

        auto batch_size = t_src.size(0);
        auto seq_len = t_src.size(1);
        auto head_cnt = m_config.head_cnt;
        auto head_size = m_config.head_size;
        auto present_kv_len = t_cos.size(1);

        parallel_for3d(batch_size, seq_len, head_cnt, [&](size_t b, size_t p, size_t h) {
            size_t sincos_pos;
            if (gather) {
                if (gather.m_rank == 4) {
                    sincos_pos = gather.at<int32_t>({b, h, p, 0}, true);
                } else {
                    sincos_pos = gather.at<int32_t>({b, p}, true);
                }
            } else {
                sincos_pos = present_kv_len - seq_len + p;
            }

            auto* src = t_src.ptr<T>(b, p, h * head_size);
            auto* cos = &t_cos.at<float>({b, sincos_pos, h, 0}, true);
            auto* sin = &t_sin.at<float>({b, sincos_pos, h, 0}, true);
            auto* dst = t_dst.ptr<T>(b, p, h);

            if (m_rotaryKernel) {
                execJitKernel(m_rotaryKernel, src, dst, cos, sin);
            } else {
                auto half_rotary_dims = rotary_dims / 2;
                size_t i = 0;
                for (; i < half_rotary_dims; i++) {
                    auto s0 = src[i];
                    auto s1 = src[i + half_rotary_dims];
                    dst[i] = cos[i] * s0 - sin[i] * s1;
                    dst[i + half_rotary_dims] = cos[i + half_rotary_dims] * s1 + sin[i + half_rotary_dims] * s0;
                }
            }

            memcpy(dst + rotary_dims, src + rotary_dims, (head_size - rotary_dims) * sizeof(T));
        });
    }
};

std::shared_ptr<ov::intel_cpu::paged_attn::RopeExecutor> ov::intel_cpu::paged_attn::make_rope_executor(
    const ov::element::Type srcPrecision,
    const op::internal::RoPE::Config& config,
    bool& can_inplace) {
    auto rtPrecision = srcPrecision;
    auto CosSinPrecision = ov::element::f32;
    can_inplace = false;
    std::shared_ptr<ov::intel_cpu::paged_attn::RopeExecutor> ptr;
    if (config.is_qwen) {
        if (rtPrecision == ov::element::f16) {
            ptr = std::make_shared<ov::intel_cpu::paged_attn::RoPEExecutorQwen<ov::float16>>(config);
        } else if (rtPrecision == ov::element::bf16) {
            ptr = std::make_shared<ov::intel_cpu::paged_attn::RoPEExecutorQwen<ov::bfloat16>>(config);
        } else {
            ptr = std::make_shared<ov::intel_cpu::paged_attn::RoPEExecutorQwen<float>>(config);
            rtPrecision = ov::element::f32;
        }
    } else if (false/*config.is_chatglm*/) {
        // if (rtPrecision == ov::element::f16) {
        //     ptr = std::make_shared<RoPEExecutorChatGLM<ov::float16>>(config);
        // } else if (rtPrecision == ov::element::bf16) {
        //     ptr = std::make_shared<RoPEExecutorChatGLM<ov::bfloat16>>(config);
        // } else {
        //     ptr = std::make_shared<RoPEExecutorChatGLM<float>>(config);
        //     rtPrecision = ov::element::f32;
        // }
    } else if (false/*config.is_interleaved*/) {
        // OPENVINO_ASSERT(config.slice_start == 0, "slice_start must be 0 for interleaved mode");
        // OPENVINO_ASSERT(config.slice_stop == 0, "slice_stop must be 0 for interleaved mode");
        // OPENVINO_ASSERT(config.gather_position_arg_id == 0, "gather_position_arg_id must be 0 for interleaved mode");
        // if (rtPrecision == ov::element::f16) {
        //     ptr = std::make_shared<RoPEExecutorInterleaved<ov::float16>>(config);
        // } else if (rtPrecision == ov::element::bf16) {
        //     ptr = std::make_shared<RoPEExecutorInterleaved<ov::bfloat16>>(config);
        // } else {
        //     ptr = std::make_shared<RoPEExecutorInterleaved<float>>(config);
        //     rtPrecision = ov::element::f32;
        // }
    } else {
        can_inplace = true;
        if (rtPrecision == ov::element::f16) {
            ptr = std::make_shared<ov::intel_cpu::paged_attn::RoPEExecutorRotateHalf<ov::float16>>(config);
        } else if (rtPrecision == ov::element::bf16) {
            ptr = std::make_shared<ov::intel_cpu::paged_attn::RoPEExecutorRotateHalf<ov::bfloat16>>(config);
        } else {
            ptr = std::make_shared<ov::intel_cpu::paged_attn::RoPEExecutorRotateHalf<float>>(config);
            rtPrecision = ov::element::f32;
        }
        if (config.slice_stop - config.slice_start > 0 || config.input_trans0213) {
            can_inplace = false;
        }
    }
    return ptr;
}