// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/x64/op/flash_attention.hpp"

#include <utils/general_utils.h>

#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "flash_attention.h"
#include "ie_parallel.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "gemm/ov_cpu_gemm.h"
#include <immintrin.h>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

FlashAttention::FlashAttention(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    errorPrefix = "Interaction node with name '" + getName() + "'";
    const auto interaction = std::dynamic_pointer_cast<const FlashAttentionNode>(op);
}

// temporary split size from experimental
const std::vector<int64_t> qsplit_range{767, 191, 31};
const std::vector<int64_t> qsplit_size{256, 64, 32};
const int64_t kvsplit_size = 512;

void FlashAttention::prepareParams() {
    auto memQ = getParentEdgeAt(0)->getMemoryPtr();
    auto queryDims = memQ->getStaticDims();
    batchSize = queryDims[0];
    seqLen = queryDims[1];
    headNum = queryDims[2];
    headSize = queryDims[3];
    hiddenSize = headNum * headSize;
    std::cout << "batchSize|" << batchSize << "|seqLen|" << seqLen << "|headNum|" << headNum << "|headSize|" << headSize
              << std::endl;

    qSplitSize = seqLen;
    for (int i = 0; i < qsplit_range.size(); ++i) {
        if (seqLen > qsplit_range[i]) {
            qSplitSize = qsplit_size[i];
            break;
        }
    }
    kvSplitSize = seqLen >= kvsplit_size ? kvsplit_size : seqLen;

    qSlice = (seqLen - 1) / qSplitSize + 1;
    qTail = (seqLen - 1) % qSplitSize + 1;
    kvSlice = (seqLen - 1) / kvSplitSize + 1;
    kvTail = (seqLen - 1) % kvSplitSize + 1;
    auto updateMemoryPtr = [&](const InferenceEngine::Precision& prc, const intel_cpu::Shape& shape, MemoryPtr& ptr) {
        if (ptr == nullptr) {
            ptr = std::make_shared<Memory>(getEngine());
            ptr->Create(intel_cpu::DnnlBlockedMemoryDesc(prc, shape));
        } else {
            MemoryDescPtr memDesc = std::make_shared<CpuBlockedMemoryDesc>(prc, shape);
            ptr->redefineDesc(memDesc);
        }
    };
    const size_t threadNum = parallel_get_max_threads();
    updateMemoryPtr(Precision::FP32, intel_cpu::Shape{threadNum, static_cast<size_t>(qSplitSize), static_cast<size_t>(kvSplitSize)}, bufferQK);
    updateMemoryPtr(Precision::FP32, intel_cpu::Shape{threadNum, static_cast<size_t>(qSplitSize)}, bufferQKMax);
    updateMemoryPtr(Precision::FP32, intel_cpu::Shape{threadNum, static_cast<size_t>(qSplitSize)}, bufferQKSum);
    updateMemoryPtr(Precision::FP32, intel_cpu::Shape{threadNum, static_cast<size_t>(qSplitSize), static_cast<size_t>(headSize)}, bufferPreOutput);
}

void FlashAttention::initSupportedPrimitiveDescriptors() {
    addSupportedPrimDesc(
        {
            {LayoutType::ncsp, Precision::FP32},
            {LayoutType::ncsp, Precision::FP32},
            {LayoutType::ncsp, Precision::FP32},
            {LayoutType::ncsp, Precision::FP32}
        },
        {{LayoutType::ncsp, Precision::FP32}},
        ref_any);
}

inline __m512 _loadu(const float* data_base) {
  return _mm512_loadu_ps(data_base);
}

inline __m512 _maskz_loadu(const float* data_base, __mmask16 mask) {
  return _mm512_maskz_loadu_ps(mask, data_base);
}

inline void _mask_storeu(float* data_base, __m512 a, __mmask16 mask) {
  _mm512_mask_storeu_ps(data_base, mask, a);
}

inline void _storeu(float* data_base, __m512 a) {
  _mm512_storeu_ps(data_base, a);
}

inline __m512 _dil_exp_kernel(__m512 vec_src) {
  static __m512 vec_factorial_1 =
      _mm512_set1_ps(0.999999701f); // 1/factorial(1)
  static __m512 vec_factorial_2 =
      _mm512_set1_ps(0.499991506f); // 1/factorial(2)
  static __m512 vec_factorial_3 =
      _mm512_set1_ps(0.166676521f); // 1/factorial(3)
  static __m512 vec_factorial_4 =
      _mm512_set1_ps(0.0418978221f); // 1/factorial(4)
  static __m512 vec_factorial_5 =
      _mm512_set1_ps(0.00828929059f); // 1/factorial(5)
  static __m512 vec_exp_log2ef =
      (__m512)_mm512_set1_epi32(0x3fb8aa3b); // log2(e)
  static __m512 vec_half = _mm512_set1_ps(0.5f);
  static __m512 vec_one = _mm512_set1_ps(1.f);
  static __m512 vec_zero = _mm512_set1_ps(0.f);
  static __m512 vec_two = _mm512_set1_ps(2.f);
  static __m512 vec_ln2f = (__m512)_mm512_set1_epi32(0x3f317218); // ln(2)
  static __m512 vec_ln_flt_min = (__m512)_mm512_set1_epi32(0xc2aeac50);
  static __m512 vec_ln_flt_max = (__m512)_mm512_set1_epi32(0x42b17218);
  static __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
  static int n_mantissa_bits = 23;

  // exp(x) =
  // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
  // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

  auto less_ln_flt_min_mask =
      _mm512_cmp_ps_mask(vec_src, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);
  vec_src = _mm512_min_ps(vec_src, vec_ln_flt_max);
  vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

  // fx = floorf(x * log2ef + 0.5)
  auto vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
  auto vec_fx_i = _mm512_cvt_roundps_epi32(
      vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

  // x = x - fx * ln2
  auto vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

  // compute polynomial
  auto vec_res =
      _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

  // compute 2^(n-1)
  auto vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);
  auto vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);
  auto vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);
  vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
  auto vec_two_pow_n = (__m512)vec_two_pow_n_i;
  vec_two_pow_n =
      _mm512_mask_blend_ps(less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

  // y = y * 2^n
  vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
  vec_res = _mm512_mul_ps(vec_res, vec_two);
  return vec_res;
}

template <typename scalar_a, typename scalar_b>
inline void _dil_div_add_reduce_max_fusion_kernel(const scalar_a* a,
                                                  const scalar_b* b,
                                                  const int& size,
                                                  float* out,
                                                  float& max) {
    auto vec_ps_min = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    auto vec_a = vec_ps_min;
    auto vec_b = vec_ps_min;
    auto vec_out = vec_ps_min;

    int i = 0;
    for (; i <= size - 16; i += 16) {
        vec_a = _loadu(a + i);
        vec_b = _loadu(b + i);
        vec_out = _mm512_add_ps(vec_a, vec_b);
        vec_ps_min = _mm512_max_ps(vec_ps_min, vec_out);
        _mm512_storeu_ps(out + i, vec_out);
    }

    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        vec_a = _maskz_loadu(a + i, mask);
        vec_b = _maskz_loadu(b + i, mask);
        vec_out = _mm512_add_ps(vec_a, vec_b);
        vec_ps_min = _mm512_mask_max_ps(vec_ps_min, mask, vec_out, vec_ps_min);
        _mm512_mask_storeu_ps(out + i, mask, vec_out);
    }

    // NOTE: _mm512_reduce_max_ps is sequence instruction
    max = _mm512_reduce_max_ps(vec_ps_min);
}

inline void _dil_exp_reduce_sum_fusion_kernel(float* a, const int& size, float* out, float& val) {
    static auto vec_zero = _mm512_set1_ps(0.f);
    auto vec_max = _mm512_set1_ps(val);
    auto vec_sum = _mm512_set1_ps(0.f);
    __m512 vec_a = {};
    __m512 vec_out = {};

    int i = 0;
    for (; i <= size - 16; i += 16) {
        vec_a = _mm512_loadu_ps(a + i);
        vec_out = _mm512_sub_ps(vec_a, vec_max);
        vec_out = _dil_exp_kernel(vec_out);
        vec_sum = _mm512_add_ps(vec_sum, vec_out);
        _mm512_storeu_ps(out + i, vec_out);
    }

    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        auto vec_a = _mm512_mask_loadu_ps(vec_max, mask, a + i);
        auto vec_out = _mm512_sub_ps(vec_a, vec_max);
        vec_out = _dil_exp_kernel(vec_out);
        vec_sum = _mm512_mask_add_ps(vec_sum, mask, vec_sum, vec_out);
        _mm512_mask_storeu_ps(out + i, mask, vec_out);
    }

    // NOTE: _mm512_reduce_add_ps is sequence instruction
    val = _mm512_reduce_add_ps(vec_sum);
}

template <typename scalar_t>
inline void _dil_normalization_kernel(const float* a, const float& sum, const int& size, scalar_t* out) {
    auto vec_sum = _mm512_set1_ps(sum);
    __m512 vec_a = {};
    __m512 vec_out = {};

    int i = 0;
    for (; i <= size - 16; i += 16) {
        auto vec_a = _mm512_loadu_ps(a + i);
        auto vec_out = _mm512_div_ps(vec_a, vec_sum);
        _storeu(out + i, vec_out);
    }

    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        auto vec_a = _mm512_maskz_loadu_ps(mask, a + i);
        auto vec_out = _mm512_div_ps(vec_a, vec_sum);
        _mask_storeu(out + i, vec_out, mask);
    }
}

inline void _mha_update_sum_max_kernel(const float* a,
                                       const float& sum_old,
                                       const float& sum_new,
                                       const float& exp_val,
                                       const int& size,
                                       float* out) {
    auto vec_sum_old = _mm512_set1_ps(sum_old);
    auto vec_sum_new = _mm512_set1_ps(sum_new);
    auto vec_sum_cor = _mm512_div_ps(vec_sum_old, vec_sum_new);
    auto exp_vec = _mm512_set1_ps(exp_val);

    int i = 0;
    for (; i <= size - 16; i += 16) {
        auto dat = _loadu(a + i);
        auto vec_a = _mm512_mul_ps(dat, vec_sum_cor);
        auto vec_out = _mm512_mul_ps(vec_a, exp_vec);
        _storeu(out + i, vec_out);
    }
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        auto dat = _mm512_maskz_loadu_ps(mask, a + i);
        auto vec_a = _mm512_mul_ps(dat, vec_sum_cor);
        auto vec_out = _mm512_mul_ps(vec_a, exp_vec);
        _mask_storeu(out + i, vec_out, mask);
    }
}
template <typename scalar_t>
void _mha_div_add_softmax_kernel(float* a,
                                 scalar_t* b,
                                 float* dst,
                                 const scalar_t* rel_kv,
                                 float* max,
                                 float* sum,
                                 const int& qsize,    /*qBlockSize 32*/
                                 const int& kvsize,   /*kvBlockSize 128*/
                                 const int& headsize, /*64*/
                                 const int& idx) {
    float tmp_max = 0.f, tmp_sum = 0.f, sum_old = 0.f, exp_tmp = 0.f;

    for (int i = 0; i < qsize /*32*/; ++i) {
        sum_old = sum[i];

        _dil_div_add_reduce_max_fusion_kernel<float, scalar_t>(a + i * kvsize,
                                                               rel_kv,
                                                               kvsize,
                                                               a + i * kvsize,
                                                               tmp_max);
        tmp_max = max[i] > tmp_max ? max[i] : tmp_max;

        tmp_sum = tmp_max;
        _dil_exp_reduce_sum_fusion_kernel(a + i * kvsize, kvsize, a + i * kvsize, tmp_sum);
        exp_tmp = exp(max[i] - tmp_max);
        sum[i] = tmp_sum + exp_tmp * sum[i];
        max[i] = tmp_max;
        /*sum*/
        _dil_normalization_kernel<scalar_t>(a + i * kvsize, sum[i], kvsize, b + i * kvsize);

        if (idx) { /*revise last output with correct sum/max*/
            _mha_update_sum_max_kernel(dst + i * headsize, sum_old, sum[i], exp_tmp, headsize, dst + i * headsize);
        }
    }
}

template <typename scalar_t>
inline void _reorder_mha_output_kernel(
    float* src,
    scalar_t* dst,
    const int& rows,
    const int& cols,
    const int& dst_stride) {
  for (int i = 0; i < rows; ++i) {
    int j = 0;
    for (; j <= cols - 16; j += 16) {
      _storeu(dst + i * dst_stride + j, _loadu(src + i * cols + j));
    }
    if (j < cols) {
      __mmask16 mask = (1 << (cols - j)) - 1;
      _mask_storeu(dst + i * dst_stride + j, _loadu(src + i * cols + j), mask);
    }
  }
}

void FlashAttention::execute(dnnl::stream strm) {
    auto memQ = getParentEdgeAt(0)->getMemoryPtr();
    auto memK = getParentEdgeAt(1)->getMemoryPtr();
    auto memMask = getParentEdgeAt(2)->getMemoryPtr();
    auto memV = getParentEdgeAt(3)->getMemoryPtr();
    auto memOutput = getChildEdgeAt(0)->getMemoryPtr();
    auto dims2string = [](const VectorDims& dims) {
        std::stringstream ss;
        for (auto& dim : dims) {
            ss << dim << ",";
        }
        return ss.str();
    };

    float* qBasePtr = reinterpret_cast<float*>(memQ->GetPtr());
    float* kBasePtr = reinterpret_cast<float*>(memK->GetPtr());
    float* maskBasePtr = reinterpret_cast<float*>(memMask->GetPtr());
    float* vBasePtr = reinterpret_cast<float*>(memV->GetPtr());

    float* qkBasePtr = reinterpret_cast<float*>(bufferQK->GetPtr());
    float* qkMax = reinterpret_cast<float*>(bufferQKMax->GetPtr());
    float* qkSum = reinterpret_cast<float*>(bufferQKSum->GetPtr());
    float* preOut = reinterpret_cast<float*>(bufferPreOutput->GetPtr());
    float* finalOut = reinterpret_cast<float*>(memOutput->GetPtr());
    std::cout << "Inference" << std::endl;
    std::cout << "Q Shape " << dims2string(memQ->getStaticDims()) << "|" << qBasePtr << std::endl;
    std::cout << "K Shape " << dims2string(memK->getStaticDims()) << "|" << kBasePtr << std::endl;
    std::cout << "Mask Shape " << dims2string(memMask->getStaticDims()) << "|" << maskBasePtr << std::endl;
    std::cout << "V Shape " << dims2string(memV->getStaticDims()) << "|" << vBasePtr << std::endl;
    parallel_for3d(batchSize, headNum, qSlice, [&](size_t i, size_t j, size_t k) {
        int64_t qBlockSize = (k == qSlice - 1) ? qTail : qSplitSize;
        int64_t threadID = parallel_get_thread_num();
        size_t sumOffset = threadID * qSplitSize;
        std::fill_n(qkMax + sumOffset, qBlockSize, std::numeric_limits<float>::lowest());
        std::fill_n(qkSum + sumOffset, qBlockSize, 0.0);
        size_t preOutOffset = threadID * qSplitSize * headSize;
        for (int l = 0; l < kvSlice; ++l) {
            size_t qOffset = i * seqLen * hiddenSize + headSize * j + k * qSplitSize * hiddenSize;
            size_t kOffset = i * seqLen * hiddenSize + headSize * j + l * kvSplitSize * hiddenSize;
            size_t qkOffset = threadID * qSplitSize * kvSplitSize;
            size_t vOffset = i * seqLen * hiddenSize + headSize * j + l * kvSplitSize * hiddenSize;
            size_t maskOffset = i * seqLen + l * qSplitSize;
            int64_t kvBlockSize = (l == kvSlice - 1) ? kvTail : kvSplitSize;
            std::cout << threadID << "|batchSize|" << i << "|headNum|" << j << "|qSlice|" << k  << "|kvSlice|" << l << "|qoff|" << qOffset << "|kOff|" <<
                kOffset << "|vOff|" << vOffset << "|maskOff|" << maskOffset << "|qBlock|" << qBlockSize << "|kvBlock|" << kvBlockSize << std::endl;
            ov_sgemm_compute("N",
                             "T",
                             qBlockSize,
                             kvBlockSize,
                             headSize,
                             1.0f,
                             qBasePtr + qOffset,
                             hiddenSize,
                             kBasePtr + kOffset,
                             hiddenSize,
                             0.f,
                             qkBasePtr + qkOffset,
                             kvBlockSize);
            _mha_div_add_softmax_kernel<float>(qkBasePtr + qkOffset,
                                               qkBasePtr + qkOffset,
                                               preOut + preOutOffset,
                                               maskBasePtr + maskOffset,
                                               qkMax + sumOffset,
                                               qkSum + sumOffset,
                                               qBlockSize,
                                               kvBlockSize,
                                               headSize,
                                               l);
            ov_sgemm_compute("N",
                             "N",
                             qBlockSize,
                             headSize,
                             kvBlockSize,
                             1.0f,
                             qkBasePtr + qkOffset,
                             kvBlockSize,
                             vBasePtr + vOffset,
                             hiddenSize,
                             l == 0 ? 0.f : 1.f,
                             preOut + preOutOffset,
                             headSize);
        }
        _reorder_mha_output_kernel(preOut + preOutOffset,
            finalOut + i * seqLen * hiddenSize + headSize * j + k * qSplitSize * hiddenSize,
            qBlockSize,
            headSize,
            hiddenSize);
    });
    return;
}

bool FlashAttention::created() const {
    return getType() == Type::FlashAttention;
}

bool FlashAttention::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op,
                                          std::string& errorMessage) noexcept {
    try {
        const auto interaction = std::dynamic_pointer_cast<const FlashAttentionNode>(op);
        if (!interaction) {
            errorMessage = "Only FlashAttention operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void FlashAttention::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool FlashAttention::isExecutable() const {
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov