// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "nodes/kernels/scaled_attn/common.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include <cstddef>
#include <cstdint>

namespace ov::Extensions::Cpu::XARCH {

template <typename DATA_TYPE, ov::element::Type_t KEY_PREC, ov::element::Type_t VALUE_PREC>
void saged_attn(const ov::intel_cpu::PlainTensor& q,
                ov::intel_cpu::PlainTensor& k_cache,
                const ov::intel_cpu::PlainTensor& v_cache,
                const ov::intel_cpu::PlainTensor& output_emb,
                const ov::intel_cpu::PlainTensor& output_score,
                [[maybe_unused]] size_t max_context_len,
                const ov::intel_cpu::PlainTensor& past_lens,
                const ov::intel_cpu::PlainTensor& subsequence_begins,
                const ov::intel_cpu::PlainTensor& block_indices,
                const ov::intel_cpu::PlainTensor& block_indices_begins,
                const ov::intel_cpu::PlainTensor& alibi_slopes) {
    printf("Going to Do SageAttn\n");
}

template <typename DATA_TYPE, ov::element::Type_t KEY_PREC, ov::element::Type_t VALUE_PREC>
void gemm_s8s8_s32(const ov::intel_cpu::PlainTensor& q,
                   ov::intel_cpu::PlainTensor& k_cache,
                   const ov::intel_cpu::PlainTensor& v_cache,
                   const ov::intel_cpu::PlainTensor& output_emb,
                   const ov::intel_cpu::PlainTensor& output_score,
                   [[maybe_unused]] size_t max_context_len,
                   const ov::intel_cpu::PlainTensor& past_lens,
                   const ov::intel_cpu::PlainTensor& subsequence_begins,
                   const ov::intel_cpu::PlainTensor& block_indices,
                   const ov::intel_cpu::PlainTensor& block_indices_begins,
                   const ov::intel_cpu::PlainTensor& alibi_slopes) {
    printf("Going to Do SageAttn\n");
}

}  // namespace ov::Extensions::Cpu::XARCH
