// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <memory>
#include <string>
#include <vector>
#include "transformations/cpu_opset/x64/op/mha2.hpp"
#include "kernels/x64/kernels_avx2.hpp"
#include "ie_parallel.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

struct Avx2Kernels {
    std::vector<std::shared_ptr<avx2::Matmul>> ops_qk;
    std::vector<std::shared_ptr<avx2::Matmul>> ops_wv;
    std::vector<tensor2D<float>> all_qk;
    avx2::PP::None pp_none;

    Avx2Kernels() {
        auto NT = parallel_get_num_threads();
        for(int i = 0; i < NT; i++) {
            ops_qk.push_back(std::make_shared<avx2::Matmul>(false, true));
            ops_wv.push_back(std::make_shared<avx2::Matmul>(false, false));
            all_qk.emplace_back();
        }
    }

    /*
        q: M x K
        k: N x K (need transpose)
        v: N x K
    */
    void one_head_attention(int tid,
                            tensor2D<float>& q,
                            tensor2D<float>& k,
                            tensor2D<float>& v,
                            tensor2D<float>& wv,
                            int causal_m0,
                            bool causal_mask,
                            float * qk_max,
                            float * qk_sum) {
        auto M = q.dims[0];
        auto N = k.dims[0];
        auto K = v.dims[1];

        auto & qk = all_qk[tid];
        qk.resize(M, N);
        (*ops_qk[tid])(q, k, qk, 0, N, pp_none);

        // softmax per row
        if (causal_mask && M > 1) {
            for(int m = 0; m<M; m++) {
                int valid_n = std::min(N, m + causal_m0 + 1);
                avx2::functional::softmax(&qk(m,0), valid_n, qk_max + m, qk_sum + m);
                // the rest part is set as zero
                memset(&qk(m, valid_n), 0, sizeof(float)*(N - valid_n));
            }
        } else {
            for(int m = 0; m<M; m++) {
                avx2::functional::softmax(&qk(m,0), N, qk_max + m, qk_sum + m);
            }
        }
        // combine
        (*ops_wv[tid])(qk, v, wv, 0, K, pp_none);
    }


    void one_head_attention(int tid,
                            tensor2D<float>& q,
                            tensor2D<float>& k,
                            tensor2D<float>& v,
                            tensor2D<float>& wv,
                            int causal_m0,
                            bool causal_mask) {
        auto M = q.dims[0];
        auto N = k.dims[0];
        auto K = v.dims[1];

        auto & qk = all_qk[tid];
        qk.resize(M, N);
        (*ops_qk[tid])(q, k, qk, 0, N, pp_none);

        // softmax per row
        if (causal_mask && M > 1) {
            for(int m = 0; m<M; m++) {
                int valid_n = std::min(N, m + causal_m0 + 1);
                avx2::functional::softmax(&qk(m,0), valid_n);
                // the rest part is set as zero
                memset(&qk(m, valid_n), 0, sizeof(float)*(N - valid_n));
            }
        } else {
            for(int m = 0; m<M; m++) {
                avx2::functional::softmax(&qk(m,0), N);
            }
        }
        // combine
        (*ops_wv[tid])(qk, v, wv, 0, K, pp_none);
    }
};

class MHA2 : public Node {
public:
    MHA2(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    bool needShapeInfer() const override {
        return false;
    }

protected:
    std::shared_ptr<const ov::intel_cpu::MHA2Node> mha2;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool needPrepareParams() const override { return false; }

    Avx2Kernels kernels;
    bool verbose;

    // when split on N dimension
    tensor2D<float> sub_states;
    tensor2D<float> qk_max;
    tensor2D<float> qk_sum;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
