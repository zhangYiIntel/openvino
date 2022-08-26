// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnnt_update.h"

#include <dnnl_extension_utils.h>

#include <ie_ngraph_utils.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "common/blocked_desc_creator.h"
#include "ie_parallel.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace dnnl;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {
namespace node {

RnntUpdate::RnntUpdate(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr& cache)
    : Node(op, eng, cache),
      ngraphOp(op) {
    setType(Type::RnntUpdate);
    setTypeStr("RnntUpdate");
}

void RnntUpdate::getSupportedDescriptors() {}

void RnntUpdate::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inputConfigurators;
    inputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); i++) {
        inputConfigurators.emplace_back(LayoutType::ncsp,
                                        convertPrecision(ngraphOp->get_input_element_type(i)),
                                        inputShapes[i]);
    }

    std::vector<PortConfigurator> outputConfigurators;
    outputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < outputShapes.size(); i++) {
        outputConfigurators.emplace_back(LayoutType::ncsp,
                                         convertPrecision(ngraphOp->get_output_element_type(i)),
                                         outputShapes[i]);
    }

    addSupportedPrimDesc(inputConfigurators, outputConfigurators, impl_desc_type::ref);
}

void RnntUpdate::createPrimitive() {}

template <typename T>
struct BTensor {
    BTensor(MemoryPtr& pmem) {
        auto& dims = pmem->getStaticDims();
        ndims = dims.size();
        assert(ndims < 8);
        size_t s = 1;
        for (int i = ndims - 1; i >= 0; i--) {
            shape[i] = dims[i];
            strides[i] = s;
            s *= dims[i];
        }
        ptr = reinterpret_cast<T*>(pmem->GetPtr());
        elesize = sizeof(T);
    }

    T& at(int i0) {
        return *(ptr + i0 * strides[0]);
    }
    T& at(int i0, int i1) {
        return *(ptr + i0 * strides[0] + i1 * strides[1]);
    }

    size_t elesize;
    T* ptr;
    int ndims;
    size_t shape[8];
    size_t strides[8];
};

// M : type for which only data movement is performed
// V:  type for which arithematic is performed
template <typename M, typename V>
void RnntUpdate::evaluate_T() {
#define NEXT_OUT_MEMPTR(idx) getChildEdgeAt(idx)->getMemoryPtr()
#define NEXT_IN_MEMPTR(idx)  getParentEdgeAt(idx)->getMemoryPtr()

    int idx = 0;
    BTensor<int32_t> current_iter(NEXT_IN_MEMPTR(idx++));  // [1]
    BTensor<float> all_f(NEXT_IN_MEMPTR(idx++));           // N,T,1024
    BTensor<float> logits(NEXT_IN_MEMPTR(idx++));          // N
    BTensor<M> o_hs1(NEXT_IN_MEMPTR(idx++));               // N,320
    BTensor<float> o_cs1(NEXT_IN_MEMPTR(idx++));           // N,320
    BTensor<M> o_hs2(NEXT_IN_MEMPTR(idx++));               // N,320
    BTensor<float> o_cs2(NEXT_IN_MEMPTR(idx++));           // N,320

    idx = 0;
    BTensor<uint8_t> next_cond(NEXT_OUT_MEMPTR(idx++));    // [1]
    BTensor<float> f(NEXT_OUT_MEMPTR(idx++));              // N,1024
    BTensor<int32_t> last_symbol(NEXT_OUT_MEMPTR(idx++));  // N
    BTensor<M> hs1(NEXT_OUT_MEMPTR(idx++));
    BTensor<float> cs1(NEXT_OUT_MEMPTR(idx++));
    BTensor<M> hs2(NEXT_OUT_MEMPTR(idx++));
    BTensor<float> cs2(NEXT_OUT_MEMPTR(idx++));
    BTensor<uint8_t> num_symbols_generated(NEXT_OUT_MEMPTR(idx++));
    BTensor<int32_t> time_idxs(NEXT_OUT_MEMPTR(idx++));
    BTensor<uint8_t> all_predictions(NEXT_OUT_MEMPTR(idx++));  // N,1024
    BTensor<int32_t> all_length(NEXT_OUT_MEMPTR(idx++));       // N

    int N = logits.shape[0];
    int C = logits.shape[1];
    int T = all_f.shape[1];

    // std::cout << "RnntUpdate::evaluate_T current_iter=" << current_iter.at(0) << std::endl;

    std::atomic<int> total_finished{0};

#define BLANK 28

    parallel_nt(0, [&](const int th_id, const int nthreads) {
        // revert Thread didn't introduce perf drop
        // th_id = nthreads - 1 - th_id;
        int n = N / nthreads;
        int n_left = N % nthreads;
        int i_start, i_end;
        if (th_id < n_left) {
            n += 1;
            i_start = th_id * n;
            i_end = i_start + n;
        } else {
            i_start = n_left * (n + 1) + (th_id - n_left) * n;
            i_end = i_start + n;
        }

        // std::stringstream ss;
        // ss << "========= th_id " << th_id << "/" << nthreads << "        " << i_start << "+" << n;
        // std::cout << ss.str() << std::endl;

        if (current_iter.at(0) == 0) {
            // initialize states
            for (int i = i_start; i < i_end; i++) {
                memset(&hs1.at(i, 0), 0, hs1.shape[1] * hs1.elesize);
                memset(&cs1.at(i, 0), 0, cs1.shape[1] * cs1.elesize);
                memset(&hs2.at(i, 0), 0, hs2.shape[1] * hs2.elesize);
                memset(&cs2.at(i, 0), 0, cs2.shape[1] * cs2.elesize);

                num_symbols_generated.at(i) = 0;
                last_symbol.at(i) = BLANK;
                time_idxs.at(i) = 0;

                // f only update partially, initialize is needed
                memcpy(&f.at(i), &all_f.at(i, 0), f.shape[1] * f.elesize);

                all_length.at(i) = 0;
            }
        }

        int local_finished = 0;
        for (int i = i_start; i < i_end; i++) {
            // auto& k = kargmax.at(i);
            auto* p_logits = &logits.at(i);
            int k = C - 1;
            auto max = p_logits[k];
            for (int c = 0; c < C - 1; c++) {
                if (max < p_logits[c]) {
                    max = p_logits[c];
                    k = c;
                }
            }

            auto& num = num_symbols_generated.at(i);
            // auto & f_end = flag_end.at(i);
            if (k != BLANK && num < 30) {
                auto& cur_len = all_length.at(i);
                auto& pred = all_predictions.at(i, cur_len);
                pred = k;
                cur_len++;

                num++;
                last_symbol.at(i) = k;
                memcpy(&hs1.at(i, 0), &o_hs1.at(i, 0), hs1.shape[1] * hs1.elesize);
                memcpy(&cs1.at(i, 0), &o_cs1.at(i, 0), cs1.shape[1] * cs1.elesize);
                memcpy(&hs2.at(i, 0), &o_hs2.at(i, 0), hs2.shape[1] * hs2.elesize);
                memcpy(&cs2.at(i, 0), &o_cs2.at(i, 0), cs2.shape[1] * cs2.elesize);
            } else {
                auto& t = time_idxs.at(i);
                num = 0;
                t++;
                if (t < T) {
                    // update i'th item in feature batch given new time_idx
                    memcpy(&f.at(i), &all_f.at(i, t), f.shape[1] * f.elesize);
                } else {
                    local_finished++;
                }
            }
        }
        if (local_finished)
            total_finished += local_finished;
    });

    next_cond.at(0) = (total_finished < N);
    return;
}

void RnntUpdate::execute(dnnl::stream strm) {
    evaluate_T<int32_t, float>();
}

std::vector<VectorDims> RnntUpdate::shapeInfer() const {
    return Node::shapeInferGeneric(0);
}

void RnntUpdate::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool RnntUpdate::created() const {
    return getType() == Type::RnntUpdate;
}

bool RnntUpdate::needShapeInfer() const {
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
