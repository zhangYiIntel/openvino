// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/linear_attn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include <cmath>
#include <vector>

namespace {
class LinearAttentionVariableInference : virtual public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        inType = ov::element::f16;

        B = 1;
        Seq = 16;
        HK = 2;
        H = 2;
        K = 16;
        V = 16;

        auto q = std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape{B, Seq, HK, K});
        auto k = std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape{B, Seq, HK, K});
        auto v = std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape{B, Seq, H, V});
        auto g = std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape{B, Seq, H});
        auto beta = std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape{B, Seq, H});

        q->set_friendly_name("q");
        k->set_friendly_name("k");
        v->set_friendly_name("v");
        g->set_friendly_name("g");
        beta->set_friendly_name("beta");

        state_init_f = std::vector<float>(B * H * K * V, 0.0f);
        auto state_init = ov::op::v0::Constant::create(inType, ov::Shape{B, H, K, V}, to_fp16(state_init_f));
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{B, H, K, V}, inType, "state"});
        auto read = std::make_shared<ov::op::v6::ReadValue>(state_init, variable);

        auto la = std::make_shared<ov::op::LinearAttention>(ov::OutputVector{q, k, v, g, beta, read});
        auto assign = std::make_shared<ov::op::v6::Assign>(la->output(1), variable);
        auto output = std::make_shared<ov::op::v0::Result>(la->output(0));

        function = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::SinkVector{assign}, ov::ParameterVector{q, k, v, g, beta},
                                               "linear_attention_variable");

        abs_threshold = 0.05f;
        rel_threshold = 0.05f;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        auto itShape = targetInputStaticShapes.begin();

        for (const auto& param : function->get_parameters()) {
            auto tensor = ov::test::utils::create_and_fill_tensor(inType, *itShape,
                                                                  ov::test::utils::InputGenerateData(0, 8, 32, 1));
            inputs.insert({param, tensor});
            if (param->get_friendly_name() == "q") {
                q_f = to_float(tensor);
            } else if (param->get_friendly_name() == "k") {
                k_f = to_float(tensor);
            } else if (param->get_friendly_name() == "v") {
                v_f = to_float(tensor);
            } else if (param->get_friendly_name() == "g") {
                g_f = to_float(tensor);
            } else if (param->get_friendly_name() == "beta") {
                beta_f = to_float(tensor);
            }
            ++itShape;
        }
    }

    std::vector<ov::Tensor> calculate_refs() override {
        std::vector<float> output(B * Seq * H * V, 0.0f);
        auto state = state_init_f;
        run_reference(state, output);
        ref_state_f = state;
        return {ov::test::utils::create_tensor(inType, ov::Shape{B, Seq, H, V}, to_fp16(output))};
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            ov::test::utils::compare(expected[i], actual[i], inference_precision, abs_threshold, rel_threshold, topk_threshold, mvn_threshold);
        }
    }

    void validate() override {
        auto actualOutputs = get_plugin_outputs();
        auto expectedOutputs = calculate_refs();
        compare(expectedOutputs, actualOutputs);

        auto states = inferRequest.query_state();
        ASSERT_EQ(states.size(), 1u);
        auto state_tensor = states[0].get_state();
        auto ref_state_tensor = ov::test::utils::create_tensor(inType, ov::Shape{B, H, K, V}, to_fp16(ref_state_f));
        ov::test::utils::compare(ref_state_tensor, state_tensor, inference_precision, abs_threshold, rel_threshold, topk_threshold, mvn_threshold);
    }

private:
    size_t B = 0;
    size_t Seq = 0;
    size_t HK = 0;
    size_t H = 0;
    size_t K = 0;
    size_t V = 0;

    std::vector<float> q_f;
    std::vector<float> k_f;
    std::vector<float> v_f;
    std::vector<float> g_f;
    std::vector<float> beta_f;
    std::vector<float> state_init_f;
    std::vector<float> ref_state_f;

    static std::vector<ov::float16> to_fp16(const std::vector<float>& src) {
        std::vector<ov::float16> dst(src.size());
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] = ov::float16(src[i]);
        }
        return dst;
    }

    static std::vector<float> to_float(const ov::Tensor& tensor) {
        const auto* data = tensor.data<ov::float16>();
        std::vector<float> dst(tensor.get_size());
        for (size_t i = 0; i < dst.size(); i++) {
            dst[i] = static_cast<float>(data[i]);
        }
        return dst;
    }

    static float dot(const float* a, const float* b, size_t n) {
        float result = 0.0f;
        for (size_t i = 0; i < n; i++)
            result += a[i] * b[i];
        return result;
    }

    static void scale(float* a, float s, size_t n) {
        for (size_t i = 0; i < n; i++)
            a[i] *= s;
    }

    static void add(float* a, const float* b, size_t n) {
        for (size_t i = 0; i < n; i++)
            a[i] += b[i];
    }

    static void l2norm(float* a, size_t n) {
        float eps = 0.000001f;
        float sum = 0.0f;
        for (size_t j = 0; j < n; j++)
            sum += a[j] * a[j];
        sum = 1.0f / std::sqrt(sum + eps);
        for (size_t j = 0; j < n; j++)
            a[j] *= sum;
    }

    void run_reference(std::vector<float>& state, std::vector<float>& output) {
        for (size_t i_b = 0; i_b < B; i_b++) {
            for (size_t i_h = 0; i_h < H; i_h++) {
                for (size_t i_v = 0; i_v < V; i_v++) {
                    std::vector<float> init_state(K, 0.0f);
                    std::vector<float> b_k(K, 0.0f);
                    std::vector<float> b_q(K, 0.0f);

                    size_t BATCH_STRIDE_Q = HK * K * Seq;
                    size_t BATCH_STRIDE_K = HK * K * Seq;
                    size_t BATCH_STRIDE_V = H * K * Seq;
                    size_t HEAD_STRIDE = K;
                    size_t group_size = H / HK;
                    size_t i_hk = i_h / group_size;
                    const float* q_ptr = q_f.data() + i_b * BATCH_STRIDE_Q + i_hk * HEAD_STRIDE;
                    const float* k_ptr = k_f.data() + i_b * BATCH_STRIDE_K + i_hk * HEAD_STRIDE;
                    const float* v_ptr = v_f.data() + i_b * BATCH_STRIDE_V + i_h * HEAD_STRIDE;

                    for (size_t j = 0; j < K; j++) {
                        init_state[j] = state[i_b * H * K * V + i_h * K * V + j * V + i_v];
                    }

                    for (size_t i = 0; i < Seq; i++) {
                        size_t G_B_STRIDE = Seq * H;
                        float b_g = std::exp(g_f[i_b * G_B_STRIDE + i * H + i_h]);
                        float b_beta = beta_f[i_b * G_B_STRIDE + i * H + i_h];

                        for (size_t j = 0; j < K; j++) {
                            b_k[j] = k_ptr[i * K * HK + j];
                            b_q[j] = q_ptr[i * K * HK + j];
                        }

                        l2norm(b_k.data(), K);
                        l2norm(b_q.data(), K);
                        scale(b_q.data(), 1.0f / std::sqrt(static_cast<float>(K)), K);

                        scale(init_state.data(), b_g, K);
                        float h_k = dot(init_state.data(), b_k.data(), K);
                        float b_v = v_ptr[i_v + i * V * H];
                        b_v = (b_v - h_k) * b_beta;
                        scale(b_k.data(), b_v, K);
                        add(init_state.data(), b_k.data(), K);

                        float b_output = dot(init_state.data(), b_q.data(), K);
                        output[i_b * Seq * H * V + i * H * V + i_h * V + i_v] = b_output;
                    }

                    for (size_t j = 0; j < K; j++) {
                        state[i_b * H * K * V + i_h * K * V + j * V + i_v] = init_state[j];
                    }
                }
            }
        }
    }
};
}  // namespace

TEST_F(LinearAttentionVariableInference, Inference) {
    run();
}
