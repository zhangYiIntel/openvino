// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_leaky_relu.hpp"

#include "openvino/opsets/opset1.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/cpu_opset/common/op/leaky_relu.hpp"

#include "itt.hpp"

std::shared_ptr<ov::opset1::Constant> getConstant(const std::shared_ptr<ov::Node>& node, size_t index) {
    bool isConvert = ov::is_type<ov::opset1::Convert>(node->get_input_node_shared_ptr(index));
    std::shared_ptr<ov::opset1::Constant> constant = ov::as_type_ptr<ov::opset1::Constant>(
        isConvert ? node->get_input_node_shared_ptr(index)->get_input_node_shared_ptr(0)
                  : node->get_input_node_shared_ptr(index));
    return constant;
}

ov::intel_cpu::ConvertToLeakyRelu::ConvertToLeakyRelu() {
    MATCHER_SCOPE(ConvertToLeakyRelu);
    auto input = ov::pass::pattern::any_input();
    auto slope_constant = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    auto slope_convert_constant = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    auto convert = ov::pass::pattern::wrap_type<ov::opset1::Convert>({slope_convert_constant});
    auto prelu = ov::pass::pattern::wrap_type<ov::opset1::PRelu>({ input, std::make_shared<pass::pattern::op::Or>(OutputVector{slope_constant, convert}) });

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto prelu = std::dynamic_pointer_cast<ov::opset1::PRelu>(m.get_match_root());
        if (!prelu) {
            return false;
        }
        std::cout << "ConvertPReluToLeakyRelu" << std::endl;
        auto slopeNode = getConstant(prelu, 1);
        if (slopeNode != nullptr && ov::shape_size(slopeNode->get_shape()) == 1) {
            const float slope = slopeNode->cast_vector<float>()[0];
            const auto leakyRelu = std::make_shared<ov::intel_cpu::LeakyReluNode>(prelu->input(0).get_source_output(), slope,
                                                                                 prelu->output(0).get_element_type());
            leakyRelu->set_friendly_name(prelu->get_friendly_name());
            ov::copy_runtime_info(prelu, leakyRelu);
            ov::replace_node(prelu, leakyRelu);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(prelu, matcher_name);
    this->register_matcher(m, callback);
}
