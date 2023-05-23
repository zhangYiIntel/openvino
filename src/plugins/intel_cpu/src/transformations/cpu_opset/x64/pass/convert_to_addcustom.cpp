#include "convert_to_addcustom.hpp"

#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/x64/op/add_custom.hpp"

static bool is_conv(std::shared_ptr<ov::Node>& ptr) {
    if (ov::is_type<ov::opset8::Convolution>(ptr))  {
        return true;
    } else {
        return false;
    }
}

ov::intel_cpu::ConvertToAddCustom::ConvertToAddCustom() {
    MATCHER_SCOPE(ConvertToAddCustom);
    using namespace ov::pass::pattern;
    // auto input_m = wrap_type<ov::opset1::MatMul>();
    auto input_m = any_input();
    auto const_m = wrap_type<ov::opset1::Constant>();
    auto add_m = wrap_type<ov::opset8::Add>({input_m->output(0), const_m->output(0)});
    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input_node =  pattern_to_output.at(input_m).get_node_shared_ptr();
        if (is_conv(input_node)) {
            return false;
        }
        auto const_node =  pattern_to_output.at(const_m).get_node_shared_ptr();
        auto add_node =  pattern_to_output.at(add_m).get_node_shared_ptr();
        auto addcustom_node = std::make_shared<AddCustom>(input_node, const_node, false);
        addcustom_node->set_friendly_name(add_node->get_friendly_name());
        replace_node(add_node, addcustom_node);
        // std::cout << "Replace Add " << addcustom_node->get_friendly_name() << std::endl;
        return true;
    };

    auto m = std::make_shared<Matcher>(add_m, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::ConvertSameShapeAddCustom::ConvertSameShapeAddCustom() {
    MATCHER_SCOPE(ConvertSameAddCustom);
    using namespace ov::pass::pattern;
    auto input1_m = any_input(has_static_rank());
    auto input2_m = any_input(has_static_rank());
    auto add_m = wrap_type<ov::opset8::Add>({input1_m->output(0), input2_m->output(0)});
    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input1_node =  pattern_to_output.at(input1_m).get_node_shared_ptr();
        auto input2_node =  pattern_to_output.at(input2_m).get_node_shared_ptr();
        if (is_conv(input1_node) || is_conv(input2_node)) {
            return false;
        }
        auto add_node =  pattern_to_output.at(add_m).get_node_shared_ptr();
        auto shape1 = input1_node->get_output_partial_shape(0);
        auto shape2 = input2_node->get_output_partial_shape(0);
        std::shared_ptr<ov::Node> left_node = input1_node;
        std::shared_ptr<ov::Node> right_node = input2_node;
        auto shape_compatible = [&](const ov::PartialShape& a, const ov::PartialShape& b) {
            if (a.compatible(b)) {
                return true;
            } else {
                size_t rank_a = a.rank().get_length();
                size_t rank_b = b.rank().get_length();
                size_t rank_diff = rank_a > rank_b ? (rank_a - rank_b) : (rank_b - rank_a);
                if (rank_diff == 1) {
                    left_node = rank_a > rank_b ? input1_node : input2_node;
                    right_node = rank_a > rank_b ? input2_node : input1_node;
                    return true;
                }
                return false;
            }
        };
        if (!shape_compatible(shape1, shape2)) {
            return false;
        }
        auto addcustom_node = std::make_shared<AddCustom>(left_node, right_node);
        addcustom_node->set_friendly_name(add_node->get_friendly_name());
        replace_node(add_node, addcustom_node);
        return true;
    };

    auto m = std::make_shared<Matcher>(add_m, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::FuseAddCustom::FuseAddCustom() {
    MATCHER_SCOPE(FuseAddCustom);
    using namespace ov::pass::pattern;
    auto input1_m = wrap_type<ov::intel_cpu::AddCustom>();
    auto input2_m = wrap_type<ov::opset1::MatMul>();
    auto const_m = wrap_type<ov::opset1::Constant>();
    auto add1_m = wrap_type<AddCustom>({input2_m->output(0), const_m->output(0)});
    auto add2_m = wrap_type<AddCustom>({input1_m->output(0), add1_m->output(0)});
    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input1_node =  pattern_to_output.at(input1_m).get_node_shared_ptr();
        auto input2_node =  pattern_to_output.at(input2_m).get_node_shared_ptr();
        auto add1_node =  pattern_to_output.at(add1_m).get_node_shared_ptr();
        auto add2_node =  pattern_to_output.at(add2_m).get_node_shared_ptr();
        auto const_node =  pattern_to_output.at(const_m).get_node_shared_ptr();
        auto shape1 = input1_node->get_output_partial_shape(0);
        auto shape2 = add1_node->get_output_partial_shape(0);
        if (!shape1.compatible(shape2)) {
            // std::cout << "Not suitbale Add Node Fuse AddCustom" << add2_node->get_friendly_name() << std::endl;
            return false;
        }
        auto addcustom_node = std::make_shared<AddCustom>(input1_node, input2_node, const_node);
        addcustom_node->set_friendly_name(add2_node->get_friendly_name());
        replace_node(add2_node, addcustom_node);
        // std::cout << "FuseAddCustom|" << addcustom_node->get_friendly_name() << std::endl;
        return true;
    };

    auto m = std::make_shared<Matcher>(add2_m, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::FuseAddCustomGelu::FuseAddCustomGelu() {
    MATCHER_SCOPE(FuseAddCustomGelu);
    using namespace ov::pass::pattern;
    auto input1_m = wrap_type<ov::intel_cpu::AddCustom>();
    auto gelu_m = wrap_type<ov::opset7::Gelu>({input1_m});
    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input1_node =  pattern_to_output.at(input1_m).get_node_shared_ptr();
        auto gelu_node =  pattern_to_output.at(gelu_m).get_node_shared_ptr();
        auto addcustom_node = std::make_shared<AddCustom>(input1_node->get_input_source_output(0), input1_node->get_input_source_output(1), true);
        addcustom_node->set_friendly_name(gelu_node->get_friendly_name());
        // std::cout << "FuseAddCustomGelu|" << addcustom_node->get_friendly_name() << std::endl;
        replace_node(gelu_node, addcustom_node);
        return true;
    };
    auto m = std::make_shared<Matcher>(gelu_m, matcher_name);
    this->register_matcher(m, callback);
}