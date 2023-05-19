#include "convert_to_layernorm.hpp"

#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/x64/op/layer_norm.hpp"

ov::intel_cpu::ConvertToLayerNorm::ConvertToLayerNorm() {
    MATCHER_SCOPE(ConvertToLayerNorm);
    using namespace ov::pass::pattern;
    auto input_m = any_input(has_static_rank());
    auto axes_m = wrap_type<ov::opset8::Constant>();
    auto mvn_m = wrap_type<ov::opset6::MVN>({input_m, axes_m});
    auto mul_const_m =  wrap_type<ov::opset8::Constant>();
    auto mul = wrap_type<ov::opset1::Multiply>({mvn_m->output(0), mul_const_m->output(0)});
    auto add_const_m =  wrap_type<ov::opset8::Constant>();
    auto add_m = wrap_type<ov::opset1::Add>({mul->output(0), add_const_m->output(0)});
    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input_node =  pattern_to_output.at(input_m).get_node_shared_ptr();
        auto mvn_node =  pattern_to_output.at(mvn_m).get_node_shared_ptr();
        auto axes_node = pattern_to_output.at(axes_m).get_node_shared_ptr();
        auto const_node = ov::as_type_ptr<ov::opset8::Constant>(axes_node);
        auto mul_const_node = pattern_to_output.at(mul_const_m).get_node_shared_ptr();
        auto add_const_node = pattern_to_output.at(add_const_m).get_node_shared_ptr();
        auto add_node = pattern_to_output.at(add_m).get_node_shared_ptr();
        auto axes_vec = const_node->get_vector<int32_t>();
        size_t input_rank = input_node->get_input_partial_shape(0).rank().get_length();
        bool suitable_axes = axes_vec.size() == 1 && (axes_vec[0] == -1 || axes_vec[0] == (input_rank - 1));
        if (!suitable_axes) {
            std::cout << "Not Suitable Layernorm|" << mvn_node->get_friendly_name() << std::endl;
            return false;
        }
        auto layer_norm = std::make_shared<LayerNorm>(input_node, mul_const_node, add_const_node);
        layer_norm->set_friendly_name(add_node->get_friendly_name());
        replace_node(add_node, layer_norm);
        std::cout << "match Layernorm|" << mvn_node->get_friendly_name() << std::endl;
        return true;
    };

    auto m = std::make_shared<Matcher>(add_m, matcher_name);
    this->register_matcher(m, callback);
}