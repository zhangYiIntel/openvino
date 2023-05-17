#include "convert_to_addcustom.hpp"

#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/x64/op/add_custom.hpp"

ov::intel_cpu::ConvertToAddCustom::ConvertToAddCustom() {
    MATCHER_SCOPE(ConvertToInteraction);
    using namespace ov::pass::pattern;
    auto input_m = any_input();
    auto const_m = wrap_type<ov::opset1::Constant>();
    auto add_m = wrap_type<ov::opset8::Add>({input_m->output(0), const_m->output(0)});
    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input_node =  pattern_to_output.at(input_m).get_node_shared_ptr();
        auto const_node =  pattern_to_output.at(const_m).get_node_shared_ptr();
        auto add_node =  pattern_to_output.at(add_m).get_node_shared_ptr();
        auto addcustom_node = std::make_shared<AddCustom>(input_node, const_node);
        addcustom_node->set_friendly_name(add_node->get_friendly_name());
        replace_node(add_node, addcustom_node);
        std::cout << "Replace Add " << addcustom_node->get_friendly_name() << std::endl;
        return true;
    };

    auto m = std::make_shared<Matcher>(add_m, matcher_name);
    this->register_matcher(m, callback);
}