#include "transformations/common_optimizations/fuse_linear_attention.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/linear_attn.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/util/op_types.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::pass;

namespace {

using InputDesc = ov::op::util::MultiSubGraphOp::InputDescription;

bool is_slice_desc(const std::shared_ptr<InputDesc>& desc,
				   uint64_t input_index,
				   int64_t axis,
				   int64_t part_size) {
	auto slice = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::SliceInputDescription>(desc);
	if (!slice) {
		return false;
	}
	return slice->m_input_index == input_index && slice->m_axis == axis && slice->m_part_size == part_size &&
		   slice->m_stride == 1 && slice->m_start == 0 && slice->m_end == -1;
}

bool is_merged_desc(const std::shared_ptr<InputDesc>& desc, uint64_t input_index) {
	auto merged = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::MergedInputDescription>(desc);
	return merged && merged->m_input_index == input_index;
}


bool matches_linear_attention_loop(const std::shared_ptr<ov::op::v5::Loop>& loop) {
	if (!loop) {
		return false;
	}

	if (loop->get_input_size() < 9 || loop->get_output_size() != 2) {
		return false;
	}

	const auto& body = loop->get_function();
	if (!body) {
		return false;
	}

	if (body->get_parameters().size() < 8 || body->get_results().size() < 3) {
		return false;
	}

	const auto& input_descs = loop->get_input_descriptions();
	if (input_descs.size() < 7) {
		return false;
	}

	for (uint64_t idx = 2; idx <= 6; ++idx) {
		bool found_slice = false;
		for (const auto& desc : input_descs) {
			if (is_slice_desc(desc, idx, 1, 1)) {
				found_slice = true;
				break;
			}
		}
		if (!found_slice) {
			return false;
		}
	}

	bool has_merged_state = false;
	bool has_merged_output = false;
	for (const auto& desc : input_descs) {
		if (is_merged_desc(desc, 7)) {
			has_merged_state = true;
		}
		if (is_merged_desc(desc, 8)) {
			has_merged_output = true;
		}
	}
	if (!has_merged_state || !has_merged_output) {
		return false;
	}

	const auto& output_descs = loop->get_output_descriptions();
	if (output_descs.size() != 2) {
		return false;
	}

	// bool has_scatter = false;
	// bool has_add = false;
	// for (const auto& result : body->get_results()) {
	// 	auto result_node = result->input_value(0).get_node_shared_ptr();
	// 	if (std::dynamic_pointer_cast<ov::op::v3::ScatterUpdate>(result_node) != nullptr) {
	// 		has_scatter = true;
	// 	}
	// 	if (is_add_node(result_node)) {
	// 		has_add = true;
	// 	}
	// }
	// if (!has_scatter || !has_add) {
	// 	return false;
	// }

	// if (!has_body_node_type(body, ov::op::v0::Exp::get_type_info_static()) ||
	// 	!has_body_node_type(body, ov::op::v1::ReduceSum::get_type_info_static()) ||
	// 	!has_body_node_type(body, ov::op::v1::Multiply::get_type_info_static())) {
	// 	return false;
	// }

	return true;
}

}  // namespace
using namespace ov::gen_pattern;
using namespace ov::pass::pattern;
ov::pass::LinearAttentionFusion::LinearAttentionFusion() {
	auto key = ov::pass::pattern::any_input();
	auto query = ov::pass::pattern::any_input();
	auto value = ov::pass::pattern::any_input();
	auto axis_q_const = pattern::wrap_type<opset1::Constant>();
	auto axis_q_convert = pattern::wrap_type<opset1::Convert>({axis_q_const});
	auto axis_q = std::make_shared<pattern::op::Or>(OutputVector{axis_q_const, axis_q_convert});

	auto eps_q_const = pattern::wrap_type<opset1::Constant>();
	auto eps_q_convert = pattern::wrap_type<opset1::Convert>({eps_q_const});
	auto eps_q = std::make_shared<pattern::op::Or>(OutputVector{eps_q_const, eps_q_convert});

	auto inv_const_q_const = pattern::wrap_type<opset1::Constant>();
	auto inv_const_q_convert = pattern::wrap_type<opset1::Convert>({inv_const_q_const});
	auto inv_const_q = std::make_shared<pattern::op::Or>(OutputVector{inv_const_q_const, inv_const_q_convert});

	auto axis_k_const = pattern::wrap_type<opset1::Constant>();
	auto axis_k_convert = pattern::wrap_type<opset1::Convert>({axis_k_const});
	auto axis_k = std::make_shared<pattern::op::Or>(OutputVector{axis_k_const, axis_k_convert});

	auto eps_k_const = pattern::wrap_type<opset1::Constant>();
	auto eps_k_convert = pattern::wrap_type<opset1::Convert>({eps_k_const});
	auto eps_k = std::make_shared<pattern::op::Or>(OutputVector{eps_k_const, eps_k_convert});

	auto inv_const_k_const = pattern::wrap_type<opset1::Constant>();
	auto inv_const_k_convert = pattern::wrap_type<opset1::Convert>({inv_const_k_const});
	auto inv_const_k = std::make_shared<pattern::op::Or>(OutputVector{inv_const_k_const, inv_const_k_convert});

	auto minus_one = pattern::wrap_type<opset1::Constant>();

	auto Multiply_14 = pattern::wrap_type<opset1::Multiply>({query, query}, {{"auto_broadcast", "numpy"}});
	auto ReduceSum_15 = pattern::wrap_type<opset1::ReduceSum>({Multiply_14, axis_q->output(0)}, {{"keep_dims", true}});
	auto Add_18 = pattern::wrap_type<opset1::Add>({ReduceSum_15, eps_q->output(0)}, {{"auto_broadcast", "numpy"}});
	auto Sqrt_19 = pattern::wrap_type<opset1::Sqrt>({Add_18});
	auto Divide_20 = pattern::wrap_type<opset1::Divide>({inv_const_q->output(0), Sqrt_19}, {{"auto_broadcast", "numpy"}});
	auto Power_20 = pattern::wrap_type<opset1::Power>({Sqrt_19, minus_one}, {{"auto_broadcast", "numpy"}});
	auto inv_sqrt_q = std::make_shared<pattern::op::Or>(OutputVector{Divide_20, Power_20});
	auto Multiply_21 = pattern::wrap_type<opset1::Multiply>({query, inv_sqrt_q->output(0)}, {{"auto_broadcast", "numpy"}});
	// q / sqrt(d)
	auto Multiply_32 = pattern::wrap_type<opset1::Multiply>({Multiply_21, any_input()}, {{"auto_broadcast", "numpy"}});

	auto Multiply_32_compressed_to_f16 = pattern::wrap_type<op::v0::Convert>({Multiply_32}, {{"destination_type", "f16"}});

	auto Multiply_22 = pattern::wrap_type<opset1::Multiply>({key, key}, {{"auto_broadcast", "numpy"}});
	auto ReduceSum_23 = pattern::wrap_type<opset1::ReduceSum>({Multiply_22, axis_k->output(0)}, {{"keep_dims", true}});
	auto Add_26 = pattern::wrap_type<opset1::Add>({ReduceSum_23, eps_k->output(0)}, {{"auto_broadcast", "numpy"}});
	auto Sqrt_27 = pattern::wrap_type<opset1::Sqrt>({Add_26});
	auto Divide_28 = pattern::wrap_type<opset1::Divide>({inv_const_k->output(0), Sqrt_27}, {{"auto_broadcast", "numpy"}});
	auto Power_28 = pattern::wrap_type<opset1::Power>({Sqrt_27, minus_one}, {{"auto_broadcast", "numpy"}});
	auto inv_sqrt_k = std::make_shared<pattern::op::Or>(OutputVector{Divide_28, Power_28});
	auto Multiply_29 = pattern::wrap_type<opset1::Multiply>({key, inv_sqrt_k->output(0)}, {{"auto_broadcast", "numpy"}});
	auto Multiply_29_compressed_to_f16 = pattern::wrap_type<op::v0::Convert>({Multiply_29}, {{"destination_type", "f16"}});

	auto q_in = std::make_shared<pattern::op::Or>(OutputVector{Multiply_32, Multiply_32_compressed_to_f16});
	auto k_in = std::make_shared<pattern::op::Or>(OutputVector{Multiply_29, Multiply_29_compressed_to_f16});

	auto loop_label = ov::pass::pattern::wrap_type<ov::op::v5::Loop>(OutputVector{
		any_input(),
		any_input(),
		q_in->output(0),
		k_in->output(0),
		value,
		any_input(),
		any_input(),
		any_input(),
		any_input()
	});

	matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
		const auto& pattern_map = m.get_pattern_value_map();
		auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(m.get_match_root());
        std::cout << "LinearAttentionFusion 11111." << std::endl;
		if (!matches_linear_attention_loop(loop)) {
			return false;
		}

		std::vector<std::shared_ptr<ov::Node>> rt_nodes{loop};

		ov::Output<Node> query_in, key_in;
		if (pattern_map.count(Multiply_32_compressed_to_f16)) {
			 query_in = pattern_map.at(Multiply_32_compressed_to_f16);
		} else {
			query_in = pattern_map.at(query);
		}

		if (pattern_map.count(Multiply_29_compressed_to_f16)) {
			 key_in = pattern_map.at(Multiply_29_compressed_to_f16);
		} else {
			key_in = pattern_map.at(key);
		}
		auto value_in = loop->input_value(4);
		ov::OutputVector inputs;
		inputs.reserve(6);

		// Loop inputs layout in the target subgraph:
		// 0: trip_count, 1: execution_condition, 2: query, 3: key, 4: value, 5: g, 6: beta, 7: initial_state, 8: init_output
		inputs.push_back(query_in);  // query
		inputs.push_back(key_in);    // key
		inputs.push_back(value_in);  // value
		inputs.push_back(loop->input_value(5));  // g
		inputs.push_back(loop->input_value(6));  // beta
		inputs.push_back(loop->input_value(7));  // initial_state

		auto linear_attn = std::make_shared<ov::op::LinearAttention>(inputs);
		linear_attn->set_friendly_name(loop->get_friendly_name());

		ov::copy_runtime_info(rt_nodes, linear_attn);
		ov::replace_node(loop, linear_attn);
        std::cout << "LinearAttentionFusion applied." << std::endl;
		register_new_node(linear_attn);
		return true;
	};

	auto m = std::make_shared<ov::pass::pattern::Matcher>(loop_label, "LinearAttentionFusion");
	register_matcher(m, callback);
}
