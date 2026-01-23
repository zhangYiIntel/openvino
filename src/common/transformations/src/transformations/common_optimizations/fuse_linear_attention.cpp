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
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/linear_attn.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/util/op_types.hpp"

using namespace ov::pass;

namespace {

using InputDesc = ov::op::util::MultiSubGraphOp::InputDescription;
using OutputDesc = ov::op::util::MultiSubGraphOp::OutputDescription;

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

bool is_concat_desc(const std::shared_ptr<OutputDesc>& desc,
					uint64_t output_index,
					int64_t axis,
					int64_t part_size) {
	auto concat = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>(desc);
	if (!concat) {
		return false;
	}
	return concat->m_output_index == output_index && concat->m_axis == axis && concat->m_part_size == part_size &&
		   concat->m_stride == 1 && concat->m_start == 0 && concat->m_end == -1;
}

bool is_body_desc(const std::shared_ptr<OutputDesc>& desc, uint64_t output_index) {
	auto body = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::BodyOutputDescription>(desc);
	return body && body->m_output_index == output_index;
}

bool has_body_node_type(const std::shared_ptr<ov::Model>& body,
						const ov::DiscreteTypeInfo& type_info) {
	for (const auto& node : body->get_ops()) {
		if (node->get_type_info() == type_info) {
			return true;
		}
	}
	return false;
}

bool is_add_node(const std::shared_ptr<ov::Node>& node) {
	return std::dynamic_pointer_cast<ov::op::v1::Add>(node) != nullptr;
}

bool is_const_one(const std::shared_ptr<ov::Node>& node) {
	auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
	if (!c) {
		return false;
	}
	if (!c->get_element_type().is_real()) {
		return false;
	}
	std::vector<float> values;
	try {
		values = c->cast_vector<float>();
	} catch (...) {
		return false;
	}
	if (values.empty()) {
		return false;
	}
	for (auto v : values) {
		if (std::abs(v - 1.0f) > 1e-6f) {
			return false;
		}
	}
	return true;
}

bool match_l2norm(const ov::Output<ov::Node>& value,
				 ov::Output<ov::Node>& original,
				 std::vector<std::shared_ptr<ov::Node>>& rt_nodes) {
	auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(value.get_node_shared_ptr());
	if (!mul) {
		return false;
	}

	for (size_t i = 0; i < 2; ++i) {
		auto x = mul->input_value(i);
		auto scale = mul->input_value(1 - i);

		auto div = std::dynamic_pointer_cast<ov::op::v1::Divide>(scale.get_node_shared_ptr());
		if (!div) {
			continue;
		}
		if (!is_const_one(div->input_value(0).get_node_shared_ptr())) {
			continue;
		}

		auto sqrt = std::dynamic_pointer_cast<ov::op::v0::Sqrt>(div->input_value(1).get_node_shared_ptr());
		if (!sqrt) {
			continue;
		}
		auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(sqrt->input_value(0).get_node_shared_ptr());
		if (!add) {
			continue;
		}

		std::shared_ptr<ov::op::v1::ReduceSum> reduce_sum;
		std::shared_ptr<ov::op::v0::Constant> eps;
		for (size_t j = 0; j < 2; ++j) {
			reduce_sum = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(add->input_value(j).get_node_shared_ptr());
			eps = std::dynamic_pointer_cast<ov::op::v0::Constant>(add->input_value(1 - j).get_node_shared_ptr());
			if (reduce_sum && eps) {
				break;
			}
		}
		if (!reduce_sum || !eps) {
			continue;
		}
		if (!reduce_sum->get_keep_dims()) {
			continue;
		}
		auto square = std::dynamic_pointer_cast<ov::op::v1::Multiply>(reduce_sum->input_value(0).get_node_shared_ptr());
		if (!square) {
			continue;
		}
		if (square->input_value(0).get_node_shared_ptr() != x.get_node_shared_ptr() ||
			square->input_value(1).get_node_shared_ptr() != x.get_node_shared_ptr()) {
			continue;
		}

		original = x;
		rt_nodes.push_back(square);
		rt_nodes.push_back(reduce_sum);
		rt_nodes.push_back(add);
		rt_nodes.push_back(sqrt);
		rt_nodes.push_back(div);
		rt_nodes.push_back(mul);
		return true;
	}

	return false;
}

ov::Output<ov::Node> strip_l2norm(const ov::Output<ov::Node>& value,
						 std::vector<std::shared_ptr<ov::Node>>& rt_nodes) {
	ov::Output<ov::Node> original;
	if (match_l2norm(value, original, rt_nodes)) {
		return original;
	}
	return value;
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

ov::pass::LinearAttentionFusion::LinearAttentionFusion() {
	auto loop_label = ov::pass::pattern::wrap_type<ov::op::v5::Loop>();

	matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& m) {
		auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(m.get_match_root());
        std::cout << "LinearAttentionFusion 11111." << std::endl;
		if (!matches_linear_attention_loop(loop)) {
			return false;
		}

		std::vector<std::shared_ptr<ov::Node>> rt_nodes{loop};

		auto query_in = strip_l2norm(loop->input_value(2), rt_nodes);
		auto key_in = strip_l2norm(loop->input_value(3), rt_nodes);
		auto value_in = strip_l2norm(loop->input_value(4), rt_nodes);

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
