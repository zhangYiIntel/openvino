// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/solve_buffer_memory.hpp"

#include "snippets/lowered/pass/allocate_buffers.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
namespace {
std::map<double, int> create_execution_number_mapping(const LinearIR& linear_ir) {
    std::map<double, int> mapping;
    int int_number = 0;
    for (const auto& expr : linear_ir) {
        const auto& double_execution_number = expr->get_exec_num();
        OPENVINO_ASSERT(mapping.count(double_execution_number) == 0, "Incorrect Expression execution numbers!");
        mapping[double_execution_number] = int_number++;
    }
    return mapping;
}
}  // namespace

std::pair<LinearIR::container, LinearIR::container> SolveBufferMemory::extract_static_and_dynamic_buffers(const LinearIR::container& buffer_expressions) {
    LinearIR::container static_buffer_exprs, dynamic_buffer_exprs;
    for (const auto& buffer_expr : buffer_expressions) {
        const auto& buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
        OPENVINO_ASSERT(buffer, "Buffer clusters expects Buffer nodes");

        auto& clusters = buffer->is_defined() ? static_buffer_exprs : dynamic_buffer_exprs;
        clusters.push_back(buffer_expr);
    }

    // Validation check that buffer cluster has only static or dynamic buffers.
    for (const auto& static_buffer : static_buffer_exprs) {
        const auto static_cluster_id = ov::as_type_ptr<op::Buffer>(static_buffer->get_node())->get_cluster_id();
        auto is_cluster_ids_the_same = [&static_cluster_id](const ExpressionPtr& expr) {
            return static_cluster_id == ov::as_type_ptr<op::Buffer>(expr->get_node())->get_cluster_id();
        };
        OPENVINO_ASSERT(std::none_of(dynamic_buffer_exprs.cbegin(), dynamic_buffer_exprs.cend(), is_cluster_ids_the_same),
                        "There is Buffer cluster with buffers which has defined and undefined allocation sizes");
    }

    return { static_buffer_exprs, dynamic_buffer_exprs };
}

std::vector<ov::MemorySolver::Box> SolveBufferMemory::init_boxes(const LinearIR::container& buffer_expressions, const LinearIR& linear_ir) {
    // ov::MemorySolver interface requires integer execution numbers (lifetime must be integer).
    // To align with ov::MemorySolver interface, we create the map [double -> integer]
    const auto int_execution_numbers = create_execution_number_mapping(linear_ir);
    auto casted_execution_number = [&](const ExpressionPtr& expr) {
        const auto double_execution_number = expr->get_exec_num();
        OPENVINO_ASSERT(int_execution_numbers.count(double_execution_number) != 0, "Expression execution number has not been found!");
        return int_execution_numbers.at(double_execution_number);
    };

    std::map<int, ov::MemorySolver::Box> map_boxes;
    for (const auto& buffer_expr : buffer_expressions) {
        const auto& buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
        OPENVINO_ASSERT(buffer, "Buffer clusters expects Buffer nodes");
        auto cluster_id = static_cast<int>(buffer->get_cluster_id());

        if (map_boxes.count(cluster_id) == 0) {
            map_boxes[cluster_id] = { std::numeric_limits<int>::max(), 0, 0, cluster_id };
        }
        auto& box = map_boxes.at(cluster_id);

        int e_start = 0, e_finish = 0;

        // life finish time - order of LoopEnd / MemoryAccess ops
        const auto& buffer_outs = buffer_expr->get_output_port_connectors();
        for (const auto& buffer_out : buffer_outs) {
            const auto consumers = buffer_out->get_consumers();
            for (const auto& consumer : consumers) {
                e_finish = std::max(e_finish, casted_execution_number(consumer.get_expr()));  // the last consumer
            }
        }
        e_start = e_finish;

        const auto& buffer_ins = buffer_expr->get_input_port_connectors();
        for (const auto& buffer_in : buffer_ins) {
            const auto& source = buffer_in->get_source();
            e_start = casted_execution_number(source.get_expr());

            const auto buffer_siblings = buffer_in->get_consumers();
            for (const auto& sibling : buffer_siblings) {
                if (const auto loop_end = ov::as_type_ptr<ov::snippets::op::LoopEnd>(sibling.get_expr()->get_node())) {
                    e_start = std::min(e_start, casted_execution_number(sibling.get_expr()));
                }
            }
        }
        OPENVINO_ASSERT(e_start <= e_finish, "Incorrect life time of buffer!");

        auto buffer_size = static_cast<int64_t>(buffer->get_byte_size());
        box.size = std::max(buffer_size, box.size);

        box.start = std::min(e_start, box.start);
        box.finish = std::max(e_finish, box.finish);
    }

    std::vector<ov::MemorySolver::Box> boxes;
    boxes.reserve(map_boxes.size());
    for (auto& p : map_boxes) {
        auto& box = p.second;
        // We use data alignment to put data in the line cache
        // TODO [143395] : Please check if alignment is really needed here
        box.size = utils::div_up(box.size, m_alignment);

        boxes.push_back(box);
    }

    return boxes;
}

void SolveBufferMemory::solve_static_buffer_memory(const LinearIR::container& static_buffer_expressions, const LinearIR& linear_ir) {
    const auto boxes = init_boxes(static_buffer_expressions, linear_ir);

    ov::MemorySolver memSolver(boxes);
    m_static_buffer_scratchpad_size = static_cast<size_t>(memSolver.solve()) * m_alignment;  // alignment in byte

    // Set offsets for Buffers
    for (const auto& buffer_expr : static_buffer_expressions) {
        const auto& buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
        OPENVINO_ASSERT(buffer, "Buffer clusters expects Buffer nodes");

        const auto offset = static_cast<size_t>(memSolver.get_offset(static_cast<int>(buffer->get_cluster_id())));
        buffer->set_offset(offset * m_alignment);  // alignment in byte
    }
}

void SolveBufferMemory::set_dynamic_buffer_offset(const LinearIR::container& dynamic_buffer_expressions) {
    size_t offset = utils::get_dynamic_value<size_t>();

    // If there are not allocated memory for static buffers in LinearIR and there is only one cluster of dynamic buffer exprs,
    // we can force offset = 0
    if (m_static_buffer_scratchpad_size == 0) {
        std::set<size_t> dynamic_clusters;
        for (const auto& dynamic_buffer_expr : dynamic_buffer_expressions) {
            const auto& buffer = ov::as_type_ptr<op::Buffer>(dynamic_buffer_expr->get_node());
            OPENVINO_ASSERT(buffer, "Buffer clusters expects Buffer nodes");
            dynamic_clusters.insert(buffer->get_cluster_id());
        }
        if (dynamic_clusters.size() == 1)
            offset = 0;
    }

    // Set offsets for Buffers
    for (const auto& buffer_expr : dynamic_buffer_expressions) {
        const auto& buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
        OPENVINO_ASSERT(buffer, "Buffer clusters expects Buffer nodes");

        buffer->set_offset(offset);
    }
}

bool SolveBufferMemory::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SolveBufferMemory");

    // TODO [143395] : MemoryManager will be able to return two containers with dynamic and static buffers
    //                 without additional `extract` functions in all passes
    LinearIR::container static_buffer_exprs, dynamic_buffer_exprs;
    std::tie(static_buffer_exprs, dynamic_buffer_exprs) = extract_static_and_dynamic_buffers(linear_ir.get_buffers());

    if (!static_buffer_exprs.empty())
        solve_static_buffer_memory(static_buffer_exprs, linear_ir);

    if (!dynamic_buffer_exprs.empty())
        set_dynamic_buffer_offset(dynamic_buffer_exprs);

    return !static_buffer_exprs.empty() && !dynamic_buffer_exprs.empty();
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
