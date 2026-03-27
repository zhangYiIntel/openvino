// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note PagedGatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing paged Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PagedGatedDeltaNet : public ov::op::Op {
public:
    OPENVINO_OP("PagedGatedDeltaNet");

    PagedGatedDeltaNet() = default;

    /// \brief Constructs a PagedGatedDeltaNet operation.
    ///
    /// \param query Query tensor input.
    /// \param key Key tensor input.
    /// \param value Value tensor input.
    /// \param recurrent_state_table Paged recurrent state table input.
    /// \param gate Gate tensor controlling state decay/update.
    /// \param beta Beta tensor scaling the delta update.
    /// \param subsequence_begins Start offsets of each sequence in token batch.
    /// \param block_indices Block indices for paged recurrent state table.
    /// \param block_indices_begins Start offsets of block index ranges per sequence.
    /// \param past_lens Number of tokens processed before current batch per sequence.
    /// \param cache_interval State caching interval per sequence.
    PagedGatedDeltaNet(const Output<Node>& query,
                       const Output<Node>& key,
                       const Output<Node>& value,
                       const Output<Node>& recurrent_state_table,
                       const Output<Node>& gate,
                       const Output<Node>& beta,
                       const Output<Node>& subsequence_begins,
                       const Output<Node>& block_indices,
                       const Output<Node>& block_indices_begins,
                       const Output<Node>& past_lens,
                       const Output<Node>& cache_interval);

    /// \brief Constructs a PagedGatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order:
    /// query, key, value, recurrent_state_table, gate, beta,
    /// subsequence_begins, block_indices, block_indices_begins, past_lens, cache_interval.
    explicit PagedGatedDeltaNet(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::op::internal
