// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>

#include "graph.hpp"
#include "openvino/openvino.hpp"
#include "repeated.hpp"  // hash
#include "utils/utils.hpp"

namespace ov {
namespace npuw {
namespace online {

class Group;  // forward declaration

namespace detail {
// At partitioning level we exclude some "non-Ops" to not interfere with the passes.
// We include some of them back to properly link everything at plugin level
bool isOp(const std::shared_ptr<ov::Node>& node);
}  // namespace detail

// Core part of the partitioning algorithm which implements a list of graph passes.
class Snapshot : public std::enable_shared_from_this<Snapshot> {
public:
    Snapshot(const std::shared_ptr<ov::Model>& model)
        : m_model(model),
          m_graph(std::make_shared<ade::Graph>()),
          m_node_to_prod_cons(std::make_shared<detail::OVNodeMap>()),
          m_node_to_gr(std::make_shared<detail::OVNodeToGroupMap>()) {}

    // Simple passes
    void singleGroup();

    // Initial OV model traversal to prepare initial groups of 1 layer each
    void buildGraph();

    // Simple passes to clean-up and reduce number of groups via merging them
    void collectLHF();
    void fuseRemnantsExtended();
    void fuseRemnants();
    void fuseInputs();

    // Advanced passes for repeated blocks algorithm
    void repeatedBlocks();
    void earlyAvoids();
    void earlyRegroup();
    void markInternalCompute();
    void resetExcludedRep();

    // Utility
    std::shared_ptr<ade::Graph> getGraph() const;
    size_t graphSize() const;
    const detail::OVNodeSet& getNodeProducers(const detail::OVNodePtr& node) const;
    const detail::OVNodeSet& getNodeConsumers(const detail::OVNodePtr& node) const;
    const detail::OVPortsMap& getPortsMap() const;
    const detail::OVNodeToGroupMapPtr& getNodeToGroupMap() const;
    const std::map<std::string, std::vector<std::set<std::string>>>& getMatches() const;
    detail::GPtrSet getRepGroups(const std::shared_ptr<Group>& group) const;
    void repeat(detail::Pass&& pass);
    void setCtx(const PassContext& ctx);

private:
    void identifyUniques();
    void mergeUniques();
    void mergeTriangles();
    void cleanUpUniques();
    void afterUniques();
    bool cleanUpUniquesImpl(const detail::GPtrSet& gset);
    std::shared_ptr<Repeated> tryGrowRepeatingGroups(const detail::GPtrSet& repeating_groups);
    std::shared_ptr<Repeated> tryMergeTriangles(const detail::GPtrSet& repeating_groups);
    std::shared_ptr<Repeated> tryMergeTriangles(const std::vector<std::shared_ptr<Group>>& prods,
                                                const std::vector<std::vector<std::shared_ptr<Group>>>& conss);
    std::shared_ptr<Repeated> tryMergeRepeating(const std::vector<std::shared_ptr<Group>>& prods,
                                                const std::vector<std::shared_ptr<Group>>& conss);
    std::unordered_map<std::shared_ptr<Repeated>, detail::GPtrSet> repeating() const;
    void completeRepeating(const std::shared_ptr<Repeated>& reptag, const detail::GPtrSet& gset);

    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<ade::Graph> m_graph;
    detail::OVNodeMapPtr m_node_to_prod_cons;
    detail::OVNodeToGroupMapPtr m_node_to_gr;
    PassContext m_ctx;

    detail::OVPortsMap m_ports_map;
    std::map<std::string, std::vector<std::set<std::string>>> m_layer_matches;
};

}  // namespace online
}  // namespace npuw
}  // namespace ov
