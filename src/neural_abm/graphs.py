"""Graph helpers for Neural ABM simulations."""

from __future__ import annotations

import networkx as nx

from neural_abm.config import GraphConfig


def build_graph(config: GraphConfig, agent_count: int, seed: int) -> nx.Graph:
    """Build the configured candidate interaction graph."""

    if config.type == "watts_strogatz":
        return nx.watts_strogatz_graph(
            n=agent_count,
            k=config.k,
            p=config.rewire_probability,
            seed=seed,
        )
    raise ValueError(f"Unsupported graph type: {config.type}")


def component_map(graph: nx.Graph) -> dict[int, int]:
    """Map node id to connected component id."""

    mapping: dict[int, int] = {}
    for component_id, nodes in enumerate(nx.connected_components(graph)):
        for node in nodes:
            mapping[int(node)] = component_id
    return mapping


def graph_from_peer_ids(agent_count: int, peer_ids: list[list[int]]) -> nx.Graph:
    """Create an undirected graph from directed peer selections."""

    graph = nx.Graph()
    graph.add_nodes_from(range(agent_count))
    for source, peers in enumerate(peer_ids):
        for target in peers:
            graph.add_edge(source, target)
    return graph
