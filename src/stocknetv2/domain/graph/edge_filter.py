from __future__ import annotations

from collections import defaultdict

from stocknetv2.domain.graph.edge import GraphEdge


def keep_top_k_per_symbol(
    edges: list[GraphEdge],
    top_k_per_symbol: int,
    *,
    reciprocal_top_k: int | None = None,
    degree_cap: int | None = None,
) -> list[GraphEdge]:
    if top_k_per_symbol <= 0 and (degree_cap is None or degree_cap <= 0):
        return edges

    symbol_edges: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        symbol_edges[edge.source_symbol].append(edge)
        symbol_edges[edge.target_symbol].append(edge)

    reciprocal_limit = top_k_per_symbol if reciprocal_top_k is None else reciprocal_top_k
    ranked_neighbors: dict[str, list[GraphEdge]] = {}
    reciprocal_neighbors: dict[str, set[tuple[str, str]]] = {}
    for symbol, symbol_edges_list in symbol_edges.items():
        ranked = sorted(symbol_edges_list, key=lambda item: item.weight, reverse=True)
        if top_k_per_symbol > 0:
            ranked = ranked[:top_k_per_symbol]
        ranked_neighbors[symbol] = ranked
        reciprocal_neighbors[symbol] = {
            tuple(sorted((edge.source_symbol, edge.target_symbol)))
            for edge in ranked[: max(reciprocal_limit, 0)]
        }

    selected_keys: set[tuple[str, str]] = set()
    for symbol, ranked in ranked_neighbors.items():
        for edge in ranked:
            edge_key = tuple(sorted((edge.source_symbol, edge.target_symbol)))
            if reciprocal_limit > 0:
                other_symbol = edge.target_symbol if edge.source_symbol == symbol else edge.source_symbol
                if edge_key not in reciprocal_neighbors.get(other_symbol, set()):
                    continue
            selected_keys.add(edge_key)

    filtered_edges = [
        edge
        for edge in edges
        if tuple(sorted((edge.source_symbol, edge.target_symbol))) in selected_keys
    ]
    if degree_cap is None or degree_cap <= 0 or len(filtered_edges) <= 1:
        return filtered_edges

    degrees: dict[str, int] = defaultdict(int)
    kept: list[GraphEdge] = []
    for edge in sorted(filtered_edges, key=lambda item: item.weight, reverse=True):
        if degrees[edge.source_symbol] >= degree_cap or degrees[edge.target_symbol] >= degree_cap:
            continue
        kept.append(edge)
        degrees[edge.source_symbol] += 1
        degrees[edge.target_symbol] += 1
    return kept
