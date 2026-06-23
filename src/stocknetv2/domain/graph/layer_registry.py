from __future__ import annotations


GRAPH_LAYER_NAMES: tuple[str, ...] = (
    "return_corr_graph",
    "dtw_return_similarity_graph",
    "flow_alignment_graph",
    "dtw_trade_flow_similarity_graph",
    "volume_expansion_graph",
    "large_trade_alignment_graph",
)


def list_graph_layers() -> list[str]:
    return list(GRAPH_LAYER_NAMES)
