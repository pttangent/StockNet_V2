from __future__ import annotations

from stocknetv2.domain.graph.layer_registry import GRAPH_LAYER_NAMES, list_graph_layers


def test_layer_registry_contains_all_six_t1_layers():
    expected = [
        "return_corr_graph",
        "dtw_return_similarity_graph",
        "flow_alignment_graph",
        "dtw_trade_flow_similarity_graph",
        "volume_expansion_graph",
        "large_trade_alignment_graph",
    ]

    assert list_graph_layers() == expected
    assert list(GRAPH_LAYER_NAMES) == expected
