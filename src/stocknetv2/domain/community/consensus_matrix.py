from __future__ import annotations

from itertools import combinations

import pandas as pd


def build_consensus_matrix(
    layer_communities: dict[str, list[list[str]]],
    layer_weights: dict[str, float],
) -> pd.DataFrame:
    symbols = sorted(
        {
            symbol
            for communities in layer_communities.values()
            for community in communities
            for symbol in community
        }
    )
    matrix = pd.DataFrame(0.0, index=symbols, columns=symbols)

    for layer_name, communities in layer_communities.items():
        weight = layer_weights.get(layer_name, 0.0)
        for community in communities:
            for symbol in community:
                matrix.loc[symbol, symbol] = 1.0
            for left_symbol, right_symbol in combinations(sorted(community), 2):
                matrix.loc[left_symbol, right_symbol] += weight
                matrix.loc[right_symbol, left_symbol] += weight

    return matrix
