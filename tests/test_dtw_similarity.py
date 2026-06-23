from __future__ import annotations

from stocknetv2.domain.graph.dtw_distance import (
    dtw_distance,
    dtw_distance_with_path_length,
    dtw_similarity,
    normalized_dtw_distance,
)


def test_dtw_distance_is_zero_for_identical_series():
    assert dtw_distance([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == 0.0
    assert normalized_dtw_distance([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == 0.0


def test_dtw_distance_reports_warping_path_length():
    distance, path_length = dtw_distance_with_path_length([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

    assert distance == 0.0
    assert path_length == 3


def test_dtw_similarity_decreases_for_more_distant_series():
    near = dtw_similarity([0.1, 0.2, 0.3], [0.1, 0.2, 0.31])
    far = dtw_similarity([0.1, 0.2, 0.3], [0.8, 0.9, 1.0])

    assert near > far
    assert 0.0 < far < 1.0


def test_normalized_dtw_distance_does_not_scale_linearly_with_repeated_path_length():
    short = normalized_dtw_distance([0.0, 1.0], [0.0, 1.2])
    repeated = normalized_dtw_distance([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.2, 1.2])

    assert abs(short - repeated) < 1e-12
