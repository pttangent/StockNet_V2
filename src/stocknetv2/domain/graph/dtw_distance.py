from __future__ import annotations


def dtw_distance_with_path_length(left: list[float], right: list[float]) -> tuple[float, int]:
    """Return DTW accumulated distance and the selected warping-path length."""

    if not left or not right:
        return float("inf"), 0

    rows = len(left)
    cols = len(right)
    costs = [[float("inf")] * (cols + 1) for _ in range(rows + 1)]
    lengths = [[0] * (cols + 1) for _ in range(rows + 1)]
    costs[0][0] = 0.0

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            predecessors = (
                (costs[row - 1][col], lengths[row - 1][col]),
                (costs[row][col - 1], lengths[row][col - 1]),
                (costs[row - 1][col - 1], lengths[row - 1][col - 1]),
            )
            previous_cost, previous_length = min(predecessors, key=lambda item: (item[0], item[1]))
            local_cost = abs(left[row - 1] - right[col - 1])
            costs[row][col] = local_cost + previous_cost
            lengths[row][col] = previous_length + 1

    return float(costs[rows][cols]), int(lengths[rows][cols])


def dtw_distance(left: list[float], right: list[float]) -> float:
    distance, _ = dtw_distance_with_path_length(left, right)
    return distance


def normalized_dtw_distance(left: list[float], right: list[float]) -> float:
    distance, path_length = dtw_distance_with_path_length(left, right)
    if distance == float("inf") or path_length <= 0:
        return float("inf")
    return distance / float(path_length)


def dtw_similarity(left: list[float], right: list[float]) -> float:
    """Convert path-length-normalized DTW distance into a bounded similarity."""

    distance = normalized_dtw_distance(left, right)
    if distance == float("inf"):
        return 0.0
    return 1.0 / (1.0 + distance)
