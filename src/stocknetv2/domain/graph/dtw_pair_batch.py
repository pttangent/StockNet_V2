from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PairComponentRecord:
    pair_key: tuple[str, str]
    component_weight: float
    left: Sequence[float]
    right: Sequence[float]
    support_points: int
