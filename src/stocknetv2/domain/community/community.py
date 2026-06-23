from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Community:
    members: list[str]
    method: str = "connected_components"
    resolution: float | None = None
    universe_ratio: float | None = None
    is_market_mode: bool = False
