from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class LayerFilterConfig:
    candidate_top_k: int = 8
    reciprocal_top_k: int = 3
    degree_cap: int = 6


@dataclass(frozen=True)
class ReturnCorrelationConfig:
    min_correlation: float = 0.70
    min_overlap_points: int = 8
    lookback_bars: int = 12
    backend: str = "cpu_numpy"
    torch_device: str = "auto"
    filter: LayerFilterConfig = field(default_factory=LayerFilterConfig)


@dataclass(frozen=True)
class FlowAlignmentConfig:
    lookback_minutes: int = 60
    min_score: float = 0.7
    min_joint_active_points: int = 8
    activity_epsilon: float = 0.05
    min_variance: float = 1e-8
    backend: str = "cpu_numpy"
    torch_device: str = "auto"
    filter: LayerFilterConfig = field(default_factory=LayerFilterConfig)


@dataclass(frozen=True)
class DTWLayerConfig:
    min_similarity: float = 0.9
    min_overlap_points: int = 8
    min_overlap_floor_points: int = 5
    min_variance: float = 1e-8
    warmup_min_minutes: int = 5
    max_lookback_minutes: int = 30
    backend: str = "cpu_python"
    torch_device: str = "auto"
    torch_batch_pair_threshold: int = 1024  # Deprecated: use torch_activation_pair_threshold instead
    torch_activation_pair_threshold: int = 1024
    torch_gpu_chunk_size: int = 8192
    filter: LayerFilterConfig = field(default_factory=LayerFilterConfig)


@dataclass(frozen=True)
class ActivityLayerConfig:
    min_score: float = 0.8
    threshold: float = 1.5
    backend: str = "cpu_numpy"
    torch_device: str = "auto"
    filter: LayerFilterConfig = field(default_factory=LayerFilterConfig)


@dataclass(frozen=True)
class CommunityDetectionConfig:
    algorithm: str = "weighted_leiden"
    resolution: float = 0.9
    min_members: int = 2
    market_mode_max_member_ratio: float = 0.15
    fallback_algorithm: str = "error"


@dataclass(frozen=True)
class ConsensusConfig:
    min_consensus_score: float = 0.35
    min_members: int = 3
    min_distinct_families: int = 2
    layer_weights: dict[str, float] = field(
        default_factory=lambda: {
            "return_corr_graph": 0.25,
            "dtw_return_similarity_graph": 0.20,
            "flow_alignment_graph": 0.20,
            "dtw_trade_flow_similarity_graph": 0.20,
            "volume_expansion_graph": 0.075,
            "large_trade_alignment_graph": 0.075,
        }
    )
    family_map: dict[str, str] = field(
        default_factory=lambda: {
            "return_corr_graph": "price",
            "dtw_return_similarity_graph": "price",
            "flow_alignment_graph": "flow",
            "dtw_trade_flow_similarity_graph": "flow",
            "volume_expansion_graph": "activity",
            "large_trade_alignment_graph": "activity",
        }
    )
    community_detection: CommunityDetectionConfig = field(
        default_factory=lambda: CommunityDetectionConfig(min_members=3)
    )


@dataclass(frozen=True)
class ThemeDiscoverySettings:
    return_corr: ReturnCorrelationConfig = field(default_factory=ReturnCorrelationConfig)
    dtw_return: DTWLayerConfig = field(default_factory=DTWLayerConfig)
    flow_alignment: FlowAlignmentConfig = field(default_factory=FlowAlignmentConfig)
    dtw_trade_flow: DTWLayerConfig = field(default_factory=DTWLayerConfig)
    volume_expansion: ActivityLayerConfig = field(default_factory=ActivityLayerConfig)
    large_trade_alignment: ActivityLayerConfig = field(
        default_factory=lambda: ActivityLayerConfig(threshold=2.0)
    )
    layer_community_detection: CommunityDetectionConfig = field(default_factory=CommunityDetectionConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_theme_discovery_settings(
    *,
    graph_backend: str,
    graph_torch_device: str,
    dtw_backend: str,
    dtw_torch_device: str,
    dtw_torch_batch_pair_threshold: int = 1024,
    torch_activation_pair_threshold: int | None = None,
    torch_gpu_chunk_size: int = 8192,
) -> ThemeDiscoverySettings:
    if torch_activation_pair_threshold is None:
        torch_activation_pair_threshold = dtw_torch_batch_pair_threshold

    base_settings = ThemeDiscoverySettings()
    return ThemeDiscoverySettings(
        return_corr=replace(
            base_settings.return_corr,
            backend=graph_backend,
            torch_device=graph_torch_device,
        ),
        flow_alignment=replace(
            base_settings.flow_alignment,
            backend=graph_backend,
            torch_device=graph_torch_device,
        ),
        volume_expansion=replace(
            base_settings.volume_expansion,
            backend=graph_backend,
            torch_device=graph_torch_device,
        ),
        large_trade_alignment=replace(
            base_settings.large_trade_alignment,
            backend=graph_backend,
            torch_device=graph_torch_device,
        ),
        dtw_return=replace(
            base_settings.dtw_return,
            backend=dtw_backend,
            torch_device=dtw_torch_device,
            torch_batch_pair_threshold=max(1, torch_activation_pair_threshold),
            torch_activation_pair_threshold=max(1, torch_activation_pair_threshold),
            torch_gpu_chunk_size=max(1, torch_gpu_chunk_size),
        ),
        dtw_trade_flow=replace(
            base_settings.dtw_trade_flow,
            backend=dtw_backend,
            torch_device=dtw_torch_device,
            torch_batch_pair_threshold=max(1, torch_activation_pair_threshold),
            torch_activation_pair_threshold=max(1, torch_activation_pair_threshold),
            torch_gpu_chunk_size=max(1, torch_gpu_chunk_size),
        ),
    )
