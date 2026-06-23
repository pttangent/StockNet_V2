from __future__ import annotations

from stocknetv2.application.services.layer_profile_service import resolve_layer_profile
from stocknetv2.domain.graph.layer_config import build_theme_discovery_settings


def _settings(*, dtw_backend: str, dtw_torch_device: str):
    return build_theme_discovery_settings(
        graph_backend="cpu_numpy",
        graph_torch_device="cpu",
        dtw_backend=dtw_backend,
        dtw_torch_device=dtw_torch_device,
        dtw_torch_batch_pair_threshold=1024,
    )


def test_resolve_layer_profile_supports_hybird_full():
    profile = resolve_layer_profile("hybird_full", _settings(dtw_backend="torch_cuda", dtw_torch_device="cuda"))

    assert profile.name == "hybird_full"
    assert profile.dtw_backend == "torch_cuda"
    assert "dtw_return_similarity_graph" in profile.selected_layers
    assert "return_corr_graph" in profile.selected_layers


def test_resolve_layer_profile_supports_gpu_only_dtw_and_cpu_only_dtw():
    gpu_profile = resolve_layer_profile("gpu_only_dtw", _settings(dtw_backend="torch_cuda", dtw_torch_device="cuda"))
    cpu_profile = resolve_layer_profile("cpu_only_dtw", _settings(dtw_backend="cpu_python", dtw_torch_device="cpu"))

    assert gpu_profile.name == "gpu_only_dtw"
    assert gpu_profile.selected_layers == ("dtw_return_similarity_graph", "dtw_trade_flow_similarity_graph")
    assert gpu_profile.dtw_backend == "torch_cuda"
    assert cpu_profile.name == "cpu_only_dtw"
    assert cpu_profile.selected_layers == ("dtw_return_similarity_graph", "dtw_trade_flow_similarity_graph")
    assert cpu_profile.dtw_backend == "cpu_python"
