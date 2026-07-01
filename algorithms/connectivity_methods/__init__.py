"""Controlled connectivity-repair strategies for ablation experiments."""

from .strategy import (
    SUPPORTED_CONNECTIVITY_STRATEGIES,
    ConnectivityResult,
    apply_connectivity_strategy,
    connected_components,
    rectangle_set_sha256,
)

__all__ = [
    "SUPPORTED_CONNECTIVITY_STRATEGIES",
    "ConnectivityResult",
    "apply_connectivity_strategy",
    "connected_components",
    "rectangle_set_sha256",
]
