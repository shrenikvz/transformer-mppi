"""Transformer-MPPI package public API.

Imports are resolved lazily to avoid eager torch initialization when only CLI/help is used.
"""

__all__ = ["MPPI", "TransformerModel", "TransformerMPPIController", "TransformerArtifacts"]


def __getattr__(name: str):
    if name == "MPPI":
        from transformer_mppi.controllers import MPPI

        return MPPI
    if name == "TransformerModel":
        from transformer_mppi.controllers import TransformerModel

        return TransformerModel
    if name == "TransformerMPPIController":
        from transformer_mppi.controllers import TransformerMPPIController

        return TransformerMPPIController
    if name == "TransformerArtifacts":
        from transformer_mppi.training import TransformerArtifacts

        return TransformerArtifacts
    raise AttributeError(f"module 'transformer_mppi' has no attribute {name!r}")
