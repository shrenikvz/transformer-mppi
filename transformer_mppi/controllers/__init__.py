from transformer_mppi.controllers.mppi import MPPI
from transformer_mppi.controllers.transformer import PositionalEncoding, TransformerModel

__all__ = ["MPPI", "PositionalEncoding", "TransformerModel", "TransformerMPPIController"]


def __getattr__(name: str):
    if name == "TransformerMPPIController":
        from transformer_mppi.controllers.transformer_mppi import TransformerMPPIController

        return TransformerMPPIController
    raise AttributeError(f"module 'transformer_mppi.controllers' has no attribute {name!r}")
