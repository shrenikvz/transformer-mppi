from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device: torch.device | None = None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        device: torch.device | None = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.src_fc = nn.Linear(input_size, hidden_size)
        self.tgt_fc = nn.Linear(output_size, hidden_size)

        self.positional_encoding = PositionalEncoding(hidden_size, dropout, device=device)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
        )
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        device = src.device

        src = self.src_fc(src) * math.sqrt(self.hidden_size)
        tgt = self.tgt_fc(tgt) * math.sqrt(self.hidden_size)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(device)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.output_fc(output)

    @torch.no_grad()
    def predict_autoregressive(self, src: torch.Tensor, horizon: int, start_token: torch.Tensor | None = None) -> torch.Tensor:
        """Autoregressive decode.

        Args:
            src: tensor of shape (src_seq_len, batch, input_size)
            horizon: target sequence length
            start_token: optional tensor of shape (1, batch, output_size)
        """
        batch_size = src.shape[1]
        if start_token is None:
            start_token = torch.zeros((1, batch_size, self.output_size), device=src.device, dtype=src.dtype)

        decoder_input = start_token
        outputs = []
        for _ in range(horizon):
            step_output = self(src, decoder_input)
            next_token = step_output[-1:, :, :]
            outputs.append(next_token)
            decoder_input = torch.cat([decoder_input, next_token], dim=0)

        return torch.cat(outputs, dim=0)


__all__ = ["TransformerModel", "PositionalEncoding"]
