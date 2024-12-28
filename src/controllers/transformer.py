import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""

    def __init__(self, d_model, dropout=0.1, max_len=5000, device=None):
        super(PositionalEncoding, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(p=dropout)
        # Create positional encoding matrix using sine and cosine functions
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor x.

        Args:
            x (Tensor): Input tensor of shape [seq_len, batch_size, d_model]

        Returns:
            Tensor: Tensor with positional encoding added, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Transformer model for sequence-to-sequence tasks."""

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=128,
        num_layers=3,
        nhead=8,
        dropout=0.1,
        device=None
    ):
        super(TransformerModel, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_size = hidden_size
        self.output_size = output_size

        # Embedding layers for source and target
        self.src_fc = nn.Linear(input_size, hidden_size)
        self.tgt_fc = nn.Linear(output_size, hidden_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size, dropout, device=device)

        # Transformer backbone
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 2,
            dropout=dropout
        )

        # Output projection layer
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt):
        """
        Forward pass through the Transformer model.

        Args:
            src (Tensor): Source sequence tensor of shape [seq_len_src, batch_size, input_size]
            tgt (Tensor): Target sequence tensor of shape [seq_len_tgt, batch_size, output_size]

        Returns:
            Tensor: Output tensor of shape [seq_len_tgt, batch_size, output_size]
        """
        device = src.device
        # Apply input embeddings and scale
        src = self.src_fc(src) * math.sqrt(self.hidden_size)
        tgt = self.tgt_fc(tgt) * math.sqrt(self.hidden_size)

        # Add positional encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Generate target mask to prevent attention to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(device)

        # Pass through Transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # Project to output size
        output = self.output_fc(output)
        return output
