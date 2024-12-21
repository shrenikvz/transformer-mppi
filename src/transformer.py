'''
Class for the transformer model

Author: Shrenik Zinage

'''


import torch
import torch.nn as nn
import math 

__all__ = ['TransformerModel', 'PositionalEncoding']


class TransformerModel(nn.Module):
    """
    A transformer model for sequence-to-sequence tasks.
    
    Args:
        input_size (int): Dimension of input features
        output_size (int): Dimension of output features 
        hidden_size (int): Dimension of transformer hidden layers (default: 512)
        num_layers (int): Number of transformer encoder/decoder layers (default: 6)
        nhead (int): Number of attention heads (default: 8)
        dropout (float): Dropout probability (default: 0.1)
    """
    def __init__(self, input_size, output_size, hidden_size=512, num_layers=6, nhead=8, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Linear layers to project input/output dimensions to hidden size
        self.src_fc = nn.Linear(input_size, hidden_size)
        self.tgt_fc = nn.Linear(output_size, hidden_size)

        # Add positional encoding to incorporate sequence order information
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)

        # Core transformer architecture
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 2,
            dropout=dropout
        )

        # Project back to output dimension
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt):
        """
        Forward pass of the transformer model.

        Args:
            src (torch.Tensor): Source sequence of shape (seq_len_src, batch_size, input_size)
            tgt (torch.Tensor): Target sequence of shape (seq_len_tgt, batch_size, output_size)

        Returns:
            torch.Tensor: Output sequence of shape (seq_len_tgt, batch_size, output_size)
        """
        # Project and scale input/output sequences
        src = self.src_fc(src) * math.sqrt(self.hidden_size)  # Scale to stabilize training
        tgt = self.tgt_fc(tgt) * math.sqrt(self.hidden_size)

        # Add positional encodings
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Create causal mask to prevent attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)

        # Pass through transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # Project to output dimension
        output = self.output_fc(output)

        return output


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings to provide position information.
    
    Args:
        d_model (int): Dimension of the model
        dropout (float): Dropout probability (default: 0.1)
        max_len (int): Maximum sequence length (default: 5000)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Add batch dimension
        
        # Register as buffer (won't be updated during backprop)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embedding_dim)
            
        Returns:
            torch.Tensor: Input with positional encoding added and dropout applied
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)