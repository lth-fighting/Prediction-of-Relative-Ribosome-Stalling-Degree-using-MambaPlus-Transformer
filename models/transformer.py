"""
Transformer-based sequence modeling modules.

This module implements positional encoding schemes and transformer blocks
for sequence modeling in the protein expression prediction task.
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as introduced in the original Transformer paper.
    
    This encoding uses sine and cosine functions of different frequencies to
    provide position information to the model without learnable parameters.
    
    Attributes:
        pe (torch.Tensor): Precomputed positional encoding buffer.
    """
    
    def __init__(self, d_model: int, max_len: int = 94):
        """
        Initialize sinusoidal positional encoding.
        
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Input with added positional encoding.
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding where position embeddings are trainable parameters.
    
    This approach allows the model to learn optimal position representations
    from the training data.
    
    Attributes:
        d_model (int): Model dimension.
        max_len (int): Maximum sequence length.
        position_embedding (nn.Parameter): Learnable position embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 94):
        """
        Initialize learnable positional encoding.
        
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable position embeddings
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # Xavier initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize position embeddings using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.position_embedding)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Input with added positional encoding.
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
        
        pos_encoding = self.position_embedding[:, :seq_len, :]
        return x + pos_encoding

    def get_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Retrieve positional encoding for a given sequence length.
        
        Args:
            seq_len: Desired sequence length.
            
        Returns:
            torch.Tensor: Positional encoding tensor.
        """
        if seq_len > self.max_len:
            raise ValueError(f"Requested sequence length {seq_len} exceeds maximum length {self.max_len}")
        return self.position_embedding[:, :seq_len, :].detach()


class SequenceTransformerBlock(nn.Module):
    """
    Single transformer block with multi-head self-attention and feed-forward network.
    
    This block follows the standard transformer architecture with pre-layer
    normalization and residual connections.
    
    Attributes:
        attention (nn.MultiheadAttention): Multi-head self-attention layer.
        linear1 (nn.Linear): First feed-forward layer.
        linear2 (nn.Linear): Second feed-forward layer.
        norm1 (nn.LayerNorm): Pre-attention layer normalization.
        norm2 (nn.LayerNorm): Pre-FFN layer normalization.
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout_rate: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension.
            nhead: Number of attention heads.
            dim_feedforward: Dimension of feed-forward network.
            dropout_rate: Dropout probability.
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout_rate,
            batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            src: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        # Self-attention with residual connection
        src2 = self.attention(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward network with residual connection
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class SequenceTransformer(nn.Module):
    """
    Stack of transformer blocks for sequence processing.
    
    This module combines positional encoding with multiple transformer blocks
    to process sequential data.
    
    Attributes:
        d_model (int): Model dimension.
        pos_encoder (LearnablePositionalEncoding): Positional encoding module.
        transformer_blocks (nn.ModuleList): List of transformer blocks.
        dropout (nn.Dropout): Dropout layer.
    """
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, 
                 dim_feedforward: int = 2048, dropout_rate: float = 0.1, 
                 max_len: int = 94):
        """
        Initialize sequence transformer.
        
        Args:
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer blocks.
            dim_feedforward: Feed-forward network dimension.
            dropout_rate: Dropout probability.
            max_len: Maximum sequence length.
        """
        super().__init__()

        self.d_model = d_model
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len)

        self.transformer_blocks = nn.ModuleList([
            SequenceTransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer stack.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        return x
