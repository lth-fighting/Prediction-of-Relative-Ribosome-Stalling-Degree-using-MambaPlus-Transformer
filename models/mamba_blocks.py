"""
Mamba-based sequence modeling modules for protein expression prediction.

This module implements Mamba (Selective State Space Model) blocks and their
bidirectional variants for efficient long-range sequence modeling. Includes
the enhanced Mamba+ architecture with gating mechanisms.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaPlusBlock(nn.Module):
    """
    Enhanced Mamba block with gating mechanism and residual connections.
    
    Mamba+ integrates a convolutional projection with the Mamba SSM and adds
    a learnable gating branch that controls the contribution of the SSM output
    versus the convolutional features.
    
    Attributes:
        d_model (int): Model dimension.
        d_inner (int): Inner dimension after expansion.
        conv1d (nn.Conv1d): Depthwise convolutional layer.
        ssm (Mamba): Mamba state space model.
        gate_linear (nn.Linear): Gating linear layer.
        output_proj (nn.Linear): Output projection layer.
    """
    
    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4, 
                 expand: int = 2, conv_kernel_size: int = 3):
        """
        Initialize the MambaPlus block.
        
        Args:
            d_model: Model dimension.
            d_state: State dimension for the SSM.
            d_conv: Convolution dimension for the SSM.
            expand: Expansion factor for the inner dimension.
            conv_kernel_size: Kernel size for the depthwise convolution.
        """
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand

        # Branch 1: Convolution + SSM
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=self.d_inner,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=d_model
        )

        self.ssm = Mamba(
            d_model=self.d_inner,
            d_state=d_state,
            d_conv=d_conv,
            expand=1
        )

        # Branch 2: Gating branch
        self.gate_linear = nn.Linear(d_model, d_model)

        self.output_proj = nn.Linear(self.d_inner, d_model)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MambaPlus block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        residual = x

        # Convolution branch
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.silu(x_conv)
        x_conv = self.layer_norm2(x_conv)

        # SSM processing
        ssm_output = self.ssm(x_conv)

        # Gating mechanism
        gate = self.sigmoid(self.gate_linear(residual))
        forget_gate = 1 - gate

        # Expand gates to match inner dimension
        gate_expanded = gate.unsqueeze(-1).repeat(1, 1, 1, self.d_inner // self.d_model)
        gate_expanded = gate_expanded.reshape(gate.size(0), gate.size(1), self.d_inner)
        
        forget_gate_expanded = forget_gate.unsqueeze(-1).repeat(1, 1, 1, self.d_inner // self.d_model)
        forget_gate_expanded = forget_gate_expanded.reshape(forget_gate.size(0), forget_gate.size(1), self.d_inner)

        # Core Mamba+ operation: y ⊗ SiLU(z) + x′ ⊗ (1 - σ(z))
        combined = ssm_output * self.silu(gate_expanded) + x_conv * forget_gate_expanded

        # Output projection and residual connection
        output = self.output_proj(combined)
        output = self.layer_norm1(output + residual)

        return output


class BiMambaPlusEncoder(nn.Module):
    """
    Bidirectional Mamba+ encoder with forward and backward processing.
    
    This encoder processes sequences in both forward and reverse directions
    using separate Mamba+ blocks, then combines their outputs with a
    feed-forward network and residual connections.
    
    Attributes:
        d_model (int): Model dimension.
        num_layers (int): Number of encoder layers.
        forward_mamba_plus (MambaPlusBlock): Forward direction block.
        backward_mamba_plus (MambaPlusBlock): Backward direction block.
        ffn (nn.Sequential): Feed-forward network.
    """
    
    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4, 
                 expand: int = 2, ff_dim: int = None, dropout_rate: float = 0.1, 
                 num_layers: int = 1):
        """
        Initialize the bidirectional Mamba+ encoder.
        
        Args:
            d_model: Model dimension.
            d_state: State dimension for SSM.
            d_conv: Convolution dimension for SSM.
            expand: Expansion factor.
            ff_dim: Feed-forward network dimension.
            dropout_rate: Dropout probability.
            num_layers: Number of encoder layers (currently fixed to 1 per block).
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        ff_dim = ff_dim or d_model * 4

        # Forward and backward Mamba+ blocks
        self.forward_mamba_plus = MambaPlusBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.backward_mamba_plus = MambaPlusBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout_rate)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the bidirectional Mamba+ encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Forward processing
        forward_output = self.forward_mamba_plus(x)

        # Backward processing (reverse, process, reverse back)
        backward_input = torch.flip(x, dims=[1])
        backward_output = self.backward_mamba_plus(backward_input)
        backward_output = torch.flip(backward_output, dims=[1])

        # Combine both directions
        combined = forward_output + backward_output

        # Residual connection and layer normalization
        x_residual = self.norm1(combined + x)
        x_residual = self.dropout(x_residual)

        # Feed-forward network
        ffn_output = self.ffn(x_residual)

        # Final output
        output = self.norm2(ffn_output + x_residual)

        return output


class BidirectionalMamba(nn.Module):
    """
    Simplified bidirectional Mamba model without the Mamba+ enhancements.
    
    This module processes sequences in both directions using standard Mamba
    blocks and fuses their outputs through a linear projection.
    
    Attributes:
        forward_mamba (Mamba): Forward direction Mamba block.
        backward_mamba (Mamba): Backward direction Mamba block.
        fusion (nn.Linear): Fusion layer for combining directions.
    """
    
    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4, expand: int = 2):
        """
        Initialize the bidirectional Mamba model.
        
        Args:
            d_model: Model dimension.
            d_state: State dimension for SSM.
            d_conv: Convolution dimension for SSM.
            expand: Expansion factor.
        """
        super().__init__()

        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.forward_layer_norm = nn.LayerNorm(d_model)
        self.backward_layer_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bidirectional Mamba.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Fused output of shape (batch_size, seq_len, d_model).
        """
        forward_output = self.forward_mamba(x)
        forward_output = self.forward_layer_norm(forward_output)

        backward_output = self.backward_mamba(torch.flip(x, dims=[1]))
        backward_output = torch.flip(backward_output, dims=[1])
        backward_output = self.backward_layer_norm(backward_output)
        
        combined = torch.cat([forward_output, backward_output], dim=-1)
        fuse_output = self.fusion(combined)

        return fuse_output


class BiMambaPlus(nn.Module):
    """
    Complete Bi-Mamba+ architecture with patching mechanism and SRA decision.
    
    This is the full implementation of the Bi-Mamba+ model, which includes
    instance normalization (RevIN), sequence patching, bidirectional Mamba+
    encoding, and output projection for regression tasks.
    
    Attributes:
        d_model (int): Model dimension.
        patch_length (int): Length of each patch.
        stride (int): Stride between consecutive patches.
        instance_norm (nn.InstanceNorm1d): Instance normalization layer.
        patch_projection (nn.Linear): Projection layer for patches.
        encoder_layers (nn.ModuleList): List of BiMambaPlusEncoder layers.
        output_proj (nn.Sequential): Output projection layers.
    """
    
    def __init__(self, input_dim: int, d_model: int = 128, d_state: int = 16, 
                 d_conv: int = 4, expand: int = 2, num_layers: int = 3, 
                 patch_length: int = 24, stride: int = 12, dropout: float = 0.1,
                 tokenization_strategy: str = None):
        """
        Initialize the complete Bi-Mamba+ model.
        
        Args:
            input_dim: Input feature dimension.
            d_model: Model dimension.
            d_state: State dimension for SSM.
            d_conv: Convolution dimension for SSM.
            expand: Expansion factor.
            num_layers: Number of encoder layers.
            patch_length: Length of each patch for sequence patching.
            stride: Stride between patches.
            dropout: Dropout probability.
            tokenization_strategy: Tokenization strategy ('channel_mixing' or 'channel_independent').
        """
        super().__init__()
        
        self.d_model = d_model
        self.patch_length = patch_length
        self.stride = stride
        self.tokenization_strategy = tokenization_strategy
        
        # Instance normalization (RevIN)
        self.instance_norm = nn.InstanceNorm1d(input_dim, affine=True)
        
        # Patch projection layer
        self.patch_projection = nn.Linear(patch_length, d_model)
        
        # Stacked Bi-Mamba+ encoders
        self.encoder_layers = nn.ModuleList([
            BiMambaPlusEncoder(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                num_layers=1
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def patchify(self, x: torch.Tensor):
        """
        Divide the input sequence into overlapping patches.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            
        Returns:
            tuple: (patches, num_patches)
                - patches: Stacked patches tensor.
                - num_patches: Number of patches generated.
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Calculate number of patches
        num_patches = (seq_len - self.patch_length) // self.stride + 1
        
        patches = []
        for i in range(num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_length
            patch = x[:, start_idx:end_idx, :]  # (batch, patch_length, input_dim)
            
            if self.tokenization_strategy == 'channel_independent':
                # Channel independent: flatten across channels
                patch = patch.reshape(batch_size, self.patch_length * input_dim)
            else:
                # Channel mixing: default strategy
                patch = patch.reshape(batch_size, self.patch_length, input_dim)
            
            patches.append(patch)
        
        # Stack patches
        patched = torch.stack(patches, dim=1)  # (batch, num_patches, ...)
        return patched, num_patches
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Bi-Mamba+ model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            
        Returns:
            torch.Tensor: Predicted target values of shape (batch,).
        """
        # Instance normalization
        x_normalized = self.instance_norm(x.transpose(1, 2)).transpose(1, 2)
        
        # Patch creation
        patches, num_patches = self.patchify(x_normalized)
        
        # Project patches to d_model dimension
        if self.tokenization_strategy == 'channel_independent':
            batch_size = patches.shape[0]
            patches = patches.reshape(batch_size, num_patches, -1)
            x_projected = self.patch_projection(patches)
        else:
            batch_size, num_patches, patch_len, input_dim = patches.shape
            patches_flat = patches.reshape(batch_size * num_patches, patch_len, input_dim)
            patches_projected = self.patch_projection(patches_flat.transpose(1, 2))
            x_projected = patches_projected.reshape(batch_size, num_patches, self.d_model)
        
        # Apply Bi-Mamba+ encoders
        for encoder in self.encoder_layers:
            x_projected = encoder(x_projected)
        
        # Global average pooling
        x_pooled = x_projected.mean(dim=1)  # (batch, d_model)
        x_pooled = self.norm(x_pooled)
        
        # Output projection
        output = self.output_proj(x_pooled)
        
        return output.squeeze(-1)
