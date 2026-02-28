"""
model.py — ColorPredictor CNN for the Synesthesia CIFAR-10 project.

Architecture summary (default configuration)
---------------------------------------------
Layer block 1  : Conv2d(2,  64, 3, padding='same') → BN → LeakyReLU → Dropout(0.2)
Layer block 2  : Conv2d(64, 128, 3, padding='same') → BN → LeakyReLU → Dropout(0.2)
Layer block 3  : Conv2d(128, 64, 3, padding='same') → BN → LeakyReLU → Dropout(0.2)
Output layer   : Conv2d(64,  1,  1, padding='same') → Sigmoid

Both BatchNorm and Dropout can be toggled independently via constructor flags
so that ablation studies are easy to run.

- Input shape  : (B, 2, 32, 32)
- Output shape : (B, 1, 32, 32)
- All spatial dimensions preserved via padding='same'.
- Sigmoid output maps predictions to [0, 1] matching normalised targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A single convolution block with **configurable** regularisation layers.

    Conv2d → [BatchNorm2d] → LeakyReLU → [Dropout2d]

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    dropout_rate : float
        Dropout probability (ignored when use_dropout=False).
    use_batchnorm : bool
        Include BatchNorm2d after Conv2d.
    use_dropout : bool
        Include Dropout2d after activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        use_batchnorm: bool = True,
        use_dropout: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        
        # Convolution layer with 'same' padding to preserve spatial dimensions
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,   # Disable bias if BN is used (BN has its own bias)
            )
        )
        
        # BatchNorm2d: normalizes activations, helps with training stability
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # LeakyReLU: allows small negative gradients (slope=0.1) to avoid dying ReLU
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        
        # Dropout2d: randomly zeros entire feature maps for regularization
        if use_dropout and dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ColorPredictor(nn.Module):
    """
    Convolutional model that predicts one missing RGB channel from the other two.

    Parameters
    ----------
    in_channels : int
        Number of input channels (2 for two-channel input).
    out_channels : int
        Number of output channels (1 for single channel prediction).
    dropout_rate : float
        Dropout probability applied after each conv block (0.0 disables).
    use_batchnorm : bool
        Whether to include BatchNorm2d layers (True by default).
    use_dropout : bool
        Whether to include Dropout2d layers (True by default).

    Input
    -----
    x : torch.Tensor of shape (B, in_channels, 32, 32)

    Output
    ------
    torch.Tensor of shape (B, out_channels, 32, 32) with values in [0, 1].
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        dropout_rate: float = 0.2,
        use_batchnorm: bool = True,
        use_dropout: bool = True,
    ) -> None:
        super().__init__()
        # Store config for inspection (used in notebook demonstrations)
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        # Shared kwargs for all ConvBlocks to ensure consistent regularization
        blk_kwargs = dict(
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            use_dropout=use_dropout,
        )

        # ── Three-layer feature extraction pipeline ────────────────────────
        # Architecture: 2 → 64 → 128 → 64 channels
        # Each block applies: Conv3x3 → [BN] → LeakyReLU → [Dropout]
        self.block1 = ConvBlock(in_channels, 64,  kernel_size=3, **blk_kwargs)  # Extract low-level features
        self.block2 = ConvBlock(64,          128, kernel_size=3, **blk_kwargs)  # Expand representational capacity
        self.block3 = ConvBlock(128,         64,  kernel_size=3, **blk_kwargs)  # Compress back down

        # ── Output projection: 64 channels → 1 channel ──────────────────────
        # 1×1 convolution acts as a learned linear combination of features
        # Sigmoid maps outputs to [0, 1] to match normalized pixel values
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1, padding="same"),
            nn.Sigmoid(),  # Ensures output is in valid pixel range
        )

        # Initialize weights using Kaiming (He) initialization for LeakyReLU
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize network weights using appropriate schemes for each layer type.
        
        - Conv2d: Kaiming normal initialization scaled for LeakyReLU (prevents
          vanishing/exploding gradients by accounting for negative slope)
        - BatchNorm2d: Standard initialization (weight=1, bias=0)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming init: variance scaled by fan_out and nonlinearity
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # Standard BN init: scale=1, shift=0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor, shape (B, 2, 32, 32)
            Input with 2 RGB channels (e.g., Red and Green).
            
        Returns
        -------
        torch.Tensor, shape (B, 1, 32, 32)
            Predicted target channel (e.g., Blue) with values in [0, 1].
        """
        # Sequential feature extraction (spatial dimensions preserved throughout)
        x = self.block1(x)       # (B, 2, 32, 32) → (B, 64, 32, 32)
        x = self.block2(x)       # (B, 64, 32, 32) → (B, 128, 32, 32)
        x = self.block3(x)       # (B, 128, 32, 32) → (B, 64, 32, 32)
        x = self.output_conv(x)  # (B, 64, 32, 32) → (B, 1, 32, 32)
        return x

    # ── Convenience helpers ──────────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_summary(self) -> str:
        """Human-readable one-liner of this model's configuration."""
        return (
            f"ColorPredictor(BN={self.use_batchnorm}, "
            f"Drop={self.use_dropout}, "
            f"params={self.count_parameters():,})"
        )


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import src.config as cfg

    model = ColorPredictor().to(cfg.DEVICE)
    print(model)
    print(f"\nTrainable parameters: {model.count_parameters():,}")

    dummy = torch.randn(4, 2, 32, 32, device=cfg.DEVICE)
    out = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")    # Expected: (4, 1, 32, 32)
    assert out.shape == (4, 1, 32, 32), "Unexpected output shape!"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output outside [0,1]!"
    print("model.py smoke test passed ✓")
