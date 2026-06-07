"""
cnn.py
------
1-D Convolutional Neural Network for theft-detection binary classification.

Architecture
------------
  Input  : (B, seq_len, input_dim)   [batch-first]
  Reshape: (B, input_dim, seq_len)   [channels-first for Conv1d]
  3 × Conv-Block:
      Conv1d → BatchNorm1d → ReLU → MaxPool1d → Dropout
  Global Average Pooling  → (B, out_channels[-1])
  FC head:
      Linear → ReLU → Dropout → Linear(1) → Sigmoid
  Output : (B, 1)
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Single convolutional block: Conv1d → BN → ReLU → MaxPool → Dropout.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output (filter) channels.
    kernel_size : int
        Convolution kernel size. Default ``3``.
    pool_size : int
        Max-pooling window size. Default ``2``.
    dropout : float
        Dropout probability. Default ``0.25``.
    use_batch_norm : bool
        Whether to apply BatchNorm1d. Default ``True``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.25,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # same-padding heuristic
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=not use_batch_norm,
        )
        self.bn: Optional[nn.BatchNorm1d] = (
            nn.BatchNorm1d(out_channels) if use_batch_norm else None
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CNN1D(nn.Module):
    """Three-block 1-D CNN binary classifier.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep.
    seq_len : int
        Number of timesteps in the input sequence.
    channels : List[int]
        Output channels for each of the 3 conv blocks.
        Defaults to ``[64, 128, 256]``.
    kernel_size : int
        Convolution kernel width. Default ``3``.
    pool_size : int
        MaxPool window / stride. Default ``2``.
    dropout : float
        Dropout probability used in conv blocks and FC head. Default ``0.3``.
    fc_hidden : int
        Hidden size of the FC classification head. Default ``128``.
    use_batch_norm : bool
        Toggle BatchNorm in every conv block. Default ``True``.

    Examples
    --------
    >>> model = CNN1D(input_dim=32, seq_len=50)
    >>> x = torch.randn(16, 50, 32)   # (B, seq_len, input_dim)
    >>> out = model(x)
    >>> out.shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.3,
        fc_hidden: int = 128,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]
        if len(channels) != 3:
            raise ValueError(
                f"Expected exactly 3 channel sizes, got {len(channels)}."
            )

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.channels = channels

        # Build 3 convolutional blocks
        in_ch = input_dim
        conv_blocks: List[nn.Module] = []
        for out_ch in channels:
            conv_blocks.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                )
            )
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Global average pooling collapses temporal dimension → (B, C)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # FC classification head
        self.fc_head = nn.Sequential(
            nn.Linear(channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )

        self._init_weights()
        logger.info(
            "CNN1D initialised | input_dim=%d | seq_len=%d | channels=%s "
            "| params=%d",
            input_dim,
            seq_len,
            channels,
            self.count_parameters(),
        )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming uniform for Conv layers; Xavier for Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            Sigmoid probability, shape ``(B, 1)``.
        """
        # (B, seq_len, input_dim) → (B, input_dim, seq_len)
        x = x.permute(0, 2, 1)

        # 3 Conv blocks
        x = self.conv_blocks(x)  # (B, channels[-1], T')

        # Global average pooling → (B, channels[-1], 1) → (B, channels[-1])
        x = self.global_avg_pool(x).squeeze(-1)

        # FC head → (B, 1)
        x = self.fc_head(x)
        return x

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns
        -------
        int
            Count of parameters requiring gradients.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_feature_maps(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return intermediate feature maps and final logits (pre-sigmoid).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(B, seq_len, input_dim)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(feature_maps, logits)`` where feature_maps is the output of the
            last conv block ``(B, channels[-1], T')``, and logits is the
            linear output before sigmoid ``(B, 1)``.
        """
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)
        feature_maps = x
        x = self.global_avg_pool(x).squeeze(-1)
        # Run FC head but intercept before Sigmoid
        for layer in list(self.fc_head.children())[:-1]:
            x = layer(x)
        logits = x
        return feature_maps, logits
