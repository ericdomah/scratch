"""
tcn.py
------
Temporal Convolutional Network (TCN) binary classifier for theft detection.

Architecture
------------
  Input  : (B, seq_len, input_dim)
  Reshape: (B, input_dim, seq_len)   [channels-first for Conv1d]

  Stack of TCNBlock modules:
      Each block uses *dilated causal* Conv1d with:
          - Weight Norm (nn.utils.weight_norm)
          - ReLU activation
          - Dropout
          - Residual (1×1 downsample conv when channel dims differ)
      Dilation doubles at every block: 1, 2, 4, 8, …

  Global Average Pooling  → (B, num_channels[-1])
  FC head:
      Linear(num_channels[-1] → 64) → ReLU → Dropout
      → Linear(64 → 1) → Sigmoid
  Output : (B, 1)

References
----------
Bai, S., Kolter, J. Z., & Koltun, V. (2018).
"An empirical evaluation of generic convolutional and recurrent networks
for sequence modeling."  arXiv:1803.01271.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TCNBlock
# ---------------------------------------------------------------------------

class TCNBlock(nn.Module):
    """Single Temporal Convolutional Network block.

    Applies two dilated causal convolutions with weight normalisation,
    ReLU activation, and dropout.  A residual skip connection (with
    optional 1×1 downsampling) is added to the block output.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel width. Default ``3``.
    dilation : int
        Dilation factor for causal convolution. Default ``1``.
    dropout : float
        Dropout probability. Default ``0.2``.

    Notes
    -----
    Causal padding = ``(kernel_size - 1) * dilation`` is applied to the left
    only, then truncated after the convolution to preserve sequence length.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.causal_padding = (kernel_size - 1) * dilation

        # First dilated causal conv
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=0,   # we handle padding manually (causal)
            )
        )

        # Second dilated causal conv
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=0,
            )
        )

        self.dropout = nn.Dropout(p=dropout)

        # 1×1 projection for residual when channel dims differ
        self.downsample: Optional[nn.Conv1d] = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming uniform for the weight-normed convolutions."""
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        if self.downsample is not None:
            nn.init.kaiming_uniform_(
                self.downsample.weight, nonlinearity="relu"
            )
            nn.init.zeros_(self.downsample.bias)

    # ------------------------------------------------------------------
    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """Apply left-only (causal) padding then convolution.

        Parameters
        ----------
        conv : nn.Conv1d
            Convolution module to apply.
        x : torch.Tensor
            Input ``(B, C, T)``.

        Returns
        -------
        torch.Tensor
            Output ``(B, C_out, T)`` — same length as input.
        """
        # Pad left only
        x = F.pad(x, (self.causal_padding, 0))
        return conv(x)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input ``(B, in_channels, T)``.

        Returns
        -------
        torch.Tensor
            Output ``(B, out_channels, T)``.
        """
        residual = x

        out = self._causal_conv(self.conv1, x)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        out = self._causal_conv(self.conv2, out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        return F.relu(out + residual, inplace=True)


# ---------------------------------------------------------------------------
# TCN model
# ---------------------------------------------------------------------------

class TCNClassifier(nn.Module):
    """Temporal Convolutional Network binary classifier.

    Stacks multiple ``TCNBlock`` modules with exponentially increasing
    dilation factors, followed by global average pooling and a small FC head.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep.
    seq_len : int
        Input sequence length (informational; TCN handles variable T).
    num_channels : List[int]
        Output channels for each TCN block. The number of blocks is
        ``len(num_channels)``. Default ``[64, 64, 128, 128]``.
    kernel_size : int
        Convolution kernel width used in every TCN block. Default ``3``.
    dropout : float
        Dropout probability used in every TCN block and FC head. Default ``0.2``.
    fc_hidden : int
        Hidden size of the FC head. Default ``64``.

    Examples
    --------
    >>> model = TCNClassifier(input_dim=32, seq_len=50)
    >>> x = torch.randn(16, 50, 32)
    >>> out = model(x)
    >>> out.shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        fc_hidden: int = 64,
    ) -> None:
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 128, 128]
        if len(num_channels) == 0:
            raise ValueError("num_channels must contain at least one element.")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_channels = num_channels

        # Build TCN blocks with exponentially increasing dilation
        layers: List[nn.Module] = []
        in_ch = input_dim
        for block_idx, out_ch in enumerate(num_channels):
            dilation = 2 ** block_idx  # 1, 2, 4, 8, …
            layers.append(
                TCNBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)

        # Global average pool collapses temporal dim → (B, C)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # FC classification head
        self.fc_head = nn.Sequential(
            nn.Linear(num_channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )

        # Final FC Linear weight init
        for m in self.fc_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        logger.info(
            "TCNClassifier initialised | input_dim=%d | seq_len=%d "
            "| num_channels=%s | kernel_size=%d | params=%d",
            input_dim,
            seq_len,
            num_channels,
            kernel_size,
            self.count_parameters(),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(B, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            Sigmoid probability ``(B, 1)``.
        """
        # (B, seq_len, input_dim) → (B, input_dim, seq_len)
        x = x.permute(0, 2, 1)

        # Pass through TCN blocks
        x = self.network(x)  # (B, num_channels[-1], T)

        # Global average pooling → (B, num_channels[-1])
        x = self.global_avg_pool(x).squeeze(-1)

        # FC head → (B, 1)
        out = self.fc_head(x)
        return out

    # ------------------------------------------------------------------
    def get_receptive_field(self) -> int:
        """Compute the theoretical receptive field of the TCN.

        Returns
        -------
        int
            Number of past timesteps visible from any output position.
        """
        kernel_size = 3  # default; better to store & retrieve from blocks
        total = 1
        for block_idx in range(len(self.num_channels)):
            dilation = 2 ** block_idx
            total += 2 * (kernel_size - 1) * dilation  # 2 convs per block
        return total

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns
        -------
        int
            Number of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
