"""
cnn_lstm.py
-----------
CNN-LSTM hybrid binary classifier for theft detection.

Architecture
------------
  Input   : (B, seq_len, input_dim)
  Reshape : (B, input_dim, seq_len)   [channels-first for Conv1d]

  CNN feature extractor (2 blocks):
      Block 1: Conv1d(input_dim → cnn_channels[0]) + BN + ReLU + MaxPool + Dropout
      Block 2: Conv1d(cnn_channels[0] → cnn_channels[1]) + BN + ReLU + MaxPool + Dropout
  Output: (B, cnn_channels[1], T') — permuted to (B, T', cnn_channels[1])

  LSTM temporal encoder:
      nn.LSTM(cnn_channels[1] → lstm_hidden, num_layers, dropout)
      Last hidden state → (B, lstm_hidden)

  Optional BatchNorm1d on hidden state

  FC head:
      Linear(lstm_hidden → 64) → ReLU → Dropout → Linear(64 → 1) → Sigmoid
  Output : (B, 1)
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _CNNBlock(nn.Module):
    """Conv1d → BN → ReLU → MaxPool → Dropout."""

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
        padding = kernel_size // 2
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
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = self.drop(x)
        return x


class CNNLSTMClassifier(nn.Module):
    """CNN-LSTM hybrid binary classifier.

    A two-block CNN extracts local temporal features, then an LSTM models
    longer-range dependencies across the CNN output frames.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep.
    seq_len : int
        Input sequence length (informational; architecture handles variable T).
    cnn_channels : List[int]
        Output channels for the two CNN blocks. Default ``[64, 128]``.
    cnn_kernel_size : int
        Convolution kernel width for both CNN blocks. Default ``3``.
    cnn_pool_size : int
        MaxPool window size for both CNN blocks. Default ``2``.
    lstm_hidden : int
        LSTM hidden dimension. Default ``128``.
    lstm_num_layers : int
        Number of stacked LSTM layers. Default ``2``.
    dropout : float
        Dropout probability throughout. Default ``0.3``.
    use_batch_norm : bool
        Apply BatchNorm in CNN blocks and on LSTM output. Default ``True``.
    fc_hidden : int
        FC head hidden size. Default ``64``.

    Examples
    --------
    >>> model = CNNLSTMClassifier(input_dim=32, seq_len=50)
    >>> x = torch.randn(16, 50, 32)
    >>> out = model(x)
    >>> out.shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        cnn_channels: Optional[List[int]] = None,
        cnn_kernel_size: int = 3,
        cnn_pool_size: int = 2,
        lstm_hidden: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        fc_hidden: int = 64,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128]
        if len(cnn_channels) != 2:
            raise ValueError(
                f"cnn_channels must have exactly 2 elements, got {len(cnn_channels)}."
            )

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden

        # ---- CNN ----
        self.cnn_block1 = _CNNBlock(
            in_channels=input_dim,
            out_channels=cnn_channels[0],
            kernel_size=cnn_kernel_size,
            pool_size=cnn_pool_size,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        self.cnn_block2 = _CNNBlock(
            in_channels=cnn_channels[0],
            out_channels=cnn_channels[1],
            kernel_size=cnn_kernel_size,
            pool_size=cnn_pool_size,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

        # ---- LSTM ----
        lstm_dropout = dropout if lstm_num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cnn_channels[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False,
        )

        self.bn_hidden: Optional[nn.BatchNorm1d] = (
            nn.BatchNorm1d(lstm_hidden) if use_batch_norm else None
        )

        # ---- FC head ----
        self.fc_head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )

        self._init_weights()
        logger.info(
            "CNNLSTMClassifier initialised | input_dim=%d | seq_len=%d "
            "| cnn_channels=%s | lstm_hidden=%d | params=%d",
            input_dim,
            seq_len,
            cnn_channels,
            lstm_hidden,
            self.count_parameters(),
        )

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming for Conv; Xavier/orthogonal for LSTM; Xavier for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)  # forget gate

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

        # CNN feature extraction
        x = self.cnn_block1(x)   # (B, cnn_channels[0], T1)
        x = self.cnn_block2(x)   # (B, cnn_channels[1], T2)

        # (B, cnn_channels[1], T2) → (B, T2, cnn_channels[1])
        x = x.permute(0, 2, 1)

        # LSTM temporal modelling
        _, (h_n, _) = self.lstm(x)
        hidden = h_n[-1]  # (B, lstm_hidden)

        if self.bn_hidden is not None:
            hidden = self.bn_hidden(hidden)

        out = self.fc_head(hidden)
        return out

    # ------------------------------------------------------------------
    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return CNN feature maps (before LSTM).

        Parameters
        ----------
        x : torch.Tensor
            Input ``(B, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            CNN output ``(B, T2, cnn_channels[1])``.
        """
        x = x.permute(0, 2, 1)
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        return x.permute(0, 2, 1)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns
        -------
        int
            Number of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
