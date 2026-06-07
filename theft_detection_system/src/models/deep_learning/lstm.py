"""
lstm.py
-------
Long Short-Term Memory (LSTM) binary classifier for theft detection.

Architecture
------------
  Input  : (B, seq_len, input_dim)
  nn.LSTM: configurable hidden_dim, num_layers, dropout
  Last hidden state h_n[-1] → (B, hidden_dim)
  FC head:
      Linear(hidden_dim → 128) → ReLU → Dropout
      → Linear(128 → 64) → ReLU → Dropout
      → Linear(64 → 1) → Sigmoid
  Output : (B, 1)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LSTMClassifier(nn.Module):
    """LSTM-based binary classifier.

    Parameters
    ----------
    input_dim : int
        Number of features per timestep.
    seq_len : int
        Sequence length (informational; not used by LSTM directly).
    hidden_dim : int
        Number of hidden units in each LSTM cell. Default ``128``.
    num_layers : int
        Number of stacked LSTM layers. Default ``2``.
    dropout : float
        Dropout applied between LSTM layers (ignored when num_layers==1)
        and in the FC head. Default ``0.3``.
    use_batch_norm : bool
        Apply BatchNorm1d on the extracted hidden state before FC head.
        Default ``True``.
    fc_hidden1 : int
        First FC hidden size. Default ``128``.
    fc_hidden2 : int
        Second FC hidden size. Default ``64``.
    bidirectional : bool
        Whether to use bidirectional LSTM. Default ``False``.
        (kept for subclass reuse)

    Examples
    --------
    >>> model = LSTMClassifier(input_dim=32, seq_len=50)
    >>> x = torch.randn(16, 50, 32)
    >>> out = model(x)
    >>> out.shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        fc_hidden1: int = 128,
        fc_hidden2: int = 64,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        fc_in = hidden_dim * self.num_directions
        self.bn: Optional[nn.BatchNorm1d] = (
            nn.BatchNorm1d(fc_in) if use_batch_norm else None
        )

        self.fc_head = nn.Sequential(
            nn.Linear(fc_in, fc_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()
        logger.info(
            "%s initialised | input_dim=%d | hidden_dim=%d | layers=%d "
            "| bidirectional=%s | params=%d",
            self.__class__.__name__,
            input_dim,
            hidden_dim,
            num_layers,
            bidirectional,
            self.count_parameters(),
        )

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Orthogonal init for LSTM weights; Xavier for Linear weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 (common trick)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        for m in self.fc_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _extract_last_hidden(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run LSTM and return the last-layer hidden state.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, hidden_dim * num_directions)``.
        """
        # output: (B, seq_len, hidden_dim * directions)
        # h_n  : (num_layers * directions, B, hidden_dim)
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # Concat last-layer forward and backward hidden states
            h_fwd = h_n[-2]  # (B, hidden_dim)
            h_bwd = h_n[-1]  # (B, hidden_dim)
            hidden = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*hidden_dim)
        else:
            hidden = h_n[-1]  # (B, hidden_dim)
        return hidden

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
        hidden = self._extract_last_hidden(x)

        if self.bn is not None:
            hidden = self.bn(hidden)

        out = self.fc_head(hidden)
        return out

    # ------------------------------------------------------------------
    def get_sequence_output(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full LSTM output sequence and final prediction.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(B, seq_len, input_dim)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(lstm_out, pred)`` where lstm_out has shape
            ``(B, seq_len, hidden_dim * directions)``.
        """
        lstm_out, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            hidden = h_n[-1]
        if self.bn is not None:
            hidden = self.bn(hidden)
        pred = self.fc_head(hidden)
        return lstm_out, pred

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns
        -------
        int
            Number of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
