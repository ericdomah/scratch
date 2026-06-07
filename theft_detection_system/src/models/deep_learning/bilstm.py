"""
bilstm.py
---------
Bidirectional LSTM binary classifier for theft detection.

Architecture
------------
  Input  : (B, seq_len, input_dim)
  BiLSTM : bidirectional=True, hidden_dim per direction
  Last hidden: concat(h_fwd, h_bwd) → (B, hidden_dim * 2)
  Optional BatchNorm1d
  FC head:
      Linear(hidden_dim*2 → 128) → ReLU → Dropout
      → Linear(128 → 64)        → ReLU → Dropout
      → Linear(64 → 1)          → Sigmoid
  Output : (B, 1)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM binary classifier.

    Concatenates the final forward and backward hidden states to form a
    rich context vector, then passes it through a 3-layer FC head.

    Parameters
    ----------
    input_dim : int
        Number of features per timestep.
    seq_len : int
        Sequence length (informational only; BiLSTM handles variable lengths).
    hidden_dim : int
        Hidden units *per direction*. Total context size = ``hidden_dim * 2``.
        Default ``128``.
    num_layers : int
        Number of stacked BiLSTM layers. Default ``2``.
    dropout : float
        Dropout between layers (skipped when num_layers==1) and in FC head.
        Default ``0.3``.
    use_batch_norm : bool
        Apply BatchNorm1d on the concatenated hidden state. Default ``True``.
    fc_hidden1 : int
        First FC hidden size. Default ``128``.
    fc_hidden2 : int
        Second FC hidden size. Default ``64``.

    Examples
    --------
    >>> model = BiLSTMClassifier(input_dim=32, seq_len=50)
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
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )

        fc_in = hidden_dim * 2  # forward + backward concatenated
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
            "BiLSTMClassifier initialised | input_dim=%d | hidden_dim=%d "
            "| layers=%d | context_size=%d | params=%d",
            input_dim,
            hidden_dim,
            num_layers,
            fc_in,
            self.count_parameters(),
        )

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier init for input-hidden weights; orthogonal for hidden-hidden.

        Forget gate biases are set to 1.0 to help with long-range dependencies.
        """
        for name, param in self.bilstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Forget gate occupies [n/4 : n/2] in the gate ordering
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        for m in self.fc_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _context_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Run BiLSTM and return concatenated last hidden states.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, hidden_dim * 2)``.
        """
        # h_n: (num_layers * 2, B, hidden_dim)
        _, (h_n, _) = self.bilstm(x)
        # Last layer: forward at index -2, backward at index -1
        h_fwd = h_n[-2]  # (B, hidden_dim)
        h_bwd = h_n[-1]  # (B, hidden_dim)
        context = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, hidden_dim*2)
        return context

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
        context = self._context_vector(x)

        if self.bn is not None:
            context = self.bn(context)

        out = self.fc_head(context)
        return out

    # ------------------------------------------------------------------
    def get_sequence_output(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the full BiLSTM output sequence and prediction.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(B, seq_len, input_dim)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(bilstm_out, pred)`` where bilstm_out has shape
            ``(B, seq_len, hidden_dim * 2)`` (fwd+bwd at each timestep).
        """
        bilstm_out, (h_n, _) = self.bilstm(x)
        context = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        if self.bn is not None:
            context = self.bn(context)
        pred = self.fc_head(context)
        return bilstm_out, pred

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns
        -------
        int
            Number of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
