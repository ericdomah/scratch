"""
transformer.py
--------------
Transformer Encoder binary classifier for theft detection.

Architecture
------------
  Input          : (B, seq_len, input_dim)
  CLS token      : prepend learnable token → (B, seq_len+1, d_model)
  Input projection : Linear(input_dim → d_model)
  Positional encoding (sinusoidal, fixed)
  TransformerEncoder:
      nhead, num_encoder_layers, dim_feedforward, dropout
  Extract CLS token representation → (B, d_model)
  FC head:
      Linear(d_model → 128) → ReLU → Dropout
      → Linear(128 → 1) → Sigmoid
  Output : (B, 1)

  Attention weights are stored per-layer for post-hoc visualisation.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Adds position-dependent sinusoid signals to token embeddings so that
    the model can leverage sequence ordering without learning position
    embeddings from data.

    Parameters
    ----------
    d_model : int
        Embedding / model dimension.
    max_len : int
        Maximum supported sequence length (including CLS). Default ``5000``.
    dropout : float
        Dropout applied after adding positional encoding. Default ``0.1``.

    References
    ----------
    Vaswani et al. (2017) "Attention Is All You Need", NeurIPS.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the sinusoidal matrix once and register as a buffer (no grad)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings tensor ``(B, T, d_model)``.

        Returns
        -------
        torch.Tensor
            Positionally-encoded tensor ``(B, T, d_model)``.
        """
        x = x + self.pe[:, : x.size(1), :]  # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Attention-capturing TransformerEncoderLayer wrapper
# ---------------------------------------------------------------------------

class _AttentionCapturingLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer that stores its own attention weights.

    PyTorch's vanilla TransformerEncoderLayer discards attention weights
    when ``need_weights=False`` (faster).  This subclass overrides forward
    to capture and cache the weights for visualisation.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Ensure batch_first=True
        kwargs.setdefault("batch_first", True)
        super().__init__(*args, **kwargs)
        self._attn_weights: Optional[torch.Tensor] = None  # (B, T, T)

    def forward(  # type: ignore[override]
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward with attention weight capture."""
        # self-attention sub-layer
        attn_out, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self._attn_weights = attn_weights.detach()  # (B, T, T)

        # Standard residual + norm + FFN (mirrors TransformerEncoderLayer)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # Feed-forward sub-layer
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        return src


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """Transformer Encoder binary classifier with CLS-token pooling.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep.
    seq_len : int
        Input sequence length (used only for logging; model handles variable T).
    d_model : int
        Internal transformer embedding dimension. Default ``128``.
    nhead : int
        Number of attention heads. ``d_model`` must be divisible by ``nhead``.
        Default ``8``.
    num_encoder_layers : int
        Number of stacked encoder layers. Default ``4``.
    dim_feedforward : int
        Feedforward network dimension inside each encoder layer. Default ``256``.
    dropout : float
        Dropout probability throughout. Default ``0.1``.
    max_len : int
        Maximum sequence length for positional encoding. Default ``5000``.
    fc_hidden : int
        FC head hidden size. Default ``128``.

    Attributes
    ----------
    attention_weights : Dict[int, torch.Tensor]
        Attention weight maps per layer, populated after each forward call.
        Keys are layer indices (0-indexed); values are ``(B, T+1, T+1)``.

    Examples
    --------
    >>> model = TransformerClassifier(input_dim=32, seq_len=50)
    >>> x = torch.randn(16, 50, 32)
    >>> out = model(x)
    >>> out.shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 5000,
        fc_hidden: int = 128,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        # ---- Learnable CLS token ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---- Input projection ----
        self.input_projection = nn.Linear(input_dim, d_model)

        # ---- Positional encoding ----
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        # ---- Transformer encoder ----
        encoder_layers = nn.ModuleList(
            [
                _AttentionCapturingLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="relu",
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.encoder_layers: nn.ModuleList = encoder_layers
        self.encoder_norm = nn.LayerNorm(d_model)

        # ---- FC head ----
        self.fc_head = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )

        # Attention weight storage
        self.attention_weights: Dict[int, torch.Tensor] = {}

        self._init_weights()
        logger.info(
            "TransformerClassifier initialised | input_dim=%d | d_model=%d "
            "| nhead=%d | layers=%d | params=%d",
            input_dim,
            d_model,
            nhead,
            num_encoder_layers,
            self.count_parameters(),
        )

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier uniform for Linear layers; LayerNorm default initialisation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(B, seq_len, input_dim)``.
        src_key_padding_mask : torch.Tensor, optional
            Boolean mask ``(B, seq_len)`` where ``True`` marks padded positions.
            A leading ``False`` column is prepended internally for the CLS token.

        Returns
        -------
        torch.Tensor
            Sigmoid probability ``(B, 1)``.
        """
        B = x.size(0)

        # 1. Project input features to d_model
        x = self.input_projection(x)  # (B, seq_len, d_model)

        # 2. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)           # (B, seq_len+1, d_model)

        # 3. Positional encoding
        x = self.pos_encoding(x)  # (B, seq_len+1, d_model)

        # 4. Extend key_padding_mask to account for CLS token
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(
                B, 1, dtype=torch.bool, device=x.device
            )  # CLS is never masked
            src_key_padding_mask = torch.cat(
                [cls_mask, src_key_padding_mask], dim=1
            )  # (B, seq_len+1)

        # 5. Pass through encoder layers, capturing attention weights
        self.attention_weights.clear()
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            if layer._attn_weights is not None:
                self.attention_weights[idx] = layer._attn_weights

        x = self.encoder_norm(x)  # (B, seq_len+1, d_model)

        # 6. Extract CLS token representation
        cls_repr = x[:, 0, :]  # (B, d_model)

        # 7. FC head
        out = self.fc_head(cls_repr)  # (B, 1)
        return out

    # ------------------------------------------------------------------
    def get_attention_weights(self) -> Dict[int, torch.Tensor]:
        """Return cached attention weights from the last forward pass.

        Returns
        -------
        Dict[int, torch.Tensor]
            Mapping from layer index to attention weight tensor
            ``(B, seq_len+1, seq_len+1)``.  The first row/column corresponds
            to the CLS token.
        """
        return dict(self.attention_weights)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns
        -------
        int
            Number of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
