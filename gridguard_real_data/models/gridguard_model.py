"""
GridGuard AI — Model Definitions (Phase 1: Real-Data Training)
==============================================================
Contains:
  - TCNBlock
  - GridGuardUniversalHybrid   (thesis two-tier hybrid model, DO NOT MODIFY)
  - AsymmetricFocalLoss         (thesis loss function, DO NOT MODIFY)
  - BiGRUBiLSTMBaseline         (walk-forward significance baseline)

Architecture is frozen per thesis specification:
  TCN (2-block, dilation=[1,2]) + Bi-LSTM (h=64, 2-layer)
  + Transformer Encoder (d=128, 4-head, 2-layer)
  Late fusion: 0.70 × P_DL + 0.30 × P_XGB, τ = 0.5270
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  TCN Block
# ─────────────────────────────────────────────

class TCNBlock(nn.Module):
    """Dilated causal temporal convolutional block with residual connection."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]   # causal trim
        out = F.relu(out)
        out = self.drop(out)
        res = self.res(x) if self.res else x
        return F.relu(out + res)


# ─────────────────────────────────────────────
#  GridGuardUniversalHybrid  (DO NOT CHANGE)
# ─────────────────────────────────────────────

class GridGuardUniversalHybrid(nn.Module):
    """
    Two-tier hybrid model: TCN → Bi-LSTM → Transformer Encoder.
    Input shape : (B, T=26, 2)   channel-0 = kWh, channel-1 = GLI
    Output shape: (B, 1)          raw sigmoid probability
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64,
                 num_heads: int = 4, num_lstm_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        # TCN branch
        self.tcn = nn.Sequential(
            TCNBlock(input_dim,  hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout),
        )
        self.tcn_pool = nn.AdaptiveAvgPool1d(1)

        # Bi-LSTM branch
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Transformer Encoder branch (operates on LSTM output)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),             nn.ReLU(),
            nn.Linear(32, 1),              nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TCN expects (B, C, T)
        tcn_in  = x.transpose(1, 2)
        tcn_out = self.tcn(tcn_in)
        tcn_vec = self.tcn_pool(tcn_out).squeeze(-1)      # (B, hidden_dim)

        # Bi-LSTM + Transformer operate on (B, T, C)
        lstm_out, _ = self.lstm(x)                         # (B, T, hidden_dim)
        attn_out    = self.transformer(lstm_out)            # (B, T, hidden_dim)
        trans_vec   = attn_out[:, -1, :]                   # (B, hidden_dim)

        fused = torch.cat([tcn_vec, trans_vec], dim=1)     # (B, hidden_dim*2)
        return self.fc(fused)                               # (B, 1)


# ─────────────────────────────────────────────
#  Asymmetric Focal Loss  (DO NOT CHANGE)
# ─────────────────────────────────────────────

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric focal loss for severe class imbalance (5% theft in SGCC).
    α=0.80 upweights theft class; γ_neg=4.0 > γ_pos=2.0 to suppress easy negatives.
    """

    def __init__(self, alpha: float = 0.80,
                 gamma_pos: float = 2.0, gamma_neg: float = 4.0):
        super().__init__()
        self.alpha     = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds   = preds.squeeze()
        bce     = F.binary_cross_entropy(preds, targets, reduction='none')
        p_t     = torch.where(targets == 1, preds, 1 - preds)
        gamma   = torch.where(
            targets == 1,
            torch.tensor(self.gamma_pos, device=preds.device),
            torch.tensor(self.gamma_neg, device=preds.device),
        )
        focal_w = (1 - p_t) ** gamma
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha,     device=preds.device),
            torch.tensor(1 - self.alpha, device=preds.device),
        )
        return (alpha_t * focal_w * bce).mean()


# ─────────────────────────────────────────────
#  BiGRU-BiLSTM Baseline  (walk-forward comparison)
# ─────────────────────────────────────────────

class BiGRUBiLSTMBaseline(nn.Module):
    """
    Bidirectional GRU → Bidirectional LSTM baseline.
    Same input/output contract as GridGuardUniversalHybrid.
    Used for significance comparison in Experiment 2 (walk-forward).
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),          nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out,  _ = self.gru(x)
        lstm_out, _ = self.lstm(gru_out)
        return self.fc(lstm_out[:, -1, :])


# ─────────────────────────────────────────────
#  Convenience factory
# ─────────────────────────────────────────────

def build_model(model_type: str = 'gridguard', device: str = 'cpu') -> nn.Module:
    """Return a freshly initialised model on the specified device."""
    if model_type == 'gridguard':
        model = GridGuardUniversalHybrid()
    elif model_type == 'bigru_bilstm':
        model = BiGRUBiLSTMBaseline()
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    return model.to(device)
