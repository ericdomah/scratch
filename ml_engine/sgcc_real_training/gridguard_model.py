"""
GridGuard AI — Model Definitions
==================================
Exact thesis architecture — DO NOT MODIFY.

  TCNBlock (dilated causal conv + residual)
  GridGuardUniversalHybrid  (TCN -> Bi-LSTM -> Transformer, late fusion with XGB)
  BiGRUBiLSTMBaseline       (walk-forward comparison baseline)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]
        out = F.relu(out)
        out = self.drop(out)
        res = self.res(x) if self.res else x
        return F.relu(out + res)


class GridGuardUniversalHybrid(nn.Module):
    """
    Two-tier hybrid: TCN -> Bi-LSTM -> Transformer Encoder.
    Input shape : (B, T=26, 2)   channel-0=kWh, channel-1=GLI
    Output shape: (B, 1)          raw sigmoid probability
    """
    def __init__(self, input_dim=2, hidden_dim=64,
                 num_heads=4, num_lstm_layers=2, dropout=0.2):
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

        # Transformer Encoder (operates on LSTM output)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),             nn.ReLU(),
            nn.Linear(32, 1),              nn.Sigmoid(),
        )

    def forward(self, x):
        # TCN expects (B, C, T)
        tcn_in  = x.transpose(1, 2)
        tcn_out = self.tcn(tcn_in)
        tcn_vec = self.tcn_pool(tcn_out).squeeze(-1)    # (B, hidden_dim)

        # Bi-LSTM + Transformer operate on (B, T, C)
        lstm_out, _ = self.lstm(x)                       # (B, T, hidden_dim)
        attn_out    = self.transformer(lstm_out)          # (B, T, hidden_dim)
        trans_vec   = attn_out[:, -1, :]                 # (B, hidden_dim)

        fused = torch.cat([tcn_vec, trans_vec], dim=1)   # (B, hidden_dim*2)
        return self.fc(fused)                             # (B, 1)


class BiGRUBiLSTMBaseline(nn.Module):
    """Baseline model for walk-forward significance comparison."""
    def __init__(self, input_dim=2, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim // 2,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=dropout,
        )
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        gru_out,  _ = self.gru(x)
        lstm_out, _ = self.lstm(gru_out)
        return self.fc(lstm_out[:, -1, :])


def build_model(model_type: str = "gridguard", device: str = "cpu") -> nn.Module:
    """Return a fresh model on the specified device."""
    if model_type == "gridguard":
        model = GridGuardUniversalHybrid()
    elif model_type == "bigru_bilstm":
        model = BiGRUBiLSTMBaseline()
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    return model.to(device)


if __name__ == "__main__":
    m = GridGuardUniversalHybrid()
    x = torch.randn(4, 26, 2)
    out = m(x)
    print(f"[OK] GridGuardUniversalHybrid output: {out.shape}")
    assert out.shape == (4, 1)

    b = BiGRUBiLSTMBaseline()
    out2 = b(x)
    print(f"[OK] BiGRUBiLSTMBaseline output: {out2.shape}")
    assert out2.shape == (4, 1)
