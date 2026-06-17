import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        out = self.conv(x)[:, :, :-self.conv.padding[0]]
        out = F.relu(self.drop(out))
        return F.relu(out + (self.res(x) if self.res else x))

class GridGuardUniversalHybridLegacy(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_heads=4,
                 num_lstm_layers=2, dropout=0.2):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_dim,  hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout),
        )
        self.tcn_pool = nn.AdaptiveAvgPool1d(1)
        self.lstm = nn.LSTM(input_dim, hidden_dim//2, num_layers=num_lstm_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if num_lstm_layers > 1 else 0.0)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                               dim_feedforward=hidden_dim*4,
                                               dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        tcn_vec  = self.tcn_pool(self.tcn(x.transpose(1,2))).squeeze(-1)
        lstm_out, _ = self.lstm(x)
        trans_vec   = self.transformer(lstm_out)[:, -1, :]
        return self.fc(torch.cat([tcn_vec, trans_vec], dim=1))
