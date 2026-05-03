import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block to capture local anomalies 
    and sharp consumption drops (Micro-patterns).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        out = self.conv(x)
        # Remove padding from the end (causal convolution)
        if self.conv.padding[0] > 0:
            out = out[:, :, :-self.conv.padding[0]]
        return self.relu(self.bn(out))

class GridGuardUniversalHybrid(nn.Module):
    """
    TRIPLE-HYBRID ARCHITECTURE (SOTA):
    1. TCN Head          - Detects local spikes and sudden bypasses (Anomaly Expert)
    2. Bi-LSTM Head      - Detects long-term downward trends (Trend Expert)
    3. Transformer Head  - Detects global correlations (Relationship Expert)
    """
    def __init__(self, window_size=30, input_dim=1, hidden_dim=64):
        super(GridGuardUniversalHybrid, self).__init__()
        
        # --- 1. TCN HEAD (Local Features) ---
        self.tcn_head = nn.Sequential(
            TCNBlock(input_dim, 32, kernel_size=3, dilation=1),
            TCNBlock(32, 64, kernel_size=3, dilation=2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # --- 2. BI-LSTM HEAD (Sequential Features) ---
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.2)
        
        # --- 3. TRANSFORMER HEAD (Global Features) ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim*2, nhead=4, 
                                                  dim_feedforward=256, dropout=0.2, 
                                                  batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # --- 4. FEATURE FUSION & CLASSIFICATION ---
        # Fusion of TCN (64) + LSTM/Transformer (128)
        self.fusion_dim = 64 + (hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        
        # TCN path (needs batch, channels, seq_len)
        x_tcn = x.transpose(1, 2) 
        tcn_out = self.tcn_head(x_tcn).squeeze(-1) # (batch, 64)
        
        # LSTM path
        lstm_out, _ = self.lstm(x) # (batch, seq_len, 128)
        
        # Transformer path (takes LSTM output for richer temporal context)
        trans_out = self.transformer_encoder(lstm_out)
        trans_out = trans_out[:, -1, :] # Last time step (batch, 128)
        
        # --- MULTI-SCALE FUSION ---
        combined = torch.cat([tcn_out, trans_out], dim=1) # (batch, 192)
        
        return self.classifier(combined)

if __name__ == "__main__":
    model = GridGuardUniversalHybrid(window_size=30)
    dummy_input = torch.randn(8, 30, 1)
    output = model(dummy_input)
    print(f"Model Forward Pass Successful! Output shape: {output.shape}")
