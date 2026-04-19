import torch
import torch.nn as nn
import torch.nn.functional as F

class GridGuardUniversalHybrid(nn.Module):
    """
    The ultimate "Super-Hybrid" architecture for GridGuard AI.
    Unifies:
    1. LSTM (Sequential dependencies)
    2. Transformer (Global seasonal patterns)
    3. Temporal Fusion concepts (Gated residual connections)
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_heads=4, dropout=0.1):
        super(GridGuardUniversalHybrid, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Feature Projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2. Sequential Backbone (LSTM)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Adjust hidden_dim for bidirectional output
        self.bi_hidden = hidden_dim * 2
        
        # 3. Transformer Branch (Global Context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.bi_hidden,
            nhead=num_heads,
            dim_feedforward=self.bi_hidden * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. Temporal Fusion Branch (TFT-style Attention)
        self.tft_attn = nn.MultiheadAttention(
            embed_dim=self.bi_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 5. Gating Mechanism (GLU-like)
        self.gating = nn.Sequential(
            nn.Linear(self.bi_hidden, self.bi_hidden),
            nn.Sigmoid()
        )
        
        # 6. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.bi_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_attention=False):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Project and encode
        x_proj = self.input_proj(x)
        lstm_out, _ = self.lstm(x_proj) # (batch, seq, bi_hidden)
        
        # Branch 1: Transformer
        trans_out = self.transformer_encoder(lstm_out)
        
        # Branch 2: TFT Attention
        tft_out, tft_weights = self.tft_attn(lstm_out, lstm_out, lstm_out)
        
        # Fusion via Gating
        # We use the gated sum of both branches
        gate = self.gating(lstm_out)
        fused = gate * trans_out + (1 - gate) * tft_out
        
        # Global Pooling (Temporal Mean)
        pooled = torch.mean(fused, dim=1)
        
        # Final Logits
        logits = self.classifier(pooled)
        
        if return_attention:
            return logits, tft_weights
        return logits

    def get_attention_weights(self, x):
        self.eval()
        with torch.no_grad():
            _, attn = self.forward(x, return_attention=True)
            return attn

if __name__ == "__main__":
    # Sanity Check
    model = GridGuardUniversalHybrid()
    sample = torch.randn(8, 30, 1)
    output = model(sample)
    print(f"Universal Hybrid Output Shape: {output.shape}")
    assert output.shape == (8, 1), "Output shape mismatch"
    print("[OK] Universal Hybrid Model verification successful.")
