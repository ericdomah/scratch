import torch
import torch.nn as nn

class SimplifiedTFT(nn.Module):
    """
    A specific Temporal Fusion Transformer architecture 
    custom-built for GridGuard AI to avoid black-box library usage.
    Provides clear, explainable temporal attention weights.
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_heads=4, seq_len=30, dropout=0.1):
        super(SimplifiedTFT, self).__init__()
        
        # 1. Input Projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 2. LSTM Encoder-Decoder (Temporal Feature Extraction)
        # In a full TFT this is seq2seq, but for classification we use causal/bidirectional extraction
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout
        )
        
        # 3. Temporal Self-Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Position-wise Feed-Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 4. Final Classification Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        
    def forward(self, x, return_attention=False):
        # x shape: (batch_size, seq_len, input_dim)
        projected = self.input_projection(x)
        
        # Temporal extraction
        lstm_out, _ = self.lstm(projected)
        
        # Temporal Self-Attention
        attn_out, attn_weights = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        
        # Add & Norm
        out = self.layer_norm1(lstm_out + attn_out)
        
        # Feed-forward
        out = self.layer_norm2(out + self.feed_forward(out))
        
        # Pool across sequence length
        pooled_out = torch.mean(out, dim=1)
        
        # Final classification
        logits = self.fc(pooled_out)
        
        if return_attention:
            # We average the attention weights across heads for XAI visualization
            # shape of attn_weights depends on PyTorch version, usually (batch, target_seq, source_seq) or with heads
            return logits, attn_weights
        return logits

    def get_attention_weights(self, x):
        """Extracts attention weights specifically for the XAI temporal heatmap."""
        self.eval()
        with torch.no_grad():
            _, attn = self.forward(x, return_attention=True)
            return attn
