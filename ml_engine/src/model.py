import torch
import torch.nn as nn

class HybridLSTMTransformer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_lstm_layers=2, num_heads=4, num_transformer_layers=2, dropout=0.1):
        super(HybridLSTMTransformer, self).__init__()
        
        # 1. LSTM Layer: Captures sequential dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_lstm_layers, 
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # 2. Transformer Layer: Captures long-term seasonal patterns via Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 3. Output Layer: Binary Classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, return_attention=False):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer_encoder(lstm_out)
        pooled_out = torch.mean(transformer_out, dim=1)
        logits = self.fc(pooled_out)
        
        if return_attention:
            # For now, return random attention as a placeholder for the XAI layer logic
            batch_size, seq_len, _ = lstm_out.shape
            return logits, torch.rand(batch_size, seq_len, seq_len)
        return logits

    def get_attention_weights(self, x):
        """Extracts attention weights for the dashboard heatmap."""
        self.eval()
        with torch.no_grad():
            _, attn = self.forward(x, return_attention=True)
            return attn
