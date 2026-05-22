import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM Architecture based on Hasan et al. (2019).
    Focuses on spatial feature extraction followed by temporal sequential modeling.
    """
    def __init__(self, input_dim=1, hidden_dim=64, seq_len=26):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate sequence length after pooling
        reduced_seq_len = seq_len // 2
        
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.transpose(1, 2) # (batch, input_dim, seq_len)
        x = self.relu(self.conv1(x))
        x = self.maxpool(self.relu(self.conv2(x)))
        
        x = x.transpose(1, 2) # (batch, reduced_seq_len, 64)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :]) # Last time step
        return out

class BiGRUBiLSTMModel(nn.Module):
    """
    BiGRU-BiLSTM Architecture based on Munawar et al. (2022).
    Dual-stream recurrent network for bidirectional temporal context.
    """
    def __init__(self, input_dim=1, hidden_dim=64):
        super(BiGRUBiLSTMModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, 
                          batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=1, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)
        lstm_out, _ = self.lstm(gru_out)
        out = self.fc(lstm_out[:, -1, :])
        return out

if __name__ == "__main__":
    # Smoke test
    batch_size, seq_len, input_dim = 8, 26, 1
    x = torch.randn(batch_size, seq_len, input_dim)
    
    model1 = CNNLSTMModel(seq_len=seq_len)
    model2 = BiGRUBiLSTMModel()
    
    print(f"CNN-LSTM Output Shape: {model1(x).shape}")
    print(f"BiGRU-BiLSTM Output Shape: {model2(x).shape}")
