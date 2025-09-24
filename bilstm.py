class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- Model 1: BiLSTM Solver ---
class BiLSTMSolver(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, padding_idx, dropout=0.3):
        super(BiLSTMSolver, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=n_layers,
            bidirectional=True, batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x) * math.sqrt(self.embedding_dim)
        pos_encoded = self.pos_encoder(embedded)
        lstm_out, _ = self.lstm(pos_encoded)
        return self.fc(lstm_out)
