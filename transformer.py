import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import os
from typing import List, Tuple, Dict

# Custom module imports
from utils.data_utils import create_char_mappings, load_words, MaskedWordDataset, pad_collate_fn
from utils.training import train_model
DATA_PATH="words_250000_train.txt"

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

class TransformerSolver(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_heads, n_encoder_layers, dim_feedforward, padding_idx, dropout=0.3):
        super(TransformerSolver, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.padding_idx = padding_idx

    def forward(self, src):
        src_key_padding_mask = (src == self.padding_idx)
        embedded = self.embedding(src) * math.sqrt(self.embedding_dim)
        pos_encoded = self.pos_encoder(embedded)
        transformer_out = self.transformer_encoder(pos_encoded, src_key_padding_mask=src_key_padding_mask)
        return self.fc(transformer_out)
    

MODEL_PATH_TRANSFORMER = 'best_transformer_solver.pth'
TARGET_GPU_ID = 3 # Or a different GPU if you have one

# Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 256 # Transformers might need smaller batch sizes
NUM_EPOCHS = 60
EMBEDDING_DIM = 768
N_HEADS = 12
N_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.2

# --- Setup ---
device = torch.device(f'cuda:{TARGET_GPU_ID}' if torch.cuda.is_available() else 'cpu')
char_to_idx, _, vocab_size, mask_token_idx, padding_idx = create_char_mappings()
all_words = load_words(DATA_PATH)
train_dataset = MaskedWordDataset(all_words, char_to_idx, mask_token_idx)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=lambda b: pad_collate_fn(b, padding_idx),
    num_workers=2
)

# --- Model Initialization ---
transformer_model = TransformerSolver(
    vocab_size, EMBEDDING_DIM, N_HEADS, N_LAYERS, DIM_FEEDFORWARD, padding_idx, DROPOUT
).to(device)

optimizer = optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)

# --- Run Training ---
train_model(transformer_model, train_dataloader, optimizer, scheduler, criterion, device, NUM_EPOCHS, MODEL_PATH_TRANSFORMER)