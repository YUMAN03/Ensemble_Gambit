import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import List, Tuple, Dict

# Custom module imports
from utils.data_utils import create_char_mappings, load_words, MaskedWordDataset, pad_collate_fn
from utils.training import train_model
DATA_PATH="words_250000_train.txt"

class CharCNNSolver(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, padding_idx, dropout=0.5):
        super(CharCNNSolver, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # Repeat output for each position to maintain a consistent output shape
        return self.fc(cat).unsqueeze(1).repeat(1, x.shape[1], 1)


MODEL_PATH_CHARCNN = 'best_charcnn_solver.pth'
TARGET_GPU_ID = 3

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 512
NUM_EPOCHS = 50
EMBEDDING_DIM = 768
NUM_FILTERS = 128
KERNEL_SIZES = [2, 3, 4, 5]
DROPOUT = 0.5

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
charcnn_model = CharCNNSolver(
    vocab_size, EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZES, padding_idx, DROPOUT
).to(device)

optimizer = optim.Adam(charcnn_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)

# --- Run Training ---
train_model(charcnn_model, train_dataloader, optimizer, scheduler, criterion, device, NUM_EPOCHS, MODEL_PATH_CHARCNN)