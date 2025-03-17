import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import random

# Установка случайного зерна для воспроизводимости
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка датасета перевода (Hugging Face datasets)
dataset = load_dataset("opus_books", "en-fr")
train_data = list(dataset["train"])[:10000]  # Преобразуем в список

# Токенизация и построение словарей
from collections import Counter
from itertools import chain

def tokenize_text(text):
    return text.lower().split()

src_counter = Counter(chain.from_iterable(map(lambda x: tokenize_text(x["translation"]["en"]), train_data)))
tgt_counter = Counter(chain.from_iterable(map(lambda x: tokenize_text(x["translation"]["fr"]), train_data)))

src_vocab = {word: i+1 for i, (word, _) in enumerate(src_counter.most_common(25000))}
tgt_vocab = {word: i+1 for i, (word, _) in enumerate(tgt_counter.most_common(25000))}

src_vocab["<unk>"] = 0
tgt_vocab["<unk>"] = 0
tgt_vocab["<sos>"] = len(tgt_vocab)
tgt_vocab["<eos>"] = len(tgt_vocab) + 1

SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = hidden.view(N_LAYERS, 2, hidden.shape[1], hidden.shape[2])
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        cell = cell.view(N_LAYERS, 2, cell.shape[1], cell.shape[2])
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        return outputs, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt):
        encoder_outputs, hidden, cell = self.encoder(src)
        input = tgt[:, 0]
        outputs = torch.zeros(tgt.shape[0], tgt.shape[1], TGT_VOCAB_SIZE).to(device)
        for t in range(1, tgt.shape[1]):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            input = tgt[:, t]
        return outputs

def train(model, dataloader, optimizer, criterion):
    model.train()
    for src, tgt in dataloader:
        optimizer.zero_grad()
        output = model(src, tgt)
        output = output[:, 1:].reshape(-1, output.shape[-1])
        tgt = tgt[:, 1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()

criterion = nn.CrossEntropyLoss(ignore_index=0)

encoder = Encoder(SRC_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
seq2seq_attention = Seq2Seq(encoder, AttentionDecoder(TGT_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device))

optimizer_attention = optim.Adam(seq2seq_attention.parameters())

print("Training Attention Model")
for epoch in range(5):
    train(seq2seq_attention, train_dataloader, optimizer_attention, criterion)
    print(f"Epoch {epoch+1} completed")
