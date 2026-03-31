import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка датасета
url = "https://www.gutenberg.org/files/1342/1342-0.txt"
response = requests.get(url)
text = response.text[:50000]

# Словарь
chars = list(set(text))
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
encoded_text = np.array([char2idx[ch] for ch in text])

# Параметры
seq_length = 100
batch_size = 64
hidden_size = 128
embedding_size = 64
num_epochs = 10
learning_rate = 0.005
vocab_size = len(chars)

# Подготовка данных
def create_batches(data, seq_length, batch_size):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length])
    X_tensor = torch.tensor(X, dtype=torch.long, device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataloader = create_batches(encoded_text, seq_length, batch_size)

# Attention Model
class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(AttentionRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=2, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)

        out = self.fc(context)
        return out

model = AttentionRNN(vocab_size, embedding_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
def train_model():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')

train_model()

# Генерация текста
def generate_text(model, start_str, length=200):
    model.eval()
    input_seq = torch.tensor([[char2idx[ch] for ch in start_str]], dtype=torch.long, device=device)
    generated_text = start_str

    for _ in range(length):
        with torch.no_grad():
            output = model(input_seq)
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_char = idx2char[predicted_idx]
            generated_text += predicted_char

            input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[predicted_idx]], device=device)], dim=1)

    return generated_text

# Пример генерации
print(generate_text(model, "You must write", 100))
