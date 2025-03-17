# Bidirectional LSTM (bidirectional=True) для лучшего понимания контекста.
# Dropout (0.3) для предотвращения переобучения.
# Динамическую температуру, которая уменьшается по мере генерации текста.
# Top-k sampling (k=5), чтобы избежать слишком случайных слов.

import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests
from collections import Counter
from torchsummary import summary
from gensim.models import Word2Vec
import gensim.downloader as api

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем текстовый датасет из интернета
url = "https://www.gutenberg.org/files/1342/1342-0.txt"  # Роман "Гордость и предубеждение"
print("Model loading...")
response = requests.get(url)
text = response.text[:50000]  # Берем часть текста, чтобы не перегружать память
print("Model loaded")
# Создаем словарь символов
words = text.split()
char2idx = {word: idx for idx, word in enumerate(set(words))}
idx2char = {idx: word for word, idx in char2idx.items()}

# Преобразуем текст в числовые индексы
encoded_text = np.array([char2idx[word] for word in words if word in char2idx])

# Параметры обучения
seq_length = 100  # Длина последовательности для входа в RNN
batch_size = 64  # Размер батча
hidden_size = 128  # Размер скрытого слоя RNN
embedding_size = 64  # Размерность Embedding слоя
num_epochs = 10  # Количество эпох
learning_rate = 0.005
vocab_size = len(char2idx)

print('Transform done')

# Подготовка данных (батчинг)
def create_batches(data, seq_length, batch_size):
    num_samples = len(data) - seq_length
    X = np.zeros((num_samples, seq_length), dtype=np.int64)
    Y = np.array(data[seq_length:], dtype=np.int64)  # Ускоряем выделение Y
    
    for i in range(num_samples):
        X[i] = data[i:i+seq_length]
    
    print('Create tensors')
    X_tensor = torch.tensor(X, dtype=torch.long, device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataloader = create_batches(encoded_text, seq_length, batch_size)
print("Batches created")

# Определяем модель на основе RNN с Embedding
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(WordRNN, self).__init__()
        self.hidden_size = hidden_size
        if os.path.exists("word2vec.model"):
            word2vec_model = Word2Vec.load("word2vec.model")
        else:
            word2vec_model = api.load("word2vec-google-news-300")
            word2vec_model.save("word2vec.model")
        embedding_matrix = np.zeros((vocab_size, embedding_size))
        for char, idx in char2idx.items():
            if char in word2vec_model.wv:
                embedding_matrix[idx] = word2vec_model.wv[char]
            else:
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_size,))

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=3, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, len(char2idx))  # Прогнозируем вероятность каждого символа
    
    def forward(self, x, hidden):
        x = self.embedding(x)  # Применяем Embedding слой
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # Берем последний выходной вектор для прогноза
        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(6, batch_size, self.hidden_size, device=device),
                torch.zeros(6, batch_size, self.hidden_size, device=device))

# Инициализация модели
model = WordRNN(vocab_size, embedding_size, hidden_size).to(device)
print('Model summary:')
#summary(model, (1, seq_length))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model():
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            hidden = model.init_hidden(X_batch.size(0))
            hidden = tuple(h.detach() for h in hidden)
            optimizer.zero_grad()
            outputs, hidden = model(X_batch, hidden)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')

print('Training model')
train_model()
print('Model trained')

# Функция для генерации текста
def generate_text(model, start_str, length=50):
    model.eval()
    hidden = model.init_hidden(1)
    input_seq = torch.tensor([char2idx[word] for word in start_str.split() if word in char2idx], dtype=torch.long, device=device).unsqueeze(0)
    generated_text = start_str
    
    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            temperature = 0.7
            probs = torch.nn.functional.softmax(output / temperature, dim=1)
            top_p = 0.9  # Используем nucleus sampling вместо top-k
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=1)
            mask = cumulative_probs < top_p
            mask[:, 0] = True  # Гарантируем, что хотя бы один элемент останется
            cumulative_probs = torch.cumsum(sorted_probs, dim=1)
            filtered_indices = torch.masked_select(sorted_indices, mask).unsqueeze(0) if torch.any(mask) else sorted_indices[:, :1]
            filtered_probs = torch.masked_select(sorted_probs, mask).unsqueeze(0) if torch.any(mask) else sorted_probs[:, :1]
            predicted_idx = filtered_indices.squeeze(0)[torch.multinomial(filtered_probs.squeeze(0), num_samples=1)].item()
            predicted_char = idx2char[predicted_idx]
            generated_text += ' ' + predicted_char
            input_seq = torch.tensor([[predicted_idx]], dtype=torch.long, device=device)
    
    return generated_text

# Генерируем текст
print(generate_text(model, "You must write", 10))

