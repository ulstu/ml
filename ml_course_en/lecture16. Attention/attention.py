import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data, datasets
import random

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Определение токенизатора и загрузка данных
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

# Построение словаря
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

# Создание итераторов данных
BATCH_SIZE = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, sort_within_batch=True, device=device
)

# Определение модели LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)) if self.lstm.bidirectional else hidden[-1]
        return self.fc(hidden)

# Определение модели с вниманием
class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.attn = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        attn_weights = torch.tanh(self.attn(output)).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(2)
        weighted_output = (output * attn_weights).sum(dim=1)
        return self.fc(weighted_output)

# Функция тренировки
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss, epoch_acc = 0, 0
    for batch in iterator:
        text, text_lengths = batch.text
        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = ((predictions > 0.5) == batch.label).sum().float() / batch.label.shape[0]
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Функция оценки модели
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = ((predictions > 0.5) == batch.label).sum().float() / batch.label.shape[0]
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Гиперпараметры
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

# Инициализация моделей
lstm_model = LSTMClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
attn_model = AttentionClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)

# Оптимизаторы и функция потерь
optimizer_lstm = optim.Adam(lstm_model.parameters())
optimizer_attn = optim.Adam(attn_model.parameters())
criterion = nn.BCEWithLogitsLoss().to(device)

# Обучение моделей
N_EPOCHS = 5
for model, optimizer, name in [(lstm_model, optimizer_lstm, "LSTM"), (attn_model, optimizer_attn, "Attention")]:
    print(f"Training {name} Model")
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.3f}, Val Acc: {valid_acc:.3f}')

# Оценка моделей на тесте
test_loss_lstm, test_acc_lstm = evaluate(lstm_model, test_iterator, criterion)
test_loss_attn, test_acc_attn = evaluate(attn_model, test_iterator, criterion)

print(f'LSTM Test Accuracy: {test_acc_lstm:.3f}')
print(f'Attention Test Accuracy: {test_acc_attn:.3f}')

