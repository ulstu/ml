from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

BASE_DIR = "ml_course_en/lecture17. Transformer/jane_austen/"

# 1. Загрузка модели и токенизатора
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 не имеет токена pad — надо задать его явно
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# 2. Подготовка датасета
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

train_dataset = load_dataset(f"{BASE_DIR}pride_and_prejudice.txt", tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 3. Настройки обучения
training_args = TrainingArguments(
    output_dir=f"./{BASE_DIR}gpt2-jane-austen",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100
)

# 4. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# 5. Дообучение
trainer.train()

# 6. Сохранение модели
trainer.save_model(f"./{BASE_DIR}gpt2-jane-austen")
tokenizer.save_pretrained(f"./{BASE_DIR}gpt2-jane-austen")
