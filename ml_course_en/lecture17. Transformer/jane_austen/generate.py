from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

BASE_DIR = "ml_course_en/lecture17. Transformer/jane_austen/"


# Загрузка дообученной модели
model = GPT2LMHeadModel.from_pretrained(f"./{BASE_DIR}gpt2-jane-austen")
tokenizer = GPT2Tokenizer.from_pretrained(f"./{BASE_DIR}gpt2-jane-austen")

prompt = "It is a truth universally acknowledged"

inputs = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=150,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
