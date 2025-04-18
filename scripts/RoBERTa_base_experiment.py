#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment: RoBERTa_base_experiment

def main():
    print(f"Running experiment: RoBERTa_base_experiment")
    # Добавьте ваш код здесь
    pass

if __name__ == "__main__":
    main()
import time
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from torch.cuda.amp import autocast
import psutil
import GPUtil

# Настройки
model_name = "roberta-base"
batch_size = 16
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка датасета IMDb
dataset = load_dataset("imdb")

# Загрузка токенизатора и модели
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Предобработка данных
def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")

encoded_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["test"]

# Настройка метрик
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "f1": f1}

# Настройка обучения
training_args = TrainingArguments(
    output_dir="./results_roberta",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_roberta",
    logging_steps=100,
    fp16=True,  # Mixed precision для NVIDIA A40
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Обучение модели
print("Начало обучения...")
start_train_time = time.time()
trainer.train()
end_train_time = time.time()
training_time = end_train_time - start_train_time
print(f"Обучение завершено. Время обучения: {training_time:.2f} секунд.")

# Оценка модели
print("Оценка модели...")
eval_results = trainer.evaluate()
accuracy = eval_results["eval_accuracy"]
f1 = eval_results["eval_f1"]
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")

# Измерение времени инференса
model.eval()
input_batch = eval_dataset.select(range(batch_size))  # Выбираем батч
inputs = {
    "input_ids": torch.tensor(input_batch["input_ids"]).to(device),
    "attention_mask": torch.tensor(input_batch["attention_mask"]).to(device),
}
start_time = time.time()
with torch.no_grad(), autocast():
    outputs = model(**inputs)
end_time = time.time()
inference_time = (end_time - start_time) * 1000 / batch_size  # мс на пример
print(f"Время инференса: {inference_time:.2f} мс/пример")

# Измерение потребления памяти GPU
gpus = GPUtil.getGPUs()
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0  # Перевод в ГБ
print(f"Потребление памяти GPU: {memory_used:.2f} ГБ")

# Вывод результатов для таблицы
print("\nРезультаты для таблицы:")
print(f"| Модель        | Accuracy | F1-score | Время инференса (мс) | Память (ГБ) | Время обучения (с) |")
print(f"|---------------|----------|----------|---------------------|-------------|--------------------|")
print(f"| RoBERTa-base  | {accuracy:.4f}   | {f1:.4f}   | {inference_time:.2f}               | {memory_used:.2f}         | {training_time:.2f}          |")