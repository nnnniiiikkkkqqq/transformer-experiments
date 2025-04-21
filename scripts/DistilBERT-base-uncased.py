import time
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import GPUtil

# Настройки
model_name = "distilbert-base-uncased"
batch_size = 16
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка датасета IMDb
dataset = load_dataset("imdb")

# Загрузка токенизатора и модели
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

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
    output_dir="./results_distilbert",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_distilbert",
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
end_train_time = time.time()  # Исправлено: определение end_train_time
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
input_batch = eval_dataset.select(range(batch_size))
inputs = {
    "input_ids": torch.tensor(input_batch["input_ids"]).to(device),
    "attention_mask": torch.tensor(input_batch["attention_mask"]).to(device),
}
start_time = time.time()
with torch.no_grad():
    outputs = model(**inputs)
end_time = time.time()
inference_time = (end_time - start_time) * 1000 / batch_size  # мс на пример
print(f"Время инференса: {inference_time:.2f} мс/пример")

# Измерение потребления памяти GPU
gpus = GPUtil.getGPUs()
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0  # Перевод в ГБ
print(f"Потребление памяти GPU: {memory_used:.2f} ГБ")

# Вывод результатов
print("\nРезультаты для таблицы:")
print(f"| Модель        | Accuracy | F1-score | Время инференса (мс) | Память (ГБ) | Время обучения (с) |")
print(f"|---------------|----------|----------|---------------------|-------------|--------------------|")
print(f"| DistilBERT-base | {accuracy:.4f} | {f1:.4f} | {inference_time:.2f} | {memory_used:.2f} | {training_time:.2f} |")

# Загрузка датасета Yelp Polarity
yelp_dataset = load_dataset("yelp_polarity")

# Предобработка Yelp
encoded_yelp_dataset = yelp_dataset.map(preprocess_function, batched=True)
train_yelp_dataset = encoded_yelp_dataset["train"]
eval_yelp_dataset = encoded_yelp_dataset["test"]

# Настройка дообучения
training_args_yelp = TrainingArguments(
    output_dir="./results_distilbert_yelp",
    num_train_epochs=1,  # Меньше эпох для дообучения
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_distilbert_yelp",
    logging_steps=100,
    fp16=True,
    learning_rate=2e-5,  # Меньший learning rate
    load_best_model_at_end=True,
)

trainer_yelp = Trainer(
    model=model,  # Используем модель после обучения на IMDb
    args=training_args_yelp,
    train_dataset=train_yelp_dataset,
    eval_dataset=eval_dataset,  # Оцениваем на IMDb для сравнения
    compute_metrics=compute_metrics,
)

# Дообучение
print("Начало дообучения на Yelp...")
start_train_time = time.time()
trainer_yelp.train()
end_train_time = time.time()  # Исправлено
training_time_yelp = end_train_time - start_train_time
print(f"Дообучение завершено. Время дообучения: {training_time_yelp:.2f} секунд.")

# Оценка на IMDb
eval_results = trainer_yelp.evaluate()
accuracy = eval_results["eval_accuracy"]
f1 = eval_results["eval_f1"]
print(f"Accuracy после дообучения: {accuracy:.4f}")
print(f"F1-score после дообучения: {f1:.4f}")

# Время инференса и память
model.eval()
start_time = time.time()
with torch.no_grad():
    outputs = model(**inputs)
end_time = time.time()
inference_time = (end_time - start_time) * 1000 / batch_size
print(f"Время инференса: {inference_time:.2f} мс/пример")
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0
print(f"Потребление памяти GPU: {memory_used:.2f} ГБ")

# Результаты
print("\nРезультаты для таблицы:")
print(f"| Модель        | Accuracy | F1-score | Время инференса (мс) | Память (ГБ) | Время дообучения (с) |")
print(f"| DistilBERT (дообученная) | {accuracy:.4f} | {f1:.4f} | {inference_time:.2f} | {memory_used:.2f} | {training_time_yelp:.2f} |")