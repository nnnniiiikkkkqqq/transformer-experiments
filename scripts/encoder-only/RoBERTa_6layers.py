#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment: RoBERTa-6-layers on IMDb and Yelp Polarity for Binary Sentiment Classification

import time
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments, RobertaConfig
from datasets import load_dataset
import evaluate
from torch.cuda.amp import autocast
import psutil
import GPUtil

# Settings
model_name = "roberta-base"
batch_size = 16
num_epochs_imdb = 3
num_epochs_yelp = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a custom configuration with 6 layers
config = RobertaConfig.from_pretrained(model_name, num_hidden_layers=6, num_labels=2)

# Load tokenizer and model with modified configuration
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, config=config).to(device)

# Data preprocessing
def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")

# Load and preprocess IMDb dataset
imdb_dataset = load_dataset("imdb")
encoded_imdb_dataset = imdb_dataset.map(preprocess_function, batched=True)
train_imdb_dataset = encoded_imdb_dataset["train"]
eval_imdb_dataset = encoded_imdb_dataset["test"]

# Metric setup
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "f1": f1}

# Training setup for IMDb
training_args_imdb = TrainingArguments(
    output_dir="./../../results/results_roberta_6_layers_imdb",
    num_train_epochs=num_epochs_imdb,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_roberta_6_layers_imdb",
    logging_steps=100,
    fp16=True,  # Mixed precision for NVIDIA A40
    learning_rate=5e-5,
    load_best_model_at_end=True,
)

trainer_imdb = Trainer(
    model=model,
    args=training_args_imdb,
    train_dataset=train_imdb_dataset,
    eval_dataset=eval_imdb_dataset,
    compute_metrics=compute_metrics,
)

# Train the model on IMDb
print("Starting training on IMDb...")
start_train_time_imdb = time.time()
trainer_imdb.train()
end_train_time_imdb = time.time()
training_time_imdb = end_train_time_imdb - start_train_time_imdb
print(f"Training finished. Training time: {training_time_imdb:.2f} seconds.")

# Evaluate the model on IMDb
print("Evaluating the model on IMDb test set...")
eval_results_imdb = trainer_imdb.evaluate()
accuracy_imdb = eval_results_imdb["eval_accuracy"]
f1_imdb = eval_results_imdb["eval_f1"]
print(f"Accuracy: {accuracy_imdb:.4f}")
print(f"F1-score: {f1_imdb:.4f}")

# Measure inference time on IMDb
model.eval()
input_batch_imdb = eval_imdb_dataset.select(range(batch_size))
inputs_imdb = {
    "input_ids": torch.tensor(input_batch_imdb["input_ids"]).to(device),
    "attention_mask": torch.tensor(input_batch_imdb["attention_mask"]).to(device),
}
start_time = time.time()
with torch.no_grad(), autocast():
    outputs = model(**inputs_imdb)
end_time = time.time()
inference_time_imdb = (end_time - start_time) * 1000 / batch_size  # ms per example
print(f"Inference time: {inference_time_imdb:.2f} ms/example")

# Measure GPU memory consumption
gpus = GPUtil.getGPUs()
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0  # Convert to GB
print(f"GPU memory consumption: {memory_used:.2f} GB")

# Output results for the IMDb table
print("\nResults for the IMDb table:")
print(f"| Model            | Accuracy | F1-score | Inference Time (ms) | Memory (GB) | Training Time (s) |")
print(f"|------------------|----------|----------|---------------------|-------------|-------------------|")
print(f"| RoBERTa-6-layers | {accuracy_imdb:.4f}   | {f1_imdb:.4f}   | {inference_time_imdb:.2f}               | {memory_used:.2f}         | {training_time_imdb:.2f}         |")

# --- Fine-tuning on Yelp Polarity ---
print("\nStarting fine-tuning on Yelp Polarity...")
yelp_dataset = load_dataset("yelp_polarity")
encoded_yelp_dataset = yelp_dataset.map(preprocess_function, batched=True)
train_yelp_dataset = encoded_yelp_dataset["train"]
eval_yelp_dataset = encoded_yelp_dataset["test"]

training_args_yelp = TrainingArguments(
    output_dir="./../../results/results_roberta_6_layers_yelp",
    num_train_epochs=num_epochs_yelp,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_roberta_6_layers_yelp",
    logging_steps=100,
    fp16=True,
    learning_rate=2e-5,  # Lower learning rate for fine-tuning
    load_best_model_at_end=True,
)

trainer_yelp = Trainer(
    model=model,  # Use the model after training on IMDb
    args=training_args_yelp,
    train_dataset=train_yelp_dataset,
    eval_dataset=eval_yelp_dataset,
    compute_metrics=compute_metrics,
)

start_train_time_yelp = time.time()
trainer_yelp.train()
end_train_time_yelp = time.time()
training_time_yelp = end_train_time_yelp - start_train_time_yelp
print(f"Fine-tuning finished. Fine-tuning time: {training_time_yelp:.2f} seconds.")

print("Evaluating the fine-tuned model on Yelp test set...")
eval_results_yelp = trainer_yelp.evaluate()
accuracy_yelp = eval_results_yelp["eval_accuracy"]
f1_yelp = eval_results_yelp["eval_f1"]
print(f"Accuracy on Yelp after fine-tuning: {accuracy_yelp:.4f}")
print(f"F1-score on Yelp after fine-tuning: {f1_yelp:.4f}")

# Inference time and memory for Yelp
model.eval()
yelp_input_batch = eval_yelp_dataset.select(range(batch_size))
yelp_inputs = {
    "input_ids": torch.tensor(yelp_input_batch["input_ids"]).to(device),
    "attention_mask": torch.tensor(yelp_input_batch["attention_mask"]).to(device),
}
start_time = time.time()
with torch.no_grad(), autocast():
    outputs = model(**yelp_inputs)
end_time = time.time()
inference_time_yelp = (end_time - start_time) * 1000 / batch_size
print(f"Inference time (Yelp batch): {inference_time_yelp:.2f} ms/example")
memory_used_yelp = gpus[0].memoryUsed / 1024 if gpus else 0
print(f"GPU memory consumption: {memory_used_yelp:.2f} GB")

print(f"\nResults table for Yelp fine-tuning evaluation:")
print(f"| Model                     | Accuracy (Yelp) | F1-score (Yelp) | Inference Time (ms) | Memory (GB) | Fine-tuning Time (s) |")
print(f"|---------------------------|-----------------|-----------------|---------------------|-------------|----------------------|")
print(f"| RoBERTa-6-layers (fine-tuned) | {accuracy_yelp:.4f}          | {f1_yelp:.4f}          | {inference_time_yelp:.2f}          | {memory_used_yelp:.2f}       | {training_time_yelp:.2f}         |")