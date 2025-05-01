#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment: BERT-base-uncased on Fill-Mask with IMDb dataset

import time
import torch
from transformers import BertForMaskedLM, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from torch.cuda.amp import autocast
import psutil
import GPUtil
import random

# Settings
model_name = "bert-base-uncased"
batch_size = 16
num_epochs = 3
mask_probability = 0.15  # As in BERT MLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name).to(device)

# Load IMDb dataset
imdb_dataset = load_dataset("imdb")

# Data preprocessing with synthetic masking
def preprocess_function(examples):
    # Tokenize text
    encodings = tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")
    input_ids = encodings["input_ids"]
    labels = input_ids.copy()  # Labels are the original tokens
    
    # Randomly mask 15% of tokens
    for i in range(len(input_ids)):
        for j in range(len(input_ids[i])):
            if random.random() < mask_probability and input_ids[i][j] not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                input_ids[i][j] = tokenizer.mask_token_id
            else:
                labels[i][j] = -100  # Ignore non-masked tokens in loss
    
    encodings["input_ids"] = input_ids
    encodings["labels"] = labels
    return encodings

encoded_imdb_dataset = imdb_dataset.map(preprocess_function, batched=True)
train_dataset = encoded_imdb_dataset["train"]
eval_dataset = encoded_imdb_dataset["test"]

# Remove unnecessary columns
encoded_imdb_dataset = encoded_imdb_dataset.remove_columns(["text", "label", "token_type_ids"])

# Metric setup
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    # Only evaluate masked positions (labels != -100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    return {"accuracy": accuracy}

# Training setup
training_args = TrainingArguments(
    output_dir="./../../results/results_bert_fill_mask",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_bert_fill_mask",
    logging_steps=100,
    fp16=True,  # Mixed precision for NVIDIA A40
    learning_rate=5e-5,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training on IMDb (Fill-Mask)...")
start_train_time = time.time()
trainer.train()
end_train_time = time.time()
training_time = end_train_time - start_train_time
print(f"Training finished. Training time: {training_time:.2f} seconds.")

# Evaluate the model
print("Evaluating the model on IMDb test set...")
eval_results = trainer.evaluate()
accuracy = eval_results["eval_accuracy"]
print(f"Accuracy: {accuracy:.4f}")

# Measure inference time
model.eval()
input_batch = eval_dataset.select(range(batch_size))
inputs = {
    "input_ids": torch.tensor(input_batch["input_ids"]).to(device),
    "attention_mask": torch.tensor(input_batch["attention_mask"]).to(device),
}
start_time = time.time()
with torch.no_grad(), autocast():
    outputs = model(**inputs)
end_time = time.time()
inference_time = (end_time - start_time) * 1000 / batch_size  # ms per example
print(f"Inference time: {inference_time:.2f} ms/example")

# Measure GPU memory consumption
gpus = GPUtil.getGPUs()
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0  # Convert to GB
print(f"GPU memory consumption: {memory_used:.2f} GB")

# Output results for the table
print("\nResults for the IMDb Fill-Mask table:")
print(f"| Model         | Accuracy | Inference Time (ms) | Memory (GB) | Training Time (s) |")
print(f"|---------------|----------|---------------------|-------------|-------------------|")
print(f"| BERT-base     | {accuracy:.4f}   | {inference_time:.2f}          | {memory_used:.2f}       | {training_time:.2f}      |")