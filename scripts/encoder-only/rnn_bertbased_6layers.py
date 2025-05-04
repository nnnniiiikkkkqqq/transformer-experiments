#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment: RNN-bertbased-1layer on IMDb and Yelp Polarity for Binary Sentiment Classification

import time
import torch
import torch.nn as nn
from transformers import BertTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
from torch.cuda.amp import autocast
import psutil
import GPUtil
import numpy as np

# --- Model Definition: RNN for Sequence Classification ---
class RNNForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embed_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=0,  # No dropout for single layer
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        embedded = self.dropout(self.embedding(input_ids))
        outputs, (hidden, cell) = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        logits = self.fc(hidden)
        
        # Initialize output dictionary
        output = {'logits': logits}
        
        # Compute loss if labels are provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output['loss'] = loss
        
        return output

# --- Settings ---
tokenizer_name = "bert-base-uncased"
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
OUTPUT_DIM = 2
N_LAYERS = 1  # Reduced from 2 to 1
BIDIRECTIONAL = True
DROPOUT = 0.5
batch_size = 16
num_epochs_imdb = 3
num_epochs_yelp = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load IMDb dataset ---
imdb_dataset = load_dataset("imdb")

# --- Load Tokenizer ---
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
PAD_IDX = tokenizer.pad_token_id
VOCAB_SIZE = tokenizer.vocab_size

# --- Data Preprocessing ---
def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")

print("Preprocessing IMDb dataset...")
encoded_imdb_dataset = imdb_dataset.map(preprocess_function, batched=True)
train_imdb_dataset = encoded_imdb_dataset["train"]
eval_imdb_dataset = encoded_imdb_dataset["test"]

train_imdb_dataset = train_imdb_dataset.map(lambda e: {'labels': e['label']}, batched=True)
eval_imdb_dataset = eval_imdb_dataset.map(lambda e: {'labels': e['label']}, batched=True)
train_imdb_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_imdb_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# --- Model Initialization ---
print("Initializing RNN model with 1 layer...")
model = RNNForSequenceClassification(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    n_layers=N_LAYERS,
    bidirectional=BIDIRECTIONAL,
    dropout=DROPOUT,
    pad_idx=PAD_IDX
).to(device)

# --- Metric setup ---
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if isinstance(logits, torch.Tensor):
        predictions = torch.argmax(logits.cpu().float(), dim=-1)
        labels = torch.tensor(labels).cpu()
    else:
        predictions = np.argmax(logits.astype(np.float32), axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "f1": f1}

# --- Training setup for IMDb ---
training_args_imdb = TrainingArguments(
    output_dir="./../../results/results_rnn_1layer_imdb",
    num_train_epochs=num_epochs_imdb,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_rnn_1layer_imdb",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
)

trainer_imdb = Trainer(
    model=model,
    args=training_args_imdb,
    train_dataset=train_imdb_dataset,
    eval_dataset=eval_imdb_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# --- Train the model on IMDb ---
print("Starting RNN training on IMDb...")
start_train_time_imdb = time.time()
trainer_imdb.train()
end_train_time_imdb = time.time()
training_time_imdb = end_train_time_imdb - start_train_time_imdb
print(f"Training finished. Training time: {training_time_imdb:.2f} seconds.")

# --- Evaluate the model on IMDb ---
print("Evaluating the RNN model on IMDb...")
eval_results_imdb = trainer_imdb.evaluate()
accuracy_imdb = eval_results_imdb["eval_accuracy"]
f1_imdb = eval_results_imdb["eval_f1"]
print(f"Accuracy: {accuracy_imdb:.4f}")
print(f"F1-score: {f1_imdb:.4f}")

# --- Measure inference time on IMDb ---
model.eval()
input_batch_imdb = eval_imdb_dataset.select(range(batch_size))
inputs_imdb = {
    "input_ids": input_batch_imdb["input_ids"].to(device),
    "attention_mask": input_batch_imdb["attention_mask"].to(device),
}
start_time = time.time()
with torch.no_grad():
    if torch.cuda.is_available():
        with autocast():
            outputs = model(**inputs_imdb)
    else:
        outputs = model(**inputs_imdb)
end_time = time.time()
inference_time_imdb = (end_time - start_time) * 1000 / batch_size
print(f"Inference time: {inference_time_imdb:.2f} ms/example")

# --- Measure GPU memory consumption ---
gpus = GPUtil.getGPUs() if GPUtil is not None else []
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0
print(f"GPU memory consumption: {memory_used:.2f} GB")

# --- Output results for IMDb ---
print(f"\n| Model            | Accuracy | F1-score | Inference Time (ms) | Memory (GB) | Training Time (s) |")
print(f"|------------------|----------|----------|---------------------|-------------|-------------------|")
print(f"| RNN-1-layer (LSTM)| {accuracy_imdb:.4f}   | {f1_imdb:.4f}   | {inference_time_imdb:.2f}               | {memory_used:.2f}         | {training_time_imdb:.2f}         |")

# --- Fine-tuning on Yelp Polarity ---
print("\nStarting fine-tuning on Yelp Polarity...")
yelp_dataset = load_dataset("yelp_polarity")
encoded_yelp_dataset = yelp_dataset.map(preprocess_function, batched=True)
train_yelp_dataset = encoded_yelp_dataset["train"]
eval_yelp_dataset = encoded_yelp_dataset["test"]

train_yelp_dataset = train_yelp_dataset.map(lambda e: {'labels': e['label']}, batched=True)
eval_yelp_dataset = eval_yelp_dataset.map(lambda e: {'labels': e['label']}, batched=True)
train_yelp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_yelp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

training_args_yelp = TrainingArguments(
    output_dir="./../../results/results_rnn_1layer_yelp",
    num_train_epochs=num_epochs_yelp,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_rnn_1layer_yelp",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
)

trainer_yelp = Trainer(
    model=model,
    args=training_args_yelp,
    train_dataset=train_yelp_dataset,
    eval_dataset=eval_yelp_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
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

# --- Inference time and memory for Yelp ---
model.eval()
yelp_input_batch = eval_yelp_dataset.select(range(batch_size))
yelp_inputs = {
    "input_ids": yelp_input_batch["input_ids"].to(device),
    "attention_mask": yelp_input_batch["attention_mask"].to(device),
}
start_time = time.time()
with torch.no_grad():
    if torch.cuda.is_available():
        with autocast():
            outputs = model(**yelp_inputs)
    else:
        outputs = model(**yelp_inputs)
end_time = time.time()
inference_time_yelp = (end_time - start_time) * 1000 / batch_size
print(f"Inference time (Yelp batch): {inference_time_yelp:.2f} ms/example")
memory_used_yelp = gpus[0].memoryUsed / 1024 if gpus else 0
print(f"GPU memory consumption: {memory_used_yelp:.2f} GB")

print(f"\nResults table for Yelp fine-tuning evaluation:")
print(f"| Model                     | Accuracy (Yelp) | F1-score (Yelp) | Inference Time (ms) | Memory (GB) | Fine-tuning Time (s) |")
print(f"|---------------------------|-----------------|-----------------|---------------------|-------------|----------------------|")
print(f"| RNN-1-layer (LSTM, fine-tuned) | {accuracy_yelp:.4f}   | {f1_yelp:.4f}   | {inference_time_yelp:.2f}   | {memory_used_yelp:.2f}   | {training_time_yelp:.2f}   |")