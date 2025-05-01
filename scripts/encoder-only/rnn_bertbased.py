import time
import torch
import torch.nn as nn
from transformers import BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import psutil
import GPUtil
import subprocess

# Custom RNN Model
class RNNForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_layers=2, dropout=0.1, num_labels=2):
        super(RNNForSequenceClassification, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)  # Исправлено: hidden_dim * 2
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, lengths=None, labels=None):
        embedded = self.embedding(input_ids)
        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(-1).float()
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            rnn_output, (hidden, cell) = self.rnn(packed)
            hidden, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        else:
            rnn_output, (hidden, cell) = self.rnn(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)  # [batch_size, hidden_dim * 2]
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

# Settings
model_name = "bert-base-uncased"
batch_size = 8
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to log GPU memory
def log_gpu_memory(step_name):
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv'], capture_output=True, text=True)
    print(f"GPU Memory at {step_name}:\n{result.stdout}")

# Clear GPU memory
torch.cuda.empty_cache()
log_gpu_memory("before_start")

# Load IMDb dataset
dataset = load_dataset("imdb")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Initialize model
vocab_size = tokenizer.vocab_size
model = RNNForSequenceClassification(
    vocab_size=vocab_size,
    embedding_dim=300,
    hidden_dim=256,
    num_layers=2,
    dropout=0.1,
    num_labels=2
).to(device)

# Data preprocessing with lengths
def preprocess_function(examples):
    result = tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")
    result["lengths"] = [sum(mask) for mask in result["attention_mask"]]
    return result

encoded_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["test"]

# Metric setup
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "f1": f1}

# Training setup
training_args = TrainingArguments(
    output_dir="./../../results/results_rnn_corrected",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_rnn_corrected",
    logging_steps=100,
    fp16=True,
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
print("Starting training...")
log_gpu_memory("before_training")
start_train_time = time.time()
trainer.train()
end_train_time = time.time()
training_time = end_train_time - start_train_time
print(f"Training finished. Training time: {training_time:.2f} seconds.")
log_gpu_memory("after_training")

# Evaluate the model
print("Evaluating the model...")
eval_results = trainer.evaluate()
accuracy = eval_results["eval_accuracy"]
f1 = eval_results["eval_f1"]
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")

# Measure inference time
model.eval()
input_batch = eval_dataset.select(range(batch_size))
inputs = {
    "input_ids": torch.tensor(input_batch["input_ids"]).to(device),
    "attention_mask": torch.tensor(input_batch["attention_mask"]).to(device),
    "lengths": torch.tensor(input_batch["lengths"]).to(device),
}
start_time = time.time()
with torch.amp.autocast('cuda'):  # Обновлено
    outputs = model(**inputs)
end_time = time.time()
inference_time = (end_time - start_time) * 1000 / batch_size
print(f"Inference time: {inference_time:.2f} ms/example")

# Measure GPU memory consumption
torch.cuda.empty_cache()
log_gpu_memory("after_inference")
gpus = GPUtil.getGPUs()
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0
print(f"GPU memory consumption: {memory_used:.2f} GB")

# Output results for the table
print(f"| Model         | Accuracy | F1-score | Inference Time (ms) | Memory (GB) | Training Time (s) |")
print(f"|---------------|----------|----------|---------------------|-------------|-------------------|")
print(f"| RNN (BiLSTM)  | {accuracy:.4f}   | {f1:.4f}   | {inference_time:.2f}               | {memory_used:.2f}         | {training_time:.2f}         |")

# --- Transfer Learning: Fine-tune on Yelp Polarity ---
print("\nStarting fine-tuning on Yelp Polarity...")
yelp_dataset = load_dataset("yelp_polarity")
encoded_yelp_dataset = yelp_dataset.map(preprocess_function, batched=True)
train_yelp_dataset = encoded_yelp_dataset["train"]
eval_yelp_dataset = encoded_yelp_dataset["test"]

training_args_yelp = TrainingArguments(
    output_dir="./results_rnn_yelp_corrected",
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_rnn_yelp_corrected",
    logging_steps=100,
    fp16=True,
    learning_rate=2e-5,
    load_best_model_at_end=True,
)

trainer_yelp = Trainer(
    model=model,
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
    "lengths": torch.tensor(yelp_input_batch["lengths"]).to(device),
}
start_time = time.time()
with torch.amp.autocast('cuda'):  # Обновлено
    outputs = model(**yelp_inputs)
end_time = time.time()
inference_time_yelp = (end_time - start_time) * 1000 / batch_size
print(f"Inference time (Yelp batch): {inference_time_yelp:.2f} ms/example")
torch.cuda.empty_cache()
log_gpu_memory("after_yelp_inference")
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0
print(f"GPU memory consumption: {memory_used:.2f} GB")

print(f"\nResults table for Yelp fine-tuning evaluation:")
print(f"| Model                 | Accuracy (Yelp) | F1-score (Yelp) | Inference Time (ms) | Memory (GB) | Fine-tuning Time (s) |")
print(f"|-----------------------|-----------------|-----------------|---------------------|-------------|----------------------|")
print(f"| RNN (BiLSTM, fine-tuned) | {accuracy_yelp:.4f}          | {f1_yelp:.4f}          | {inference_time_yelp:.2f}          | {memory_used:.2f}       | {training_time_yelp:.2f}         |")