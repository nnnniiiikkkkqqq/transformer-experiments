import time
import torch
import torch.nn as nn
from transformers import BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from torch.cuda.amp import autocast
import psutil
import GPUtil
import uuid

# Custom RNN Model
class RNNForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=512, num_layers=2, dropout=0.1, num_labels=2):
        super(RNNForSequenceClassification, self).__init__()
        # Use BERT's embedding layer for fair comparison
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Load pre-trained BERT embeddings if available
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        # Classification head: account for bidirectional (2 * hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get embeddings
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Apply attention mask to zero out padded tokens
        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(-1).float()

        # RNN forward pass
        rnn_output, (hidden, cell) = self.rnn(embedded)  # rnn_output: [batch_size, seq_len, hidden_dim * 2]
        
        # Use the final hidden state from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)  # [batch_size, hidden_dim * 2]
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)  # [batch_size, num_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

# Settings
model_name = "bert-base-uncased"  # For tokenizer and embedding initialization
batch_size = 16
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load IMDb dataset
dataset = load_dataset("imdb")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Initialize model
vocab_size = tokenizer.vocab_size
model = RNNForSequenceClassification(
    vocab_size=vocab_size,
    embedding_dim=768,
    hidden_dim=512,
    num_layers=2,
    dropout=0.1,
    num_labels=2
).to(device)

# Load pre-trained BERT embeddings into the embedding layer
from transformers import BertModel
bert_model = BertModel.from_pretrained(model_name)
model.embedding.weight.data.copy_(bert_model.embeddings.word_embeddings.weight.data)
model.embedding.weight.requires_grad = False  # Freeze embeddings for efficiency

# Data preprocessing
def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")

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
    output_dir="./../../results/results_rnn",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_rnn",
    logging_steps=100,
    fp16=True,  # Mixed precision for NVIDIA A40
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
start_train_time = time.time()
trainer.train()
end_train_time = time.time()
training_time = end_train_time - start_train_time
print(f"Training finished. Training time: {training_time:.2f} seconds.")

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
    output_dir="./results_rnn_yelp",
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_rnn_yelp",
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
}
start_time = time.time()
with torch.no_grad(), autocast():
    outputs = model(**yelp_inputs)
end_time = time.time()
inference_time_yelp = (end_time - start_time) * 1000 / batch_size
print(f"Inference time (Yelp batch): {inference_time_yelp:.2f} ms/example")
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0
print(f"GPU memory consumption: {memory_used:.2f} GB")

print(f"\nResults table for Yelp fine-tuning evaluation:")
print(f"| Model                 | Accuracy (Yelp) | F1-score (Yelp) | Inference Time (ms) | Memory (GB) | Fine-tuning Time (s) |")
print(f"|-----------------------|-----------------|-----------------|---------------------|-------------|----------------------|")
print(f"| RNN (BiLSTM, fine-tuned) | {accuracy_yelp:.4f}          | {f1_yelp:.4f}          | {inference_time_yelp:.2f}          | {memory_used:.2f}       | {training_time_yelp:.2f}         |")