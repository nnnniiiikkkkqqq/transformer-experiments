import time
import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset # Removed load_metric
import evaluate # Changed from 'import test'
from torch.cuda.amp import autocast
import psutil
import GPUtil

# Settings
model_name = "bert-base-uncased"
batch_size = 16
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load IMDb dataset
dataset = load_dataset("imdb")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Data preprocessing
def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")

encoded_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["test"]

# Metric setup
accuracy_metric = evaluate.load("accuracy") # Changed from test.load
f1_metric = evaluate.load("f1") # Changed from test.load

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    # Use the loaded evaluate objects directly
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"] # Specify average for f1 if needed, e.g., "weighted"
    return {"accuracy": accuracy, "f1": f1}

# Training setup
training_args = TrainingArguments(
    output_dir="./../../results/results_bertbase",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
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
# Retrieve metrics from eval_results instead of recalculating
accuracy = eval_results["eval_accuracy"]
f1 = eval_results["eval_f1"]
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")

# Measure inference time
model.eval()
input_batch = eval_dataset.select(range(batch_size))  # Select a batch
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
print(f"| BERT-base     | {accuracy:.4f}   | {f1:.4f}   | {inference_time:.2f}               | {memory_used:.2f}         | {training_time:.2f}         |")

# --- Transfer Learning: Fine-tune on Yelp Polarity ---
print("\nStarting fine-tuning on Yelp Polarity...")
from datasets import load_dataset as load_dataset_yelp  # To avoid confusion if needed

yelp_dataset = load_dataset_yelp("yelp_polarity")
encoded_yelp_dataset = yelp_dataset.map(preprocess_function, batched=True)
train_yelp_dataset = encoded_yelp_dataset["train"]
eval_yelp_dataset = encoded_yelp_dataset["test"]

training_args_yelp = TrainingArguments(
    output_dir="./results_bertbase_yelp",
    num_train_epochs=1,  # Fewer epochs for fine-tuning
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_bertbase_yelp",
    logging_steps=100,
    fp16=True,
    learning_rate=2e-5,  # Lower learning rate
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
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0
print(f"GPU memory consumption: {memory_used:.2f} GB")

print(f"\nResults table for Yelp fine-tuning evaluation:")
print(f"| Model                 | Accuracy (Yelp) | F1-score (Yelp) | Inference Time (ms) | Memory (GB) | Fine-tuning Time (s) |")
print(f"|-----------------------|-----------------|-----------------|---------------------|-------------|----------------------|")
print(f"| BERT-base (fine-tuned) | {accuracy_yelp:.4f}          | {f1_yelp:.4f}          | {inference_time_yelp:.2f}          | {memory_used:.2f}       | {training_time_yelp:.2f}         |")

# --- Train BERT-base on Yelp from Scratch ---
print("\n---\nStarting training BERT-base on Yelp from scratch...")
model_yelp_scratch = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
training_args_yelp_scratch = TrainingArguments(
    output_dir="./../../results/results_bertbase_yelp_scratch",
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_bertbase_yelp_scratch",
    logging_steps=100,
    fp16=True,
    learning_rate=2e-5,
    load_best_model_at_end=True,
)
trainer_yelp_scratch = Trainer(
    model=model_yelp_scratch,
    args=training_args_yelp_scratch,
    train_dataset=train_yelp_dataset,
    eval_dataset=eval_yelp_dataset,
    compute_metrics=compute_metrics,
)
start_train_time_scratch = time.time()
trainer_yelp_scratch.train()
end_train_time_scratch = time.time()
training_time_yelp_scratch = end_train_time_scratch - start_train_time_scratch
print(f"Training from scratch finished. Training time: {training_time_yelp_scratch:.2f} seconds.")

print("Evaluating the model trained from scratch on Yelp test set...")
eval_results_scratch = trainer_yelp_scratch.evaluate()
accuracy_scratch = eval_results_scratch["eval_accuracy"]
f1_scratch = eval_results_scratch["eval_f1"]
print(f"Accuracy on Yelp (from scratch): {accuracy_scratch:.4f}")
print(f"F1-score on Yelp (from scratch): {f1_scratch:.4f}")

model_yelp_scratch.eval()
yelp_input_batch_scratch = eval_yelp_dataset.select(range(batch_size))
yelp_inputs_scratch = {
    "input_ids": torch.tensor(yelp_input_batch_scratch["input_ids"]).to(device),
    "attention_mask": torch.tensor(yelp_input_batch_scratch["attention_mask"]).to(device),
}
start_time_scratch = time.time()
with torch.no_grad(), autocast():
    outputs_scratch = model_yelp_scratch(**yelp_inputs_scratch)
end_time_scratch = time.time()
inference_time_scratch = (end_time_scratch - start_time_scratch) * 1000 / batch_size
print(f"Inference time (Yelp batch, from scratch): {inference_time_scratch:.2f} ms/example")
print(f"GPU memory consumption (approx.): {memory_used:.2f} GB")

print(f"\nResults table for Yelp training from scratch:")
print(f"| Model                       | Accuracy (Yelp) | F1-score (Yelp) | Inference Time (ms) | Memory (GB) | Training Time (s) |")
print(f"|-----------------------------|-----------------|-----------------|---------------------|-------------|-------------------|")
print(f"| BERT-base (Yelp Scratch) | {accuracy_scratch:.4f}          | {f1_scratch:.4f}          | {inference_time_scratch:.2f}          | {memory_used:.2f}       | {training_time_yelp_scratch:.2f}      |")