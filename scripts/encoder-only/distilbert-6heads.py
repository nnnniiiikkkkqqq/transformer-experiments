import time
import torch
import subprocess
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from torch.cuda.amp import autocast
import psutil
import GPUtil

# Settings
model_name = "distilbert-base-uncased"
batch_size = 16
num_epochs = 3
num_attention_heads = 6  # Reduced from 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to log GPU memory
def log_gpu_memory(step_name):
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv'], capture_output=True, text=True)
    print(f"GPU Memory at {step_name}:\n{result.stdout}")

# Clear GPU memory before starting
torch.cuda.empty_cache()
log_gpu_memory("before_start")

# Load IMDb dataset
dataset = load_dataset("imdb")

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Configure model with reduced attention heads
config = DistilBertConfig.from_pretrained(model_name, num_attention_heads=num_attention_heads, num_labels=2)
model = DistilBertForSequenceClassification.from_pretrained(model_name, config=config).to(device)

# Verify model configuration
print(f"Model config: {model.config}")

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
    output_dir="./../../results/results_distilbert_6heads",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_distilbert_6heads",
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
}
start_time = time.time()
with torch.no_grad(), autocast():
    outputs = model(**inputs)
end_time = time.time()
inference_time = (end_time - start_time) * 1000 / batch_size  # ms per example
print(f"Inference time: {inference_time:.2f} ms/example")

# Measure GPU memory consumption
torch.cuda.empty_cache()  # Clear memory before measurement
log_gpu_memory("after_inference")
gpus = GPUtil.getGPUs()
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0  # Convert to GB
print(f"GPU memory consumption: {memory_used:.2f} GB")

# Output results for the table
print("\nResults for the IMDb table:")
print(f"| Model                     | Accuracy | F1-score | Inference Time (ms) | Memory (GB) | Training Time (s) |")
print(f"|---------------------------|----------|----------|---------------------|-------------|-------------------|")
print(f"| DistilBERT-6-heads        | {accuracy:.4f}   | {f1:.4f}   | {inference_time:.2f}               | {memory_used:.2f}         | {training_time:.2f}         |")

# --- Transfer Learning: Fine-tune on Yelp Polarity ---
print("\nStarting fine-tuning on Yelp Polarity...")
yelp_dataset = load_dataset("yelp_polarity")
encoded_yelp_dataset = yelp_dataset.map(preprocess_function, batched=True)
train_yelp_dataset = encoded_yelp_dataset["train"]
eval_yelp_dataset = encoded_yelp_dataset["test"]

training_args_yelp = TrainingArguments(
    output_dir="./results_distilbert_6heads_yelp",
    num_train_epochs=1,  # Fewer epochs for fine-tuning
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_distilbert_6heads_yelp",
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
torch.cuda.empty_cache()  # Clear memory before measurement
log_gpu_memory("after_yelp_inference")
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0
print(f"GPU memory consumption: {memory_used:.2f} GB")

print(f"\nResults table for Yelp fine-tuning evaluation:")
print(f"| Model                          | Accuracy (Yelp) | F1-score (Yelp) | Inference Time (ms) | Memory (GB) | Fine-tuning Time (s) |")
print(f"|-------------------------------|-----------------|-----------------|---------------------|-------------|----------------------|")
print(f"| DistilBERT-6-heads (fine-tuned) | {accuracy_yelp:.4f}          | {f1_yelp:.4f}          | {inference_time_yelp:.2f}          | {memory_used:.2f}       | {training_time_yelp:.2f}         |")