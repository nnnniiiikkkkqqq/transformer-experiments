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
        # Using BERT's vocab size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # Using LSTM as the RNN layer with 6 layers
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,  # Dropout only between layers
            batch_first=True  # Input shape: (batch_size, seq_len, embed_dim)
        )
        # Linear layer for classification
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)  # x2 if bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        # input_ids = [batch size, seq len]
        embedded = self.dropout(self.embedding(input_ids))
        # embedded = [batch size, seq len, embed dim]
        
        # outputs, (hidden, cell) output
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # Concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        if self.rnn.bidirectional:
            # hidden = [num layers * 2, batch size, hid dim] -> get last layer's forward and backward states
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            # hidden = [num layers, batch size, hid dim] -> get last layer's state
            hidden = self.dropout(hidden[-1,:,:])
        
        # hidden = [batch size, hid dim * num directions]
        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]
        
        return {'logits': prediction}  # Returning dict to mimic HF model output structure

# --- Settings ---
tokenizer_name = "bert-base-uncased"
# RNN Model Hyperparameters
EMBEDDING_DIM = 256  # Smaller embedding than BERT's 768
HIDDEN_DIM = 512     # RNN hidden dimension
OUTPUT_DIM = 2       # Binary classification (positive/negative)
N_LAYERS = 6         # Increased to 6 LSTM layers
BIDIRECTIONAL = True
DROPOUT = 0.5

batch_size = 16
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load IMDb dataset ---
dataset = load_dataset("imdb")

# --- Load Tokenizer ---
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
PAD_IDX = tokenizer.pad_token_id  # Important for embedding layer
VOCAB_SIZE = tokenizer.vocab_size  # Get vocab size from the tokenizer

# --- Data Preprocessing ---
def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")

print("Preprocessing dataset...")
encoded_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["test"]

# Ensure datasets output tensors, remove text columns
train_dataset = train_dataset.map(lambda e: {'labels': e['label']}, batched=True)
eval_dataset = eval_dataset.map(lambda e: {'labels': e['label']}, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# --- Model Initialization ---
print("Initializing RNN (6-layer BiLSTM) model...")
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

# --- Training setup ---
training_args = TrainingArguments(
    output_dir="./results_rnn_6layers",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_rnn_6layers",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,  # Consistent with BERT/RoBERTa experiments
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
    max_grad_norm=1.0,  # Gradient clipping for stability with deeper LSTM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# --- Train the model ---
print("Starting RNN (6-layer BiLSTM) training...")
start_train_time = time.time()
trainer.train()
end_train_time = time.time()
training_time = end_train_time - start_train_time
print(f"RNN Training finished. Training time: {training_time:.2f} seconds.")

# --- Evaluate the model ---
print("Evaluating the RNN (6-layer BiLSTM) model...")
eval_results = trainer.evaluate()
accuracy = eval_results["eval_accuracy"]
f1 = eval_results["eval_f1"]
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")

# --- Measure inference time ---
model.eval()
input_batch = eval_dataset.select(range(batch_size))
inputs = {
    "input_ids": input_batch["input_ids"].to(device),
    "attention_mask": input_batch["attention_mask"].to(device),
}

start_time = time.time()
with torch.no_grad():
    if torch.cuda.is_available() and training_args.fp16:
        with autocast():
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)
end_time = time.time()
inference_time = (end_time - start_time) * 1000 / batch_size  # ms per example
print(f"Inference time: {inference_time:.2f} ms/example")

# --- Measure GPU memory consumption ---
gpus = GPUtil.getGPUs() if GPUtil is not None else []
memory_used = gpus[0].memoryUsed / 1024 if gpus else 0  # Convert MB to GB
print(f"GPU memory consumption (at evaluation): {memory_used:.2f} GB")

# --- Output results for the table ---
print(f"\n| Model                 | Accuracy | F1-score | Inference Time (ms) | Memory (GB) | Training Time (s) |")
print(f"|-----------------------|----------|----------|---------------------|-------------|-------------------|")
print(f"| BERT-base             |  ...     |  ...     |  ...                |  ...        |  ...              |")  # Placeholder for BERT results
print(f"| RNN-6-layers (BiLSTM) | {accuracy:.4f}   | {f1:.4f}   | {inference_time:.2f}               | {memory_used:.2f}         | {training_time:.2f}         |")