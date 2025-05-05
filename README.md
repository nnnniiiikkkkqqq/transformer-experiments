# IMDb and Yelp Dataset Experiments

## IMDb Dataset Experiments

### Baseline Models

| Model           | Accuracy | F1-score | Inference Time (ms) | Memory (GB) | Training Time (s) |
| --------------- | -------- | -------- | ------------------- | ----------- | ----------------- |
| RoBERTa-base    | 0.9304   | 0.9303   | 0.79                | 9.62        | 1160.75           |
| BERT-base       | 0.9161   | 0.9158   | 0.84                | 9.45        | 1187.78           |
| DistilBERT-base | 0.9173   | 0.9172   | 0.28                | 5.26        | 609.53            |
| RNN (BiLSTM)    | 0.6468   | 0.6466   | 1.62                | 3.23        | 470.22            |

### 6-Head Models

| Model              | Accuracy | F1-score | Inference Time (ms) | Memory (GB) | Training Time (s) |
| ------------------ | -------- | -------- | ------------------- | ----------- | ----------------- |
| BERT-6-heads       | 0.9214   | 0.9213   | 0.79                | 7.31        | 1016.64           |
| DistilBERT-6-heads | 0.8901   | 0.8897   | 0.41                | 2.10        | 532.58            |
| RoBERTa-6-heads    | 0.8989   | 0.8988   | 0.79                | 7.38        | 1004.12           |

### 384-Embedding Models

| Model                    | Accuracy | F1-score | Inference Time (ms) | Memory (GB) | Training Time (s) |
| ------------------------ | -------- | -------- | ------------------- | ----------- | ----------------- |
| BERT-384-embedding       | 0.8581   | 0.8577   | 0.75                | 7.40        | 751.79            |
| DistilBERT-384-embedding | 0.8647   | 0.8642   | 0.44                | 4.31        | 442.08            |
| RoBERTa-384-embedding    | 0.8653   | 0.8652   | 0.77                | 7.10        | 737.69            |

***

## Yelp Dataset Experiments

### Baseline Models

|Model|Accuracy|F1-score|Inference Time (ms)|Memory (GB)|Fine-tuning Time (s)|
|---|---|---|---|---|---|
|RNN (BiLSTM, fine-tuned)|0.8957|0.8957|1.67|4.23|2400.21|
|RoBERTa-base (fine-tuned)|0.9792|0.9792|0.82|9.62|4578.58|
|BERT-base (fine-tuned)|0.9769|0.9769|0.91|9.45|4675.41|
|DistilBERT-base (fine-tuned)|0.9738|0.9738|0.29|5.26|2400.17|

### 6-Head Models

|Model|Accuracy|F1-score|Inference Time (ms)|Memory (GB)|Fine-tuning Time (s)|
|---|---|---|---|---|---|
|BERT-6-heads (fine-tuned)|0.9737|0.9737|0.78|7.31|3795.71|
|DistilBERT-6-heads (fine-tuned)|0.9694|0.9694|0.44|2.10|1973.77|
|RoBERTa-6-heads (fine-tuned)|0.9713|0.9713|1.08|7.38|3775.07|

### 384-Embedding Models

|Model|Accuracy|F1-score|Inference Time (ms)|Memory (GB)|Fine-tuning Time (s)|
|---|---|---|---|---|---|
|BERT-384-embedding (fine-tuned)|0.9359|0.9359|0.77|7.40|3586.41|
|DistilBERT-384-embedding (fine-tuned)|0.9368|0.9368|0.46|4.31|1984.68|
|RoBERTa-384-embedding (fine-tuned)|0.9376|0.9375|0.78|7.10|3554.23|
