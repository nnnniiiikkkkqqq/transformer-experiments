# Transformer Optimization Experiments

This repository contains the code and instructions for reproducing experiments on optimizing encoder-only transformer models (BERT, RoBERTa, DistilBERT) for binary sentiment classification, as described in the bachelor thesis. The experiments evaluate the impact of reducing architectural parameters (attention heads, embedding size, hidden layers) on performance and efficiency using the IMDb and Yelp Polarity datasets.

### System Requirements

- To run the experiments, ensure your system meets the following requirements:





- Operating System: Linux with bash shell (required for running experiment scripts and SLURM jobs).



- Hardware: Nvidia A100 GPU with 40 GB VRAM.



- Software:





    - conda for managing Python environments and dependencies.



    - SLURM workload manager for submitting and managing GPU jobs on a cluster.



    - Python 3.8 or higher, along with necessary libraries (specified in the environment file).


# User guide
Repository has following structure, where :

-   Directory ```logs``` contains error and output logs, produced by evaluated ```slurm script```

-   Directory ```scripts``` contains code in format ```.py``` and ```slurm sript``` for launching.
-   Directory ```templates``` containes a template for creating a certain ```slurm script```, when creating new experiment with ```create_experiment.sh```
-   Directory ```yml-files``` contains a yml-file, for setting conda-environment


```├── create_experiment.sh
├── logs
│   └── encoder-only
│       ├── bert_384embedding
│       │   └── output.log
│       ├── bert_base_6heads
│       │   └── output.log
        ....
├── README.md
├── scripts
│   ├── encoder-only
│   │   ├── bert_384embedding.py
        ....
│   └── results
├── templates
│   └── slurm_template.sh
├── word_embedding.py
└── yml-files
    └── environment.yml
```
## Step-by-Step Instructions

### Step 1: Set Up the Conda Environment

To create and activate the conda environment with the required dependencies:

1. Navigate to the repository's root directory:
   ```bash
   cd /path/to/repository
   ```
2. Create the environment using the provided `.yml` file:
   ```bash
   conda env create -f yml-files/environment.yml
   ```
3. Activate the environment:
   ```bash
   conda activate transformer-experiments
   ```

### Step 2: Create a New Experiment

To create a new experiment:

1. From the root directory, run the `create_experiment.sh` script with the appropriate arguments:
   ```bash
   ./create_experiment.sh <type> "<experiment_name>"
   ```
   * `<type>`: Use `1` for encoder-only models, `2` for decoder-only models, or `3` for sequence-to-sequence models.
   * `<experiment_name>`: A descriptive name for the experiment (e.g., `bert_6heads`).
   * Example:
     ```bash
     ./create_experiment.sh 1 "bert_6heads"
     ```
2. This script generates a Python script (e.g., `scripts/encoder-only/bert_6heads.py`) and a corresponding SLURM script (e.g., `scripts/encoder-only/bert_6heads.sh`) based on the `templates/slurm_template.sh`.

### Step 3: Run the Experiment

To execute the experiment on a SLURM-managed cluster:

1. Navigate to the appropriate scripts directory (e.g., for encoder-only models):
   ```bash
   cd scripts/encoder-only
   ```
2. Submit the SLURM job using the generated `.sh` file:
   ```bash
   sbatch <experiment_name>.sh
   ```
Example:
     ```bash
     sbatch bert_6heads.sh
     ```
1. The SLURM job will execute the corresponding Python script and save the output to the `logs/` directory.

### Step 4: View the Results

To check the results of the experiment:

1. Navigate to the `logs/` directory corresponding to your experiment (e.g., `logs/encoder-only/bert_6heads/`):
   ```bash
   cd logs/encoder-only/bert_6heads
   ```
2. Inspect the `output.log` file for training and evaluation metrics, including accuracy, F1-score, inference time, memory usage, and training time:
   ```bash
   cat output.log
   ```
3. Additional results, such as model checkpoints, are saved in the `scripts/results/` directory.


# IMDb and Yelp Dataset Experiments

## IMDb Dataset Experiments

### Baseline Models

| Model           | Accuracy | F1-score | Inference Time (ms) | Memory (GB) | Training Time (s) |
| --------------- | -------- | -------- | ------------------- | ----------- | ----------------- |
| RNN (BiLSTM)    | 0.6468   | 0.6466   | 1.62                | 3.23        | 470.22            |
| RoBERTa-base    | 0.9304   | 0.9303   | 0.79                | 9.62        | 1160.75           |
| BERT-base       | 0.9161   | 0.9158   | 0.84                | 9.45        | 1187.78           |
| DistilBERT-base | 0.9173   | 0.9172   | 0.28                | 5.26        | 609.53            |


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
|RNN (BiLSTM) |0.8957|0.8957|1.67|4.23|2400.21|
|RoBERTa-base |0.9792|0.9792|0.82|9.62|4578.58|
|BERT-base |0.9769|0.9769|0.91|9.45|4675.41|
|DistilBERT-base 0.9738|0.9738|0.29|5.26|2400.17|

### 6-Head Models

|Model|Accuracy|F1-score|Inference Time (ms)|Memory (GB)|Fine-tuning Time (s)|
|---|---|---|---|---|---|
|BERT-6-heads |0.9737|0.9737|0.78|7.31|3795.71|
|DistilBERT-6-heads |0.9694|0.9694|0.44|2.10|1973.77|
|RoBERTa-6-heads |0.9713|0.9713|1.08|7.38|3775.07|

### 384-Embedding Models

|Model|Accuracy|F1-score|Inference Time (ms)|Memory (GB)|Fine-tuning Time (s)|
|---|---|---|---|---|---|
|BERT-384-embedding |0.9359|0.9359|0.77|7.40|3586.41|
|DistilBERT-384-embedding |0.9368|0.9368|0.46|4.31|1984.68|
|RoBERTa-384-embedding |0.9376|0.9375|0.78|7.10|3554.23|
