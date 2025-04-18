#!/bin/bash
#SBATCH --job-name=bert_base_uncased_experiment
#SBATCH --output=../logs/bert_base_uncased_experiment/output.log
#SBATCH --error=../logs/bert_base_uncased_experiment/error.log
#SBATCH --partition=dgx  # Použitie GPU partície
#SBATCH --gres=gpu:1  # Pridelenie jednej GPU
#SBATCH --cpus-per-task=4  # Počet CPU jadier na jednu úlohu
#SBATCH --mem=16G  # Alokovaná pamäť pre úlohu
#SBATCH --time=01:00:00  # Maximálny čas behu (1 hodina)

# Активация conda-окружения
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bert-imdb-env

# Выполнение Python-скрипта
python ../scripts/bert_base_uncased_experiment.py