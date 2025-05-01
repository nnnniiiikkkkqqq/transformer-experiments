#!/bin/bash
#SBATCH --job-name=distilbert-6heads       # Название эксперимента
#SBATCH --output=../../logs/encoder-only/distilbert-6heads/output.log  # Путь к логу вывода
#SBATCH --error=../../logs/encoder-only/distilbert-6heads/error.log    # Путь к логу ошибок
#SBATCH --partition=dgx  # Použitie GPU partície
#SBATCH --gres=gpu:1  # Pridelenie jednej GPU
#SBATCH --cpus-per-task=4  # Počet CPU jadier na jednu úlohu
#SBATCH --mem=16G  # Alokovaná pamäť pre úlohu
#SBATCH --time=10:00:00  # Maximálny čas behu (1 hodina)

# Активация conda-окружения
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bert-imdb-env

# Выполнение Python-скрипта (замените на ваш код)
module load python/3.8 cuda/11.2
nvidia-smi --query-gpu=memory.used --format=csv > memory_log_start.txt
python distilbert-6heads.py
nvidia-smi --query-gpu=memory.used --format=csv > memory_log_end.txt

# Пример: Сохранение результатов
# cp model.pth ../../results/encoder-only/distilbert-6heads/