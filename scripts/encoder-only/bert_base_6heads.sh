#!/bin/bash
#SBATCH --job-name=bert_base_6heads       # Название эксперимента
#SBATCH --output=../../logs/encoder-only/bert_base_6heads/output.log  # Путь к логу вывода
#SBATCH --error=../../logs/encoder-only/bert_base_6heads/error.log    # Путь к логу ошибок
#SBATCH --partition=dgx  # Použitie GPU partície
#SBATCH --gres=gpu:1  # Pridelenie jednej GPU
#SBATCH --cpus-per-task=4  # Počet CPU jadier na jednu úlohu
#SBATCH --mem=16G  # Alokovaná pamäť pre úlohu
#SBATCH --time=10:00:00  # Maximálny čas behu (1 hodina)

# Активация conda-окружения
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bert-imdb-env

# Выполнение Python-скрипта (замените на ваш код)
python bert_base_6heads.py

# Пример: Сохранение результатов
# cp model.pth ../../results/encoder-only/bert_base_6heads/