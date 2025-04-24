#!/bin/bash

# Проверка, переданы ли аргументы
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Ошибка: Укажите тип модели (1: encoder-only, 2: decoder-only, 3: sequence-to-sequence) и имя эксперимента."
  echo "Пример: ./create_experiment.sh 1 my_bert_experiment"
  exit 1
fi

MODEL_TYPE_NUM=$1
EXPERIMENT_NAME=$2
MODEL_TYPE_DIR=""

# Определение директории по типу модели
case $MODEL_TYPE_NUM in
  1)
    MODEL_TYPE_DIR="encoder-only"
    ;;
  2)
    MODEL_TYPE_DIR="decoder-only"
    ;;
  3)
    MODEL_TYPE_DIR="sequence-to-sequence"
    ;;
  *)
    echo "Ошибка: Неверный тип модели. Используйте 1, 2 или 3."
    exit 1
    ;;
esac

# Создание директорий
mkdir -p scripts/$MODEL_TYPE_DIR
mkdir -p logs/$MODEL_TYPE_DIR/$EXPERIMENT_NAME
mkdir -p results/$MODEL_TYPE_DIR/$EXPERIMENT_NAME

# Копирование шаблона SLURM-скрипта
SLURM_SCRIPT=scripts/$MODEL_TYPE_DIR/$EXPERIMENT_NAME.sh
cp templates/slurm_template.sh $SLURM_SCRIPT

# Замена плейсхолдеров в SLURM-скрипте
sed -i "s/{EXPERIMENT_NAME}/$EXPERIMENT_NAME/g" $SLURM_SCRIPT
sed -i "s|{MODEL_TYPE_DIR}|$MODEL_TYPE_DIR|g" $SLURM_SCRIPT # Use | as delimiter for sed

# Создание пустого Python-скрипта
PYTHON_SCRIPT=scripts/$MODEL_TYPE_DIR/$EXPERIMENT_NAME.py
if [ ! -f "$PYTHON_SCRIPT" ]; then
  cat << EOF > $PYTHON_SCRIPT
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment: $EXPERIMENT_NAME ($MODEL_TYPE_DIR)

def main():
    print(f"Running experiment: $EXPERIMENT_NAME")
    # Добавьте ваш код здесь
    pass

if __name__ == "__main__":
    main()
EOF
fi

echo "Эксперимент '$EXPERIMENT_NAME' ($MODEL_TYPE_DIR) создан:"
echo "- SLURM-скрипт: $SLURM_SCRIPT"
echo "- Python-скрипт: $PYTHON_SCRIPT"
echo "- Логи: logs/$MODEL_TYPE_DIR/$EXPERIMENT_NAME/"
echo "- Результаты: results/$MODEL_TYPE_DIR/$EXPERIMENT_NAME/"

# Делаем SLURM-скрипт исполняемым
chmod +x $SLURM_SCRIPT