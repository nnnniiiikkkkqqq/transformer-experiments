#!/bin/bash

# Проверка, передан ли аргумент с именем эксперимента
if [ -z "$1" ]; then
  echo "Ошибка: Укажите имя эксперимента, например: ./create_experiment.sh experiment_1"
  exit 1
fi

EXPERIMENT_NAME=$1

# Создание директорий
mkdir -p scripts
mkdir -p logs/$EXPERIMENT_NAME
mkdir -p results/$EXPERIMENT_NAME

# Копирование шаблона SLURM-скрипта
SLURM_SCRIPT=scripts/$EXPERIMENT_NAME.sh
cp templates/slurm_template.sh $SLURM_SCRIPT

# Замена {EXPERIMENT_NAME} в SLURM-скрипте
sed -i "s/{EXPERIMENT_NAME}/$EXPERIMENT_NAME/g" $SLURM_SCRIPT

# Создание пустого Python-скрипта (опционально)
PYTHON_SCRIPT=scripts/$EXPERIMENT_NAME.py
if [ ! -f "$PYTHON_SCRIPT" ]; then
  cat << EOF > $PYTHON_SCRIPT
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment: $EXPERIMENT_NAME

def main():
    print(f"Running experiment: $EXPERIMENT_NAME")
    # Добавьте ваш код здесь
    pass

if __name__ == "__main__":
    main()
EOF
fi

echo "Эксперимент '$EXPERIMENT_NAME' создан:"
echo "- SLURM-скрипт: $SLURM_SCRIPT"
echo "- Python-скрипт: $PYTHON_SCRIPT"
echo "- Логи: logs/$EXPERIMENT_NAME/"
echo "- Результаты: results/$EXPERIMENT_NAME/"

# Делаем SLURM-скрипт исполняемым
chmod +x $SLURM_SCRIPT