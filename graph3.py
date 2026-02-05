import json
import matplotlib.pyplot as plt
import numpy as np

def plot_protein_analysis(file_path):
    # 1. Загрузка данных
    with open(file_path, 'r', encoding='utf-8') as f:
        experiments = json.load(f)

    if not experiments:
        print("Файл пуст!")
        return

    # Подготовка данных для графиков
    # Индексы BGR: 0=Blue, 1=Green, 2=Red
    channels = [('Red', 2, 'red'), ('Green', 1, 'green'), ('Blue', 0, 'blue')]
    names = ['Ref1', 'Ref2', 'Ref3', 'Ref4', 'Ref5', 'Test']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    #fig.suptitle('Сравнение каналов BGR для анализа белка (разные условия)', fontsize=16)
    fig.suptitle('Сравнение каналов BGR для анализа кетонов (разные условия)', fontsize=16)

    for i, (ch_name, ch_idx, color) in enumerate(channels):
        ax = axes[i]
        
        # Для каждого эксперимента в логе рисуем свою линию
        for idx, exp in enumerate(experiments):
            # Собираем значения: 5 референсов + 1 тест
            values = [ref[ch_idx] for ref in exp['reference']] + [exp['test'][ch_idx]]
            
            label = f"{exp['metadata'].get('angle', 'n/a')} : {exp['metadata'].get('lighting', 'n/a')}"
            ax.plot(names, values, marker='o', label=label, alpha=0.7)
        
        ax.set_title(f'Канал: {ch_name}')
        ax.set_ylabel('Интенсивность (0-255)')
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 2: # Добавим легенду только к последнему графику
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Запуск ---
# Убедись, что файл 'data_log.json' лежит в той же папке
#plot_protein_analysis('data_log_кетоны_lab.json')
plot_protein_analysis('data_log_кетоны_rgb.json')