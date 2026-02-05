import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

def transform_json_to_flat_structure(file_path):
    # 1. Читаем наш лог
    with open(file_path, 'r', encoding='utf-8') as f:
        experiments = json.load(f)

    # 2. Инициализируем структуру (предположим, у нас всегда 5 референсов)
    # Мы используем RMS или какой-то один канал? 
    # В примере ниже я считаю RMS (как в твоем прошлом вопросе), 
    # чтобы получить одно число для каждого измерения.
    
    result = {
        'Test': [],
        'Ref1': [],
        'Ref2': [],
        'Ref3': [],
        'Ref4': [],
        'Ref5': []
    }

    for exp in experiments:
        # Извлекаем тест
        t_bgr = exp['test']
        rms_test = (sum(c**2 for c in t_bgr) / 3)**0.5
        result['Test'].append(round(rms_test, 2))

        # Извлекаем референсы (проходим циклом по списку из 5 элементов)
        for i, ref_bgr in enumerate(exp['reference']):
            ref_key = f'Ref{i+1}'
            if ref_key in result:
                rms_ref = (sum(c**2 for c in ref_bgr) / 3)**0.5
                result[ref_key].append(round(rms_ref, 2))

    return result



print('---')
# 1. Исходные данные для белков
data_rgb_p = {
    'Test': [95.71, 163.47, 179.59, 106.73, 169.85, 209.56],
    'Ref1': [59.85, 106.44, 162.66, 73.13, 95.05, 114.07],
    'Ref2': [59.91, 127.18, 174.53, 76.73, 126.60, 134.06],
    'Ref3': [72.14, 140.66, 167.77, 114.16, 196.27, 303.88],
    'Ref4': [73.46, 138.36, 150.02, 92.90, 144.62, 148.26],
    'Ref5': [69.91, 143.37, 155.11, 77.06, 163.42, 190.65]
}

# --- Использование ---
data_rgb_p = transform_json_to_flat_structure('data_log_protein_rgb.json')
print(data_rgb_p)

# Ваши новые метки измерений
measurement_labels = ['L0Up', 'L05Up', 'L1Up', 'L0Ang', 'L05Ang', 'L1Ang']

df = pd.DataFrame(data_rgb_p)
# Назначаем метки вместо цифр 0-5
df.index = measurement_labels
df.index.name = 'Condition'

# 2. Преобразование данных для построения (из широкого в длинный формат)
df_plot = df.reset_index().melt(id_vars='Condition', var_name='Strip', value_name='RMS')

# Изменение данных на 1/X (где X - RMS)
df_plot['RMS'] = 1 / df_plot['RMS']

# Сопоставление названий колонок с номерами на оси X
strip_map = {'Test': 0, 'Ref1': 1, 'Ref2': 2, 'Ref3': 3, 'Ref4': 4, 'Ref5': 5}
df_plot['Strip_Index'] = df_plot['Strip'].map(strip_map)

# 3. Построение графика
plt.figure(figsize=(11, 6))

# Используем параметр hue='Condition', чтобы легенда подписалась вашими именами
sns.lineplot(data=df_plot, x='Strip_Index', y='RMS', hue='Condition', 
             marker='o', palette='Set2', linewidth=2.5)

plt.title('Яркость полосок при разных условиях освещения и углах')
plt.xlabel('Номер полоски (0 = Test, 1-5 = Reference)')
plt.ylabel('Обратная яркость (1/RMS)')
plt.xticks(range(6))
plt.grid(True, linestyle='--', alpha=0.5)

# Размещаем легенду сбоку, чтобы не мешала
#plt.legend(title='Условия съемки', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(title='Условия съемки', loc='upper left')
plt.tight_layout()

plt.show()