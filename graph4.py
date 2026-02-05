import json
import matplotlib.pyplot as plt
import numpy as np

def plot_normalized_index(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        experiments = json.load(f)

    names = ['Ref1', 'Ref2', 'Ref3', 'Ref4', 'Ref5', 'Test']
    names = [0.15, 0.3, 0.5, 1.0, 2.0] # Концентрации белков в г/л для референсов
    names = [0.5, 1.5, 4.0, 8.0, 16.0] # Концентрации кетонов в ммоль/л для референсов

    plt.figure(figsize=(10, 6))

    for idx, exp in enumerate(experiments):
        # Функция для расчета индекса R/B (индекс 2 / индекс 0)
        def calc_index(bgr_list):
            # Добавляем 0.1, чтобы избежать деления на ноль, если данные шумные
            #return bgr_list[1] / (bgr_list[0] + bgr_list[1]  + bgr_list[2])
            return 1 / np.sqrt(sum(c ** 2 for c in bgr_list))
            #return np.sqrt(sum(c ** 2 for c in bgr_list))

        # Считаем индексы для референсов и теста
        indices = [calc_index(ref) for ref in exp['reference']] # + [calc_index(exp['test'])]

        label = f"{exp['metadata'].get('angle', 'n/a')}: {exp['metadata'].get('lighting', 'n/a')}"
        plt.plot(names, indices, marker='s', linestyle='--', label=label)

    #plt.title('Нормализованный индекс белка (R / (B + G + R))')
    #plt.ylabel('Значение индекса (Отношение Red/(Blue + Green + Red))')
    # plt.title('Нормализованный индекс белка (1 / (B^2 + G^2 + R^2)^0.5)')
    # plt.ylabel('Значение индекса (Отношение (1 / (B^2 + G^2 + R^2)^0.5))')
    #plt.title('Нормализованный индекс кетонов (R / (B + G + R))')
    #plt.ylabel('Значение индекса (Отношение Red/(Blue + Green + Red))')
    plt.title('Нормализованный индекс кетонов (1 / (B^2 + G^2 + R^2)^0.5)')
    plt.ylabel('Значение индекса (Отношение (1 / (B^2 + G^2 + R^2)^0.5))')
    #     
    plt.xlabel('Зоны (0 = Норма, 16 = Высокие кетоны)')
    #plt.xlable('Зоны (0 = Норма, 2.0 =  Высокие белки)')
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()
    plt.show()

# Запуск
#plot_normalized_index('data_log_protein_rgb.json')
plot_normalized_index('data_log_кетоны_rgb.json')