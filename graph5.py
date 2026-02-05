import matplotlib.pyplot as plt
import numpy as np

# Данные из твоей серии L0Up (Exp 0)
# Концентрации с упаковки (г/л)
# Используем 0.01 вместо 0 для корректного отображения на логарифмической шкале
x_concs = [0.01, 0.15, 0.3, 0.5, 1.0, 2.0]

# Примерные значения индекса R/(R+G+B) из твоего Figure_br2 для Exp 0
y_indices = [0.437, 0.442, 0.420, 0.395, 0.385, 0.376]
test_index = 0.315 # Значение твоего теста (явно ниже Ref5)

# Дополнительная кривая (второй набор измерений)
y_indices2 = [0.430, 0.438, 0.425, 0.402, 0.390, 0.382]  # примерные значения для L0Ang

plt.figure(figsize=(10, 6))

# Строим калибровочные кривые
plt.plot(x_concs, y_indices, 'o-', color='teal', label='Калибровка (L0Up)')
plt.plot(x_concs, y_indices2, 's--', color='orange', label='Калибровка (L0Ang)')

# Отмечаем уровень теста
plt.axhline(y=test_index, color='red', linestyle='--', label=f'Тест (Индекс: {test_index})')

# Оформление
#plt.xscale('log') # Логарифмическая шкала лучше всего подходит для хим. концентраций
plt.xticks(x_concs, ['0', '0.15', '0.3', '0.5', '1.0', '2.0'])
plt.xlabel('Концентрация белка (г/л)')
plt.ylabel('Нормированный индекс R / (R+G+B)')
plt.title('Определение концентрации белка по калибровочной кривой')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()

plt.show()