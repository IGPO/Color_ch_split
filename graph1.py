import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Создаем пример данных (имитация ваших 6 измерений)
data = []

# Test and reference distancies in RGB space for proteins
name = ['Test', 'Ref1', 'Ref2', 'Ref3', 'Ref4', 'Ref5']
L0Up = [0, 59.85295932777454, 59.90671703205755, 72.13597844426256, 73.46342881454733, 69.91364677184471]
L05Up = [0, 106.43765860038121, 127.17650506417704, 140.65635374623795, 138.36076269731066, 143.3715526606556]
L1Up = [0, 162.65635606302234, 174.52663312115595, 167.77183337427505, 150.01831139801848, 155.1078753905787]
L0Ang = [0, 73.12923525237458, 76.73489067646665, 114.15713187989518, 92.90228300259854, 77.06372952953153]
L05Ang = [0, 95.04984842161078, 126.5978819560344, 196.26757831006776, 144.61973895750512, 163.41548936178046]
L1Ang = [0, 114.06509411494349, 134.06184305694782, 303.88094174327574, 148.25989402813715, 190.65252834218452]

# Test BGR values of proteins
RGB_p = ([47.66371812, 55.06707946, 62.09789179],
        [87.62119032, 104.68920863, 89.92164814],
        [101.43202052, 118.94648157, 88.40018108],
        [51.96665217, 63.47426417, 68.28389155],
        [93.52502828, 106.09799208, 94.0436934],
        [120.52722265, 134.44071746, 106.37094838])
test = [np.sqrt(r**2 + g**2 + b**2) for r, g, b in RGB_p]
print("Test BGR RMS values:", test)
print(np.sqrt(28**2 + 46**2 + 49**2))
df = pd.DataFrame([L0Up, L05Up, L1Up, L0Ang, L05Ang, L1Ang], columns=name)
df["Test"] = test
print(df)
    # Уровни концентрации (Strip_ID): 0 - тест, 1-5 - референс
strip_ids = [0, 1, 2, 3, 4, 5]
# Имитируем 6 разных условий съемки (lighting conditions)
for m in range(1, 7):
    for s_id in strip_ids:
        # Генерируем случайные R, G, B, которые немного "шумят" от измерения к измерению
        # Для теста (0) заложим стабильное значение, для референсов - нарастающее
        base_val = 100 if s_id == 0 else s_id * 40 
        r = base_val + np.random.normal(0, 5)
        g = base_val * 0.8 + np.random.normal(0, 5)
        b = base_val * 0.6 + np.random.normal(0, 5)
        
        # Считаем RMS (яркость)
        rms = np.sqrt(r**2 + g**2 + b**2)
        
        data.append({
            'Measurement': m,
            'Strip_ID': s_id,
            'RMS': rms,
            'Type': 'Test' if s_id == 0 else 'Reference'
        })

df = pd.DataFrame(data)

# 2. Построение графика
plt.figure(figsize=(10, 6))

# Рисуем все точки (измерения), чтобы видеть разброс
sns.stripplot(data=df, x='Strip_ID', y='RMS', hue='Type', dodge=False, alpha=0.5, palette='viridis')

# Рисуем линию со средними значениями и доверительным интервалом (ошибка освещения)
sns.pointplot(data=df, x='Strip_ID', y='RMS', capsize=.1, errorbar='sd', color='black', markers='d')

plt.title('Зависимость яркости (RMS) от номера полоски\n(6 разных условий освещения)')
plt.xlabel('Номер референсной полоски (0 = Тестовая)')
plt.ylabel('Величина яркости (RMS)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Тип полоски')

plt.savefig('ketone_chart.png')