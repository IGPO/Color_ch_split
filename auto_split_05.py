
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline


# -----------------------------
# Параметры (Parameters)
# -----------------------------
A_min_ratio = 0.001        # минимальная площадь квадрата
A_max_ratio = 0.05         # максимальная площадь (чтобы отсечь упаковку)
epsilon_ratio = 0.05       # точность approxPolyDP
side_ratio_thresh = 1.4    # допуск по сторонам (учет перспективы)
angle_cos_thresh = 0.4     # допуск по углам
N = 128                    # размер выпрямленного квадрата
margin_ratio = 0.1         # отступ от краёв для анализа цвета
expected_count = 5         # мы знаем, что квадратов должно быть 5
concentrations = np.array([0.5, 1.5, 4.0, 8.0, 16.0])
# --- 1. Определение цветов (BGR) ---
COLOR_SCALE = (0, 255, 0)   # Зеленый для эталонной шкалы
COLOR_TEST = (255, 165, 0)  # Оранжевый (или синий (255,0,0)) для тестов
TEXT_COLOR = (255, 255, 255) # Белый цвет для текста

# -----------------------------
# Функции (Functions)
# -----------------------------
def create_model_visualization(model, width=800, height=100):
    """Создает изображение-градиент на основе построенной модели."""
    fit_l, fit_a, fit_b = model
    
    # Создаем сетку концентраций от 0.5 до 16.0
    c_grid = np.linspace(concentrations[0], concentrations[-1], width)
    
    # Вычисляем LAB значения для всей сетки
    l_vals = fit_l(c_grid)
    a_vals = fit_a(c_grid)
    b_vals = fit_b(c_grid)
    
    # Формируем массив в формате LAB для OpenCV
    # Важно: ограничиваем значения (clip), чтобы не выйти за пределы 0-255
    lab_gradient = np.zeros((height, width, 3), dtype=np.uint8)
    lab_gradient[:, :, 0] = np.clip(l_vals, 0, 255)
    lab_gradient[:, :, 1] = np.clip(a_vals, 0, 255)
    lab_gradient[:, :, 2] = np.clip(b_vals, 0, 255)
    
    # Переводим обратно в BGR для отображения
    bgr_gradient = cv2.cvtColor(lab_gradient, cv2.COLOR_LAB2BGR)
    
    # Добавляем текстовые метки эталонных точек
    for c in concentrations:
        # Находим X-координату для этой концентрации
        #x = int((c - concentrations[0]) / (concentrations[-1] - concentrations[0]) * (width - 1))
        x = int((c - concentrations[0]) / (concentrations[-1] - concentrations[0]) * (width - 1))
        cv2.line(bgr_gradient, (x, 0), (x, height), (255, 255, 255), 2)
        cv2.putText(bgr_gradient, str(c), (x + 10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(bgr_gradient, "16.0", (width - 50, height - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)    
    return bgr_gradient

def is_square_geometry(approx, side_ratio_thresh=1.3, angle_cos_thresh=0.3):
    pts = approx.reshape(-1, 2)
    if len(pts) < 4: return False
    
    # Для контуров > 4 точек используем ограничивающий прямоугольник
    rect = cv2.minAreaRect(pts)
    (w_r, h_r) = rect[1]
    if min(w_r, h_r) == 0: return False
    aspect_ratio = max(w_r, h_r) / min(w_r, h_r)
    
    if aspect_ratio > side_ratio_thresh:
        return False
    return True

def order_points(pts):
    # pts: (4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_square_from_rect(center, size, angle, img_shape):
    # Создает 4 точки "идеального" квадрата на основе параметров
    cx, cy = center
    w_h = size / 2
    pts = np.array([
        [cx - w_h, cy - w_h],
        [cx + w_h, cy - w_h],
        [cx + w_h, cy + w_h],
        [cx - w_h, cy + w_h]
    ], dtype=np.float32)
    
    # Поворот (если нужно, пока упростим)
    return pts

def is_colored(img_lab, threshold=50):
    # Извлекаем средние значения каналов a и b
    q75, q25 = np.percentile(img_lab[:,:,1], [75 ,25])
    iqr_a = q75 - q25
    q75, q25 = np.percentile(img_lab[:,:,2], [75 ,25])
    iqr_b = q75 - q25
    print("Цвет найден: a IQR =", iqr_a, ", b IQR =", iqr_b)
    # Если отклонение от нейтрального серого (128) маленькое — это шум/штрих-код
    if abs(iqr_a) < threshold and abs(iqr_b) < threshold:        
        return True
    return False

def extract_robust_lab(square_img):
    """Извлекает цвет в LAB, игнорируя блики и тени."""
    lab = cv2.cvtColor(square_img, cv2.COLOR_BGR2LAB)
    
    # Отступаем от краев
    m = int(N * margin_ratio)
    inner = lab[m:-m, m:-m]
    pixels = inner.reshape(-1, 3)
    
    # Фильтрация бликов по каналу L (яркость)
    # Отсекаем верхние 20% (блики) и нижние 5% (шум)
    l_channel = pixels[:, 0]
    lower_b = np.percentile(l_channel, 5)
    upper_b = np.percentile(l_channel, 80)
    
    mask = (l_channel > lower_b) & (l_channel < upper_b)
    robust_pixels = pixels[mask]
    
    if len(robust_pixels) == 0: # Если всё выбито бликом, берем просто медиану
        return np.median(pixels, axis=0)
        
    return np.median(robust_pixels, axis=0)

def build_color_law(scale_squares):
    """Строит математическую модель изменения цвета."""
    raw_labs = np.array([extract_robust_lab(img) for img in scale_squares])
    
    # k=2 для плавности без осцилляций, s=smoothing factor
    # s подбирается экспериментально (чем больше бликов, тем выше s)
    fit_l = UnivariateSpline(concentrations, raw_labs[:, 0], k=2, s=10)
    fit_a = UnivariateSpline(concentrations, raw_labs[:, 1], k=2, s=10)
    fit_b = UnivariateSpline(concentrations, raw_labs[:, 2], k=2, s=10)
    print("Модель цветовой зависимости построена.")
    print(f"Пример предсказания для концентрации 5.0: L={fit_l(5.0):.2f}, a={fit_a(5.0):.2f}, b={fit_b(5.0):.2f}")
    return (fit_l, fit_a, fit_b), raw_labs

def estimate_concentration(test_img, model):
    """Сравнивает цвет полоски с моделью шкалы."""
    test_lab = extract_robust_lab(test_img)
    fit_l, fit_a, fit_b = model
    
    # Создаем плотную сетку значений для поиска
    c_grid = np.linspace(concentrations[0], concentrations[-1], 200)
    
    best_c = c_grid[0]
    min_dist = float('inf')
    
    for c in c_grid:
        pred_lab = np.array([fit_l(c), fit_a(c), fit_b(c)])
        # Delta E (евклидово расстояние в LAB)
        dist = np.linalg.norm(pred_lab - test_lab)
        if dist < min_dist:
            min_dist = dist
            best_c = c
            
    return best_c, test_lab

# -----------------------------
# Загрузка (Loading)
# -----------------------------
# Замените на путь к вашему файлу
image_path = 'кетоны/свет0.5-вид сверху/full.jpg'
img = cv2.imread(image_path)
if img is None:
    # Создаем заглушку если файла нет в текущей директории для теста
    img = np.zeros((600, 800, 3), dtype=np.uint8)

img_draw = img.copy()
h_img, w_img = img.shape[:2]
A_min = h_img * w_img * A_min_ratio
A_max = h_img * w_img * A_max_ratio

# 1. Подготовка маски (HSV Saturation)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s_channel = hsv[:,:,1]
s_thresh = cv2.adaptiveThreshold(s_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 31, -2)

# Морфология
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
mask_closed = cv2.morphologyEx(s_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

# 2. Поиск кандидатов
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
candidates = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < A_min or area > A_max:
        continue
    
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
    
    # Проверяем геометрию (даже если точек > 4 из-за шума)
    if is_square_geometry(approx, side_ratio_thresh):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        center = rect[0]
        side = np.mean(rect[1])
        candidates.append({'center': center, 'side': side, 'box': box, 'area': area})

#print(f"Found {len(candidates)} candidates, filtering down to {candidates[0]} after geometry check.")
#print(candidates)
found_squares = []  # Сохраняем для дальнейшей обработки
for c in candidates:
    found_squares.append(order_points(c['box']))
    #print(f"Candidate at {c['center']} with side {c['side']} has box points: {order_points(c['box'])}")
# 1. Сначала превращаем список найденных контуров в список массивов координат
# (убеждаемся, что работаем именно с координатами точек)
# found_squares должен содержать элементы, полученные через order_points()

if len(found_squares) > 0:
    # Сортируем все найденные объекты по вертикали (Y), чтобы отделить ряды
    # np.mean(s[:, 1]) считает среднюю высоту квадрата
    found_squares.sort(key=lambda s: np.mean(np.array(s)[:, 1]))

    # Берем первые 5 квадратов, которые находятся выше всего (наша шкала)
    # Или используем медиану Y, чтобы найти все объекты на одной линии
    median_y = np.median([np.mean(np.array(s)[:, 1]) for s in found_squares])
    
    # Оставляем только те, что лежат на линии шкалы (отклонение не более 20% высоты квадрата)
    # Это отсечет тестовые полоски, которые лежат ниже
    scale_squares = [s for s in found_squares if abs(np.mean(np.array(s)[:, 1]) - median_y) < N*0.5]

    # Теперь сортируем только шкалу по горизонтали (X) слева направо
    scale_squares.sort(key=lambda s: np.mean(np.array(s)[:, 0]))
    
    # Квадраты тестовых полосок (те, что ниже медианы Y)
    test_strips = [s for s in found_squares if np.mean(np.array(s)[:, 1]) > median_y + N]
    test_strips.sort(key=lambda s: np.mean(np.array(s)[:, 0]))

# 3. Интеллектуальное восстановление ряда
# Сортируем по X
candidates.sort(key=lambda c: c['center'][0])

# Убираем дубликаты или слишком близкие объекты (если есть)
unique_candidates = []
if candidates:
    unique_candidates.append(candidates[0])
    for i in range(1, len(candidates)):
        if np.linalg.norm(np.array(candidates[i]['center']) - np.array(unique_candidates[-1]['center'])) > unique_candidates[-1]['side'] * 0.8:
            unique_candidates.append(candidates[i])

# Если нашли хотя бы 2, можем построить ряд
final_rois = []
if len(unique_candidates) >= 2:
    # Считаем средний Y и средний размер
    avg_y = np.mean([c['center'][1] for c in unique_candidates])
    avg_side = np.mean([c['side'] for c in unique_candidates])
    print(f"Average Y: {avg_y:.2f}, Average side: {avg_side:.2f}")
    # Считаем средний шаг между соседними
    steps = []
    for i in range(len(unique_candidates)-1):
        steps.append(unique_candidates[i+1]['center'][0] - unique_candidates[i]['center'][0])
    
    # Берем медианный шаг (он самый надежный)
    avg_step = np.median(steps) if steps else avg_side * 1.5
    
    # Пытаемся сопоставить найденные с сеткой из 5 элементов
    # Начинаем от самого левого (предполагаем, что это один из 5)
    # Это упрощенная логика: в реальности лучше искать "лучшее покрытие"
    first_center = unique_candidates[0]['center']
    
    # Генерируем 5 центров на основе первого найденного и шага
    # (нужно определить, какой это по счету квадрат - допустим 1-й или 2-й)
    # Для простоты: возьмем все уникальные и достроим недостающие
    
    # В этой версии просто выведем те что нашли, но отсортированные
    for c in unique_candidates:
        pts_src = order_points(c['box'])
        final_rois.append(pts_src)
else:
    # Если нашли мало, просто берем что есть
    for c in unique_candidates:
        final_rois.append(order_points(c['box']))

# -----------------------------

# Функция-помощник для вырезки
def warp_square(source_img, points, size):
    pts_dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(points.astype(np.float32), pts_dst)
    return cv2.warpPerspective(source_img, H, (size, size))

# 1. Вырезаем изображения только для шкалы (их должно быть 5)
scale_imgs = [warp_square(img, pts, N) for pts in scale_squares]

# 2. Вырезаем изображения для тестовых полосок
test_imgs = [warp_square(img, pts, N) for pts in test_strips]

# --- 5. Построение модели и расчет ---
results = []
if len(scale_imgs) == expected_count:
    # Передаем в модель ТОЛЬКО изображения шкалы
    model, labs = build_color_law(scale_imgs)
    print(model, labs)
    print("Модель успешно построена по 5 квадратам шкалы.")
    
    # Если нашли хотя бы одну тестовую полоску, считаем её концентрацию
    if len(test_imgs) > 0:
        for i, t_img in enumerate(test_imgs):
            res_c, t_lab = estimate_concentration(t_img, model)
            if is_colored(t_img):
                results.append(res_c)
            else:
                test_imgs[i] = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY) # для отладки, показать что это не цвет
                results.append(0.0)  # или можно пометить как "неоп
            print(f"Полоска {i+1}: Расчетная концентрация = {res_c:.2f}")
else:
    print(f"Ошибка: Найдено {len(scale_imgs)} квадратов шкалы вместо {expected_count}")

# Сборка финальной картинки

# --- Использование в основном коде ---
if len(scale_imgs) == expected_count:
    model, labs = build_color_law(scale_imgs)
    
    # Создаем картинку модели
    model_vis = create_model_visualization(model)
    #cv2.imshow("Математическая модель шкалы", model_vis)
    cv2.imwrite("model_gradient.jpg", model_vis)
    

# 4. Визуализация и вырезка
square_imgs = []
for pts_src in final_rois:
    pts_dst = np.array([[0,0],[N-1,0],[N-1,N-1],[0,N-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(pts_src, pts_dst)
    square_img = cv2.warpPerspective(img, H, (N,N))
    square_imgs.append(square_img)
    
    # Рисуем на оригинале
# 1. Отрисовка шкалы
for i, pts in enumerate(scale_squares):
    pts_int = pts.astype(int)
    cv2.polylines(img_draw, [pts_int], True, COLOR_SCALE, 3)
    cv2.putText(img_draw, f"Scale {i+1}", (pts_int[0][0], pts_int[0][1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SCALE, 2)

# 2. Отрисовка тестов с результатами
# Предположим, результаты хранятся в списке results = [6.81, 4.55, 0.50]
#results = res_c # подставьте свои переменные из цикла расчета

for i, pts in enumerate(test_strips):
    pts_int = pts.astype(int)
    cv2.polylines(img_draw, [pts_int], True, COLOR_TEST, 3)
    
    # Выводим концентрацию прямо над квадратом
    conc_text = f"TEST {i+1}: {results[i]:.2f}"
    cv2.putText(img_draw, conc_text, (pts_int[0][0], pts_int[0][1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEST, 2)

if square_imgs:
    bottom_row = cv2.hconcat(square_imgs)
    # Resize to match top row width
    bottom_row = cv2.resize(bottom_row, (img_draw.shape[1], N))
    model_vis = cv2.resize(model_vis, (img_draw.shape[1], model_vis.shape[0]))  # Подгоняем модель по ширине
    final_vis = cv2.vconcat([img_draw, bottom_row, model_vis])  # Добавляем модель внизу
    cv2.imshow(f"Square {len(found_squares)}", final_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite('result_improved.jpg', final_vis)
    print(f"Total squares found and processed: {len(final_rois)}")
else:
    cv2.imwrite('result_improved.jpg', img_draw)
    print("No squares found.")
