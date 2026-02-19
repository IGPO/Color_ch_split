"""
Концентрационная шкала: автоматическое обнаружение цветных квадратов и оценка концентрации.

Алгоритм:
1. Загрузить изображение
2. Обнаружить референсные квадраты шкалы (большие, сверху) с порогами площади A_MIN_RATIO - A_MAX_RATIO
3. Обнаружить тестовый квадрат (маленький, внизу, в 4 раза меньше) с порогами A_MIN_RATIO_TEST - A_MAX_RATIO_TEST
4. Выпрямить перспективу квадратов
5. Построить сплайн-модель цвета для каждого LAB канала на основе референсных квадратов
6. Оценить концентрацию тестового квадрата
"""

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline


# ============================================================================
# ПАРАМЕТРЫ КОНФИГУРАЦИИ
# ============================================================================

# Геометрия квадратов
SQUARE_SIZE = 128              # размер выпрямленного квадрата (N)
MARGIN_RATIO = 0.1             # доля отступа от краёв при анализе цвета

# Фильтрация контуров
A_MIN_RATIO = 0.003            # минимальная доля площади изображения для референсных (уменьшено)
A_MAX_RATIO = 0.01             # максимальная доля площади (отсечь упаковку)
A_MIN_RATIO_TEST = 0.0001     # минимальная доля для тестового (уменьшено)
A_MAX_RATIO_TEST = 0.03      # максимальная доля для тестового
EPSILON_RATIO = 0.05           # точность аппроксимации contour (approxPolyDP)
SIDE_RATIO_THRESH = 1.4        # допуск по соотношению сторон (перспектива)
ANGLE_COS_THRESH = 0.4         # допуск по углам четырёхугольника

# Концентрационная шкала
EXPECTED_SQUARE_COUNT = 5      # ожидаемое число квадратов в шкале
CONCENTRATIONS = np.array([0.5, 1.5, 4.0, 8.0, 16.0])  # концентрации для каждого

# Сплайн-подгонка
SPLINE_SMOOTHING = 10          # фактор сглаживания (выше = более гладко)

# Цвета визуализации (BGR)
COLOR_SCALE = (0, 255, 0)      # зеленый - эталонная шкала
COLOR_TEST = (255, 165, 0)     # оранжевый - тестовые полоски
COLOR_TEXT = (255, 255, 255)   # белый - текст

# Пути по умолчанию
folder_path = 'кетоны/19_febr/'
file = 'day_light_rot1'
ext = '.jpg'
DEFAULT_IMAGE_PATH = folder_path + file + ext

# ============================================================================
# УТИЛИТЫ РАБОТЫ С ЦВЕТОМ
# ============================================================================

def extract_robust_lab(square_img):
    """Извлекает LAB цвет квадрата, игнорируя блики и тени.
    
    Args:
        square_img: Изображение квадрата (БГР)
        
    Returns:
        np.array: [L, a, b] в диапазоне [0-255]
    """
    lab = cv2.cvtColor(square_img, cv2.COLOR_BGR2LAB)
    
    # Отступаем от краёв
    margin = int(SQUARE_SIZE * MARGIN_RATIO)
    inner = lab[margin:-margin, margin:-margin]
    pixels = inner.reshape(-1, 3)
    
    # Фильтруем блики по каналу L (яркость)
    # Отсекаем верхние 20% (блики) и нижние 5% (тени/шум)
    l_channel = pixels[:, 0]
    lower_percentile = np.percentile(l_channel, 5)
    upper_percentile = np.percentile(l_channel, 80)
    
    mask = (l_channel > lower_percentile) & (l_channel < upper_percentile)
    robust_pixels = pixels[mask]
    
    if len(robust_pixels) == 0:
        # Если все выбито бликом, берём медиану всех пикселей
        return np.median(pixels, axis=0)
    
    return np.median(robust_pixels, axis=0)


def build_color_law(scale_imgs):
    """Строит математическую модель изменения цвета в зависимости от концентрации.
    
    Args:
        scale_imgs: Список из 5 изображений квадратов шкалы
        
    Returns:
        tuple: (model, raw_labs) где model = (fit_l, fit_a, fit_b) - сплайны для LAB
               raw_labs: массив исходных LAB значений
    """
    raw_labs = np.array([extract_robust_lab(img) for img in scale_imgs])
    
    # k=2 для гладкости без осцилляций
    # s - параметр сглаживания (выше = более гладко, но менее точно)
    fit_l = UnivariateSpline(CONCENTRATIONS, raw_labs[:, 0], k=2, s=SPLINE_SMOOTHING)
    fit_a = UnivariateSpline(CONCENTRATIONS, raw_labs[:, 1], k=2, s=SPLINE_SMOOTHING)
    fit_b = UnivariateSpline(CONCENTRATIONS, raw_labs[:, 2], k=2, s=SPLINE_SMOOTHING)
    
    print("✓ Модель цветовой зависимости построена")
    print(f"  Пример: c=5.0 → L={fit_l(5.0):.1f}, a={fit_a(5.0):.1f}, b={fit_b(5.0):.1f}")
    
    return (fit_l, fit_a, fit_b), raw_labs


def estimate_concentration(test_img, model):
    """Оценивает концентрацию тестовой полоски по цвету.
    
    Args:
        test_img: Изображение тестовой полоски
        model: Кортеж (fit_l, fit_a, fit_b) от build_color_law()
        
    Returns:
        tuple: (estimated_concentration, test_lab_color)
    """
    test_lab = extract_robust_lab(test_img)
    fit_l, fit_a, fit_b = model
    
    # Поиск минимума евклидова расстояния в LAB пространстве
    c_grid = np.linspace(CONCENTRATIONS[0], CONCENTRATIONS[-1], 200)
    
    best_c = c_grid[0]
    min_dist = float('inf')
    
    for c in c_grid:
        pred_lab = np.array([fit_l(c), fit_a(c), fit_b(c)])
        dist = np.linalg.norm(pred_lab - test_lab)  # Евклидово расстояние
        if dist < min_dist:
            min_dist = dist
            best_c = c
    
    return best_c, test_lab


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

def create_model_visualization(model, width=800, height=100):
    """Создаёт изображение-градиент на основе построенной модели цвета.
    
    Args:
        model: Кортеж (fit_l, fit_a, fit_b) от build_color_law()
        width: Ширина картинки
        height: Высота картинки
        
    Returns:
        np.ndarray: Изображение градиента в BGR
    """
    fit_l, fit_a, fit_b = model
    
    # Сетка концентраций
    c_grid = np.linspace(CONCENTRATIONS[0], CONCENTRATIONS[-1], width)
    l_vals = fit_l(c_grid)
    a_vals = fit_a(c_grid)
    b_vals = fit_b(c_grid)
    
    # Создаём LAB изображение
    lab_gradient = np.zeros((height, width, 3), dtype=np.uint8)
    lab_gradient[:, :, 0] = np.clip(l_vals, 0, 255)
    lab_gradient[:, :, 1] = np.clip(a_vals, 0, 255)
    lab_gradient[:, :, 2] = np.clip(b_vals, 0, 255)
    
    # Переводим в BGR для отображения
    bgr_gradient = cv2.cvtColor(lab_gradient, cv2.COLOR_LAB2BGR)
    
    # Добавляем маркеры эталонных точек
    for c in CONCENTRATIONS:
        x = int((c - CONCENTRATIONS[0]) / (CONCENTRATIONS[-1] - CONCENTRATIONS[0]) * (width - 1))
        cv2.line(bgr_gradient, (x, 0), (x, height), COLOR_TEXT, 2)
        cv2.putText(bgr_gradient, str(c), (x + 10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    
    return bgr_gradient


# ============================================================================
# ОБНАРУЖЕНИЕ КВАДРАТОВ
# ============================================================================

def is_square_geometry(approx, side_ratio_thresh=1.3):
    """Проверяет, является ли контур квадратом по форме.
    
    Args:
        approx: Аппроксимированный контур (из approxPolyDP)
        side_ratio_thresh: Максимальное соотношение сторон
        
    Returns:
        bool: True если это похоже на квадрат
    """
    pts = approx.reshape(-1, 2)
    if len(pts) < 4:
        return False
    
    rect = cv2.minAreaRect(pts)
    w, h = rect[1]
    if min(w, h) == 0:
        return False
    
    aspect_ratio = max(w, h) / min(w, h)
    return aspect_ratio <= side_ratio_thresh


def order_points(pts):
    """Упорядочивает 4 точки прямоугольника в порядке: TL, TR, BR, BL.
    
    Args:
        pts: массив из 4 точек
        
    Returns:
        np.ndarray: упорядоченные точки
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL - минимальная сумма координат
    rect[2] = pts[np.argmax(s)]  # BR - максимальная сумма координат
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR - минимальная разница
    rect[3] = pts[np.argmax(diff)]  # BL - максимальная разница
    return rect


def warp_square(source_img, points, size):
    """Выпрямляет (приводит к фронтальному виду) перспективный квадрат.
    
    Args:
        source_img: Исходное изображение
        points: 4 точки угла квадрата в исходной системе
        size: Размер выходного квадрата
        
    Returns:
        np.ndarray: Выпрямленный квадрат
    """
    pts_dst = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]], 
                       dtype=np.float32)
    H = cv2.getPerspectiveTransform(points.astype(np.float32), pts_dst)
    return cv2.warpPerspective(source_img, H, (size, size))


def find_squares_in_image(img, a_min_ratio, a_max_ratio):
    """Находит цветные квадраты на изображении с заданными порогами площади.
    
    Алгоритм:
    1. Выделяем цветные области по HSV насыщенности
    2. Применяем морфологию для очистки
    3. Ищем контуры с параметрами квадрата
    
    Args:
        img: Исходное изображение (BGR)
        a_min_ratio: Минимальная доля площади
        a_max_ratio: Максимальная доля площади
        
    Returns:
        list: Список упорядоченных точек (4x2 каждый) для каждого найденного квадрата
    """
    h_img, w_img = img.shape[:2]

    a_min = h_img * w_img * a_min_ratio
    a_max = h_img * w_img * a_max_ratio
    print(f"  Поиск квадратов: a_min_ratio={a_min_ratio:.5f}, a_min={a_min:.5f}")
    print(f"                 : a_max_ratio={a_max_ratio:.5f}, a_max={a_max:.5f}")
    print(f"  Размер изображения: {w_img}x{h_img}, площадь: {h_img * w_img}") 
    # 1. Маска по HSV насыщенности
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    s_thresh = cv2.adaptiveThreshold(s_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, -2)
    
    # 2. Морфологическая очистка
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_closed = cv2.morphologyEx(s_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Поиск контуров
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"    Найдено контуров: {len(contours)}")
    
    # Для каждого контура выводим информацию
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    
    if all_areas:
        all_areas.sort()
        print(f"    Площади всех контуров (min-max): {all_areas[0]:.1f} - {all_areas[-1]:.1f}")
        print(f"    Медиана площадей: {all_areas[len(all_areas)//2]:.1f}")
    
    candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < a_min or area > a_max:
            continue
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPSILON_RATIO * peri, True)
        
        if is_square_geometry(approx, SIDE_RATIO_THRESH):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            candidates.append({
                'center': rect[0],
                'side': np.mean(rect[1]),
                'box': box,
                'area': area
            })
            print(f"      Найден квадрат, площадь: {area}")
    
    print(f"    Кандидатов-квадратов: {len(candidates)}")
    
    # Преобразуем в упорядоченные точки
    found_squares = [order_points(c['box']) for c in candidates]
    return found_squares


def find_reference_squares(img):
    """Находит референсные квадраты шкалы (большие, сверху, на одном уровне)."""
    all_squares = find_squares_in_image(img, A_MIN_RATIO, A_MAX_RATIO)
    
    # Фильтруем по положению: только верхняя половина изображения
    h_img = img.shape[0]
    top_squares = [s for s in all_squares if np.mean(s[:, 1]) < h_img * 0.5]
    
    if len(top_squares) < EXPECTED_SQUARE_COUNT:
        # Если меньше 5, возвращаем все
        top_squares.sort(key=lambda s: np.mean(s[:, 0]))
        return top_squares
    
    # Вычисляем средние Y-координаты
    y_means = [np.mean(s[:, 1]) for s in top_squares]
    median_y = np.median(y_means)
    
    # Отсеиваем квадраты, которые сильно отклоняются от медианной Y (более чем на 10% высоты квадрата)
    tolerance = SQUARE_SIZE * 0.1
    level_squares = [s for s, y in zip(top_squares, y_means) if abs(y - median_y) < tolerance]
    
    # Сортируем по площади (убыванию) и берём топ EXPECTED_SQUARE_COUNT
    level_squares_with_area = []
    for s in level_squares:
        # Приблизительная площадь как среднее расстояние между точками
        side1 = np.linalg.norm(s[0] - s[1])
        side2 = np.linalg.norm(s[1] - s[2])
        area = side1 * side2
        level_squares_with_area.append((area, s))
    
    level_squares_with_area.sort(key=lambda x: x[0], reverse=True)
    selected_squares = [s for _, s in level_squares_with_area[:EXPECTED_SQUARE_COUNT]]
    
    # Сортируем по горизонтали (слева направо)
    selected_squares.sort(key=lambda s: np.mean(s[:, 0]))
    
    return selected_squares


def find_test_square(img, reference_colors=None):
    """Находит тестовый квадрат (маленький, внизу, из пула референсных цветов, однородный).
    
    Returns:
        tuple: (selected_squares, all_candidates) - выбранный и все кандидаты
    """
    all_squares = find_squares_in_image(img, A_MIN_RATIO_TEST, A_MAX_RATIO_TEST)
    
    # Фильтруем по положению: только нижняя половина изображения
    h_img = img.shape[0]
    w_img = img.shape[1]
    bottom_squares = [s for s in all_squares if np.mean(s[:, 1]) > h_img * 0.5]
    
    # Опционально: ограничение по горизонтали (например, для правого угла)
    # Раскомментируйте для фильтрации: bottom_squares = [s for s in bottom_squares if np.mean(s[:, 0]) > w_img * 0.5]
    
    if not bottom_squares:
        print("  Тестовый квадрат не найден в нижней половине изображения")
        return [], []
    
    valid_squares = []
    for s in bottom_squares:
        # Выпрямляем и извлекаем цвет
        test_img_warped = warp_square(img, s, SQUARE_SIZE)
        test_lab = extract_robust_lab(test_img_warped)
        
        # Проверяем однородность: дисперсия L канала после фильтрации
        lab_full = cv2.cvtColor(test_img_warped, cv2.COLOR_BGR2LAB)
        margin = int(SQUARE_SIZE * MARGIN_RATIO)
        inner = lab_full[margin:-margin, margin:-margin]
        l_channel = inner[:, :, 0].flatten()
        print(f"  Тестовый кандидат: средний L={np.mean(l_channel):.1f}, std L={np.std(l_channel):.1f}")
        l_filtered = l_channel[(l_channel > np.percentile(l_channel, 5)) & (l_channel < np.percentile(l_channel, 80))]
        if len(l_filtered) > 0:
            l_std = np.std(l_filtered)
            print(f"  Тестовый кандидат: L_std={l_std:.2f}")
            # Если дисперсия L > 60, считаем неоднородным
            if l_std > 60:
                print("    Отсеян по однородности")
                continue
        
        # Если есть референсные цвета, фильтруем по близости цвета
        # Временно убрана проверка цвета для диагностики
        # if reference_colors is not None and len(reference_colors) > 0:
        #     min_dist = float('inf')
        #     for ref_lab in reference_colors:
        #         dist = np.linalg.norm(test_lab - ref_lab)
        #         min_dist = min(min_dist, dist)
        #     
        #     print(f"  Тестовый кандидат: min_dist={min_dist:.2f}")
        #     # Если расстояние больше порога, пропускаем
        #     if min_dist >= 20:
        #         print("    Отсеян по цвету")
        #         continue
        
        valid_squares.append(s)
    
    # Если несколько, берём самый нижний
    if len(valid_squares) > 1:
        valid_squares.sort(key=lambda s: np.mean(s[:, 1]), reverse=True)
    
    return valid_squares[:1], bottom_squares  # Возвращаем выбранный и всех кандидатов


def separate_scale_and_tests(found_squares):
    """Устаревшая функция - теперь поиск разделён на отдельные функции."""
    # Эта функция больше не используется, но оставлена для совместимости
    return [], []


# ============================================================================
# ОСНОВНОЙ КОНВЕЙЕР (MAIN PIPELINE)
# ============================================================================

def process_image(image_path):
    """Полный конвейер обработки: обнаружение, разделение, расчёт.
    
    Args:
        image_path: Путь к изображению
        
    Returns:
        dict: Результаты с ключами:
            - 'scale_squares': список упорядоченных точек для шкалы
            - 'test_strips': список упорядоченных точек для тестов
            - 'scale_imgs': список выпрямленных изображений шкалы
            - 'test_imgs': список выпрямленных изображений тестов
            - 'model': модель цвета (или None)
            - 'raw_labs': исходные LAB значения шкалы (или None)
            - 'results': список оценённых концентраций
    """
    results = {
        'scale_squares': [],
        'test_strips': [],
        'test_candidates': [],  # Все кандидаты на тестовый квадрат
        'scale_imgs': [],
        'test_imgs': [],
        'model': None,
        'raw_labs': None,
        'results': []
    }
    
    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠ Не удалось загрузить {image_path}")
        return results
    
    print(f"✓ Загружено изображение {image_path}")
    
    # 1. Обнаруживаем референсные квадраты шкалы
    scale_squares = find_reference_squares(img)
    print(f"✓ Найдено {len(scale_squares)} референсных квадратов")
    if len(scale_squares) > 0:
        areas = [np.prod([np.linalg.norm(s[i] - s[(i+1)%4]) for i in range(4)]) / 4 for s in scale_squares]  # approximate area
        print(f"  Площади референсных: {[f'{a:.0f}' for a in areas]}")
    
    # Выпрямляем референсные квадраты для извлечения цветов
    scale_imgs_temp = [warp_square(img, pts, SQUARE_SIZE) for pts in scale_squares]
    reference_colors = []
    for img_sq in scale_imgs_temp:
        lab = extract_robust_lab(img_sq)
        reference_colors.append(lab)
        
        # Отладка однородности референсных
        lab_full = cv2.cvtColor(img_sq, cv2.COLOR_BGR2LAB)
        margin = int(SQUARE_SIZE * MARGIN_RATIO)
        inner = lab_full[margin:-margin, margin:-margin]
        l_channel = inner[:, :, 0].flatten()
        l_filtered = l_channel[(l_channel > np.percentile(l_channel, 5)) & (l_channel < np.percentile(l_channel, 80))]
        if len(l_filtered) > 0:
            l_std = np.std(l_filtered)
            print(f"  Референсный: L_std={l_std:.2f}")
    
    # 2. Обнаруживаем тестовый квадрат (с проверкой цвета)
    test_strips, test_candidates = find_test_square(img, reference_colors)
    print(f"✓ Найдено {len(test_strips)} тестовых квадратов")
    print(f"  Всего кандидатов: {len(test_candidates)}")
    if len(test_strips) > 0:
        areas_test = [np.prod([np.linalg.norm(s[i] - s[(i+1)%4]) for i in range(4)]) / 4 for s in test_strips]
        print(f"  Площади тестовых: {[f'{a:.0f}' for a in areas_test]}")
    else:
        print("  Тестовый квадрат не найден или не прошел проверки цвета/однородности")
    
    results['test_candidates'] = test_candidates
    
    if len(scale_squares) == 0:
        print("⚠ Референсные квадраты не найдены")
        return results
    
    results['scale_squares'] = scale_squares
    results['test_strips'] = test_strips
    
    # 3. Выпрямляем изображения
    scale_imgs = [warp_square(img, pts, SQUARE_SIZE) for pts in scale_squares]
    test_imgs = [warp_square(img, pts, SQUARE_SIZE) for pts in test_strips]
    
    results['scale_imgs'] = scale_imgs
    results['test_imgs'] = test_imgs
    
    # 4. Строим модель (если достаточно данных)
    if len(scale_imgs) == EXPECTED_SQUARE_COUNT:
        try:
            model, raw_labs = build_color_law(scale_imgs)
            results['model'] = model
            results['raw_labs'] = raw_labs
        except Exception as e:
            print(f"⚠ Ошибка построения модели: {e}")
            return results
    else:
        print(f"⚠ Найдено {len(scale_imgs)} квадратов, ожидалось {EXPECTED_SQUARE_COUNT}")
        return results
    
    # 5. Оцениваем концентрацию для тестов
    test_results = []
    if len(test_imgs) > 0:
        for i, test_img in enumerate(test_imgs):
            c_est, test_lab = estimate_concentration(test_img, model)
            test_results.append(c_est)
            print(f"  Тест {i+1}: концентрация = {c_est:.2f}")
    
    results['results'] = test_results
    return results


def visualize_results(img, results):
    """Визуализирует результаты обнаружения и оценки.
    
    Args:
        img: Исходное изображение (BGR)
        results: Результаты от process_image()
        
    Returns:
        np.ndarray: Визуализированное изображение
    """
    img_draw = img.copy()
    
    # Отрисовка шкалы (зеленая)
    for i, pts in enumerate(results['scale_squares']):
        pts_int = pts.astype(int)
        cv2.polylines(img_draw, [pts_int], True, COLOR_SCALE, 3)
        cv2.putText(img_draw, f"Scale {i+1}", (pts_int[0][0], pts_int[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SCALE, 2)
    
    # Отрисовка кандидатов на тестовый квадрат (синий)
    for i, pts in enumerate(results.get('test_candidates', [])):
        pts_int = pts.astype(int)
        # Жирная синяя рамка для всех кандидатов
        cv2.polylines(img_draw, [pts_int], True, (255, 0, 0), 2)
        cv2.putText(img_draw, f"Cand {i+1}", (pts_int[0][0], pts_int[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Отрисовка ВЫБРАННЫХ тестов (ярко-оранжевая) с результатами - ПОВЕРХ кандидатов
    for i, pts in enumerate(results['test_strips']):
        pts_int = pts.astype(int)
        # ТОЛСТАЯ ярко-оранжевая рамка для выбранного
        cv2.polylines(img_draw, [pts_int], True, COLOR_TEST, 5)
        cv2.rectangle(img_draw, pts_int[0], pts_int[2], COLOR_TEST, 5)
        
        if i < len(results['results']) and results['results'][i] is not None:
            conc_text = f"✓ TEST {i+1}: {results['results'][i]:.2f}"
        else:
            conc_text = f"✓ TEST {i+1}: No data"
        
        cv2.putText(img_draw, conc_text, (pts_int[0][0], pts_int[0][1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEST, 3)

    
    return img_draw


def create_composite_visualization(img, results):
    """Собирает полную визуализацию: оригинал + выпрямленные + модель.
    
    Args:
        img: Исходное изображение (BGR)
        results: Результаты от process_image()
        
    Returns:
        np.ndarray или None: Составное изображение
    """
    if len(results['scale_imgs']) == 0:
        return None
    
    # Верхняя часть: аннотированное исходное изображение
    img_draw = visualize_results(img, results)
    
    # Средняя часть: выпрямленные квадраты
    square_imgs = results['scale_imgs'] + results['test_imgs']
    if square_imgs:
        bottom_row = cv2.hconcat(square_imgs)
        bottom_row = cv2.resize(bottom_row, (img_draw.shape[1], SQUARE_SIZE))
    else:
        return img_draw
    
    # Нижняя часть: модель цвета (если есть)
    composite = cv2.vconcat([img_draw, bottom_row])
    
    if results['model'] is not None:
        model_vis = create_model_visualization(results['model'])
        model_vis = cv2.resize(model_vis, (img_draw.shape[1], model_vis.shape[0]))
        composite = cv2.vconcat([composite, model_vis])
    
    return composite

# ============================================================================
# ТОЧКА ВХОДА (ENTRY POINT)
# ============================================================================

if __name__ == '__main__':
    """
    Основной конвейер обработки:
    1. Загружает изображение
    2. Обнаруживает и разделяет квадраты (шкала + тесты)
    3. Выпрямляет перспективу
    4. Строит модель цвета и оценивает концентрацию
    5. Визуализирует результаты
    """
    
    print("=" * 70)
    print("АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ КОНЦЕНТРАЦИИ ПО ЦВЕТУ")
    print("=" * 70)
    
    # Обработка изображения
    results = process_image(DEFAULT_IMAGE_PATH)
    
    # Загружаем исходное изображение для визуализации
    img = cv2.imread(DEFAULT_IMAGE_PATH)
    if img is None:
        print("\n⚠ ОШИБКА: Не удалось загрузить изображение")
        print(f"  Путь: {DEFAULT_IMAGE_PATH}")
        exit(1)
    
    # Создаём составную визуализацию
    composite = create_composite_visualization(img, results)
    
    if composite is None:
        print("\n⚠ ОШИБКА: Не удалось создать визуализацию (нет квадратов)")
        exit(1)
    
    # Выводим результаты
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 70)
    print(f"  Найдено квадратов шкалы: {len(results['scale_squares'])}")
    print(f"  Найдено тестовых полосок: {len(results['test_strips'])}")
    
    if results['model'] is not None:
        print(f"  ✓ Модель успешно построена")
        print(f"\n  Оценённые концентрации:")
        for i, conc in enumerate(results['results'], 1):
            if conc is not None:
                print(f"    Полоска {i}: {conc:.2f}")
            else:
                print(f"    Полоска {i}: не определена")
    else:
        print(f"  ⚠ Модель не построена (недостаточно квадратов шкалы)")
    
    # Сохраняем результат
    output_path = folder_path + "proc/" + file + '_proc.jpg'
    cv2.imwrite(output_path, composite)
    print(f"\n✓ Результат сохранён: {output_path}")
    
    # Показываем результат (окно закроется при нажатии любой клавиши)
    print("\nОкно с результатом откроется. Нажмите любую клавишу для закрытия...")
    cv2.imshow('Результат обработки', composite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("✓ ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 70)