"""
Концентрационная шкала: автоматическое обнаружение цветных квадратов и оценка концентрации.

Алгоритм:
1. Загрузить изображение
2. Выделить цветные квадраты по маске насыщенности (HSV)
3. Разделить на шкалу (5 эталонных квадратов) и тесты (полоски ниже)
4. Построить сплайн-модель цвета для каждого LAB канала
5. Оценить концентрацию тестовых полосок
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
A_MIN_RATIO = 0.001            # минимальная доля площади изображения
A_MAX_RATIO = 0.05             # максимальная доля площади (отсечь упаковку)
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
DEFAULT_IMAGE_PATH = 'кетоны/свет0.5-вид сверху/full.jpg'


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


def find_squares_in_image(img):
    """Находит все цветные квадраты на изображении.
    
    Алгоритм:
    1. Выделяем цветные области по HSV насыщенности
    2. Применяем морфологию для очистки
    3. Ищем контуры с параметрами квадрата
    
    Args:
        img: Исходное изображение (BGR)
        
    Returns:
        list: Список упорядоченных точек (4x2 каждый) для каждого найденного квадрата
    """
    h_img, w_img = img.shape[:2]
    a_min = h_img * w_img * A_MIN_RATIO
    a_max = h_img * w_img * A_MAX_RATIO
    
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
    
    # Преобразуем в упорядоченные точки
    found_squares = [order_points(c['box']) for c in candidates]
    return found_squares


def separate_scale_and_tests(found_squares):
    """Разделяет найденные квадраты на шкалу (сверху) и тесты (снизу).
    
    Args:
        found_squares: Список упорядоченных точек квадратов
        
    Returns:
        tuple: (scale_squares, test_strips) - два списка упорядоченных точек
    """
    if len(found_squares) == 0:
        return [], []
    
    # Сортируем по вертикали
    found_squares_sorted = sorted(found_squares, key=lambda s: np.mean(s[:, 1]))
    
    # Находим медианную Y координату
    median_y = np.median([np.mean(s[:, 1]) for s in found_squares_sorted])
    
    # Разделяем: шкала выше медианы, тесты ниже
    scale_squares = [s for s in found_squares_sorted 
                     if abs(np.mean(s[:, 1]) - median_y) < SQUARE_SIZE * 0.5]
    test_strips = [s for s in found_squares_sorted 
                   if np.mean(s[:, 1]) > median_y + SQUARE_SIZE]
    
    # Сортируем по горизонтали (слева направо)
    scale_squares.sort(key=lambda s: np.mean(s[:, 0]))
    test_strips.sort(key=lambda s: np.mean(s[:, 0]))
    
    return scale_squares, test_strips


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
    
    # 1. Обнаруживаем все квадраты
    found_squares = find_squares_in_image(img)
    print(f"✓ Найдено {len(found_squares)} квадратов")
    
    if len(found_squares) == 0:
        print("⚠ Квадраты не найдены")
        return results
    
    # 2. Разделяем на шкалу и тесты
    scale_squares, test_strips = separate_scale_and_tests(found_squares)
    print(f"  Шкала: {len(scale_squares)}, Тесты: {len(test_strips)}")
    
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
    
    # Отрисовка тестов (оранжевая) с результатами
    for i, pts in enumerate(results['test_strips']):
        pts_int = pts.astype(int)
        cv2.polylines(img_draw, [pts_int], True, COLOR_TEST, 3)
        
        if i < len(results['results']) and results['results'][i] is not None:
            conc_text = f"TEST {i+1}: {results['results'][i]:.2f}"
        else:
            conc_text = f"TEST {i+1}: No data"
        
        cv2.putText(img_draw, conc_text, (pts_int[0][0], pts_int[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEST, 2)
    
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
    output_path = 'result_improved.jpg'
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