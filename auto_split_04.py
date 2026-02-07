
import cv2
import numpy as np

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

# -----------------------------
# Функции (Functions)
# -----------------------------
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

print(f"Found {len(candidates)} candidates, filtering down to {candidates[0]} after geometry check.")
#print(candidates)
found_squares = []  # Сохраняем для дальнейшей обработки
for c in candidates:
    found_squares.append(order_points(c['box']))
    print(f"Candidate at {c['center']} with side {c['side']} has box points: {order_points(c['box'])}")
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

# 4. Визуализация и вырезка
square_imgs = []
for pts_src in final_rois:
    pts_dst = np.array([[0,0],[N-1,0],[N-1,N-1],[0,N-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(pts_src, pts_dst)
    square_img = cv2.warpPerspective(img, H, (N,N))
    square_imgs.append(square_img)
    
    # Рисуем на оригинале
    pts_int = pts_src.astype(int)
    cv2.polylines(img_draw, [pts_int], isClosed=True, color=(0,255,0), thickness=3)

# Сборка финальной картинки
# (код для показа аналогичен вашему)
if square_imgs:
    bottom_row = cv2.hconcat(square_imgs)
    # Resize to match top row width
    bottom_row = cv2.resize(bottom_row, (img_draw.shape[1], N))
    final_vis = cv2.vconcat([img_draw, bottom_row])
    cv2.imshow(f"Square {len(found_squares)}", final_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite('result_improved.jpg', final_vis)
    print(f"Total squares found and processed: {len(final_rois)}")
else:
    cv2.imwrite('result_improved.jpg', img_draw)
    print("No squares found.")
