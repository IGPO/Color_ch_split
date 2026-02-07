import cv2
import numpy as np

# -----------------------------
# Параметры
# -----------------------------
A_min_ratio = 0.001        # минимальная площадь квадрата
epsilon_ratio = 0.05       # точность approxPolyDP
side_ratio_thresh = 1.3
angle_cos_thresh = 0.3
N = 128                    # размер выпрямленного квадрата
margin_ratio = 0.08        # отступ от краёв
adaptive_thresh_C = 5       # параметр adaptive threshold для маски

# -----------------------------
# Функции
# -----------------------------
def is_square_geometry(approx, side_ratio_thresh=1.15, angle_cos_thresh=0.3):
    pts = approx.reshape(4, 2)
    sides = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
    if max(sides) / min(sides) > side_ratio_thresh:
        return False
    for i in range(4):
        p0 = pts[i]
        p1 = pts[i-1]
        p2 = pts[(i+1)%4]
        v1 = p1 - p0
        v2 = p2 - p0
        cos = abs(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))
        if cos > angle_cos_thresh:
            return False
    return True

def order_points(approx):
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

# -----------------------------
# Загрузка изображения
# -----------------------------
img = cv2.imread('кетоны/свет0.5-вид сверху/full.jpg')
# img = cv2.imread('кетоны/свет0-вид под углом/full.jpg')
img_draw = img.copy()
h, w = img.shape[:2]
A_min = h * w * A_min_ratio

# -----------------------------
# 1. CLAHE + GaussianBlur
# -----------------------------
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A_chan, B_chan = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
L_clahe = clahe.apply(L)
L_blur = cv2.GaussianBlur(L_clahe, (5,5), 0)

# # -----------------------------
# # 2. Canny + Morphology
# # -----------------------------
# edges = cv2.Canny(L_blur, 40, 120)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
# -----------------------------

# -----------------------------
# 2. HSV Saturation + Улучшенная морфология
# -----------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Адаптивный порог: увеличиваем block_size до 31, чтобы он был больше букв текста
# C=2 помогает отсечь слабый фон
s_thresh = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 31, -5)

# 1. Убираем мелкий текст (OPENING)
kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
mask_no_text = cv2.morphologyEx(s_thresh, cv2.MORPH_OPEN, kernel_small, iterations=2)

# 2. Склеиваем разрывы внутри квадратов от бликов (CLOSING)
# Используем ядро побольше, чтобы соединить "ошметки" на 4-м и 5-м квадратах
kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
edges_closed = cv2.morphologyEx(mask_no_text, cv2.MORPH_CLOSE, kernel_big, iterations=2)

# Размытие, чтобы findContours не цеплялся за "лесенку" пикселей
edges_closed = cv2.GaussianBlur(edges_closed, (7,7), 0)
# -----------------------------
# 3. Найти контуры
# -----------------------------
contours, _ = cv2.findContours(edges_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
found_squares = []
pixels_list = []
square_imgs = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < A_min:
        continue
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
    if len(approx) != 4:
        continue
    if not cv2.isContourConvex(approx):
        continue
    if not is_square_geometry(approx, side_ratio_thresh, angle_cos_thresh):
        continue

    pts_src = order_points(approx)
    found_squares.append(pts_src)

    # Perspective warp
    pts_dst = np.array([[0,0],[N-1,0],[N-1,N-1],[0,N-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(pts_src, pts_dst)
    square_img = cv2.warpPerspective(img, H, (N,N))

    # -----------------------------
    # Маска внутри квадрата (устойчивость к бликам)
    # -----------------------------
    hsv = cv2.cvtColor(square_img, cv2.COLOR_BGR2HSV)
    H_chan, S_chan, V_chan = cv2.split(hsv)
    mask = cv2.adaptiveThreshold(V_chan, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 15, adaptive_thresh_C)
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # убрать края (margin)
    margin = int(N * margin_ratio)
    inner = square_img[margin:-margin, margin:-margin]
    pixels = inner.reshape(-1,3)
    pixels_list.append(pixels)

    # нарисовать квадрат на исходном изображении
    pts_int = pts_src.astype(int)
    cv2.polylines(img_draw, [pts_int], isClosed=True, color=(0,255,0), thickness=2)

    # для визуализации сохраняем квадрат и маску
    square_imgs.append(cv2.hconcat([square_img, mask_color]))

# -----------------------------
# 4. Компоновка изображений для визуализации
# -----------------------------
# верхняя строка: CLAHE + Blur, Canny, Morphology
top_row = cv2.hconcat([img_draw,
                       cv2.cvtColor(s_thresh, cv2.COLOR_GRAY2BGR),
                       cv2.cvtColor(edges_closed, cv2.COLOR_GRAY2BGR)])

# нижняя строка: квадраты + их маски
if square_imgs:
    bottom_row = cv2.hconcat(square_imgs) if len(square_imgs) > 1 else square_imgs[0]
else:
    bottom_row = np.zeros((N, N*2,3), dtype=np.uint8)  # пустое место

# приводим bottom_row к ширине top_row
w_top = top_row.shape[1]
bottom_row = cv2.resize(bottom_row, (w_top, bottom_row.shape[0]))

# убедимся, что оба изображения BGR
if len(top_row.shape) == 2:
    top_row = cv2.cvtColor(top_row, cv2.COLOR_GRAY2BGR)
if len(bottom_row.shape) == 2:
    bottom_row = cv2.cvtColor(bottom_row, cv2.COLOR_GRAY2BGR)

# объединяем вертикально
final_vis = cv2.vconcat([top_row, bottom_row])

cv2.imshow("Processing Pipeline", final_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Found {len(found_squares)} squares.")
