import cv2
import numpy as np

# -----------------------------
# Параметры
# -----------------------------
A_min_ratio = 0.0008  # минимальная площадь квадрата относительно изображения
epsilon_ratio = 0.1  # точность approxPolyDP
side_ratio_thresh = 1.15
angle_cos_thresh = 0.5
N = 16  # размер выпрямленного квадрата для пикселей
margin_ratio = 0.008  # отступ от границы

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
img = cv2.imread('кетоны/свет0-вид сверху/full.jpg')
# img = cv2.imread('белки/свет0.5-вид сверху/full.jpg')
img_draw = img.copy()
h, w = img.shape[:2]
A_min = h * w * A_min_ratio

# -----------------------------
# Преобразование для поиска контуров
# -----------------------------
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A_chan, B_chan = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
L_clahe = clahe.apply(L)
L_blur = cv2.GaussianBlur(L_clahe, (5,5), 0)
edges = cv2.Canny(L_blur, 20, 180)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

# -----------------------------
# Найти контуры
# -----------------------------
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

found_squares = []
pixels_list = []

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

    # perspective warp
    pts_dst = np.array([
        [0, 0],
        [N-1, 0],
        [N-1, N-1],
        [0, N-1]
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(pts_src, pts_dst)
    square_img = cv2.warpPerspective(img, H, (N,N))
    
    # убрать рамку
    margin = int(N * margin_ratio)
    inner = square_img[margin:-margin, margin:-margin]
    pixels = inner.reshape(-1, 3)
    pixels_list.append(pixels)

    # нарисовать квадрат на исходном изображении
    pts = pts_src.astype(int)
    cv2.polylines(img_draw, [pts], isClosed=True, color=(0,255,0), thickness=2)

# -----------------------------
# Показ результата
# -----------------------------
cv2.imshow("Squares Found", img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Found {len(found_squares)} squares.")
