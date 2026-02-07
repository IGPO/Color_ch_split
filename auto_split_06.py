import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def get_robust_color(img_block):
    # Переход в LAB и очистка от бликов
    lab = cv2.cvtColor(img_block, cv2.COLOR_BGR2LAB)
    l_chan = lab[:,:,0]
    mask = (l_chan > np.percentile(l_chan, 5)) & (l_chan < np.percentile(l_chan, 85))
    pixels = lab[mask] if any(mask.flatten()) else lab.reshape(-1, 3)
    return np.median(pixels, axis=0)

# 1. Загрузка
img = cv2.imread('кетоны/свет0.5-вид сверху/full.jpg')
h_img, w_img = img.shape[:2]
img_draw = img.copy()

# 2. Поиск маски (Saturation + Adaptive)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s_channel = hsv[:,:,1]
# Делаем порог очень чувствительным (C=10), чтобы поймать бледные цвета
mask = cv2.adaptiveThreshold(s_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 3. Поиск кандидатов
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
candidates = []
N = 128

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < (h_img * w_img * 0.0005): continue
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    w, h = rect[1]
    if min(w,h) > 0 and (max(w,h)/min(w,h)) < 1.6:
        candidates.append({'center': rect[0], 'box': box, 'size': np.mean(rect[1])})

# 4. Логика "5 квадратов" (Априорная информация)
# Сортируем кандидатов по Y, находим группу с похожей высотой
candidates.sort(key=lambda x: x['center'][1])
# Группируем те, что в одном ряду (допустим, верхний ряд - это шкала)
scale_candidates = sorted(candidates[:10], key=lambda x: x['center'][0]) 

# (Упрощенно: берем первые 5 найденных в ряду)
final_squares = []
for i, c in enumerate(scale_candidates[:5]):
    pts_src = order_points(c['box'])
    M = cv2.getPerspectiveTransform(pts_src, np.array([[0,0],[N-1,0],[N-1,N-1],[0,N-1]], dtype="float32"))
    sq_img = cv2.warpPerspective(img, M, (N, N))
    final_squares.append(sq_img)
    cv2.polylines(img_draw, [pts_src.astype(int)], True, (0, 255, 0), 2)

# 5. Визуализация
top = cv2.resize(img_draw, (w_img//2, h_img//2))
mask_vis = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (w_img//2, h_img//2))
top_row = cv2.hconcat([top, mask_vis])

if final_squares:
    bottom_row = cv2.hconcat(final_squares)
    bottom_row = cv2.resize(bottom_row, (top_row.shape[1], N))
    result = cv2.vconcat([top_row, bottom_row])
else:
    result = top_row

cv2.imshow(f"Square {len(final_squares)}", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result_v6.jpg', result)
print("Готово! Результат в файле result_v6.jpg")