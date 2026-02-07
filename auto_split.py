import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_perspective_warped(img, points):
    rect_pts = order_points(points.astype("float32"))
    (tl, tr, br, bl) = rect_pts
    width = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    height = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect_pts, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    if height > width: warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

def get_hybrid_rois(warped_img, num_expected=6):
    """Ищет реальные контуры, а если не находит - строит сетку"""
    h_strip, w_strip = warped_img.shape[:2]
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_rois = []
    
    # Пытаемся найти реальные контуры (смягчили фильтры)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (h_strip * 0.2 < h < h_strip * 1.0) and (0.5 < w/h < 2.0):
            detected_rois.append((x, y, x + w, y + h))
    
    detected_rois = sorted(detected_rois, key=lambda r: r[0])

    # ЕСЛИ НАШЛИ МАЛО (или 0): строим сетку-заплатку
    if len(detected_rois) < num_expected:
        print(f"Автодетекция нашла {len(detected_rois)}, строим сетку-заплатку...")
        final_rois = []
        seg_w = w_strip / num_expected
        for i in range(num_expected):
            # Сетка по центру сегмента
            x1 = int(i * seg_w + (seg_w * 0.15))
            x2 = int((i + 1) * seg_w - (seg_w * 0.15))
            y1, y2 = int(h_strip * 0.2), int(h_strip * 0.8)
            final_rois.append((x1, y1, x2, y2))
        return final_rois
    
    return detected_rois

def process_and_save(image_path, output_dir="extracted"):
    img = cv2.imread(image_path)
    if img is None: return print("Ошибка: Кадр не загружен")
    
    # 1. Детекция всей полоски
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return print("Полоска не найдена на общем фоне")
    strip_cnt = max(contours, key=cv2.contourArea)
    
    # 2. Выпрямление
    rect_min = cv2.minAreaRect(strip_cnt)
    box = cv2.boxPoints(rect_min)
    warped = get_perspective_warped(img, box)
    
    # 3. Поиск квадратов (Гибридный метод)
    rois = get_hybrid_rois(warped)
    
    # 4. Вырезание и Сохранение
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    final_view = warped.copy()
    
    for i, (x1, y1, x2, y2) in enumerate(rois):
        patch = warped[y1:y2, x1:x2]
        cv2.imwrite(f"{output_dir}/patch_{i}.png", patch)
        cv2.rectangle(final_view, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
    print(f"Успех! В папке '{output_dir}' сохранено {len(rois)} квадратов.")
    return final_view

# ЗАПУСК
result = process_and_save('кетоны/свет0-вид под углом/full.jpg')        

if result is not None:
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()