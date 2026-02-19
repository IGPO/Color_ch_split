import cv2
import numpy as np

# Load the masks
mask_clean = cv2.imread('debug_masks/04_mask_clean.jpg', cv2.IMREAD_GRAYSCALE)
mask_combined = cv2.imread('debug_masks/03_mask_combined.jpg', cv2.IMREAD_GRAYSCALE)
edges = cv2.imread('debug_masks/02_canny_edges.jpg', cv2.IMREAD_GRAYSCALE)
hsv_sat = cv2.imread('debug_masks/01_hsv_saturation.jpg', cv2.IMREAD_GRAYSCALE)

print("Статистика масок:")
print(f"HSV Saturation: белых пикселей = {np.sum(hsv_sat > 128)} ({100*np.sum(hsv_sat > 128)/(mask_clean.shape[0]*mask_clean.shape[1]):.2f}%)")
print(f"Canny Edges: белых пикселей = {np.sum(edges > 128)} ({100*np.sum(edges > 128)/(mask_clean.shape[0]*mask_clean.shape[1]):.2f}%)")
print(f"Mask Combined: белых пикселей = {np.sum(mask_combined > 128)} ({100*np.sum(mask_combined > 128)/(mask_clean.shape[0]*mask_clean.shape[1]):.2f}%)")
print(f"Mask Clean (final): белых пикселей = {np.sum(mask_clean > 128)} ({100*np.sum(mask_clean > 128)/(mask_clean.shape[0]*mask_clean.shape[1]):.2f}%)")

# Анализ размещение белых пикселей на маске clean
h, w = mask_clean.shape
top_half_white = np.sum(mask_clean[:h//2, :] > 128)
bottom_half_white = np.sum(mask_clean[h//2:, :] > 128)
print(f"\nВерхняя половина: {top_half_white} белых пикселей")
print(f"Нижняя половина (где должен быть тест): {bottom_half_white} белых пикселей")

# Анализ contoured
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nКонтуры из mask_clean: найдено {len(contours)}")

# Анализ размера контуров в нижней половине
print("\nКонтуры в нижней половине изображения:")
for i, cnt in enumerate(contours):
    h_cnt = cnt.shape[0]
    area = cv2.contourArea(cnt)
    y_mean = np.mean(cnt[:, 0, 1])
    if y_mean > h/2:
        print(f"  Контур #{i}: площадь={area:.0f}, средний Y={y_mean:.0f}")
