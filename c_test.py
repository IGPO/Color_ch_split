import cv2
import numpy as np
from skimage import color  # для deltaE_ciede2000
from scipy.spatial import distance
import json
import os

folder = 'свет0-вид сверху'
folder = 'свет0.5-вид сверху'
folder = 'свет1-вид сверху'
folder = 'свет0-вид под углом'
folder = 'свет0.5-вид под углом'
folder = 'свет1-вид под углом'
light = "1" # "0", "0.5", "1"
angle = "Ang" # "Up" - вид сверху, "Ang" - вид под углом
# test_folder = "белки" # "test"
test_folder = "кетоны" # "test"
f_path = test_folder + '/' + folder + '/'


def save_experiment_data(file_path, ref_colors, test_color, lighting_type, angle):
    # Конвертируем numpy в обычные списки для JSON
    new_entry = {
        "metadata": {
            "lighting": lighting_type, # например, "daylight", "led"
            "angle": angle,           # например, "top", "45_deg"
        },
        "reference": [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in ref_colors],
        "test": test_color.tolist() if isinstance(test_color, np.ndarray) else test_color
    }

    # 1. Загружаем существующие данные
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data] # На случай, если в файле был один объект, а не список
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # 2. Добавляем новую запись
    data.append(new_entry)

    # 3. Сохраняем обратно
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Данные успешно добавлены. Всего записей: {len(data)}")


def load_image(test_image_path):
    """Load an image from a file."""
    image = cv2.imread(test_image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{test_image_path}' not found.")
    return image


# ---------- BGR ----------
def mean_color_bgr(image):
    """Calculate the mean BGR color of an image."""
    return image.mean(axis=0).mean(axis=0)


def all_distances_bgr(test_color, reference_colors):
    """Compute all Euclidean distances in BGR space."""
    return [distance.euclidean(test_color, ref) for ref in reference_colors]


# ---------- Lab ----------
def mean_color_lab(image):
    """Calculate the mean LAB color of an image (normalized for CIEDE2000)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean = lab.mean(axis=0).mean(axis=0)  # (L, a, b) в OpenCV
    # нормализация
    L = mean[0] / 255 * 100
    a = mean[1] - 128
    b = mean[2] - 128
    return np.array([L, a, b])


def all_distances_lab(test_color, reference_colors):
    """Compute all CIEDE2000 distances in Lab space."""

    return [
        color.deltaE_ciede2000(test_color, ref)
        #float(color.deltaE_ciede2000(test_color[np.newaxis, :], ref[np.newaxis, :]))
        for ref in reference_colors
    ]


# ---------- Main ----------
test_image_path = f_path + 'test.jpg'
if test_folder == "кетоны":
    reference_image_paths = [f_path + '1.jpg', f_path + '2.jpg', f_path + '3.jpg', f_path + '4.jpg', f_path + '5.jpg']   # for ketones
else:
    reference_image_paths = [f_path + '0.15.jpg', f_path + '0.3.jpg', f_path + '0.5.jpg', f_path + '1.jpg', f_path + '2.jpg'] # for proteins


# Load & compute mean colors
reference_colors_bgr = [mean_color_bgr(load_image(p)) for p in reference_image_paths]
test_color_bgr = mean_color_bgr(load_image(test_image_path))
reference_colors_bgr_RMS = [np.sqrt(r**2 + g**2 + b**2) for r, g, b in reference_colors_bgr]
data_to_send_rgb = {
    "reference": [arr.tolist() for arr in reference_colors_bgr],
    "test": test_color_bgr.tolist()
}
save_experiment_data(f'data_log_{test_folder}_rgb.json', reference_colors_bgr, test_color_bgr, light, angle)
print('')
# print('reference_colors_bgr:', reference_colors_bgr)
print('data_to_send_rgb:', data_to_send_rgb)
#print("test_color_bgr_RMS:", np.sqrt(test_color_bgr[0]**2 + test_color_bgr[1]**2 + test_color_bgr[2]**2))
#print("reference_colors_bgr_RMS:", reference_colors_bgr_RMS)
print('')
reference_colors_lab = [mean_color_lab(load_image(p)) for p in reference_image_paths]
test_color_lab = mean_color_lab(load_image(test_image_path))
save_experiment_data(f'data_log_{test_folder}_lab.json', reference_colors_lab, test_color_lab, light, angle)
data_to_send_lab = {
    "reference": [arr.tolist() for arr in reference_colors_lab],
    "test": test_color_lab.tolist()
}
print('data_to_send_lab:', data_to_send_lab)
print('')
# Distances
distances_bgr = all_distances_bgr(test_color_bgr, reference_colors_bgr)
distances_lab = all_distances_lab(test_color_lab, reference_colors_lab)
#print(type(distances_lab))
closest_index_bgr = int(np.argmin(distances_bgr))
closest_index_lab = int(np.argmin(distances_lab))

# ---------- Results ----------

#print("---------- Results ----------")
print(" folder is ", "<", folder, ">")
for i, d in enumerate(distances_lab, 1):
    print(f"Reference {i}:: distance LAB = {d:.2f}, RGB = {distances_bgr[i-1]:.2f}")

print("=== Results in RGB (Euclidean) ===")
print(f"Closest reference image: {closest_index_bgr + 1}")

print(f"Closest mean BGR: {reference_colors_bgr[closest_index_bgr]}")
print(f"Test mean BGR: {test_color_bgr}")
print(f"reference_colors_bgr_1: {reference_colors_bgr[0]}")

print("=== Results in Lab (CIEDE2000) ===")
#for i, d in enumerate(distances_lab, 1):
#    print(f"Reference {i}: distance = {d:.2f}")
print(f"Closest reference image: {closest_index_lab + 1}")

print(f"Closest mean Lab: {reference_colors_lab[closest_index_lab]}")
print(f"Test mean Lab: {test_color_lab}")
print(f"reference_colors_lab_1: {reference_colors_lab[0]}")