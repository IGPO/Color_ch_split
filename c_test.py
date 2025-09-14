import cv2
import numpy as np
from skimage import color  # для deltaE_ciede2000
from scipy.spatial import distance

folder = 'свет0.5-вид под углом'

f_path = 'test/' + folder + '/'

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
reference_image_paths = [f_path + '1.jpg', f_path + '2.jpg', f_path + '3.jpg', f_path + '4.jpg', f_path + '5.jpg']

# Load & compute mean colors
reference_colors_bgr = [mean_color_bgr(load_image(p)) for p in reference_image_paths]
test_color_bgr = mean_color_bgr(load_image(test_image_path))

reference_colors_lab = [mean_color_lab(load_image(p)) for p in reference_image_paths]
test_color_lab = mean_color_lab(load_image(test_image_path))

# Distances
distances_bgr = all_distances_bgr(test_color_bgr, reference_colors_bgr)
distances_lab = all_distances_lab(test_color_lab, reference_colors_lab)
#print(type(distances_lab))
closest_index_bgr = int(np.argmin(distances_bgr))
closest_index_lab = int(np.argmin(distances_lab))

# ---------- Results ----------

print("---------- Results ----------")
print(" folder is ", "<", folder, ">")
for i, d in enumerate(distances_lab, 1):
    print(f"Reference {i}:: distance LAB = {d:.2f}, RGB = {distances_bgr[i-1]:.2f}")

print("\n=== Results in RGB (Euclidean) ===")
print(f"Closest reference image: {closest_index_bgr + 1}")
print(f"Test mean BGR: {test_color_bgr}")
print(f"Closest mean BGR: {reference_colors_bgr[closest_index_bgr]}")
print()

print("=== Results in Lab (CIEDE2000) ===")
#for i, d in enumerate(distances_lab, 1):
#    print(f"Reference {i}: distance = {d:.2f}")
print(f"Closest reference image: {closest_index_lab + 1}")
print(f"Test mean Lab: {test_color_lab}")
print(f"Closest mean Lab: {reference_colors_lab[closest_index_lab]}")
