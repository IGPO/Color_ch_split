import cv2
import numpy as np
from scipy.spatial import distance


def load_image(test_image_path):
    """Load an image from a file."""
    image = cv2.imread(test_image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{test_image_path}' not found.")
    return image


def mean_color(image):
    """Calculate the mean color of an image."""
    mean_color = image.mean(axis=0).mean(axis=0)
    return mean_color


def closest_color(test_color, reference_colors):
    """Find the reference color closest to the test color."""
    distances = [distance.euclidean(test_color, ref_color) for ref_color in reference_colors]
    print(distances)
    closest_index = np.argmin(distances)
    return closest_index, distances[closest_index]


# Paths to the reference images and the test image
test_image_path = 'test_img/03.png'
reference_image_paths = ['ref_img/00.png', 'ref_img/01.png', 'ref_img/03.png', 'ref_img/10.png', 'ref_img/30.png',
                         'ref_img/100.png']

# Load images and compute mean colors
reference_colors = [mean_color(load_image(path)) for path in reference_image_paths]
test_color = mean_color(load_image(test_image_path))

# Find the closest reference color to the test image color
closest_index, closest_distance = closest_color(test_color, reference_colors)

# Print the results
print(f"The test image is closest to reference image {closest_index + 1} with a distance of {closest_distance}.")
print(f"Mean color of the test image: {test_color}")
print(f"Mean color of the closest reference image: {reference_colors[closest_index]}")
