# Color Matching Algorithm - AI Agent Instructions

## Project Overview
This codebase implements a color matching algorithm that compares test images to reference images using two color spaces: BGR (Euclidean distance) and Lab (CIEDE2000 metric). The algorithm identifies the reference image most similar in average color to a test image.

## Architecture
- **Main Algorithm**: `c_test.py` - Loads images, computes mean colors, calculates distances, outputs results
- **Visualization**: `graph.py` - Contains `plot_xy()` function for scatter plots of color data
- **Data Structure**: Images organized in nested folders by category (e.g., `test/`, `белки/`), lighting conditions (`свет0-`, `свет0.5-`, `свет1-`), and view angles (`вид сверху`, `вид под углом`)

## Key Dependencies
- `cv2` (OpenCV): Image loading and BGR↔Lab conversion
- `numpy`: Array operations and mean calculations
- `skimage.color`: CIEDE2000 distance computation
- `scipy.spatial.distance`: Euclidean distance in BGR space

## File Naming Conventions
- **Test Images**: `test.jpg` in each subfolder
- **Reference Images**: 
  - `test/` category: `1.jpg`, `2.jpg`, `3.jpg`, `4.jpg`, `5.jpg`
  - `белки/` category: `0.15.jpg`, `0.3.jpg`, `0.5.jpg`, `1.jpg`, `2.jpg`
- **Folders**: Cyrillic names like `свет1-вид под углом` (light intensity + view angle)

## Workflow Patterns
- Modify `folder` and `test_folder` variables in `c_test.py` to switch between test conditions
- Run `c_test.py` directly with Python - no build step required
- Use `graph.py` for plotting color distance data with `plot_xy(x, y, labels=labels)`

## Color Space Handling
- **BGR Mean**: Direct averaging of pixel values across image
- **Lab Mean**: Convert to Lab, average, then normalize L∈[0,100], a,b∈[-128,127] for CIEDE2000 compatibility
- Always compute both metrics - they can give different closest matches

## Common Modifications
- Add new reference images following naming patterns
- Extend distance calculations (currently only min distance, could add ranking)
- Visualize results using `graph.py` plotting functions
- Handle different image formats beyond JPG

## Data Organization
Images are grouped by:
- **Category**: `test/` (ketones?) vs `белки/` (proteins)
- **Lighting**: `свет0` (no light), `свет0.5` (half light), `свет1` (full light)
- **View**: `вид сверху` (top view), `вид под углом` (angled view)

Reference `c_test.py` lines 8-12 for current folder configuration.