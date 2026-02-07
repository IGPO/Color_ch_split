# Color Matching Algorithm - AI Agent Instructions

## Project Overview
This codebase implements color matching and concentration estimation for test strip imagery. It combines two parallel approaches:
1. **Color reference matching** (`c_test.py`): Compares test image colors to reference sets using BGR (Euclidean) and Lab (CIEDE2000)
2. **Concentration estimation** (`auto_split_*.py` series): Detects and analyzes color strips to estimate analyte concentration via spline-fitted color laws

## Core Workflows

### Color Matching (Legacy/Reference Path)
- **Entry point**: `c_test.py` (155 lines)
- **Process**: Load test image → compute mean BGR/Lab colors → compare against reference set using two metrics
- **Output**: Distance matrix for both color spaces, identifies closest match
- **Key insight**: Both metrics can give different results; always report both

### Concentration Estimation (Active Development)
- **Evolution**: `auto_split.py` → `auto_split_01.py` → ... → `auto_split_05.py` (latest, 401 lines)
- **New approach** (auto_split_05.py): Detects colored squares → extracts robust Lab colors → fits spline model to concentration scale → estimates unknown samples
- **Critical function**: `build_color_law()` - creates UnivariateSpline model from 5-point concentration scale
- **Robustness**: `extract_robust_lab()` filters out highlights/shadows using percentile-based masking on L channel

## Architecture & Key Functions

### Color Space Conversions (universal pattern)
- Always use `cv2.cvtColor(img, cv2.COLOR_BGR2LAB)` for BGR→Lab
- Lab normalization for CIEDE2000: L∈[0,100], a,b∈[-128,127] (standard ranges)
- Both color spaces needed because they weight colors differently

### Square Detection & Geometry (auto_split_05.py)
- `is_square_geometry()`: Uses min bounding rect + aspect ratio check (side_ratio_thresh=1.4)
- `extract_robust_lab()`: Crops margins (10% default), percentile-filters L channel (keep 5-80th percentile) to remove highlights/noise
- Parameter tuning: epsilon_ratio=0.05, angle_cos_thresh=0.4, A_min/max_ratio=0.001-0.05

### Color Model Building (auto_split_05.py)
- `build_color_law()`: Creates 3 UnivariateSpline objects (L, a, b channels) from 5 reference squares at concentrations [0.5, 1.5, 4.0, 8.0, 16.0]
- Smoothing factor s=10 (tuned for typical highlight/blur noise)
- Returns model tuple + raw_labs for comparison/visualization

## Data Organization & Naming

### Directory Structure
```
категория/
  свет{0|0.5|1}-вид{сверху|под углом}/
    test.jpg (test image)
    reference_images (for color matching)
```

### Categories & Metadata
- **Protein scale** (`белки/`): References `0.15.jpg, 0.3.jpg, 0.5.jpg, 1.jpg, 2.jpg`
- **Ketone scale** (`кетоны/`): Multiple sub-datasets
- **Lighting**: `свет0` (none), `свет0.5` (50%), `свет1` (full)
- **Angle**: `вид сверху` (top), `вид под углом` (angled 45°)

### Data Persistence
- Color matching results → `data_log_*.json` (stores ref/test colors + metadata)
- Fields: lighting, angle, reference [LAB arrays], test [LAB array]

## Development Patterns

### Iterative Refinement (auto_split versioning)
- `auto_split_01.py`: Basic contour detection
- `auto_split_02.py`: Add perspective warping
- `auto_split_03.py`: Hybrid ROI detection (contours + grid fallback)
- `auto_split_04.py`: Full pipeline with visualization
- `auto_split_05.py` (current): Robust color extraction + spline model + visualization gradient

**Pattern**: Each version addresses previous failure modes; preserve working functions when iterating

### Configuration via Module Variables
- Top of scripts: Modify `folder`, `test_folder`, `light`, `angle`, `concentrations`, thresholds
- No config files; inline params for rapid experimentation
- Example: `A_min_ratio=0.001, A_max_ratio=0.05, margin_ratio=0.1`

### Visualization & Debugging
- `create_model_visualization()`: Renders spline model as BGR gradient image with concentration tick marks
- `graph.py`: Provides `plot_xy()` for scatter plots
- Intermediate images saved to workspace (warped strips, detected squares, gradient visualizations)

## Common Modifications
1. **Tune detection**: Adjust epsilon_ratio, side_ratio_thresh, A_min/max_ratio in auto_split script headers
2. **Adjust color robustness**: Change percentile thresholds in `extract_robust_lab()` (currently 5-80 for L)
3. **Smooth fitting**: Modify `s` parameter in UnivariateSpline call (higher = smoother but less detail)
4. **New scales**: Update `concentrations` array and reference image count
5. **Visualization**: Call `create_model_visualization(model, width, height)` to render gradient

## Dependencies
- `cv2`: Image I/O, color conversion, contour/geometry operations
- `numpy`: Array math, percentile calculations
- `scipy.interpolate.UnivariateSpline`: Smooth curve fitting
- `skimage.color.deltaE_ciede2000`: Color distance metric
- `scipy.spatial.distance.euclidean`: BGR distance
- `matplotlib`: Plotting (optional, for graph.py)