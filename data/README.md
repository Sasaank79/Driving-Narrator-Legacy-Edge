# Dataset

The LISA Traffic Sign dataset (1.1 GB) is not included in this repository to keep the repo size manageable.

## Download

**Augmented version (used in this project):**  
https://universe.roboflow.com/suryaworkspace/lisa-road-signs-ftd0q/dataset/2

**Original LISA dataset:**  
https://universe.roboflow.com/dakota-smith/lisa-road-signs

## Setup

1. Download the dataset from Roboflow in YOLO format
2. Extract to this `data/` folder
3. Update paths in `LISA_Training.ipynb` if needed

## Dataset Statistics

- **Total Images:** 16,544 (after augmentation)
- **Train/Val/Test Split:** 90% / 6% / 4%
- **Classes:** 47 (US MUTCD regulatory, warning, guide signs)
- **Resolution:** 640×640 (normalized)
