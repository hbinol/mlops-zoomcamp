# mlops-zoomcamp

assert '_distutils' in core.__file__, assert 'setuptools._distutils.log' not in sys.modules
AssertionError: /opt/conda/python3.10/distutils/core.py

usually comes from a conflict between Python’s native distutils and setuptools’ internal _distutils in newer Python environments. It often surfaces when PyTorch (especially with CUDA) and other C/C++ extension packages are installed in an inconsistent Python environment (e.g., mixing conda and pip or messing with distutils manually).

# Solution: Docker-based PyTorch 2.6.0 Setup Cleanly
FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn8-runtime

# Optional: upgrade pip & install system packages
RUN apt-get update && apt-get install -y git curl && \
    pip install --upgrade pip setuptools

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_yellow_objects(image, debug=False):
    """
    Detects yellow-colored objects in the image using HSV thresholding.
    Returns a binary mask.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define yellow color range (tweakable)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Yellow Object Mask")
        plt.show()

    return mask

def measure_yellow_grapes(image, pixels_per_cm, debug=False):
    """
    Detects yellow grapes and computes area in cm² using pixel_per_cm scale.
    """
    mask = detect_yellow_objects(image, debug=debug)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    sizes_cm2 = []
    for i in range(1, num_labels):  # skip background
        area_pixels = stats[i, cv2.CC_STAT_AREA]
        area_cm2 = area_pixels / (pixels_per_cm ** 2)
        sizes_cm2.append(area_cm2)

    print(f"Detected {len(sizes_cm2)} yellow grapes.")
    return sizes_cm2
