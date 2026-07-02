import cv2
import numpy as np

def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_gaussian_blur(image: np.ndarray, ksize=(5, 5), sigma=1) -> np.ndarray:
    return cv2.GaussianBlur(image, ksize, sigma)

def apply_median_blur(image: np.ndarray, ksize=5) -> np.ndarray:
    return cv2.medianBlur(image, ksize)
