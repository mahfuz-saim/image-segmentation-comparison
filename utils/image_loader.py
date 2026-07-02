import cv2
import numpy as np
import os

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return img

def save_image(img: np.ndarray, path: str):
    cv2.imwrite(path, img)
