import cv2
import numpy as np

def morphological_closing(mask: np.ndarray, ksize=(5, 5)) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def morphological_opening(mask: np.ndarray, ksize=(5, 5)) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def fill_contours(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    return filled_mask

def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    return cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
