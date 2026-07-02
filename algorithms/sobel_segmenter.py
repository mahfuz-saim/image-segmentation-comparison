import cv2
import numpy as np
import os
from .base_segmenter import Segmenter
from utils.preprocessing import to_grayscale, apply_gaussian_blur
from utils.postprocessing import morphological_closing, overlay_mask

class SobelSegmenter(Segmenter):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def segment(self, image: np.ndarray) -> np.ndarray:
        # Pipeline: Grayscale -> Gaussian Blur -> Sobel -> Magnitude -> Otsu -> Closing
        gray = to_grayscale(image)
        blurred = apply_gaussian_blur(gray)
        
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        
        _, mask = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        final_mask = morphological_closing(mask)
        overlay = overlay_mask(image, final_mask)

        # Save intermediates
        os.makedirs(self.output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.output_dir, "original.png"), image)
        cv2.imwrite(os.path.join(self.output_dir, "edges.png"), magnitude)
        cv2.imwrite(os.path.join(self.output_dir, "mask.png"), final_mask)
        cv2.imwrite(os.path.join(self.output_dir, "overlay.png"), overlay)
        
        return final_mask
