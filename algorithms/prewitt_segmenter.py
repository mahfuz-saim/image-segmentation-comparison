import cv2
import numpy as np
import os
from .base_segmenter import Segmenter
from utils.preprocessing import to_grayscale, apply_median_blur
from utils.postprocessing import fill_contours

class PrewittSegmenter(Segmenter):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def segment(self, image: np.ndarray) -> np.ndarray:
        # Pipeline: Grayscale -> Median Blur -> Prewitt -> Magnitude -> Threshold -> Contour Fill
        gray = to_grayscale(image)
        blurred = apply_median_blur(gray)
        
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        gx = cv2.filter2D(blurred, cv2.CV_64F, kernel_x)
        gy = cv2.filter2D(blurred, cv2.CV_64F, kernel_y)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        
        _, mask = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        
        final_mask = fill_contours(mask)

        # Save intermediates
        os.makedirs(self.output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.output_dir, "prewitt_edges.png"), magnitude)
        cv2.imwrite(os.path.join(self.output_dir, "prewitt_thresh.png"), mask)
        cv2.imwrite(os.path.join(self.output_dir, "final_mask.png"), final_mask)
        
        return final_mask
