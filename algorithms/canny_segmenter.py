import cv2
import numpy as np
import os
from .base_segmenter import Segmenter
from utils.preprocessing import to_grayscale, apply_gaussian_blur
from utils.postprocessing import morphological_closing, fill_contours

class CannySegmenter(Segmenter):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def segment(self, image: np.ndarray) -> np.ndarray:
        # Pipeline: Grayscale -> Gaussian Blur -> Canny -> Closing -> Contour Fill
        gray = to_grayscale(image)
        blurred = apply_gaussian_blur(gray)
        
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        final_mask = fill_contours(closed)

        # Save intermediates
        os.makedirs(self.output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.output_dir, "canny_edges.png"), edges)
        cv2.imwrite(os.path.join(self.output_dir, "canny_closed.png"), closed)
        cv2.imwrite(os.path.join(self.output_dir, "final_mask.png"), final_mask)
        
        return final_mask
