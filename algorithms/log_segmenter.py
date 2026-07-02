import cv2
import numpy as np
import os
from .base_segmenter import Segmenter
from utils.preprocessing import to_grayscale, apply_gaussian_blur
from utils.postprocessing import morphological_opening, morphological_closing

class LoGSegmenter(Segmenter):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def segment(self, image: np.ndarray) -> np.ndarray:
        # Pipeline: Grayscale -> Gaussian Blur -> Laplacian -> Zero Crossing -> Binary Mask -> Morph Cleanup
        gray = to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # Efficient Zero Crossing detection using NumPy
        # A zero crossing occurs where signs change between adjacent pixels
        zero_cross = np.zeros(laplacian.shape, dtype=np.uint8)
        
        # Check horizontal and vertical neighbors for sign changes
        # Use a small threshold to avoid noise in flat areas
        thresh = 0.01 
        
        # Shifted versions to compare neighbors
        h_shift = np.zeros_like(laplacian)
        h_shift[:, 1:] = laplacian[:, :-1]
        v_shift = np.zeros_like(laplacian)
        v_shift[1:, :] = laplacian[:-1, :]
        
        # Sign change detection
        z_c_h = ((laplacian > thresh) & (h_shift < -thresh)) | ((laplacian < -thresh) & (h_shift > thresh))
        z_c_v = ((laplacian > thresh) & (v_shift < -thresh)) | ((laplacian < -thresh) & (v_shift > thresh))
        
        zero_cross[z_c_h | z_c_v] = 255
        
        # Morphological cleanup (using smaller kernels to preserve edges before filling)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(zero_cross, cv2.MORPH_CLOSE, kernel)
        
        # To turn edges into a segmentation mask, we fill the contours
        from utils.postprocessing import fill_contours
        final_mask = fill_contours(mask)

        # Save intermediates
        os.makedirs(self.output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.output_dir, "laplacian.png"), np.uint8(np.absolute(laplacian)))
        cv2.imwrite(os.path.join(self.output_dir, "zero_cross.png"), zero_cross)
        cv2.imwrite(os.path.join(self.output_dir, "final_mask.png"), final_mask)
        
        return final_mask
