import cv2
import numpy as np
import os
from .base_segmenter import Segmenter
from utils.preprocessing import to_grayscale, apply_gaussian_blur
from utils.postprocessing import fill_contours


class WatershedSegmenter(Segmenter):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def segment(self, image: np.ndarray) -> np.ndarray:
        # Pipeline: Grayscale -> Blur -> Threshold -> Distance Transform -> Markers -> Watershed
        gray = to_grayscale(image)
        blurred = apply_gaussian_blur(gray)

        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sure_bg = cv2.dilate(thresh, kernel, iterations=3)

        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist, 0.7 * dist.max(), 255, cv2.THRESH_BINARY
        )
        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(image, markers)

        final_mask = np.zeros_like(gray)
        final_mask[markers > 1] = 255
        final_mask = fill_contours(final_mask)

        os.makedirs(self.output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.output_dir, "thresh.png"), thresh)
        cv2.imwrite(
            os.path.join(self.output_dir, "dist_transform.png"),
            np.uint8(dist),
        )
        cv2.imwrite(os.path.join(self.output_dir, "final_mask.png"), final_mask)

        return final_mask
