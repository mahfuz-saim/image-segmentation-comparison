import cv2
import numpy as np
import os
from .base_segmenter import Segmenter
from utils.preprocessing import to_grayscale, apply_gaussian_blur


class RegionGrowingSegmenter(Segmenter):
    def __init__(self, output_dir: str, threshold: int = 25):
        self.output_dir = output_dir
        self.threshold = threshold

    def segment(self, image: np.ndarray) -> np.ndarray:
        # Pipeline: Grayscale -> Gaussian Blur -> Seed at center -> Flood-Fill with intensity tolerance
        gray = to_grayscale(image)
        blurred = apply_gaussian_blur(gray)

        h, w = blurred.shape
        seed = (w // 2, h // 2)
        seed_value = int(blurred[seed[1], seed[0]])

        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = (
            cv2.FLOODFILL_FIXED_RANGE
            | cv2.FLOODFILL_MASK_ONLY
        )
        cv2.floodFill(
            blurred,
            mask,
            seed,
            newVal=255,
            loDiff=self.threshold,
            upDiff=self.threshold,
            flags=flags,
        )

        region = mask[1:-1, 1:-1]

        final_mask = np.where(region > 0, 255, 0).astype(np.uint8)

        os.makedirs(self.output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.output_dir, "blurred.png"), blurred)
        cv2.imwrite(os.path.join(self.output_dir, "region.png"), final_mask)

        return final_mask
