from abc import ABC, abstractmethod
import numpy as np

class Segmenter(ABC):
    """Abstract Base Class for Image Segmenters."""
    
    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform segmentation on the input image.
        Returns:
            np.ndarray: The final binary mask.
        """
        pass

    def save_intermediate(self, image: np.ndarray, path: str):
        """Utility to save intermediate processing steps."""
        import cv2
        cv2.imwrite(path, image)
