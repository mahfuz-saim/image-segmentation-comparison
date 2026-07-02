import os
import time
import cv2
import numpy as np
import config
from algorithms.sobel_segmenter import SobelSegmenter
from algorithms.prewitt_segmenter import PrewittSegmenter
from algorithms.log_segmenter import LoGSegmenter
from algorithms.canny_segmenter import CannySegmenter
from utils.image_loader import load_image
from utils.visualization import create_comparison_plot
from utils.metrics import get_basic_stats

def main():
    # 1. Setup folders
    input_files = [f for f in os.listdir(config.INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not input_files:
        print("No input images found in data/input. Generating a sample image...")
        sample_img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(sample_img, (200, 200), 100, (255, 255, 255), -1)
        cv2.rectangle(sample_img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.imwrite(os.path.join(config.INPUT_DIR, "sample.png"), sample_img)
        input_files = ["sample.png"]

    # 2. Initialize Segmenters
    segmenters = {
        "Sobel": SobelSegmenter(config.SOBEL_RES),
        "Prewitt": PrewittSegmenter(config.PREWITT_RES),
        "LoG": LoGSegmenter(config.LOG_RES),
        "Canny": CannySegmenter(config.CANNY_RES)
    }

    for filename in input_files:
        print(f"\nProcessing {filename}...")
        img_path = os.path.join(config.INPUT_DIR, filename)
        image = load_image(img_path)
        
        results = []
        titles = []
        metrics_table = []

        for name, segmenter in segmenters.items():
            start_time = time.perf_counter()
            mask = segmenter.segment(image)
            end_time = time.perf_counter()
            
            duration_ms = (end_time - start_time) * 1000
            stats = get_basic_stats(mask)
            
            results.append(mask)
            titles.append(name)
            metrics_table.append({
                "Algorithm": name,
                "Time(ms)": f"{duration_ms:.2f}",
                "Regions": stats["num_regions"],
                "Edge Pixels": stats["edge_pixels"]
            })
            
            print(f"{name:8} | Time: {duration_ms:8.2f}ms | Regions: {stats['num_regions']:3} | Edges: {stats['edge_pixels']:6}")

        # 3. Visualization
        comparison_path = os.path.join(config.COMPARISON_RES, f"comparison_{filename}")
        create_comparison_plot(image, results, titles, comparison_path)
        print(f"Comparison saved to {comparison_path}")

        # 4. Save Table to Text
        table_path = os.path.join(config.COMPARISON_RES, f"metrics_{filename}.txt")
        with open(table_path, "w") as f:
            f.write(f"{'Algorithm':12} | {'Time(ms)':10} | {'Regions':8} | {'Edge Pixels':12}\n")
            f.write("-" * 55 + "\n")
            for row in metrics_table:
                f.write(f"{row['Algorithm']:12} | {row['Time(ms)']:10} | {row['Regions']:8} | {row['Edge Pixels']:12}\n")

if __name__ == "__main__":
    main()
