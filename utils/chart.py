import os
import re
import matplotlib.pyplot as plt
import numpy as np


METRIC_COLUMNS = ["Time(ms)", "Regions", "Edge Pixels"]


def parse_metrics_file(path: str):
    """Parse a metrics_*.txt file into a dict {algorithm: {metric: value}}."""
    results = {}
    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines[2:]:
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            continue
        algo, time_ms, regions, edges = parts
        try:
            results[algo] = {
                "Time(ms)": float(time_ms),
                "Regions": int(regions),
                "Edge Pixels": int(edges),
            }
        except ValueError:
            continue
    return results


def collect_metrics(comparison_dir: str):
    """
    Read every metrics_*.txt file in the comparison folder.
    Returns:
        image_metrics: {image_name: {algorithm: {metric: value}}}
        algorithms: ordered list of algorithm names
    """
    image_metrics = {}
    algorithms = []

    files = sorted(
        f for f in os.listdir(comparison_dir)
        if f.startswith("metrics_") and f.endswith(".txt")
    )

    for fname in files:
        path = os.path.join(comparison_dir, fname)
        parsed = parse_metrics_file(path)
        if not parsed:
            continue

        # Image name = filename without "metrics_" prefix and ".txt" suffix
        image_name = fname[len("metrics_"):-len(".txt")]

        # Keep only images that contain the full current algorithm set
        # (skip stale files from earlier runs that still list LoG)
        for algo in parsed:
            if algo not in algorithms:
                algorithms.append(algo)

        image_metrics[image_name] = parsed

    return image_metrics, algorithms


def create_comparison_barcharts(comparison_dir: str, save_dir: str):
    """
    Build one bar chart per metric (Time, Regions, Edge Pixels).
    Each chart shows that metric for every algorithm across all images.
    """
    os.makedirs(save_dir, exist_ok=True)

    image_metrics, algorithms = collect_metrics(comparison_dir)
    if not image_metrics:
        print("No metrics files found.")
        return

    image_names = list(image_metrics.keys())
    n_images = len(image_names)
    n_algorithms = len(algorithms)

    bar_width = 0.8 / max(n_algorithms, 1)
    x = np.arange(n_images)

    # Stable color per algorithm
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_algorithms)]

    for metric in METRIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(max(8, n_images * 1.5), 5))

        for i, algo in enumerate(algorithms):
            values = []
            for img in image_names:
                algo_data = image_metrics[img].get(algo)
                if algo_data is None:
                    values.append(0)
                else:
                    values.append(algo_data.get(metric, 0))
            ax.bar(
                x + i * bar_width - 0.4 + bar_width / 2,
                values,
                width=bar_width,
                label=algo,
                color=colors[i],
            )

        ax.set_title(f"{metric} per Image")
        ax.set_xlabel("Image")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, rotation=30, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        safe_metric = metric.replace("(", "").replace(")", "").replace(" ", "_")
        out_path = os.path.join(save_dir, f"barchart_{safe_metric}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")