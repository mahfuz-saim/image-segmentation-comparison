import math
import matplotlib.pyplot as plt
import cv2


def create_comparison_plot(original, results, titles, save_path):
    """
    Build a grid layout: 2 rows x 3 columns.
    First slot shows the original image, remaining slots show the results.
    results: list of masks or images
    titles: list of titles
    """
    items = [("Original", original)] + list(zip(titles, results))
    total = len(items)
    cols = 3
    rows = math.ceil(total / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, (title, img) in zip(axes, items):
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
