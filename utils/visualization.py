import matplotlib.pyplot as plt
import cv2

def create_comparison_plot(original, results, titles, save_path):
    """
    results: list of masks or images
    titles: list of titles
    """
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(20, 5))
    
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    for i, (res, title) in enumerate(zip(results, titles)):
        if len(res.shape) == 3:
            axes[i+1].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        else:
            axes[i+1].imshow(res, cmap='gray')
        axes[i+1].set_title(title)
        axes[i+1].axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
