import numpy as np
import cv2

def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union != 0 else 0

def calculate_dice(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    sum_masks = gt_mask.sum() + pred_mask.sum()
    return (2. * intersection) / sum_masks if sum_masks != 0 else 0

def calculate_precision_recall(gt_mask, pred_mask):
    tp = np.logical_and(gt_mask, pred_mask).sum()
    fp = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return precision, recall, f1

def get_basic_stats(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_pixels = np.sum(cv2.Canny(mask, 100, 200) > 0)
    return {
        "num_regions": len(contours),
        "edge_pixels": int(edge_pixels)
    }
