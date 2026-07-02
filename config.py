import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Algorithm Results Directories
SOBEL_RES = os.path.join(RESULTS_DIR, "sobel")
PREWITT_RES = os.path.join(RESULTS_DIR, "prewitt")
LOG_RES = os.path.join(RESULTS_DIR, "log")
CANNY_RES = os.path.join(RESULTS_DIR, "canny")
COMPARISON_RES = os.path.join(RESULTS_DIR, "comparison")

# Parameters
GAUSSIAN_KERNEL = (5, 5)
MORPH_KERNEL = (5, 5)
CANNY_THRESH1 = 50
CANNY_THRESH2 = 150
