import cv2
import numpy as np
from typing import Tuple


def resize_for_check(frame, size: Tuple[int, int] = (128, 128)):
    small = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def brightness_mean(gray):
    return float(np.mean(gray))


def std_dev(gray):
    return float(np.std(gray))

def laplacian_variance(gray):
    # ensure uint8 input to avoid OpenCV unsupported format errors
    gray_u8 = gray.astype(np.uint8)
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    return float(np.var(lap))


def histogram_spread(gray):
    hist = cv2.calcHist([gray.astype('uint8')], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    cdf = np.cumsum(hist)

    p5 = np.searchsorted(cdf, 0.05)
    p95 = np.searchsorted(cdf, 0.95)

    return float(p95 - p5)


def block_variances(gray, blocks=(8, 8)):
    h, w = gray.shape
    bh = h // blocks[0]
    bw = w // blocks[1]

    vars = np.zeros(blocks, dtype=np.float32)

    for r in range(blocks[0]):
        for c in range(blocks[1]):
            block = gray[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            vars[r, c] = np.std(block)

    return vars
