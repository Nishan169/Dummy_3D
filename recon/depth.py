"""Monocular metric depth (Depth Anything V2, indoor-metric variant)."""
import cv2
import numpy as np
import torch
from transformers import pipeline

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class DepthEstimator:
    def __init__(self, model=MODEL):
        self.pipe = pipeline("depth-estimation", model=model,
                             device=0 if torch.cuda.is_available() else -1)

    def __call__(self, pil_img):
        d = self.pipe(pil_img)["predicted_depth"].squeeze().cpu().numpy()
        return cv2.resize(d, pil_img.size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
