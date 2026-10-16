"""Find named things (door, chair, ...) from the prompt in the photos, in 3D."""
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from .geometry import lift


def _inward(p, center):
    d = np.asarray(center, float) - np.asarray(p, float)
    n = np.linalg.norm(d)
    return (d / n).tolist() if n > 1e-6 else [0.0, 1.0]


def locate(frames, poses, M, labels, center, model="google/owlvit-base-patch32", min_score=0.08):
    """Open-vocabulary detection per image; the best box per label is lifted to a floor
    position (box bottom-centre + depth), then moved into the aligned world frame."""
    det = pipeline("zero-shot-object-detection", model=model,
                   device=0 if torch.cuda.is_available() else -1)
    best = {}
    for idx, T in poses.items():
        f = frames[idx]
        H, W = f.depth.shape
        for r in det(Image.fromarray(f.img), candidate_labels=[f"a {l}" for l in labels]):
            name = r["label"][2:]
            if r["score"] < min_score or r["score"] <= best.get(name, (0,))[0]:
                continue
            b = r["box"]
            u, v = (b["xmin"] + b["xmax"]) / 2, min(b["ymax"] - 3, H - 1)
            patch = f.depth[int(max(v - 4, 0)):int(v + 4), int(max(u - 4, 0)):int(u + 4)]
            z = float(np.median(patch))
            p = np.array([(u - f.K[0, 2]) / f.K[0, 0] * z, (v - f.K[1, 2]) / f.K[1, 1] * z, z])
            w = (M @ T) @ np.append(p, 1.0)
            size = [(b["xmax"] - b["xmin"]) / f.K[0, 0] * z, (b["ymax"] - b["ymin"]) / f.K[1, 1] * z]
            best[name] = (r["score"], [float(w[0]), float(w[2])], size)
    anchors = {n: dict(pos=p, inward=_inward(p, center), size=s, score=round(float(sc), 3))
               for n, (sc, p, s) in best.items()}
    return anchors, [l for l in labels if l not in anchors]
