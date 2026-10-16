import os
import numpy as np
from PIL import Image, ImageOps
from .geometry import intrinsics
from .register import make_frame, build_poses
from .layout import fuse, align_to_floor, apply, crop_room


def load_frames(photo_dir, f_scale=0.9, max_side=640):
    from .depth import DepthEstimator
    paths = sorted(os.path.join(photo_dir, p) for p in os.listdir(photo_dir)
                   if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    if len(paths) < 2:
        raise SystemExit(f"Need at least 2 photos in {photo_dir}")
    est, frames = DepthEstimator(), []
    for p in paths:
        im = ImageOps.exif_transpose(Image.open(p).convert("RGB"))
        im.thumbnail((max_side, max_side))
        print(f"[depth] {os.path.basename(p)}")
        frames.append(make_frame(np.array(im), est(im), intrinsics(im.width, im.height, f_scale)))
    return frames


def reconstruct(frames):
    """photos -> (poses, floor-alignment M, cloud pts, colours, room bounds) in a Y-up metric-ish frame."""
    poses = build_poses(frames)
    pts, cols = fuse(frames, poses)
    M = align_to_floor(pts)
    pts, cols, bounds = crop_room(apply(M, pts), cols)
    return poses, M, pts, cols, bounds
