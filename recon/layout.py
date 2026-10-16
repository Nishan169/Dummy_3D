"""Fuse registered depth maps into one coloured cloud and level it on the floor."""
import numpy as np
from .geometry import backproject


def fuse(frames, poses, stride=3, voxel=0.02, max_depth=12.0):
    P, C = [], []
    for i, T in poses.items():
        f = frames[i]
        d = f.depth
        grad = np.abs(np.gradient(d, axis=0)) + np.abs(np.gradient(d, axis=1))
        ok = (d > 0.2) & (d < max_depth) & (grad < 0.08 * d)  # drop flying pixels at edges
        xyz = backproject(d, f.K)
        sel = np.zeros_like(ok)
        sel[::stride, ::stride] = True
        m = ok & sel
        P.append(xyz[m] @ T[:3, :3].T + T[:3, 3])
        C.append(f.img[m])
    P, C = np.concatenate(P), np.concatenate(C)
    _, idx = np.unique(np.floor(P / voxel).astype(np.int64), axis=0, return_index=True)
    return P[idx], C[idx]


def align_to_floor(pts, up=(0.0, -1.0, 0.0), iters=800, tol=0.04, seed=0):
    """RANSAC the floor plane (normal near `up`; camera y points down, so up=-y) and
    return a 4x4 that maps points into a Y-up frame with the floor at y=0."""
    rng = np.random.default_rng(seed)
    up = np.asarray(up, float)
    h = pts @ up
    low = pts[h < np.percentile(h, 35)]
    sub = pts[rng.choice(len(pts), min(40000, len(pts)), replace=False)]
    best, best_c = None, 0
    for _ in range(iters):
        s = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ up < 0:
            n = -n
        if n @ up < 0.92:
            continue
        c = int((np.abs((sub - s[0]) @ n) < tol).sum())
        if c > best_c:
            best, best_c = (n, s[0]), c
    if best is None:
        raise RuntimeError("Could not find a floor. Make sure the floor is visible in the photos.")
    n, p0 = best
    inl = pts[np.abs((pts - p0) @ n) < tol]
    c = inl.mean(0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][-1]  # refit with SVD
    n = n if n @ up > 0 else -n
    y = np.array([0.0, 1.0, 0.0])
    v, cos = np.cross(n, y), float(n @ y)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx / (1 + cos)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -(R @ c)[1] * y  # floor -> y = 0
    return M


def apply(M, pts):
    return pts @ M[:3, :3].T + M[:3, 3]


def room_bounds(pts):
    return np.percentile(pts[:, [0, 2]], 1, axis=0), np.percentile(pts[:, [0, 2]], 99, axis=0)


def in_room(pts, bounds, pad=0.0):
    lo, hi = bounds
    return ((pts[:, 1] > -0.15) & (pts[:, 1] < 4.0) & (pts[:, 0] > lo[0] - pad) & (pts[:, 0] < hi[0] + pad)
            & (pts[:, 2] > lo[1] - pad) & (pts[:, 2] < hi[1] + pad))


def crop_room(pts, cols):
    b = room_bounds(pts)
    m = in_room(pts, b)
    return pts[m], cols[m], b
