"""Camera + similarity-transform maths (all from scratch, numpy only)."""
import numpy as np


def intrinsics(w, h, f_scale=0.9):
    """Guess a pinhole camera: focal ~0.9*max(w,h) is a typical phone wide lens."""
    f = f_scale * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)


def lift(uv, depth, K):
    """Pixels (N,2) + depth map -> 3D points (N,3) in camera frame (x right, y down, z fwd)."""
    h, w = depth.shape
    u, v = uv[:, 0], uv[:, 1]
    z = depth[np.clip(v.astype(int), 0, h - 1), np.clip(u.astype(int), 0, w - 1)]
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)


def backproject(depth, K):
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)


def umeyama(src, dst):
    """Least-squares similarity transform: dst ~= s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var = (s0 ** 2).sum() / len(src)
    U, D, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / var
    return s, R, mu_d - s * R @ mu_s


def ransac_sim3(src, dst, iters=800, thresh=0.1, seed=0):
    """Robust 3D-3D alignment. Returns (4x4 matrix, n_inliers) or None."""
    n = len(src)
    if n < 8:
        return None
    rng = np.random.default_rng(seed)
    best, best_c = None, 0
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        s, R, t = umeyama(src[idx], dst[idx])
        if not (0.5 < s < 2.0):
            continue
        inl = np.linalg.norm(s * src @ R.T + t - dst, axis=1) < thresh
        c = int(inl.sum())
        if c > best_c:
            best, best_c = inl, c
    if best is None or best_c < 8:
        return None
    s, R, t = umeyama(src[best], dst[best])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, t
    return T, best_c
