"""Pose-free multi-view registration.

No COLMAP: for every image pair we match SIFT features, lift both keypoints to 3D with
the depth maps, and solve a RANSAC similarity transform (Umeyama). Pairwise transforms
are chained along a maximum-inlier spanning tree into one common frame.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import lift, ransac_sim3


@dataclass
class Frame:
    img: np.ndarray    # HxWx3 uint8 RGB
    depth: np.ndarray  # HxW float32 (metres)
    K: np.ndarray
    kp: np.ndarray     # Nx2
    des: np.ndarray


def make_frame(img, depth, K):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kps, des = cv2.SIFT_create(4000).detectAndCompute(gray, None)
    return Frame(img, depth, K, np.array([k.pt for k in kps], np.float32), des)


def _refine(T0, pb, ua, za, K, w_depth=10.0):
    """Polish a b->a similarity with 2D reprojection error in image a plus a depth-consistency
    term. A 3D-vs-3D threshold hides small camera motion (e.g. a slow dolly); pixels do not."""
    s0 = float(np.cbrt(abs(np.linalg.det(T0[:3, :3]))))
    x0 = np.r_[Rotation.from_matrix(T0[:3, :3] / s0).as_rotvec(), T0[:3, 3], np.log(s0)]

    def res(x):
        q = np.exp(x[6]) * pb @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:6]
        z = np.maximum(q[:, 2], 1e-3)
        return np.r_[K[0, 0] * q[:, 0] / z + K[0, 2] - ua[:, 0], K[1, 1] * q[:, 1] / z + K[1, 2] - ua[:, 1],
                     w_depth * (np.log(z) - np.log(za))]

    x = least_squares(res, x0, loss="soft_l1", f_scale=2.0, max_nfev=60).x
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.exp(x[6]) * Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6]
    r = res(x)
    n = len(pb)
    inl = int(((np.abs(r[:n]) < 3) & (np.abs(r[n:2 * n]) < 3) & (np.abs(r[2 * n:]) < 1.0)).sum())
    return T, inl


def pair_transform(a, b):
    """Transform mapping camera-b coords -> camera-a coords, plus inlier count."""
    if a.des is None or b.des is None:
        return None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a.des, b.des, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 12:
        return None
    ua = a.kp[[m.queryIdx for m in good]]
    ub = b.kp[[m.trainIdx for m in good]]
    _, mask = cv2.findFundamentalMat(ua, ub, cv2.FM_RANSAC, 2.0, 0.99)  # drop 2D outliers
    if mask is None:
        return None
    keep = mask.ravel() == 1
    ua, pa, pb = ua[keep], lift(ua[keep], a.depth, a.K), lift(ub[keep], b.depth, b.K)
    ok = (pa[:, 2] > 0.2) & (pb[:, 2] > 0.2) & (pa[:, 2] < 25) & (pb[:, 2] < 25)
    ua, pa, pb = ua[ok], pa[ok], pb[ok]
    if len(pa) < 8:
        return None
    init = ransac_sim3(pb, pa, thresh=0.06 * float(np.median(pa[:, 2])))
    if init is None:
        return None
    try:
        T, inl = _refine(init[0], pb, ua, pa[:, 2], a.K)
        if inl >= 0.6 * init[1]:
            return T, inl
    except Exception:
        pass
    return init


def build_poses(frames, min_inliers=20):
    """Return {frame_idx: 4x4 cam->common-frame}. Unconnected frames are dropped."""
    n, edges = len(frames), {}
    for i in range(n):
        for j in range(i + 1, n):
            r = pair_transform(frames[i], frames[j])
            if r and r[1] >= min_inliers:
                edges[(i, j)] = r
    if not edges:
        raise RuntimeError("No image pairs overlap enough. Use photos with ~40% overlap.")
    strength = np.zeros(n)
    for (i, j), (_, k) in edges.items():
        strength[i] += k
        strength[j] += k
    ref = int(strength.argmax())
    poses = {ref: np.eye(4)}
    while True:  # Prim-style: always attach via the strongest edge
        best = None
        for (i, j), (T, k) in edges.items():
            if (i in poses) != (j in poses) and (best is None or k > best[0]):
                best = (k, i, j, T)
        if best is None:
            break
        _, i, j, T = best
        if i in poses:
            poses[j] = poses[i] @ T
        else:
            poses[i] = poses[j] @ np.linalg.inv(T)
    dropped = sorted(set(range(n)) - set(poses))
    if dropped:
        print(f"[warn] images not connected to the rest and ignored: {dropped}")
    return poses
