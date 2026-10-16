"""Depth-map meshing: every registered photo becomes a textured triangle grid.

Neighbouring pixels are joined into triangles unless they straddle a depth jump (object
edges, so foreground never gets 'webbed' to the background). Views are merged in the
aligned frame; each vertex carries its photo colour."""
import cv2
import numpy as np
from .geometry import backproject
from .layout import in_room


def depth_mesh(frame, T, stride=4, max_depth=12.0, rel_jump=0.08):
    s = stride
    d = frame.depth[::s, ::s]
    h, w = d.shape
    img = cv2.resize(frame.img, (w, h), interpolation=cv2.INTER_AREA)
    K = frame.K.copy()
    K[:2] /= s
    xyz = backproject(d, K).reshape(-1, 3)
    z = d.ravel()
    idx = np.arange(h * w).reshape(h, w)
    tl, tr, bl, br = (a.ravel() for a in (idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]))

    def keep(a, b, c):
        zz = np.stack([z[a], z[b], z[c]], 1)
        ok = (zz > 0.2).all(1) & (zz < max_depth).all(1) & ((zz.max(1) - zz.min(1)) < rel_jump * zz.mean(1))
        return np.stack([a, b, c], 1)[ok]

    F = np.concatenate([keep(tl, bl, tr), keep(tr, bl, br)])
    v0, v1, v2 = xyz[F[:, 0]], xyz[F[:, 1]], xyz[F[:, 2]]  # drop faces seen almost edge-on (stretched streaks)
    n, c = np.cross(v1 - v0, v2 - v0), (v0 + v1 + v2) / 3
    F = F[np.abs((n * c).sum(1)) > 0.03 * np.linalg.norm(n, axis=1) * np.linalg.norm(c, axis=1)]
    return xyz @ T[:3, :3].T + T[:3, 3], img.reshape(-1, 3), F


def build_mesh(frames, poses, M, bounds, stride=4):
    """Merge all views -> (vertices float32, colours uint8, faces uint32) in the aligned frame."""
    Vs, Cs, Fs, off = [], [], [], 0
    for i, T in poses.items():
        V, C, F = depth_mesh(frames[i], M @ T, stride)
        F = F[in_room(V, bounds, pad=0.1)[F].all(1)]  # drop geometry outside the room
        used = np.unique(F)
        remap = np.full(len(V), -1, np.int64)
        remap[used] = np.arange(len(used))
        Vs.append(V[used]); Cs.append(C[used]); Fs.append(remap[F] + off)
        off += len(used)
    return (np.concatenate(Vs).astype(np.float32), np.concatenate(Cs).astype(np.uint8),
            np.concatenate(Fs).astype(np.uint32))
