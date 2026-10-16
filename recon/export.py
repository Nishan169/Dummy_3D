import json
import os
import shutil
from pathlib import Path
import numpy as np


def to_jsonable(obj):
    """Convert NumPy types recursively into JSON-compatible Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            str(k): to_jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    return obj


def camera_rigs(frames, poses, M):
    """Where each photo was taken, in the aligned world frame."""
    out = []

    for i in sorted(poses):
        W = M @ poses[i]

        R = W[:3, :3] / np.cbrt(
            abs(np.linalg.det(W[:3, :3]))
        )

        f = frames[i]

        out.append({
            "pos": W[:3, 3].round(4).tolist(),
            "fwd": R[:, 2].round(4).tolist(),
            "fov_y": float(
                np.degrees(
                    2 * np.arctan(
                        f.depth.shape[0] / 2 / f.K[1, 1]
                    )
                )
            ),
        })

    return out


def write_scene(
    out,
    pts,
    cols,
    anchors,
    plan,
    grid=None,
    mesh=None,
    cameras=None,
):
    os.makedirs(out, exist_ok=True)

    # --------------------------------------------------
    # Point cloud
    # --------------------------------------------------
    with open(f"{out}/cloud.bin", "wb") as f:
        # float32 XYZ
        f.write(pts.astype("<f4").tobytes())

        # uint8 RGB
        f.write(cols.astype(np.uint8).tobytes())

    # --------------------------------------------------
    # Scene metadata
    # --------------------------------------------------
    meta = {
        "n_points": int(len(pts)),
        "cloud": "cloud.bin",

        "bounds": {
            "min": pts.min(0).tolist(),
            "max": pts.max(0).tolist(),
        },

        "anchors": to_jsonable(anchors),
        "plan": to_jsonable(plan),
        "cameras": to_jsonable(cameras or []),
    }

    # --------------------------------------------------
    # Mesh
    # --------------------------------------------------
    if mesh is not None:
        V, C, F = mesh

        V = np.asarray(V)
        C = np.asarray(C)
        F = np.asarray(F)

        with open(f"{out}/mesh.bin", "wb") as f:
            # float32 XYZ
            f.write(V.astype("<f4").tobytes())

            # uint8 RGB
            f.write(C.astype(np.uint8).tobytes())

            # Padding so triangle indices start on 4-byte boundary
            f.write(b"\0" * ((-len(V) * 15) % 4))

            # uint32 triangle indices
            f.write(F.astype("<u4").tobytes())

        meta["mesh"] = {
            "file": "mesh.bin",
            "n_verts": int(len(V)),
            "n_faces": int(len(F)),
        }

    # --------------------------------------------------
    # Convert EVERYTHING before JSON serialization
    # --------------------------------------------------
    meta = to_jsonable(meta)

    with open(f"{out}/scene.json", "w") as f:
        json.dump(meta, f)

    # --------------------------------------------------
    # Navigation debug image
    # --------------------------------------------------
    if grid is not None:
        import cv2

        cv2.imwrite(
            f"{out}/nav.png",
            grid.debug_image(plan["paths"], anchors),
        )

    # --------------------------------------------------
    # Viewer
    # --------------------------------------------------
    shutil.copy(
        Path(__file__).parent.parent / "viewer" / "index.html",
        out,
    )