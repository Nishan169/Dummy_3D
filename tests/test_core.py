"""Run: python tests/test_core.py   (no models or photos needed)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recon.geometry import ransac_sim3
from recon.layout import align_to_floor, apply
from recon.navgrid import NavGrid
from recon.planner import plan
from recon.script_parser import rule_based, scene_labels
from recon.geometry import intrinsics
from recon.layout import room_bounds
from recon.mesh import build_mesh, depth_mesh
from recon.register import Frame
from run import synthetic_room, synthetic_views


def test_registration():
    rng = np.random.default_rng(1)
    src = rng.random((300, 3)) * 4
    a = 0.6
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    dst = 1.1 * src @ R.T + [0.5, 0.2, -1]
    dst[:90] += rng.normal(0, 2, (90, 3))  # 30% outliers
    T, n = ransac_sim3(src, dst)
    assert np.abs((src @ T[:3, :3].T + T[:3, 3] - dst)[100:]).max() < 1e-6 and n >= 200


def test_floor_alignment():
    pts, _, _ = synthetic_room()
    t = np.radians(8)  # camera frame: y down, 8 degree tilt, shifted
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    cam = (pts * [-1, -1, 1]) @ Rx.T + [0.3, 1.4, 2.0]
    out = apply(align_to_floor(cam), cam)
    assert np.abs(out[:60000, 1]).max() < 0.05, "floor not at y=0"  # first 60k pts are the floor
    assert 2.5 < out[:, 1].max() < 2.9, "up axis wrong"


def test_navigation_detours():
    pts, _, anchors = synthetic_room()
    grid = NavGrid.from_cloud(pts)
    path = grid.path([0.0, 1.08], [0.46, 0.17])
    assert len(path) > 2, "should bend around the cabinet"
    mid = [grid.obs[grid._ij(np.array(p))] for p in path[1:-1]]
    assert not any(mid), "waypoints must be free"


def test_plan_and_script():
    script = rule_based("A person enters through the door, walks up to the person sitting on the chair, stabs him and runs out.")
    assert scene_labels(script) == ["chair", "door"] and script["events"][-1].get("pace") == "run"
    pts, _, anchors = synthetic_room()
    p = plan(script, anchors, [0, 0], NavGrid.from_cloud(pts))
    kinds = [s["kind"] for s in p["tracks"]["victim"]]
    assert kinds == ["sit", "collapse"] and p["impacts"] and p["tracks"]["perp"][-1]["kind"] == "hidden"
    walks = [s for s in p["tracks"]["perp"] if s["kind"] == "walk"]
    assert any(w["pace"] == "run" for w in walks) and len(walks) >= 4


def test_mesh_breaks_at_depth_jumps():
    h, w = 40, 60
    d = np.full((h, w), 2.0, np.float32); d[:, 30:] = 3.0  # a step: wall at 2 m next to wall at 3 m
    fr = Frame(np.zeros((h, w, 3), np.uint8), d, intrinsics(w, h), None, None)
    V, C, F = depth_mesh(fr, np.eye(4), stride=1)
    assert len(F) == 2 * (h - 1) * (w - 2), "triangles across the jump must be dropped"
    cols = F % w
    assert not ((cols < 30).any(1) & (cols >= 30).any(1)).any()


def test_demo_mesh():
    pts, cols, _ = synthetic_room()
    frames, poses = synthetic_views(pts, cols)
    V, C, F = build_mesh(frames, poses, np.eye(4), room_bounds(pts), stride=2)
    assert len(F) > 20000 and F.max() < len(V)
    assert V[:, 0].min() > -2.7 and V[:, 0].max() < 2.7 and V[:, 1].min() > -0.2 and V[:, 1].max() < 3.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
