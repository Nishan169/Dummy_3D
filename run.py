"""python run.py --photos photos/ --prompt "A person enters through the door, ..." --out out/
   python run.py --demo --out out/        # synthetic room, no models / photos needed"""
import argparse
import functools
import http.server
import numpy as np
from recon.script_parser import parse_script, scene_labels
from recon.planner import plan
from recon.export import write_scene, camera_rigs
from recon.navgrid import NavGrid
from recon.mesh import build_mesh
from recon.layout import room_bounds
from recon.register import Frame

DEMO_PROMPT = ("A person enters through the door, walks up to the person sitting on the chair, "
               "stabs him and then leaves.")


def synthetic_room(seed=0):
    rng = np.random.default_rng(seed)
    P, C = [], []

    def quad(a, b, c, n, col):
        a, b, c = map(np.array, (a, b, c))
        u, v = rng.random((n, 1)), rng.random((n, 1))
        P.append(a + u * (b - a) + v * (c - a))
        C.append(np.clip(np.array(col) + rng.normal(0, 8, (n, 3)), 0, 255))

    quad((-2.5, 0, -2), (2.5, 0, -2), (-2.5, 0, 2), 60000, (190, 170, 140))    # floor
    quad((-2.5, 0, -2), (2.5, 0, -2), (-2.5, 2.7, -2), 30000, (205, 205, 210))  # walls
    quad((-2.5, 0, 2), (2.5, 0, 2), (-2.5, 2.7, 2), 30000, (205, 205, 210))
    quad((-2.5, 0, -2), (-2.5, 0, 2), (-2.5, 2.7, -2), 24000, (195, 200, 205))
    quad((2.5, 0, -2), (2.5, 0, 2), (2.5, 2.7, -2), 24000, (195, 200, 205))
    quad((-0.45, 0, 1.98), (0.45, 0, 1.98), (-0.45, 2.0, 1.98), 6000, (110, 70, 40))  # door
    quad((0.5, 0.45, -0.8), (1.1, 0.45, -0.8), (0.5, 0.45, -0.2), 2500, (90, 60, 40))  # chair seat
    quad((0.5, 0.45, -0.8), (1.1, 0.45, -0.8), (0.5, 1.0, -0.8), 2500, (90, 60, 40))   # chair back
    quad((-1.5, 0.75, -1.2), (-0.3, 0.75, -1.2), (-1.5, 0.75, -0.4), 5000, (120, 90, 60))  # table
    quad((-0.3, 0.9, 0.45), (0.35, 0.9, 0.45), (-0.3, 0.9, 0.75), 3000, (80, 100, 120))   # cabinet in the way
    quad((-0.3, 0, 0.75), (0.35, 0, 0.75), (-0.3, 0.9, 0.75), 3000, (80, 100, 120))
    quad((-0.3, 0, 0.45), (0.35, 0, 0.45), (-0.3, 0.9, 0.45), 3000, (80, 100, 120))
    quad((-0.3, 0, 0.45), (-0.3, 0, 0.75), (-0.3, 0.9, 0.45), 2000, (80, 100, 120))
    quad((0.35, 0, 0.45), (0.35, 0, 0.75), (0.35, 0.9, 0.45), 2000, (80, 100, 120))
    anchors = {"door": {"pos": [0.0, 1.98], "inward": [0.0, -1.0]},
               "chair": {"pos": [0.8, -0.5], "inward": [-0.8, 0.5]}}
    return np.concatenate(P), np.concatenate(C), anchors


def render_view(pts, cols, pos, target, size=(320, 240), f=280.0, splat=2):
    """Z-buffer the demo cloud from a virtual camera -> a Frame with colour + depth (no ML needed)."""
    w, h = size
    pos, fwd = np.asarray(pos, float), np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd], 1)  # camera: x right, y down, z forward
    pc = (pts - pos) @ R
    m = pc[:, 2] > 0.1
    pc, c = pc[m], cols[m]
    z = pc[:, 2]
    u, v = np.round(pc[:, 0] / z * f + w / 2).astype(int), np.round(pc[:, 1] / z * f + h / 2).astype(int)
    depth, img = np.full((h, w), np.inf, np.float32), np.zeros((h, w, 3), np.uint8)
    offs = [(du, dv) for du in range(-splat, splat + 1) for dv in range(-splat, splat + 1)]
    for du, dv in offs:
        uu, vv = u + du, v + dv
        ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        np.minimum.at(depth, (vv[ok], uu[ok]), z[ok])
    for du, dv in offs:
        uu, vv = u + du, v + dv
        ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        sel = ok.copy()
        sel[ok] = z[ok] <= depth[vv[ok], uu[ok]] + 0.01
        img[vv[sel], uu[sel]] = c[sel]
    depth[np.isinf(depth)] = 0
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, pos
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)
    return Frame(img, depth, K, None, None), T


def synthetic_views(pts, cols):
    cams = [((0, 1.6, 1.9), (0, 1.0, -1.5)), ((-2.3, 1.8, -1.8), (1, 0.8, 1)), ((2.3, 1.8, 1.8), (-1, 0.8, -1))]
    out = [render_view(pts, cols, p, t) for p, t in cams]
    return [f for f, _ in out], {i: T for i, (_, T) in enumerate(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos")
    ap.add_argument("--prompt", default=DEMO_PROMPT)
    ap.add_argument("--out", default="out")
    ap.add_argument("--demo", action="store_true", help="synthetic room, no ML models needed")
    ap.add_argument("--anchor", action="append", default=[], help="override, e.g. door=1.2,-0.4")
    ap.add_argument("--f-scale", type=float, default=0.9, help="focal length / max(w,h)")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--mesh-stride", type=int, help="pixels per mesh vertex (default 4; smaller = denser)")
    ap.add_argument("--no-mesh", action="store_true")
    a = ap.parse_args()

    script = parse_script(a.prompt)
    print("[script]", [f"{e['actor']}:{e['action']}" for e in script["events"]])

    if a.demo:
        pts, cols, anchors = synthetic_room()
        frames, poses, M, bounds = *synthetic_views(pts, cols), np.eye(4), room_bounds(pts)
    else:
        from recon.pipeline import load_frames, reconstruct
        from recon.semantic import locate
        frames = load_frames(a.photos, a.f_scale)
        poses, M, pts, cols, bounds = reconstruct(frames)
        center = np.median(pts[:, [0, 2]], axis=0)
        anchors, missing = locate(frames, poses, M, scene_labels(script), center)
        for kv in a.anchor:
            k, v = kv.split("=")
            x, z = map(float, v.split(","))
            d = center - np.array([x, z])
            anchors[k] = {"pos": [x, z], "inward": (d / (np.linalg.norm(d) + 1e-9)).tolist()}
        missing = [m for m in missing if m not in anchors]
        if missing:
            raise SystemExit(f"Could not find {missing} in the photos. Re-run with --anchor name=x,z")
    center = np.median(pts[:, [0, 2]], axis=0)
    grid = NavGrid.from_cloud(pts)
    mesh = None if a.no_mesh else build_mesh(frames, poses, M, bounds, a.mesh_stride or (2 if a.demo else 4))
    if mesh is not None:
        print(f"[mesh] {len(mesh[0]):,} vertices, {len(mesh[2]):,} triangles")
    write_scene(a.out, pts, cols, anchors, plan(script, anchors, center, grid), grid, mesh, camera_rigs(frames, poses, M))
    print(f"[done] {len(pts):,} points -> {a.out}/  (open index.html via a local server)")
    if a.serve:
        h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=a.out)
        print("Serving on http://localhost:8000")
        http.server.ThreadingHTTPServer(("", 8000), h).serve_forever()


if __name__ == "__main__":
    main()
