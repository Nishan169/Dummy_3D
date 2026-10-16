"""Action script + 3D anchors -> time-stamped tracks the viewer plays back.

Everything is in the aligned world frame (metres, Y up); tracks use floor (x, z).
With a NavGrid, walking follows A* routes around furniture."""
import numpy as np

SPEEDS = {"walk": 1.3, "run": 2.8}  # m/s


def _yaw(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    return float(np.arctan2(d[0], d[1]))


def plan(script, anchors, center, grid=None):
    actors = {a["id"]: a for a in script["actors"]}
    tracks = {k: [] for k in actors}
    paths = {k: [] for k in actors}
    dist = {k: 0.0 for k in actors}
    pos, captions, impacts, collapsed = {}, [], [], set()
    center = np.asarray(center, float)
    clock = {"t": 1.0}

    def seg(aid, kind, t0, t1, a, b, yaw, **kw):
        tracks[aid].append({"kind": kind, "t0": round(t0, 3), "t1": round(t1, 3),
                            "from": np.round(np.asarray(a, float), 3).tolist(),
                            "to": np.round(np.asarray(b, float), 3).tolist(),
                            "yaw": round(float(yaw), 3), **kw})

    def anchor(name):
        if name not in anchors:
            raise KeyError(f"'{name}' was not located in the scene; pass --anchor {name}=x,z")
        return np.array(anchors[name]["pos"], float), np.array(anchors[name]["inward"], float)

    def where(name):
        return pos[name] if name in actors else anchor(name)[0]

    def walk(aid, a, b, nav=True, pace="walk"):
        pace = pace if pace in SPEEDS else "walk"
        a, b = np.asarray(a, float), np.asarray(b, float)
        pts = grid.path(a, b) if (grid is not None and nav) else [a, b]
        paths[aid].append(np.round(np.array(pts), 3).tolist())
        for p, q in zip(pts[:-1], pts[1:]):
            d = float(np.linalg.norm(q - p))
            if d < 1e-3:
                continue
            dur = max(0.2, d / SPEEDS[pace])
            seg(aid, "walk", clock["t"], clock["t"] + dur, p, q, _yaw(p, q),
                length=round(d, 3), dist0=round(dist[aid], 3), pace=pace)
            clock["t"] += dur
            dist[aid] += d
        pos[aid] = b

    def collapse(tgt, ti):
        if tgt in collapsed or tgt not in actors:
            return
        v = tracks[tgt][-1]
        pose = "sit" if v["kind"] == "sit" else "stand"
        seg(tgt, "collapse", ti, ti + 1.3, pos[tgt], pos[tgt], v["yaw"], from_pose=pose)
        impacts.append({"t": round(ti, 3), "pos": [float(pos[tgt][0]), 0.95 if pose == "sit" else 1.2, float(pos[tgt][1])]})
        collapsed.add(tgt)

    for a in actors.values():
        s, aid = a.get("start", {}), a["id"]
        if "at" in s:
            p = anchor(s["at"])[0]
            pos[aid] = p
            seg(aid, "sit" if s.get("pose", "sitting") == "sitting" else "idle", 0, 1.0, p, p, _yaw(p, center))
        else:
            pos[aid] = None
            seg(aid, "hidden", 0, 1.0, [0, 0], [0, 0], 0)

    for ev in script["events"]:
        aid, act, tgt = ev["actor"], ev["action"], ev.get("target")
        pace, role, t = ev.get("pace", "walk"), actors[ev["actor"]]["role"], clock["t"]
        if act == "enter":
            p, inw = anchor(tgt)
            captions.append({"t": t, "text": f"The {role} enters through the {tgt}"})
            walk(aid, p - inw * 1.0, p + inw * 0.9, nav=False, pace=pace)
        elif act == "walk_to":
            tp, cur = where(tgt), pos[aid]
            d = cur - tp
            n = np.linalg.norm(d)
            d = d / n if n > 1e-6 else np.array([0.0, 1.0])
            captions.append({"t": t, "text": f"The {role} walks up to the {tgt}"})
            walk(aid, cur, tp + d * 0.75, pace=pace)
        elif act == "attack":
            tp, cur = where(tgt), pos[aid]
            yaw, style = _yaw(cur, tp), ev.get("style", "stab")
            captions.append({"t": t, "text": f"The {role} {'stabs' if style == 'stab' else 'strikes'} the {tgt}"})
            seg(aid, "attack", t, t + 1.6, cur, cur, yaw, style=style)
            collapse(tgt, t + 0.8)
            seg(aid, "idle", t + 1.6, t + 2.4, cur, cur, yaw)
            clock["t"] = t + 2.4
        elif act == "collapse":
            collapse(aid, t)
        elif act == "leave":
            p, inw = anchor(tgt)
            captions.append({"t": t, "text": f"The {role} {'runs out' if pace == 'run' else 'leaves'} through the {tgt}"})
            walk(aid, pos[aid], p + inw * 0.5, pace=pace)
            walk(aid, p + inw * 0.5, p - inw * 1.4, nav=False, pace=pace)
            seg(aid, "hidden", clock["t"], clock["t"] + 1, [0, 0], [0, 0], 0)
        elif act == "wait":
            sec = float(ev.get("seconds", 1.0))
            seg(aid, "idle", t, t + sec, pos[aid], pos[aid], 0)
            clock["t"] += sec

    captions.sort(key=lambda c: c["t"])
    return {"duration": round(clock["t"] + 1.5, 2), "tracks": tracks, "paths": paths, "captions": captions,
            "impacts": impacts, "roles": {k: a["role"] for k, a in actors.items()}}
