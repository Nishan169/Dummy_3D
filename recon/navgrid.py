"""Floor-plan occupancy grid built from the reconstructed cloud + A* path planning,
so actors walk *around* furniture instead of through it."""
import heapq
import cv2
import numpy as np
from scipy import ndimage

MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]


class NavGrid:
    def __init__(self, obstacle, origin, cell):
        self.obs, self.origin, self.cell = obstacle, np.asarray(origin, float), cell
        self.nx, self.nz = obstacle.shape
        self._nearest = None
        if obstacle.any() and not obstacle.all():  # nearest free cell for each cell
            self._nearest = ndimage.distance_transform_edt(obstacle, return_indices=True)[1]

    @classmethod
    def from_cloud(cls, pts, cell=0.1, radius=0.3, y_lo=0.25, y_hi=1.7, min_pts=3):
        """Solid stuff between knee and head height *above the local floor* blocks walking."""
        lo = pts[:, [0, 2]].min(0) - 0.2
        shape = np.ceil((pts[:, [0, 2]].max(0) + 0.2 - lo) / cell).astype(int)
        ij = np.floor((pts[:, [0, 2]] - lo) / cell).astype(int)
        ok = (ij >= 0).all(1) & (ij < shape).all(1)
        pts, ij = pts[ok], ij[ok]
        known = np.zeros(shape, int)
        np.add.at(known, (ij[:, 0], ij[:, 1]), 1)
        ymin = np.full(shape, np.inf)
        np.minimum.at(ymin, (ij[:, 0], ij[:, 1]), pts[:, 1])
        floor = ndimage.minimum_filter(ymin, size=7)  # lowest nearby point ~ floor, even under table tops
        rel = pts[:, 1] - floor[ij[:, 0], ij[:, 1]]
        sel = (rel > y_lo) & (rel < y_hi)
        cnt = np.zeros(shape, int)
        np.add.at(cnt, (ij[sel, 0], ij[sel, 1]), 1)  # sparse cells = depth noise
        raw = cnt >= min_pts
        if known.any() and raw[known > 0].mean() > 0.6:
            print("[warn] obstacle map looks unreliable (floor/scale problem); ignoring furniture for routes")
            raw[:] = False
        r = max(1, int(round(radius / cell)))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return cls(ndimage.binary_dilation(raw, structure=xx ** 2 + yy ** 2 <= r * r), lo, cell)

    def _ij(self, p):
        ij = np.floor((np.asarray(p, float) - self.origin) / self.cell).astype(int)
        return int(np.clip(ij[0], 0, self.nx - 1)), int(np.clip(ij[1], 0, self.nz - 1))

    def _xz(self, ij):
        return self.origin + (np.asarray(ij, float) + 0.5) * self.cell

    def _free(self, ij):
        if self._nearest is None or not self.obs[ij]:
            return ij
        return int(self._nearest[0][ij]), int(self._nearest[1][ij])

    def _los(self, p, q):
        n = int(max(abs(q[0] - p[0]), abs(q[1] - p[1]))) * 2 + 1
        return not any(self.obs[int(round(p[0] + (q[0] - p[0]) * t)), int(round(p[1] + (q[1] - p[1]) * t))]
                       for t in np.linspace(0, 1, n + 1))

    def path(self, a, b):
        """Smoothed obstacle-free polyline from a to b (xz). Falls back to a straight line."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        s, g = self._free(self._ij(a)), self._free(self._ij(b))
        if s == g:
            return [a, b]
        cost, came, heap = {s: 0.0}, {}, [(0.0, 0.0, s)]
        while heap:
            _, c, cur = heapq.heappop(heap)
            if cur == g:
                break
            if c > cost[cur]:
                continue
            for dx, dz, w in MOVES:
                n = (cur[0] + dx, cur[1] + dz)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.nz) or self.obs[n]:
                    continue
                if dx and dz and (self.obs[cur[0] + dx, cur[1]] or self.obs[cur[0], cur[1] + dz]):
                    continue  # no cutting corners
                nc = c + w
                if nc < cost.get(n, 1e18):
                    cost[n], came[n] = nc, cur
                    heapq.heappush(heap, (nc + np.hypot(g[0] - n[0], g[1] - n[1]), nc, n))
        if g not in came:
            print("[warn] no walkable route found; using a straight line")
            return [a, b]
        cells = [g]
        while cells[-1] != s:
            cells.append(came[cells[-1]])
        cells.reverse()
        keep, i = [cells[0]], 0  # line-of-sight shortcutting
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._los(cells[i], cells[j]):
                j -= 1
            keep.append(cells[j])
            i = j
        return [a] + [self._xz(c) for c in keep[1:-1]] + [b]

    def debug_image(self, paths, anchors, scale=6):
        """Top-down map: dark = blocked, lines = planned routes. Handy for sanity checks."""
        img = np.full((self.nz, self.nx, 3), 235, np.uint8)
        img[self.obs.T] = (95, 85, 80)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        px = lambda p: tuple(int(v) for v in (np.asarray(p) - self.origin) / self.cell * scale)
        for k, (aid, lines) in enumerate(paths.items()):
            for pl in lines:
                cv2.polylines(img, [np.array([px(p) for p in pl], np.int32)], False, [(40, 160, 230), (220, 150, 60)][k % 2], 2)
        for name, a in anchors.items():
            cv2.circle(img, px(a["pos"]), 6, (60, 60, 200), -1)
            cv2.putText(img, name, px(a["pos"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        return img
