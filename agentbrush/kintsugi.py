"""Kintsugi: a broken thing joined again with lacquer and gold, and the joins left to show.

`shatter` breaks a panel the way one blow breaks a tile: cracks run out from the blow nearly straight (`rays`),
others run round it from one crack to the next (`rings`), and the piece under the blow itself is lost. It
says which piece every point of the panel belongs to, so a scene can treat the pieces apart (cut several
pictures together). `join` then lays gold in every seam and over the lost piece, as
metal leaf on the scene with its own relief, and sets each piece a hair out of plane with its neighbours, as
mended things lie.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import raster
from .noise import fbm, smoothstep

FINE_GOLD = (1.0, 0.77, 0.36)     # gold dusted on as powder: a deeper yellow than beaten leaf (scenes.GOLD)


@dataclass
class Break:
    piece: np.ndarray     # (H, W) int32: 0 where the piece is lost, else the piece's number, from 1
    names: list           # names[i - 1] = (sector, ring) of piece i: sectors round from the first ray, rings outward
    seam: np.ndarray      # how many pixels each point lies from the nearest crack
    knots: np.ndarray     # (n, 2) the points where cracks meet each other or run off the panel
    lost: np.ndarray      # coverage of the lost piece


def shatter(W, H, blow, rays, rings, lost, seed=0, pins=None, drift=9.0):
    """Break a W x H panel with a blow at `blow` (pixels). `rays` are the bearings of the cracks that run
    out from it, in degrees counter-clockwise from east, ascending. `rings[i]` lists the cracks that cross
    the sector between ray i and the next, each as the distances from the blow at which it meets those two
    rays. `lost[i]` is how far along ray i the lost piece reaches. A ray wanders `drift` pixels (at 2048
    across) off the straight; `pins[i]` is a distance at which ray i must lie exactly on its bearing."""
    rng = np.random.default_rng(seed)
    n = len(rays)
    cx, cy = blow
    k = W / 2048.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx + 0.5 - cx, cy - (yy + 0.5)
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    reach = float(np.hypot(max(cx, W - cx), max(cy, H - cy)))
    rs = np.linspace(0.0, reach, 1024)
    bearing = np.empty((n, rs.size))
    for i, a in enumerate(rays):
        off = (rng.uniform(0.5, 1.0) * drift * k * np.sin(rs / (rng.uniform(0.28, 0.5) * reach) + rng.uniform(0.0, 6.3))
               + 0.3 * drift * k * np.sin(rs / (rng.uniform(0.07, 0.13) * reach) + rng.uniform(0.0, 6.3)))
        off -= off[0]
        if pins and i in pins:
            off -= np.interp(pins[i], rs, off) * rs / pins[i]
        bearing[i] = np.radians(a) + off / np.maximum(rs, 60.0 * k)

    def at(i, rad):
        """The point on ray i that far from the blow."""
        b = np.interp(rad, rs, bearing[i % n])
        return cx + rad * np.cos(b), cy - rad * np.sin(b)

    # which sector each point lies in, and how far across it (0 at its first ray, 1 at its second)
    two_pi = 2.0 * np.pi
    sector = np.zeros((H, W), np.int32)
    across = np.zeros((H, W), np.float32)
    first = np.interp(r, rs, bearing[0]).astype(np.float32)
    lo = first
    for i in range(n):
        hi = first if i == n - 1 else np.interp(r, rs, bearing[i + 1]).astype(np.float32)
        span = np.mod(hi - lo, two_pi)
        rel = np.mod(th - lo, two_pi)
        inside = rel < span
        sector[inside] = i
        across[inside] = (rel / np.maximum(span, 1e-6))[inside]
        lo = hi

    piece = np.zeros((H, W), np.int32)
    names, knots = [], []
    for i in range(n):
        m = sector == i
        t, ri = across[m], r[m]
        cut = np.zeros(t.shape, np.int32)
        half = 0.5 * np.radians((rays[(i + 1) % n] - rays[i]) % 360.0)
        for ra, rb in rings.get(i, ()):
            # between an arc round the blow and the straight chord, and never quite either
            sag = rng.uniform(0.3, 1.2) * 0.5 * (ra + rb) * (1.0 - np.cos(half))
            line = ra + (rb - ra) * t - sag * 4.0 * t * (1.0 - t)
            line += 0.004 * (ra + rb) * np.sin(t * rng.uniform(5.0, 11.0) + rng.uniform(0.0, 6.3)) * np.sin(np.pi * t)
            cut += ri > line
            knots += [at(i, ra), at(i + 1, rb)]
        for j in range(len(rings.get(i, ())) + 1):
            names.append((i, j))
            sub = piece[m]
            sub[cut == j] = len(names)
            piece[m] = sub

    # the piece under the blow: its corners on the rays, its sides the pieces' broken tips
    corners = []
    for i in range(n):
        corners.append(at(i, lost[i]))
        a = np.interp(lost[i], rs, bearing[i])
        b = np.interp(lost[(i + 1) % n], rs, bearing[(i + 1) % n])
        mid = a + np.mod(b - a, two_pi) * rng.uniform(0.35, 0.65)
        rad = 0.5 * (lost[i] + lost[(i + 1) % n]) * rng.uniform(0.9, 1.1)
        corners.append((cx + rad * np.cos(mid), cy - rad * np.sin(mid)))
        knots.append(at(i, lost[i]))
    gone = np.zeros((H, W), np.float32)
    shp = raster.polygon(corners, H, W)
    gone[shp[0]] = shp[1]
    piece[gone > 0.5] = 0

    for i in range(n):          # where each ray runs off the panel
        px, py = cx + rs * np.cos(bearing[i]), cy - rs * np.sin(bearing[i])
        out = np.nonzero((px < 0) | (px > W - 1) | (py < 0) | (py > H - 1))[0]
        if out.size:
            knots.append((float(np.clip(px[out[0]], 0, W - 1)), float(np.clip(py[out[0]], 0, H - 1))))

    cracks = np.zeros((H, W), bool)
    cracks[:, :-1] |= piece[:, :-1] != piece[:, 1:]
    cracks[:-1, :] |= piece[:-1, :] != piece[1:, :]
    seam = distance_transform_edt(~cracks).astype(np.float32)
    return Break(piece, names, seam, np.array(knots, np.float64), gone)


def touching(piece, least=12):
    """The pairs of pieces that share a seam at least `least` pixels long."""
    count = {}
    for a, b in ((piece[:, :-1], piece[:, 1:]), (piece[:-1, :], piece[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        code, num = np.unique(np.minimum(a[m], b[m]).astype(np.int64) * 100000 + np.maximum(a[m], b[m]), return_counts=True)
        for c, q in zip(code, num):
            count[int(c)] = count.get(int(c), 0) + int(q)
    return sorted((c // 100000, c % 100000) for c, q in count.items() if q >= least)


def four_colour(pairs, count, pins=None):
    """Give each of `count` pieces (numbered from 1) one of four numbers, 0 to 3, so that no two pieces in
    `pairs` (from `touching`) share one: four are always enough for pieces on a flat panel. `pins` fixes some
    pieces' numbers; the rest take whichever number their neighbours allow that has been used least so far,
    so the four come out about even. Returns the numbers in piece order."""
    near = {i: set() for i in range(1, count + 1)}
    for a, b in pairs:
        near[a].add(b)
        near[b].add(a)
    out = dict(pins or {})
    order = sorted((i for i in near if i not in out), key=lambda i: (-len(near[i]), i))
    for i, c in out.items():
        assert all(out.get(j) != c for j in near[i]), f'pinned piece {i} touches another pinned the same'

    def place(j):
        if j == len(order):
            return True
        i = order[j]
        used = [sum(1 for c in out.values() if c == q) for q in range(4)]
        for c in sorted(range(4), key=lambda q: (used[q], q)):
            if all(out.get(n) != c for n in near[i]):
                out[i] = c
                if place(j + 1):
                    return True
                del out[i]
        return False

    assert place(0), 'no way to give these pieces four numbers with those pins'
    return [out[i] for i in range(1, count + 1)]


def join(scene, brk, rng, width=0.0058, bead=1.5, tilt=0.006, metal=FINE_GOLD, rough=0.36, lacquer=(0.27, 0.08, 0.05),
         redrawn=None, pale=(0.97, 0.88, 0.60)):
    """Join the pieces: gold in every seam (`width` of the picture's width across, never even, wider where a
    chip was lost from an edge and where cracks meet) and gold for the lost piece. The seams stand `bead`
    proud, a hair of the lacquer under the gold shows along their edges, and each piece is tipped about
    `tilt` (a slope) out of plane, so a light breaks at every join. `redrawn` is line work (a coverage mask)
    for what the lost piece showed: it is drawn on the gold again in a paler gold, raised and burnished, as
    a restorer continues a lost pattern in maki-e."""
    H, W = scene.H, scene.W
    s = W / 2048.0
    hw0 = 0.5 * width * W
    half = hw0 * (1.0 + 0.2 * fbm(H, W, 150.0 * s, rng, 2))
    half += 0.7 * hw0 * smoothstep(1.3, 2.2, fbm(H, W, 60.0 * s, rng, 2))
    for kx, ky in brk.knots:
        rad = hw0 * rng.uniform(1.6, 2.6)
        x0, x1 = int(max(kx - 3.0 * rad, 0)), int(min(kx + 3.0 * rad + 1, W))
        y0, y1 = int(max(ky - 3.0 * rad, 0)), int(min(ky + 3.0 * rad + 1, H))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        half[y0:y1, x0:x1] += rng.uniform(0.5, 1.1) * hw0 * np.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / (rad * rad))
    seam = np.clip(half - brk.seam + 0.5, 0.0, 1.0)
    gold = np.maximum(seam, brk.lost)
    plate = raster.blur(brk.lost, 1.4 * s)
    edge = np.maximum(np.clip(half + 1.5 * s - brk.seam + 0.5, 0.0, 1.0), np.clip(4.0 * plate - 0.3, 0.0, 1.0))
    scene.overlay(0.85 * edge, lacquer)

    grain = np.clip(fbm(H, W, 2.0 * max(s, 0.6), rng, 2), -2.0, 2.0)       # gold dusted on as powder, then burnished
    scene.gild(gold, metal, rough)
    laid = scene.leaf > 0.0
    scene.leaf_color[laid] = (np.asarray(metal, np.float32) * (0.93 + 0.045 * grain)[..., None])[laid]
    scene.leaf_rough[laid] = np.clip(rough + 0.06 * grain, 0.05, 1.0)[laid]
    # the relief: a bead of lacquer under each seam; the lost piece built up level, brushed, its edge rounded
    dome = np.sqrt(np.clip(1.0 - (brk.seam / np.maximum(half, 1e-3)) ** 2, 0.0, 1.0))
    fill = smoothstep(0.1, 0.9, raster.blur(brk.lost, 3.0 * s))
    brushed = 0.22 * fbm(H, W, (2.0 * s, 46.0 * s), rng, 2) + 0.3 * fbm(H, W, 26.0 * s, rng, 2)
    scene.tool += bead * np.maximum(dome, 0.8 * fill) + brushed * brk.lost
    if redrawn is not None:
        line = np.clip(redrawn, 0.0, 1.0) * (brk.lost > 0.5)
        scene.leaf_color += (np.asarray(pale, np.float32) - scene.leaf_color) * line[..., None]
        scene.leaf_rough += (0.18 - scene.leaf_rough) * line
        scene.tool += 0.8 * bead * raster.blur(line, 0.8 * s)
    table = rng.normal(0.0, tilt, (len(brk.names) + 1, 2))
    table[0] = 0.0
    scene.tilt = table[brk.piece].astype(np.float32)
