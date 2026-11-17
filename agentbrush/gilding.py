"""Gold-leaf tooling: incised lines, punched dots and rosettes, stippling, rays and haloes.

Everything writes relief into scene.tool (dents are negative) and roughness into
scene.leaf_rough, and only where leaf is laid, so paint put afterwards wipes it."""
import numpy as np

from .noise import smoothstep
from . import raster


def _cup(r):
    k = int(np.ceil(r * 1.6)) + 1
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1].astype(np.float32)
    d = np.hypot(xx, yy) / r
    cup = -np.clip(1.0 - d * d, 0.0, None)                           # the dent
    burr = 0.2 * np.clip(1.0 - np.abs(d - 1.2) / 0.35, 0.0, None)    # metal pushed up round it
    return k, (cup + burr).astype(np.float32)


def punch(sc, pts, r, depth=1.0, rough=0.42):
    """Round punch marks at (x, y) points."""
    k, ker = _cup(r)
    scat = np.clip(-ker * 2.0, 0.0, 1.0)
    H, W = sc.H, sc.W
    for x, y in pts:
        ix, iy = int(round(x)), int(round(y))
        x0, y0 = max(ix - k, 0), max(iy - k, 0)
        x1, y1 = min(ix + k + 1, W), min(iy + k + 1, H)
        if x0 >= x1 or y0 >= y1:
            continue
        kx, ky = x0 - (ix - k), y0 - (iy - k)
        kk = ker[ky:ky + y1 - y0, kx:kx + x1 - x0]
        lf = sc.leaf[y0:y1, x0:x1]
        sc.tool[y0:y1, x0:x1] += depth * kk * lf
        rr = sc.leaf_rough[y0:y1, x0:x1]
        rr += (rough - rr) * scat[ky:ky + y1 - y0, kx:kx + x1 - x0] * lf


def incise(sc, pts, width=1.0, depth=0.6):
    """A groove along a polyline."""
    pts = list(pts)
    shape = raster.polyline(pts, np.full(len(pts), width), sc.H, sc.W)
    if shape is None:
        return
    win, m = shape[0], shape[1]
    sc.tool[win] -= depth * m * sc.leaf[win]


def incise_circle(sc, cx, cy, r, width=1.0, depth=0.6):
    """A circular groove (compass-scribed)."""
    x0, y0 = max(int(cx - r - width - 2), 0), max(int(cy - r - width - 2), 0)
    x1, y1 = min(int(cx + r + width + 3), sc.W), min(int(cy + r + width + 3), sc.H)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.abs(np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) - r)
    sc.tool[y0:y1, x0:x1] -= depth * np.clip(width + 0.5 - d, 0.0, 1.0) * sc.leaf[y0:y1, x0:x1]


def ring_of_punches(sc, cx, cy, r, spacing, pr, depth=1.0):
    n = max(6, int(2 * np.pi * r / spacing))
    a = np.arange(n) * 2 * np.pi / n
    punch(sc, zip(cx + r * np.cos(a), cy + r * np.sin(a)), pr, depth)


def rosette(sc, x, y, r, pr, petals=8, depth=1.0):
    a = np.arange(petals) * 2 * np.pi / petals
    punch(sc, zip(x + r * np.cos(a), y + r * np.sin(a)), pr, depth)
    punch(sc, [(x, y)], pr * 1.25, depth)


def lattice(sc, spacing, angle_deg=40.0, width=0.8, depth=0.45, rosette_r=8.0, pr=1.5):
    """Incised lozenge diaper over the leaf: rosettes at the crossings, a dot in each lozenge."""
    H, W = sc.H, sc.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    for s in (1.0, -1.0):
        t = xx * ca + s * yy * sa
        d = np.abs((t / spacing + 0.5) % 1.0 - 0.5) * spacing
        sc.tool -= depth * smoothstep(width + 0.8, width - 0.3, d) * sc.leaf
    # crossings: x ca + y sa = i*sp and x ca - y sa = j*sp
    ii = np.arange(-2, int((W * ca + H * sa) / spacing) + 3)
    jj = np.arange(int(-H * sa / spacing) - 3, int(W * ca / spacing) + 3)
    I, J = np.meshgrid(ii, jj)
    for off in (0.0, 0.5):
        t1, t2 = (I + off) * spacing, (J + off) * spacing
        x, y = ((t1 + t2) / (2 * ca)).ravel(), ((t1 - t2) / (2 * sa)).ravel()
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x, y = x[ok], y[ok]
        on = sc.leaf[y.astype(int), x.astype(int)] > 0.5
        for px, py in zip(x[on], y[on]):
            if off == 0.0:
                rosette(sc, px, py, rosette_r, pr)
            else:
                punch(sc, [(px, py)], pr * 1.2)


def stipple(sc, mask, rng, spacing=4.0, r=1.0, depth=0.5):
    """Granulated leaf, punched all over: matte and glittering."""
    H, W = sc.H, sc.W
    ys, xs = np.mgrid[0:H:spacing, 0:W:spacing].astype(np.float32)
    xs = xs + rng.uniform(-0.35, 0.35, xs.shape) * spacing
    ys = ys + rng.uniform(-0.35, 0.35, ys.shape) * spacing
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys = xs[ok], ys[ok]
    on = mask[ys.astype(int), xs.astype(int)] > 0.5
    punch(sc, zip(xs[on], ys[on]), r, depth, rough=0.5)


def rays(sc, cx, cy, r0, r_long, r_short, n, width=1.0, depth=0.6, pr=1.8, phase=0.0):
    """Alternating long and short incised rays, each ending in a punched dot."""
    for i in range(n):
        a = phase + 2 * np.pi * i / n
        r1 = r_long if i % 2 == 0 else r_short
        c, s = np.cos(a), np.sin(a)
        incise(sc, [(cx + r0 * c, cy + r0 * s), (cx + r1 * c, cy + r1 * s)], width, depth)
        punch(sc, [(cx + (r1 + 3 * pr) * c, cy + (r1 + 3 * pr) * s)], pr)


def halo(sc, cx, cy, r, S=1.0):
    """Punched halo: incised rims, a ring of dots, and rosettes inside."""
    incise_circle(sc, cx, cy, r, 0.9 * S, 0.6)
    incise_circle(sc, cx, cy, r - 7 * S, 0.8 * S, 0.5)
    ring_of_punches(sc, cx, cy, r - 3.5 * S, 7 * S, 1.3 * S)
    rr = r - 22 * S
    n = max(6, int(2 * np.pi * rr / (34 * S)))
    for i in range(n):
        a = 2 * np.pi * i / n
        rosette(sc, cx + rr * np.cos(a), cy + rr * np.sin(a), 6 * S, 1.4 * S, 6)
