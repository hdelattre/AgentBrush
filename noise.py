"""Noise fields and small array helpers (numpy + Pillow only)."""
import numpy as np
from PIL import Image


def C(r, g, b):
    return np.array([r, g, b], np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, np.float32) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    """Lerp that broadcasts a per-pixel weight over RGB when either end is a colour."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    t = np.asarray(t, np.float32)
    if t.ndim >= 1 and (a.shape[-1:] == (3,) or b.shape[-1:] == (3,)) and t.shape[-1:] != (3,):
        t = t[..., None]
    return a + (b - a) * t


def _cells(cell):
    if np.isscalar(cell):
        return float(cell), float(cell)
    return float(cell[0]), float(cell[1])


def value_noise(h, w, cell, rng):
    """Bicubic value noise in ~[0, 1] with features about `cell` px (scalar or (cy, cx))."""
    cy, cx = _cells(cell)
    gh = max(2, int(round(h / cy)) + 1)
    gw = max(2, int(round(w / cx)) + 1)
    grid = rng.random((gh, gw)).astype(np.float32)
    up = Image.fromarray(grid, mode='F').resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(up, np.float32)


def fbm(h, w, cell, rng, octaves=5, gain=0.5):
    """Fractal value noise normalised to zero mean, unit std."""
    cy, cx = _cells(cell)
    total = np.zeros((h, w), np.float32)
    amp = 1.0
    for _ in range(octaves):
        total += amp * value_noise(h, w, (max(cy, 1.0), max(cx, 1.0)), rng)
        amp *= gain
        cy *= 0.5
        cx *= 0.5
        if cy < 1.0 and cx < 1.0:
            break
    total -= total.mean()
    return total / (total.std() + 1e-6)


def fbm1(n, cell, rng, octaves=5):
    return fbm(1, n, (1.0, cell), rng, octaves)[0]


def hash01(a, b=0, c=0):
    """Deterministic pseudo-random value in [0, 1) per integer triple (vectorised)."""
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    c = np.asarray(c, np.int64)
    h = (a * 374761393 + b * 668265263 + c * 1442695041 + 144665) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return (h & 0xFFFFFF).astype(np.float32) / np.float32(0x1000000)


def voronoi(px, pz, seed, jitter=0.85):
    """Jittered-grid Voronoi (cell size 1). Returns nearest/second ids and distances."""
    ci = np.floor(px).astype(np.int64)
    cj = np.floor(pz).astype(np.int64)
    d1 = np.full(px.shape, 1e9, np.float32)
    d2 = d1.copy()
    i1 = np.zeros(px.shape, np.int64)
    i2 = i1.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            gi = ci + di
            gj = cj + dj
            fx = gi + 0.5 + jitter * (hash01(gi, gj, seed) - 0.5)
            fz = gj + 0.5 + jitter * (hash01(gi, gj, seed + 1) - 0.5)
            d = np.sqrt((px - fx) ** 2 + (pz - fz) ** 2).astype(np.float32)
            gid = (gi + 5000) * 10007 + (gj + 5000)
            closer = d < d1
            second = ~closer & (d < d2)
            d2 = np.where(closer, d1, np.where(second, d, d2))
            i2 = np.where(closer, i1, np.where(second, gid, i2))
            d1 = np.where(closer, d, d1)
            i1 = np.where(closer, gid, i1)
    return i1, i2, d1, d2


def catmull(points, n=12):
    """Smooth polyline through control points (Catmull-Rom)."""
    P = np.asarray(points, np.float64)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t * t * t))
    out.append(P[-2])
    return np.array(out)
