"""Field drawing: the subject is never drawn, only the field round it.

Lay a sheet of paper over a magnet and shake iron filings onto it: the filings line up along the field and
the magnet shows as the place they bristle from. That is the whole medium. A scene says which shapes are
charged and how (`solve`: the shapes are held at +1 and -1 and the paper's edge at 0, and the potential
between them is relaxed until it settles); `lines` then draws the lines of force, as many leaving each shape
as its charge deserves, so they crowd where the field is strong and never touch; and `strew` lays filings
along them as metal on the scene, slivers a few pixels long with their own relief, a few lying loose in
between. The shapes themselves stay bare paper.
"""
import numpy as np
from numba import njit
from scipy.ndimage import binary_dilation, zoom

from .raster import blur


@njit(fastmath=True, cache=True)
def _relax(phi, held, sweeps, omega, sides):
    H, W = phi.shape
    for _ in range(sweeps):
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if not held[y, x]:
                    phi[y, x] += omega * (0.25 * (phi[y, x - 1] + phi[y, x + 1] + phi[y - 1, x] + phi[y + 1, x]) - phi[y, x])
        if sides:                                           # the side edges are not earthed: no line ends on them
            for y in range(H):
                if not held[y, 0]:
                    phi[y, 0] = phi[y, 1]
                if not held[y, W - 1]:
                    phi[y, W - 1] = phi[y, W - 2]


def solve(plus, minus):
    """The potential over the sheet with `plus` held at +1, `minus` at -1 and the sheet's top and bottom edges
    at 0. The side edges are left open, for shapes that come in from the side. Worked coarse to fine, and
    returned at full size."""
    H, W = plus.shape
    phi = None
    for k in (16, 8, 4, 2):
        small = lambda m: m.reshape(H // k, k, W // k, k).mean(axis=(1, 3))
        p, m = small(plus) > 0.5, small(minus) > 0.5
        held = p | m
        held[0, :] = held[-1, :] = True
        phi = np.zeros(p.shape) if phi is None else zoom(phi, 2.0, order=1)
        phi[0, :] = phi[-1, :] = 0.0
        phi[p], phi[m] = 1.0, -1.0
        _relax(phi, held, 400 if k > 2 else 250, 1.9, True)
    return blur(zoom(phi, 2.0, order=1).astype(np.float32), 1.5)


@njit(fastmath=True, cache=True)
def _trace(ex, ey, body, x, y, sign, step, most, out):
    """Follow the field from (x, y), the way it points (sign +1) or against it. Returns how many points were
    written to `out` and where the line ended: 0 off the sheet or lost, else the number of the body it ran into."""
    H, W = ex.shape
    n = 0
    for _ in range(most):
        out[n, 0], out[n, 1] = x, y
        n += 1
        px, py = x, y
        for half in range(2):                              # midpoint rule
            ix, iy = int(px), int(py)
            if ix < 1 or iy < 1 or ix > W - 3 or iy > H - 3:
                return n, 0
            fx, fy = px - ix, py - iy
            vx = (ex[iy, ix] * (1 - fx) + ex[iy, ix + 1] * fx) * (1 - fy) + (ex[iy + 1, ix] * (1 - fx) + ex[iy + 1, ix + 1] * fx) * fy
            vy = (ey[iy, ix] * (1 - fx) + ey[iy, ix + 1] * fx) * (1 - fy) + (ey[iy + 1, ix] * (1 - fx) + ey[iy + 1, ix + 1] * fx) * fy
            m = np.sqrt(vx * vx + vy * vy)
            if m < 1e-9:
                return n, 0
            if half == 0:
                px, py = x + 0.5 * step * sign * vx / m, y + 0.5 * step * sign * vy / m
            else:
                x, y = x + step * sign * vx / m, y + step * sign * vy / m
        ix, iy = int(x), int(y)
        if ix < 1 or iy < 1 or ix > W - 3 or iy > H - 3:
            return n, 0
        if n > 6 and body[iy, ix] > 0:
            return n, body[iy, ix]
    return n, 0


def lines(phi, plus, minus, count, step=2.0):
    """The lines of force: `count` leave the plus shape, spaced round it so each carries the same share of its
    charge, and run to the minus shape or off the sheet; then the minus shape gets the lines that reach it
    from off the sheet, at the same charge a line. Returns [(points, joined)], `joined` true for a line that
    runs from one shape to the other."""
    H, W = phi.shape
    gy, gx = np.gradient(phi)
    ex, ey = np.ascontiguousarray(-gx, np.float32), np.ascontiguousarray(-gy, np.float32)
    body = np.zeros((H, W), np.int32)
    body[plus > 0.5], body[minus > 0.5] = 1, 2
    buf = np.empty((int(4 * (H + W) / step), 2))
    out = []
    share = None
    for mask, sign, other in ((plus > 0.5, 1.0, 2), (minus > 0.5, -1.0, 1)):
        ys, xs = np.nonzero(binary_dilation(mask, iterations=2) & ~mask)
        ok = (xs > 2) & (ys > 2) & (xs < W - 3) & (ys < H - 3)
        xs, ys = xs[ok], ys[ok]
        cy, cx = np.nonzero(mask)[0].mean(), np.nonzero(mask)[1].mean()
        order = np.argsort(np.arctan2(ys - cy, xs - cx))
        xs, ys = xs[order], ys[order]
        flux = np.cumsum(np.hypot(ex[ys, xs], ey[ys, xs]))
        if share is None:
            share = flux[-1] / count
        for q in (np.arange(int(flux[-1] / share)) + 0.5) * share:
            i = min(int(np.searchsorted(flux, q)), xs.size - 1)
            n, end = _trace(ex, ey, body, xs[i] + 0.5, ys[i] + 0.5, sign, step, len(buf), buf)
            if n > 8 and (sign > 0 or end != other):            # the joined ones were drawn from the plus side already
                out.append((buf[:n].copy(), end == other))
    return out


@njit(fastmath=True, cache=True)
def _strew(cov, col, tool, F):
    """Filings: a row of F is one sliver (x, y, angle, length, half-width, r, g, b, how proud it stands)."""
    H, W = cov.shape
    for i in range(F.shape[0]):
        x, y, a, L, hw = F[i, 0], F[i, 1], F[i, 2], F[i, 3], F[i, 4]
        ca, sa = np.cos(a), np.sin(a)
        reach = 0.5 * L + hw + 1.5
        for iy in range(max(int(y - reach), 0), min(int(y + reach) + 2, H)):
            for ix in range(max(int(x - reach), 0), min(int(x + reach) + 2, W)):
                dx, dy = ix + 0.5 - x, iy + 0.5 - y
                t = min(max(dx * ca + dy * sa, -0.5 * L), 0.5 * L)
                d = np.sqrt((dx - t * ca) ** 2 + (dy - t * sa) ** 2)
                c = min(max(hw - d + 0.5, 0.0), 1.0)
                if c > cov[iy, ix]:
                    cov[iy, ix] = c
                    col[iy, ix, 0], col[iy, ix, 1], col[iy, ix, 2] = F[i, 5], F[i, 6], F[i, 7]
                    tool[iy, ix] = c * F[i, 8] * (1.0 - 0.6 * (d / max(hw, 0.5)) ** 2)


def strew(scene, drawn, phi, rng, metals, loose=0.00004, rough=0.42, bodies=None, fringe=0.45):
    """Lay filings on the scene along the lines `drawn` (from `lines`): `metals` = (colour of the filings on a
    joined line, colour of the rest). Each line is a few slivers wide, never quite in step; `loose` is how many
    more lie between the lines, per pixel, turned the way the field runs there but owing nothing to any line.
    `bodies` is the charged shapes together: `fringe` of the points along their edges get a sliver standing off."""
    H, W = scene.H, scene.W
    s = W / 2048.0
    rows = []
    for pts, joined in drawn:
        seg = np.diff(pts, axis=0)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        for lane in range(3):
            at = np.arange(rng.uniform(0.0, 8.0) * s, cum[-1], 8.5 * s)
            at = at[rng.random(at.size) < 0.86]
            i = np.clip(np.searchsorted(cum, at) - 1, 0, len(seg) - 1)
            ang = np.arctan2(seg[i, 1], seg[i, 0])
            off = (lane - 1) * 1.5 * s + rng.normal(0.0, 0.6 * s, at.size)
            tone = rng.uniform(0.72, 1.0, at.size)[:, None] * np.asarray(metals[0 if joined else 1])
            rows.append(np.column_stack([np.interp(at, cum, pts[:, 0]) - np.sin(ang) * off, np.interp(at, cum, pts[:, 1]) + np.cos(ang) * off,
                                         ang + rng.normal(0.0, 0.1, at.size), rng.uniform(7.0, 15.0, at.size) * s, rng.uniform(0.6, 1.05, at.size) * s,
                                         tone, rng.uniform(0.6, 1.2, at.size)]))
    gy, gx = np.gradient(phi)
    n = int(loose * H * W / (s * s))
    x, y = rng.uniform(4, W - 4, n), rng.uniform(4, H - 4, n)
    ix, iy = x.astype(np.int64), y.astype(np.int64)
    strength = np.hypot(gx[iy, ix], gy[iy, ix])
    keep = (rng.random(n) < np.clip(strength / np.percentile(strength, 90), 0.05, 1.0)) & (np.abs(phi[iy, ix]) < 0.97)
    x, y, ix, iy = x[keep], y[keep], ix[keep], iy[keep]
    tone = rng.uniform(0.6, 0.95, x.size)[:, None] * np.asarray(metals[1])
    rows.append(np.column_stack([x, y, np.arctan2(-gy[iy, ix], -gx[iy, ix]) + rng.normal(0.0, 0.25, x.size), rng.uniform(5.0, 11.0, x.size) * s,
                                 rng.uniform(0.5, 0.9, x.size) * s, tone, rng.uniform(0.5, 1.0, x.size)]))
    if bodies is not None:
        # at a charged shape's edge the filings stand thick, end on: that fringe is all the outline it gets
        ys, xs = np.nonzero(binary_dilation(bodies > 0.5, iterations=1) & ~(bodies > 0.5))
        pick = rng.random(xs.size) < fringe
        xs, ys = xs[pick].astype(np.float64), ys[pick].astype(np.float64)
        ang = np.arctan2(gy[ys.astype(np.int64), xs.astype(np.int64)], gx[ys.astype(np.int64), xs.astype(np.int64)])
        ang = ang + np.where(phi[ys.astype(np.int64), xs.astype(np.int64)] > 0.0, np.pi, 0.0)       # pointing away from the shape
        length = rng.uniform(5.0, 13.0, xs.size) * s
        tone = rng.uniform(0.55, 0.9, xs.size)[:, None] * np.asarray(metals[1])
        rows.append(np.column_stack([xs + (0.5 * length + 1.0) * np.cos(ang), ys + (0.5 * length + 1.0) * np.sin(ang),
                                     ang + rng.normal(0.0, 0.12, xs.size), length, rng.uniform(0.55, 0.95, xs.size) * s, tone,
                                     rng.uniform(0.6, 1.2, xs.size)]))
    F = np.concatenate(rows)
    F = F[rng.permutation(len(F))]
    cov = np.zeros((H, W), np.float32)
    col = np.zeros((H, W, 3), np.float32)
    tool = np.zeros((H, W), np.float32)
    _strew(cov, col, tool, F)
    scene.overlay(0.35 * blur(cov, 1.6 * s), (0.25, 0.23, 0.22))          # what little shadow they cast on the paper
    scene.leaf = cov
    scene.leaf_color = col
    scene.leaf_rough = np.full((H, W), rough, np.float32)
    scene.tool = tool
    return len(F)
