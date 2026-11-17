"""Marbling: a picture made on water and lifted off it in one pull.

A tray is filled with size (water thickened with carrageen). Colour ground with ox gall is dropped on it and
does not sink or mix: each drop spreads to a disc, and pushes aside everything already floating. So the first
colours thrown end up squeezed into veins between the last ones (the stone pattern). The floating skin is
then drawn: a comb pulled through it turns stones into feathers and scales, and one needle draws a drop out
into a petal, a leaf, a stem. Nothing can be taken back. A sheet treated with alum is laid on the bath once,
and takes the whole picture off it.

A `Bath` records what the marbler does, in order: `drop` (and `spatter`, `row`), `tine` and `comb`, `stroke`
(the needle along a path) and `swirl`. None of it is rasterised as it happens. To pull the print, every point
of the sheet is followed backward through all of it (`_lift`): each act is undone in turn until the point
lands inside a drop at the moment that drop was let fall, and that is its colour. A drop's push and a comb's
drag are exact, area-keeping maps (Lu, Jaffer et al., "Mathematical Marbling"), so the print is sharp at any
size, down to the finest vein.

A style pulls it with one pass, `dict(kind='pull', paints={colour name: pigment parts, ...}, ...)`, on
`surface='laid'`: each colour is a film of that paint on the paper (pigments.py), and the sheet keeps what
sheets keep, the lines where the hands paused in laying it and the pinholes where air was caught. The scene
holds the bath in `scene.marks['bath']` and, if the alum stopped short of the sheet's edge, the part that took
colour in `scene.masks['sheet']`. A bath is worked in design units; `scale` is how many pixels one is.
"""
import numpy as np
from numba import njit, prange

from .noise import fbm
from .pigments import layer
from .raster import blur

DROP, COMB, STEP, SKIP, SWIRL = 0.0, 1.0, 2.0, 3.0, 4.0
FAR = 1e9


class Bath:
    def __init__(self, rng=None):
        self.ops = []
        self.colours = []
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def _colour(self, name):
        if name not in self.colours:
            self.colours.append(name)
        return float(self.colours.index(name))

    def drop(self, x, y, r, colour):
        """Let a drop fall at (x, y): it spreads to radius r, and everything already floating moves out of its way."""
        self.ops.append([DROP, x, y, r, self._colour(colour), self.rng.uniform(-1.0, 1.0)] + [0.0] * 7)

    def spatter(self, n, colour, r, box, clear=None):
        """Colour thrown from a brush: n drops of radius r = (smallest, largest), mostly small, anywhere in
        box = (x0, y0, x1, y1). `clear(x, y)` can say where none may fall."""
        rng = self.rng
        k = 0
        while k < n:
            x, y = rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])
            if clear is None or not clear(x, y):
                self.drop(x, y, r[0] + (r[1] - r[0]) * rng.random() ** 1.6, colour)
                k += 1

    def row(self, pts, r, gap, colour):
        """Drops let fall one after another along a path, `gap` apart."""
        P = np.asarray(pts, np.float64)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        for s in np.arange(0.0, cum[-1] + 1e-6, gap):
            self.drop(float(np.interp(s, cum, P[:, 0])), float(np.interp(s, cum, P[:, 1])), r, colour)

    def tine(self, x, y, ang, shift, sharp, span=None, soft=20.0):
        """One tooth drawn right across the bath through (x, y) the way `ang` points (radians, y down): the
        skin beside its line is dragged `shift` along it, less with distance (`sharp` is how far the drag
        reaches). `span` = (from, to) keeps the pull to part of the line, measured from (x, y)."""
        self.comb(x, y, ang, 0.0, shift, sharp, span, soft)

    def comb(self, x, y, ang, pitch, shift, sharp, span=None, soft=20.0, teeth=None):
        """A comb with teeth `pitch` apart, one of them through (x, y), drawn the way `ang` points. `teeth`
        = (first, last) counts teeth to either side of that one, if the comb is not as wide as the bath."""
        t0, t1 = span if span is not None else (-FAR, FAR)
        k0, k1 = teeth if teeth is not None else (-FAR, FAR)
        self.ops.append([COMB, x, y, np.cos(ang), np.sin(ang), shift, sharp, pitch, t0, t1, soft, k0, k1])

    def stroke(self, pts, reach, drag):
        """The needle drawn along a path. The skin within about `reach` of its point goes with it, and what
        lay on the path is carried about `drag` before it slips off (drag is a few times reach at most)."""
        P = np.asarray(pts, np.float64)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        hold = 1.0 - np.exp(-drag / reach)                 # how much of each step the skin at the point keeps up with
        n = max(2, int(cum[-1] / min(0.3 * reach / hold, 4.0)) + 1)
        s = np.linspace(0.0, cum[-1], n)
        q = np.stack([np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])], 1)
        for a, b in zip(q[:-1], q[1:]):
            self.ops.append([STEP, 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), hold * (b[0] - a[0]), hold * (b[1] - a[1]), reach] + [0.0] * 7)
        pad = 7.0 * reach + drag
        self.ops.append([SKIP, q[:, 0].min() - pad, q[:, 1].min() - pad, q[:, 0].max() + pad, q[:, 1].max() + pad, n - 1] + [0.0] * 7)

    def swirl(self, x, y, r, turn, sharp):
        """The needle taken once round a circle of radius r about (x, y): the skin on the circle turns `turn`
        radians with it, less to either side."""
        self.ops.append([SWIRL, x, y, r, turn, sharp] + [0.0] * 7)


@njit(parallel=True, fastmath=True, cache=True)
def _lift(ops, W, H, ss, scale, cov, tone):
    """Follow every point of the sheet (ss x ss of them to a pixel) backward through all that was done on the
    bath, undoing each act in turn, until it lies inside a drop as that drop fell. `cov[y, x, c]` gets how much
    of the pixel is colour c; `tone` a number of each drop's own, averaged."""
    n = ops.shape[0]
    w = 1.0 / (ss * ss)
    for y in prange(H):
        for x in range(W):
            for sub in range(ss * ss):
                px = (x + (sub % ss + 0.5) / ss) / scale
                py = (y + (sub // ss + 0.5) / ss) / scale
                i = n - 1
                while i >= 0:
                    k = int(ops[i, 0])
                    if k == 0:
                        dx, dy = px - ops[i, 1], py - ops[i, 2]
                        d2, r2 = dx * dx + dy * dy, ops[i, 3] * ops[i, 3]
                        if d2 <= r2:
                            cov[y, x, int(ops[i, 4])] += w
                            tone[y, x] += w * ops[i, 5]
                            break
                        s = np.sqrt(1.0 - r2 / d2)
                        px, py = ops[i, 1] + dx * s, ops[i, 2] + dy * s
                    elif k == 2:
                        lam = ops[i, 5]
                        dx, dy = px - ops[i, 1], py - ops[i, 2]
                        if abs(dx) < 7.0 * lam and abs(dy) < 7.0 * lam:
                            f = np.exp(-np.sqrt(dx * dx + dy * dy) / lam)
                            dx, dy = dx - ops[i, 3] * f, dy - ops[i, 4] * f
                            f = np.exp(-np.sqrt(dx * dx + dy * dy) / lam)
                            px, py = px - ops[i, 3] * f, py - ops[i, 4] * f
                    elif k == 3:
                        if px < ops[i, 1] or px > ops[i, 3] or py < ops[i, 2] or py > ops[i, 4]:
                            i -= int(ops[i, 5])
                    elif k == 1:
                        mx, my = ops[i, 3], ops[i, 4]
                        dx, dy = px - ops[i, 1], py - ops[i, 2]
                        s = -dx * my + dy * mx                           # across the teeth: a drag along them leaves this alone
                        pitch = ops[i, 7]
                        if pitch > 0.0:
                            s -= pitch * min(max(np.floor(s / pitch + 0.5), ops[i, 11]), ops[i, 12])
                        pull = ops[i, 5] * np.exp(-abs(s) / ops[i, 6])
                        if pull > 0.01 or pull < -0.01:
                            if ops[i, 8] > -1e8 or ops[i, 9] < 1e8:
                                t = dx * mx + dy * my
                                g = 1.0
                                for _ in range(3):
                                    g = min(max((t - g * pull - ops[i, 8]) / ops[i, 10], 0.0), 1.0) * min(max((ops[i, 9] - t + g * pull) / ops[i, 10], 0.0), 1.0)
                                pull *= g
                            px, py = px - pull * mx, py - pull * my
                    else:
                        dx, dy = px - ops[i, 1], py - ops[i, 2]
                        rho = np.sqrt(dx * dx + dy * dy)
                        a = -ops[i, 4] * np.exp(-abs(rho - ops[i, 3]) / ops[i, 5])
                        ca, sa = np.cos(a), np.sin(a)
                        px, py = ops[i, 1] + dx * ca - dy * sa, ops[i, 2] + dx * sa + dy * ca
                    i -= 1


def laid(H, W, rng):
    """Laid paper, in [0, 1]: the fine ribs of the mould's wires, a chain line every thumb's width, fibre."""
    s = W / 1024.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ribs = 0.5 + 0.5 * np.sin(2.0 * np.pi * (yy + 1.5 * s * fbm(H, W, 90.0 * s, rng, 2)) / (3.2 * max(s, 0.7)))
    chain = np.exp(-(((xx + 4.0 * s * fbm(H, W, 200.0 * s, rng, 2)) % (62.0 * s) - 31.0 * s) / (1.6 * s)) ** 2)
    g = 0.5 + 0.1 * (ribs - 0.5) - 0.12 * chain + 0.07 * fbm(H, W, 2.2 * max(s, 0.7), rng, 2) + 0.05 * fbm(H, W, 40.0 * s, rng, 2)
    return np.clip(g, 0.0, 1.0).astype(np.float32)


def pull(canvas, height, palette, scene, P, scale, rng):
    """Lay the sheet on the bath and lift it: the print, as films of paint on the paper. Returns how many
    things were done on the bath."""
    bath = scene.marks['bath']
    H, W = scene.H, scene.W
    s = scale / 1024.0
    ops = np.array(bath.ops, np.float64)
    cov = np.zeros((H, W, len(bath.colours)), np.float32)
    tone = np.zeros((H, W), np.float32)
    _lift(ops, W, H, int(P.get('fine', 2)), s, cov, tone)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # what the paper does: it takes a little more or less colour where the alum was sponged on unevenly, and the
    # colour granulates on it
    take = P.get('film', 1.0) * (1.0 + P.get('uneven', 0.15) * tone)
    take = take * (1.0 + 0.05 * fbm(H, W, (400.0 * s, 26.0 * s), rng, 2)) * (1.0 + P.get('grain', 0.1) * np.clip(fbm(H, W, 1.7 * max(s, 0.7), rng, 2), -2.0, 2.0))
    if 'sheet' in scene.masks:
        take = take * scene.masks['sheet']
    for _ in range(int(P.get('pauses', 2))):
        # where the hands paused in laying the sheet down, a hair of it took less
        y0 = rng.uniform(0.15, 0.85) * H + 6.0 * s * np.sin(xx / (rng.uniform(140.0, 320.0) * s) + rng.uniform(0.0, 6.3))
        take = take * (1.0 - rng.uniform(0.3, 0.5) * np.exp(-((yy - y0) / (rng.uniform(1.4, 3.0) * s)) ** 2))
    for _ in range(int(P.get('bubbles', 8))):
        # and where air was caught under it, none
        bx, by, br = rng.uniform(0.05, 0.95) * W, rng.uniform(0.05, 0.95) * H, rng.uniform(3.0, 9.0) * s
        take = take * (1.0 - np.clip(br - np.hypot(xx - bx, yy - by) + 0.5, 0.0, 1.0))
    # colour gathers a little where two films meet
    whole = np.maximum(cov.sum(-1), 1e-6)
    seam = blur(1.0 - (cov * cov).sum(-1) / (whole * whole), 0.6 * s)
    take = take * (1.0 + P.get('edge', 0.5) * seam)
    for c, name in enumerate(bath.colours):
        K, S = palette.recipe(**P['paints'][name])
        R, T = layer(np.asarray(K, np.float64), np.asarray(S, np.float64), (cov[..., c] * take)[..., None].astype(np.float64))
        canvas[:] = R + T * T * canvas / (1.0 - R * canvas)
    height += P.get('relief', 0.08) * whole * take
    return len(ops)
