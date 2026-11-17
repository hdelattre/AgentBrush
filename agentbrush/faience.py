"""Faience: colour brushed onto raw tin glaze and fired into it.

A tile or plaque is dipped in white tin glaze and painted while the glaze is still a dry powder on it. The
powder drinks like blotting paper: a stroke is taken up as the brush passes and cannot be moved, lifted or
gone over, so every mark is made once. In the kiln the colour sinks into the melting glaze, its edges
soften, and it comes out as stained glass over white. There is no white paint and no body colour: a light is
glaze left bare, kept clear with wax where a wash has to pass over it.

So this paints with stains (pigments.lake: cobalt, copper, antimony, iron, manganese ...) and three things
done with them:
  a wash    a flat brush drawn across in bands, each band lapping the last, the bristles leaving their lanes
            and the brush running out along each pull (`_wash`)
  a touch   one stroke of a loaded round brush, set down and drawn off: a leaf, a petal, a roof tile, a blade
            of grass, a dot. The colour gathers along its edges and where the brush leaves (`_touch`)
  a line    the long liner, for the drawing and for branches (`_draw`)
Films of stain multiply, as coloured glasses do: yellow over blue is green, and three strong ones make a black.

A scene lays its work out on a `Bench`: for each stain where its washes lie and how strong, every touch and
every line, and the wax. One bench can serve several plaques painted alike and afterwards cut together: each
wash, touch and line belongs to one of them, and `plaque` says which plaque each point of the finished piece
came from. A style then fires each stain with a pass, `dict(kind='stain', stain='zaffre', paint=dict(zaffre=1),
...)`, on `surface='glaze'` (painter.py). Lengths in a pass are fractions of the picture's width.
"""
import numpy as np
from numba import njit, prange
from scipy.ndimage import map_coordinates

from .graining import _h, _n1, _n2
from .noise import fbm
from .pigments import PIGMENTS, layer, to_display, to_linear
from .raster import blur

LEAF, TEAR, PETAL, DASH, DOT = 0.0, 1.0, 2.0, 3.0, 4.0     # the shapes one touch of the brush can leave


@njit(parallel=True, fastmath=True, cache=True)
def _wash(out, amount, u, v, band, pull, streak, lap, seed):
    """Lay each wash the way a flat brush lays it: in bands along the lines of v, a band a brush wide, drawn
    about `pull` far before the brush is filled again. A band drifts a little off its line and laps its
    neighbours by `lap` of its width, the brush is fullest where it is set down and runs out along the pull,
    its bristles leave lanes, and a little colour gathers along a band's wet edges."""
    H, W = out.shape
    for y in prange(H):
        for x in range(W):
            a = amount[y, x]
            if a < 0.002:
                continue
            uu, vv = u[y, x], v[y, x] / band
            lane0 = int(np.floor(vv))
            tot = 0.0
            for lane in range(lane0 - 1, lane0 + 2):
                c = lane + 0.5 + 0.2 * (_n1(uu / (7.0 * band), lane, seed) - 0.5)
                dv = abs(vv - c)
                if dv >= 0.5 + lap:
                    continue
                # a band is full across its middle and thins across the lap, so where two meet they leave about
                # one film between them: a little more where they crowd, a little less where they part
                across = min((0.5 + lap - dv) / (2.0 * lap), 1.0)
                e = (0.5 + lap - dv) * band                      # pixels inside this band's edge
                stag = _h(lane, 7, seed) * pull
                m0 = int(np.floor((uu + stag) / pull))
                for m in range(m0 - 1, m0 + 2):
                    a0 = (m + 0.3 * (_h(lane, m, seed + 1) - 0.5)) * pull - stag
                    b0 = (m + 1 + 0.3 * (_h(lane, m + 1, seed + 1) - 0.5)) * pull - stag
                    f = 0.6 * band                               # one pull runs into the next over about a brush width
                    if uu <= a0 - f or uu >= b0 + f:
                        continue
                    ends = min((uu - a0 + f) / (2.0 * f), 1.0) * min((b0 + f - uu) / (2.0 * f), 1.0)
                    load = (0.93 + 0.14 * _h(lane, m, seed + 2)) * (1.05 - 0.12 * (uu - a0) / (b0 - a0))
                    n = _n2(uu / (3.0 * band) + 31.0 * _h(lane, m, seed + 3), (vv - c + 0.7) * 11.0, seed + lane * 13)
                    tot += across * ends * load * (1.0 - streak + 2.0 * streak * n) * (1.0 + 0.2 * np.exp(-e / 2.5))
            out[y, x] += a * tot


@njit(fastmath=True, cache=True)
def _touch(out, field, M, streak, pool, seed, wax):
    """Single touches of a round brush. A row of M is one touch: where it is set down (x, y), the way it is
    drawn (angle), how far, its half-width at the fullest, how much it turns on the way (radians, so its
    spine is an arc), its shape, its load, the plaque it is on (-1: any), and a number of its own. With `wax`
    the touches are wax, not colour: `out` takes how well each keeps a wash off."""
    H, W = out.shape
    for i in range(M.shape[0]):
        x0, y0, ang, L, hw0, bend = M[i, 0], M[i, 1], M[i, 2], M[i, 3], M[i, 4], M[i, 5]
        kind, load, who, own = int(M[i, 6]), M[i, 7], int(M[i, 8]), int(M[i, 9])
        ca, sa = np.cos(ang), np.sin(ang)
        R = 0.0
        if kind == 4:
            mx, my, reach = x0, y0, hw0 + 2.0
        else:
            if abs(bend) < 1e-3:
                ex, ey = x0 + L * ca, y0 + L * sa
            else:
                R = L / bend
                ex, ey = x0 + R * (np.sin(ang + bend) - sa), y0 - R * (np.cos(ang + bend) - ca)
            mx, my, reach = 0.5 * (x0 + ex), 0.5 * (y0 + ey), 0.5 * L + hw0 + 2.0
        xa, xb = max(int(mx - reach), 0), min(int(mx + reach) + 2, W)
        ya, yb = max(int(my - reach), 0), min(int(my + reach) + 2, H)
        for y in range(ya, yb):
            for x in range(xa, xb):
                lab = field[y, x]
                if lab < -1 or (who >= 0 and lab != who):
                    continue
                px, py = x + 0.5 - x0, y + 0.5 - y0
                if kind == 4:
                    e = hw0 - np.sqrt(px * px + py * py)
                    if e <= -0.5:
                        continue
                    cov = min(e + 0.5, 1.0) * min(hw0 / 0.7, 1.0)
                    if wax:
                        out[y, x] = max(out[y, x], load * cov)
                    else:
                        out[y, x] += load * cov * (0.85 + 0.3 * _n2(x / 3.0, y / 3.0, seed + own)) * (0.9 + pool * np.exp(-max(e, 0.0) / 1.4))
                    continue
                if R == 0.0:
                    s, l = px * ca + py * sa, -px * sa + py * ca
                else:
                    qx, qy = px + R * sa, py - R * ca            # from the point the stroke turns about
                    rho = np.sqrt(qx * qx + qy * qy)
                    th = np.arctan2(qx, -qy) if R > 0.0 else np.arctan2(-qx, qy)
                    turn = th - ang
                    turn -= 6.283185307 * np.floor(turn / 6.283185307 + 0.5)
                    s = turn * R
                    l = R - rho if R > 0.0 else R + rho
                if s < 0.0 or s > L:
                    continue
                if kind == 0:        # a leaf: down to a point, pressed, lifted to a point
                    hw = hw0 * max(np.sin(3.14159265 * (s / L) ** 0.85), 0.0) ** 0.75
                elif kind == 3:      # a dash: the same weight all the way
                    if s < hw0:
                        hw = np.sqrt(max(hw0 * hw0 - (hw0 - s) ** 2, 0.0))
                    elif s > L - hw0:
                        hw = np.sqrt(max(hw0 * hw0 - (s - L + hw0) ** 2, 0.0))
                    else:
                        hw = hw0
                else:
                    # a tear is set down full and drawn off to a point; a petal is the same stroke made the other way
                    r = s if kind == 1 else L - s
                    if r < hw0:
                        hw = np.sqrt(max(hw0 * hw0 - (hw0 - r) ** 2, 0.0))
                    else:
                        hw = hw0 * max(1.0 - (r - hw0) / max(L - hw0, 1e-3), 0.0) ** 0.8
                e = hw - abs(l)
                if e <= -0.5:
                    continue
                cov = min(e + 0.5, 1.0) * min(hw / 0.7, 1.0)
                if wax:
                    out[y, x] = max(out[y, x], load * cov)
                    continue
                lanes = 1.0 - streak + 2.0 * streak * _n1(2.6 * l / max(hw, 0.7) + 40.0, own, seed)
                run = 1.12 - 0.3 * s / L                                        # the brush runs out along the stroke
                bead = 1.0 + 0.45 * np.exp(-(L - s) / (0.1 * L + 1.5))          # and leaves a bead where it lifts
                out[y, x] += load * cov * lanes * run * bead * (0.9 + pool * np.exp(-max(e, 0.0) / 1.5))


@njit(fastmath=True, cache=True)
def _draw(out, plaque, field, pts, offs, hws, prm, tmp, seed, wax):
    """Lines of the long liner: polylines with a half-width at every point. A row of prm is one line: its
    load, the plaque it is on (-1: any), and whether it keeps inside the picture (`field`). `tmp` is scratch,
    all zeros, and is left all zeros."""
    H, W = out.shape
    for i in range(offs.size - 1):
        load, who, inside = prm[i, 0], int(prm[i, 1]), prm[i, 2] > 0.5
        bx0, by0, bx1, by1 = W, H, 0, 0
        for j in range(offs[i], offs[i + 1] - 1):
            x0, y0, x1, y1 = pts[j, 0], pts[j, 1], pts[j + 1, 0], pts[j + 1, 1]
            h0, h1 = hws[j], hws[j + 1]
            ex, ey = x1 - x0, y1 - y0
            ll = ex * ex + ey * ey
            if ll < 1e-9:
                continue
            pad = max(h0, h1) + 1.5
            xa, xb = max(int(min(x0, x1) - pad), 0), min(int(max(x0, x1) + pad) + 2, W)
            ya, yb = max(int(min(y0, y1) - pad), 0), min(int(max(y0, y1) + pad) + 2, H)
            for y in range(ya, yb):
                for x in range(xa, xb):
                    px, py = x + 0.5, y + 0.5
                    t = min(max(((px - x0) * ex + (py - y0) * ey) / ll, 0.0), 1.0)
                    dx, dy = x0 + t * ex - px, y0 + t * ey - py
                    c = min(0.5 + h0 + (h1 - h0) * t - np.sqrt(dx * dx + dy * dy), 1.0)
                    if c > tmp[y, x]:
                        tmp[y, x] = c
            bx0, by0, bx1, by1 = min(bx0, xa), min(by0, ya), max(bx1, xb), max(by1, yb)
        for y in range(by0, by1):
            for x in range(bx0, bx1):
                c = tmp[y, x]
                if c <= 0.0:
                    continue
                tmp[y, x] = 0.0
                lab = field[y, x] if inside else plaque[y, x]
                if (inside and lab < -1) or (who >= 0 and lab != who):
                    continue
                if wax:
                    out[y, x] = max(out[y, x], load * c)
                else:
                    out[y, x] += load * c * (0.82 + 0.36 * _n2(x / 7.0, y / 7.0, seed + i))


class Bench:
    """A paintress's bench: the work laid out before anything is fired. `plaque` gives, for each point of the
    finished piece, the number of the plaque that point came from (below 0: none); `field` is the same but
    below -1 outside the picture, and touches and lines keep to it. Positions and sizes are given in the
    units the piece was designed in, `scale` pixels each."""

    def __init__(self, stains, plaque, scale=1.0, rng=None, field=None):
        self.stains = tuple(stains)
        self.plaque = np.ascontiguousarray(plaque, np.int32)
        self.field = self.plaque if field is None else np.ascontiguousarray(field, np.int32)
        self.S = scale
        self.rng = rng if rng is not None else np.random.default_rng(0)
        shape = self.plaque.shape
        self.wash = {s: np.zeros(shape, np.float32) for s in self.stains}
        self.top = {s: np.zeros(shape, np.float32) for s in self.stains}       # washes laid inside something waxed round
        self.wax = np.zeros(shape, np.float32)
        self.touches = {s: [] for s in self.stains + ('wax',)}
        self.lines = {s: [] for s in self.stains + ('wax',)}
        self._on = {}

    def _where(self, k, mask):
        """(window or None, the mask cut down to plaque k)."""
        win, m = (mask[0], mask[1]) if isinstance(mask, tuple) else (None, mask)
        if k is not None:
            if k not in self._on:
                self._on[k] = (self.plaque == k).astype(np.float32)
            m = m * (self._on[k] if win is None else self._on[k][win])
        return win, m

    def lay(self, k, paint, mask, over=False):
        """A wash on plaque k (None: on every one) where `mask` says (a whole field, or a raster shape):
        `paint` gives how many films of each stain, as a number or a field. over=True is a wash laid inside
        something kept clear with wax."""
        if mask is None:
            return
        win, m = self._where(k, mask)
        into = self.top if over else self.wash
        for stain, amount in paint.items():
            if win is None:
                into[stain] += amount * m
            else:
                into[stain][win] += amount * m

    def keep(self, k, mask, strength=1.0):
        """Wax: washes laid under it (not over=True ones) stay off."""
        if mask is None:
            return
        win, m = self._where(k, mask)
        if win is None:
            np.maximum(self.wax, strength * m, out=self.wax)
        else:
            np.maximum(self.wax[win], strength * m, out=self.wax[win])

    def touch(self, k, paint, x, y, ang, length, width, bend=0.0, kind=LEAF):
        """Touches of the round brush, one for each entry of the arrays: set down at (x, y), drawn `length`
        the way `ang` points (radians, y down) turning `bend` on the way, `width` the half-width at the
        fullest. `paint` gives each stain's load (a number or one per touch); 'wax' is a stain here too."""
        cols = np.broadcast_arrays(*(np.asarray(c, np.float64) for c in (x, y, ang, length, width, bend, kind)))
        n = cols[0].size
        if n == 0:
            return
        rows = np.empty((n, 10))
        for j, (c, unit) in enumerate(zip(cols, (self.S, self.S, 1.0, self.S, self.S, 1.0, 1.0))):
            rows[:, j] = c.ravel() * unit
        rows[:, 8] = -1.0 if k is None else k
        rows[:, 9] = self.rng.integers(0, 1 << 20, n)
        for stain, load in paint.items():
            load = np.asarray(load, np.float64)
            load = load.ravel() if load.size == n else np.broadcast_to(load, (n,))
            r = rows[load > 0.0].copy()
            r[:, 7] = load[load > 0.0]
            self.touches[stain].append(r)

    def draw(self, k, stain, pts, hw, load=1.0, taper=0.5, inside=True):
        """A line of the liner through `pts`, half-width `hw` (or (start, end)). It comes down to `taper` of
        that at each end, and the hand is never quite even."""
        P = np.asarray(pts, np.float64) * self.S
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        if cum[-1] < 0.5:
            return
        at = np.linspace(0.0, cum[-1], max(2, int(cum[-1] / 3.0) + 1))
        q = np.stack([np.interp(at, cum, P[:, 0]), np.interp(at, cum, P[:, 1])], 1)
        h0, h1 = hw if isinstance(hw, tuple) else (hw, hw)
        ends = np.minimum(np.minimum(at, cum[-1] - at) / max(min(0.5 * cum[-1], 14.0 * self.S), 1e-6), 1.0)
        w = (h0 + (h1 - h0) * at / cum[-1]) * self.S * (taper + (1.0 - taper) * ends)
        w = w * (1.0 + 0.12 * np.sin(at / (self.rng.uniform(18.0, 46.0) * self.S) + self.rng.uniform(0.0, 6.3)))
        self.lines[stain].append((q, w, load, -1.0 if k is None else k, 1.0 if inside else 0.0))

    def _touches(self, stain):
        return np.concatenate(self.touches[stain]) if self.touches[stain] else np.zeros((0, 10))

    def _lines(self, stain):
        ls = self.lines[stain]
        if not ls:
            return np.zeros((0, 2)), np.zeros(1, np.int64), np.zeros(0), np.zeros((0, 3))
        offs = np.zeros(len(ls) + 1, np.int64)
        offs[1:] = np.cumsum([len(l[0]) for l in ls])
        return np.concatenate([l[0] for l in ls]), offs, np.concatenate([l[1] for l in ls]), np.array([l[2:] for l in ls], np.float64)

    def finish(self, scene, stray=2.2):
        """Hand the work to the scene: each stain's wash as scene.masks[stain], its touches and lines as
        scene.marks[stain]. The wax is spent here, and no wash keeps quite to its drawing: every edge strays a
        little (`stray`, design units), all the stains together."""
        H, W = self.plaque.shape
        wax = self.wax
        tmp = np.zeros((H, W), np.float32)
        _touch(wax, self.field, self._touches('wax'), 0.0, 0.0, 0, True)
        _draw(wax, self.plaque, self.field, *self._lines('wax'), tmp, 0, True)
        wax = np.clip(wax, 0.0, 1.0)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        sx = xx + stray * self.S * fbm(H, W, 46.0 * self.S, self.rng, 2)
        sy = yy + stray * self.S * fbm(H, W, 46.0 * self.S, self.rng, 2)
        for stain in self.stains:
            a = self.wash[stain] * (1.0 - wax) + self.top[stain]
            scene.masks[stain] = np.ascontiguousarray(map_coordinates(a, [sy, sx], order=1, mode='nearest'), np.float32)
            scene.marks[stain] = (self._touches(stain),) + self._lines(stain)
        scene.marks['plaque'], scene.marks['field'] = self.plaque, self.field


def proof(scene, ground, stains):
    """The washes alone, flat, as display colour: roughly how the piece will fire, for a scene's reference."""
    c = np.empty((scene.H, scene.W, 3), np.float64)
    c[:] = to_linear(ground)
    for stain in stains:
        c *= np.exp(-2.0 * PIGMENTS[stain][0] * scene.masks[stain][..., None])
    return to_display(c).astype(np.float32)


def glaze(H, W, rng):
    """Fired tin glaze, in [0, 1]: a slow swell like the skin on milk, a faint orange peel, the odd pinhole."""
    s = W / 2048.0
    g = 0.5 + 0.16 * fbm(H, W, 110.0 * s, rng, 2) + 0.012 * fbm(H, W, 12.0 * s, rng, 2)
    n = int(H * W / (170.0 * s) ** 2)
    pits = np.zeros((H, W), np.float32)
    pits[rng.integers(2, H - 2, n), rng.integers(2, W - 2, n)] = rng.uniform(0.4, 1.0, n)
    return np.clip(g - 9.0 * blur(pits, 1.3 * max(s, 0.6)), 0.0, 1.0).astype(np.float32)


def fire(canvas, height, paint, scene, P, scale, rng):
    """One stain: its washes, its touches and its lines put on the raw glaze and fired. Returns how many
    touches and lines there were."""
    H, W = scene.H, scene.W
    s = scale / 2048.0
    seed = int(rng.integers(1 << 20))
    amount = scene.masks[P['stain']]
    pool = P.get('pool', 0.5)
    if pool > 0:
        # a wet wash dries from its edge inward and leaves its colour there
        amount = amount + 2.0 * pool * np.clip(amount - blur(amount, 2.2 * s), 0.0, None)
    if 'figure' in P:
        u, v = scene.figures[P['figure']]
    else:
        v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.zeros((H, W), np.float32)
    streak = P.get('streak', 0.25)
    _wash(d, np.ascontiguousarray(amount, np.float32), u, v, P.get('band', 0.03) * scale, P.get('pull', 0.4) * scale, streak,
          P.get('lap', 0.1), seed)
    touches, pts, offs, hws, prm = scene.marks[P['stain']]
    _touch(d, scene.marks['field'], touches, streak, pool, seed + 1, False)
    _draw(d, scene.marks['plaque'], scene.marks['field'], pts, offs, hws, prm, np.zeros((H, W), np.float32), seed + 2, False)
    # in the kiln the colour sinks into the melting glaze: its edges soften, and it draws up into grains
    d = blur(d, max(P.get('bleed', 0.0004) * scale, 0.3))
    d = d * (1.0 + P.get('grain', 0.08) * np.clip(fbm(H, W, 1.8 * max(s, 0.6), rng, 2), -2.0, 2.0))
    K, S = paint
    R, T = layer(np.asarray(K, np.float64), np.asarray(S, np.float64), np.maximum(d, 0.0)[..., None].astype(np.float64))
    canvas[:] = R + T * T * canvas / (1.0 - R * canvas)
    height += P.get('relief', 0.06) * d
    return len(touches) + len(prm)
