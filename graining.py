"""Graining: a picture taken out of a wet glaze instead of put on with paint.

The grainer's trade (false oak and teak on doors, shop fronts and railway carriages): a pale ground, one dark
transparent glaze rubbed over all of it, and while that is wet the figure of the wood drawn by taking glaze
off again, with steel combs, a rag over the thumb, a mottler and the fingertip, and a badger to soften it.
Nothing light is ever put on. A `Glaze` is that wet film: the paint it is made of and how thick it lies at
each point. The tools thin it, push it aside and pile it up; `dry` leaves what is left on the picture as a
film of pigment (pigments.layer), so a combed line shows the ground and what lies between the lines shows
the ground through the glaze.

The tools draw along a figure the scene supplies, `scene.figures[name] = (u, v)`: two fields in pixels, v
constant along each line of the grain and u running along it. Where the scene crowds the lines of v
together the grain runs finer, as it does beside a knot. How hard a tool bears down at each point is a mask
(`scene.masks`), and that is how a picture is drawn with them: a comb pressed lightly leaves hairlines in
the dark, pressed hard it leaves broad clean lines, lifted it leaves the glaze alone.

A style works a glaze with passes (painter.py): `dict(kind='glaze', paint=dict(umber=2, medium=5), ...)` lays
it, each `dict(kind='grain', tool=..., ...)` after it works it, and it dries before the next pass of any
other kind. Lengths in a pass are fractions of the picture's width, as brush sizes are."""
import numpy as np
from numba import njit, prange
from scipy.ndimage import map_coordinates

from noise import fbm
from pigments import layer
from raster import blur


@njit(fastmath=True, inline='always')
def _h(a, b, c):
    h = (a * 374761393 + b * 668265263 + c * 982451653 + 144665) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0


@njit(fastmath=True, inline='always')
def _n1(x, a, b):
    """Value noise along a line, in [0, 1]."""
    i = int(np.floor(x))
    f = x - i
    f = f * f * (3.0 - 2.0 * f)
    p = _h(i, a, b)
    return p + (_h(i + 1, a, b) - p) * f


@njit(fastmath=True, inline='always')
def _n2(x, y, seed):
    """Value noise over a plane, in [0, 1]."""
    i, j = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - i, y - j
    fx, fy = fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy)
    a, b = _h(i, j, seed), _h(i + 1, j, seed)
    c, d = _h(i, j + 1, seed), _h(i + 1, j + 1, seed)
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy


@njit(parallel=True, fastmath=True, cache=True)
def _comb(d, u, v, q, press, pitch, tines, pull, duty, wmax, strength, wobble, skew, clog, graduate, feather,
          pulse, register, seed):
    """Pull combs through the glaze along the lines of v. A comb has `tines` teeth `pitch` apart and is
    drawn about `pull` far before it is lifted, wiped and set down again, so the picture is covered by
    pulls side by side and end to end. Each pull has its own hand: a little out of register with its
    neighbours, wandering as a rigid thing (all its lines together), bearing harder on one side, taking
    less glaze off as its teeth fill, and trailing off where it is lifted. `q` is how many pixels one
    unit of v spans at each point."""
    H, W = d.shape
    T = float(tines)
    for y in prange(H):
        for x in range(W):
            p0 = press[y, x]
            if p0 < 0.004:
                continue
            vv = v[y, x] / pitch
            uu = u[y, x]
            px = q[y, x] * pitch                      # from one line to the next, in pixels on the picture
            lane0 = int(np.floor(vv / T))
            keep, pile = 1.0, 0.0
            for lane in range(lane0 - 1, lane0 + 2):
                stag = _h(lane, 7, seed) * pull
                m0 = int(np.floor((uu + stag) / pull))
                # what stays the same down the whole lane: the comb, and the track it is set back into after each lift
                own, l2, l3 = _h(lane, 11, seed) - 0.5, _h(lane, 12, seed), _h(lane, 13, seed)
                l5, l6 = _h(lane, 14, seed), _h(lane, 15, seed)
                for m in range(m0 - 1, m0 + 2):
                    a = (m + 0.4 * (_h(lane, m, seed + 1) - 0.5)) * pull - stag
                    b = (m + 1 + 0.4 * (_h(lane, m + 1, seed + 1) - 0.5)) * pull - stag
                    a -= (3.0 + 2.0 * _h(lane, m, seed + 2)) * pitch         # set down before the place the last pull began to lift
                    if uu < a or uu > b:
                        continue
                    t = (uu - a) / (b - a)
                    r1, r4 = _h(lane, m, seed + 3), _h(lane, m, seed + 6)
                    r7, r8, r9 = _h(lane, m, seed + 9), _h(lane, m, seed + 10), _h(lane, m, seed + 11)
                    s = vv - lane * T + register * (0.7 * own + 0.3 * (r1 - 0.5))
                    s += wobble * (np.sin(6.2832 * (uu / (pull * (0.3 + 0.5 * l2)) + l3))
                                   + 0.5 * np.sin(6.2832 * (uu / (0.13 * pull) + l2)))
                    s += skew * T * (2.0 * r4 - 1.0) * (t - 0.5)             # the comb turned a little in the hand
                    wide = 1.0
                    if graduate > 0.0:
                        # a graduated comb: its teeth stand wider apart toward one end
                        xs = s / T
                        if xs <= 0.0 or xs >= 1.0:
                            continue
                        e = 1.0 + graduate * (0.4 + 1.2 * l5)
                        if l6 < 0.5:
                            g, dg = xs ** e, e * xs ** (e - 1.0)
                        else:
                            g, dg = 1.0 - (1.0 - xs) ** e, e * (1.0 - xs) ** (e - 1.0)
                        s = g * T
                        wide = 1.0 / max(dg, 0.3)
                    if s < 0.0:
                        continue
                    k = int(np.floor(s))
                    if k >= tines:
                        continue
                    f = s - k - 0.5
                    per = px * wide
                    dist = abs(f) * per
                    # it comes down onto the glaze quickly and lifts off slowly, and no two teeth leave at the same place
                    jag = 1.5 * pitch * (_h(lane * 131 + k, m, seed + 13) - 0.5)
                    e0 = min(max((uu - a - jag) / 5.0, 0.0), 1.0)
                    e1 = min(max((b + jag - uu) / (2.5 * pitch), 0.0), 1.0)
                    ends = e0 * e0 * (3.0 - 2.0 * e0) * e1 * e1 * (3.0 - 2.0 * e1)
                    hand = (0.94 + 0.12 * r7) * (1.0 - clog * t * (0.4 + 0.9 * r8)) * (1.0 + 0.3 * (r9 - 0.5) * (2.0 * s / T - 1.0))
                    pr = min(p0 * hand, 1.0)
                    wl = duty * per * (0.3 + 0.7 * pr) * (0.35 + 0.65 * ends)
                    if pulse > 0.0:
                        wl *= 1.0 + pulse * (2.0 * _n1(uu / (0.045 * pull), lane * 31 + k, seed + 12) - 1.0)
                    wl = min(wl, wmax)
                    half = 0.5 * wl
                    cov = min(max(half - dist + 0.5, 0.0), 1.0) * min(wl, 1.0)
                    if feather > 0.0 and f > 0.0 and dist > half:
                        # wiped with the thumb and flicked off one way: one edge of the line clean, the other soft
                        cov = max(cov, 0.75 * np.exp(-(dist - half) / (0.5 * feather * per)))
                    if per < 4.0:
                        # lines too close to tell apart read as an even tone
                        vis = min(max((per - 1.5) / 2.5, 0.0), 1.0)
                        cov = cov * vis + (1.0 - vis) * min(wl / per, 1.0)
                    depth = strength * min(1.0, 0.3 + 0.9 * pr)
                    # where one pull laps the last the teeth run in the grooves already there, and take no more
                    keep = min(keep, 1.0 - depth * cov * ends)
                    edge = dist - half
                    if per >= 4.0 and edge > 0.0 and edge < 2.4:
                        pile = max(pile, 0.5 * depth * ends * (1.0 - abs(edge - 1.2) / 1.2))   # what a tooth pushes aside lies along its line
            d[y, x] *= keep * (1.0 + pile)


@njit(parallel=True, fastmath=True, cache=True)
def _rag(d, u, v, press, strength, streak, seed, taken):
    """Wipe with a rag: glaze comes off broadly, in fine streaks the way the rag travelled."""
    H, W = d.shape
    for y in prange(H):
        for x in range(W):
            p = press[y, x]
            if p < 0.004:
                continue
            n = 0.65 * _n2(u[y, x] / 70.0, v[y, x] / 2.3, seed) + 0.35 * _n2(u[y, x] / 260.0, v[y, x] / 9.0, seed + 1)
            r = min(strength * p * (1.0 - streak * n), 1.0)
            taken[y, x] = r
            d[y, x] *= 1.0 - r


@njit(parallel=True, fastmath=True, cache=True)
def _mottle(d, u, v, press, period, strength, slant, seed):
    """Press a mottler across the grain and rock it: soft bands of light, coming and going (fiddleback)."""
    H, W = d.shape
    for y in prange(H):
        for x in range(W):
            p = press[y, x]
            if p < 0.004:
                continue
            a, b = u[y, x] / period, v[y, x] / period
            ph = a + slant * b + 1.6 * (_n2(0.2 * a, 0.4 * b, seed) - 0.5)
            band = 0.5 + 0.5 * np.sin(6.2832 * ph)
            gate = min(max((_n2(a / 7.0, b / 4.0, seed + 1) - 0.3) / 0.4, 0.0), 1.0)
            d[y, x] *= 1.0 - strength * p * band * band * (0.35 + 0.65 * gate)


@njit(parallel=True, fastmath=True, cache=True)
def _flog(d, u, v, press, cu, cv, strength, chance, seed):
    """Short flecks along the grain: pores under a flogger, the silver fleck of quartered oak, glitter."""
    H, W = d.shape
    for y in prange(H):
        for x in range(W):
            p = press[y, x]
            if p < 0.004:
                continue
            j = int(np.floor(v[y, x] / cv))
            a = u[y, x] / cu + _h(j, 3, seed)
            i = int(np.floor(a))
            if _h(i, j, seed + 1) > chance * p:
                continue
            fu, fv = a - i, v[y, x] / cv - j
            du = (fu - (0.5 + 0.2 * (_h(i, j, seed + 2) - 0.5))) / (0.2 + 0.25 * _h(i, j, seed + 3))
            dv = (fv - 0.5) / (0.22 + 0.2 * _h(i, j, seed + 4))
            e = du * du + dv * dv
            if e < 1.0:
                d[y, x] *= 1.0 - strength * min(1.0, 3.0 * (1.0 - e)) * (0.6 + 0.4 * _h(i, j, seed + 5))


class Glaze:
    """A wet film of one transparent paint: its absorption and scattering (K, S, per colour channel, for
    a film one unit thick), how thick it lies at each point (`d`), and how much relief it dries to."""

    def __init__(self, K, S, d, relief):
        self.K, self.S, self.d, self.relief = K, S, d, relief
        self._scale = {}

    def spacing(self, name, v):
        """How many pixels one unit of a figure's v spans at each point."""
        if name not in self._scale:
            gy, gx = np.gradient(v)
            self._scale[name] = (1.0 / np.maximum(np.hypot(gx, gy), 0.05)).astype(np.float32)
        return self._scale[name]

    def dry(self, canvas, height):
        """Leave the film on the picture (linear reflectance) and its thickness on the relief."""
        R, T = layer(self.K, self.S, self.d[..., None].astype(np.float64))
        canvas[:] = R + T * T * canvas / (1.0 - R * canvas)
        height += self.relief * self.d


def _figure(scene, P):
    """The grain a pass draws along, or plain rows if it names none."""
    if 'figure' in P:
        return scene.figures[P['figure']]
    yy, xx = np.mgrid[0:scene.H, 0:scene.W].astype(np.float32)
    return xx, yy


def lay(paint, scene, P, scale, rng):
    """Rub the glaze over the whole picture: an even film, a little streaked the way the brush laid it off."""
    u, v = _figure(scene, P)
    seed = int(rng.integers(1 << 20))
    s = scale / 2048.0
    streak = P.get('streak', 0.1)
    n = np.empty((scene.H, scene.W), np.float32)
    _streaks(n, u, v, 1.0 / (150.0 * s), 1.0 / (2.6 * s), seed)
    d = P.get('film', 1.0) * (1.0 + streak * n + 0.4 * streak * fbm(scene.H, scene.W, 300.0 * s, rng, 2))
    return Glaze(np.asarray(paint[0], np.float64), np.asarray(paint[1], np.float64), d.astype(np.float32), P.get('relief', 0.2))


@njit(parallel=True, fastmath=True, cache=True)
def _streaks(out, u, v, fu, fv, seed):
    H, W = out.shape
    for y in prange(H):
        for x in range(W):
            out[y, x] = (1.3 * _n2(u[y, x] * fu, v[y, x] * fv, seed) + 0.7 * _n2(u[y, x] * fu * 2.3, v[y, x] * fv * 0.3, seed + 1)) - 1.0


def work(glaze, scene, P, scale, rng):
    """One tool over the wet glaze. `press` names the mask that says how hard it bears down where."""
    tool = P['tool']
    d = glaze.d
    press = np.ascontiguousarray(scene.masks[P['press']], np.float32)
    seed = int(rng.integers(1 << 20))
    if tool == 'charge':
        # more glaze stippled in: solid things, the dark behind a light
        d += P.get('amount', 1.0) * press * (1.0 + 0.2 * fbm(scene.H, scene.W, 6.0 * scale / 2048.0, rng, 2))
        return
    u, v = _figure(scene, P)
    if tool == 'comb':
        pitch = P['pitch'] * scale
        _comb(d, u, v, glaze.spacing(P.get('figure', ''), v), press, pitch, int(P.get('tines', 8)), P.get('pull', 0.25) * scale,
              P.get('duty', 0.5), P.get('max', 0.004) * scale, P.get('strength', 0.9), P.get('wobble', 0.12), P.get('skew', 0.02),
              P.get('clog', 0.35), P.get('graduate', 0.0), P.get('feather', 0.0), P.get('pulse', 0.0), P.get('register', 0.6), seed)
    elif tool == 'rag':
        taken = np.zeros_like(d)
        _rag(d, u, v, press, P.get('strength', 0.8), P.get('streak', 0.3), seed, taken)
        if P.get('rim', 0.0) > 0:
            # what the rag pushed ahead of it lies round the edge of the place it cleaned
            sigma = P.get('rim_width', 0.0012) * scale
            d *= 1.0 + P['rim'] * np.clip(2.5 * (blur(taken, sigma) - taken), 0.0, 1.0)
    elif tool == 'mottle':
        _mottle(d, u, v, press, P['period'] * scale, P.get('strength', 0.4), P.get('slant', 0.0), seed)
    elif tool == 'flog':
        cu, cv = P['cell']
        _flog(d, u, v, press, cu * scale, cv * scale, P.get('strength', 0.6), P.get('chance', 0.4), seed)
    elif tool == 'soften':
        # the badger, drawn lightly one way across the grain: each line keeps one clean edge and loses the other
        gy, gx = np.gradient(v)
        g = np.hypot(gx, gy) + 1e-6
        yy, xx = np.mgrid[0:scene.H, 0:scene.W].astype(np.float32)
        step = P.get('reach', 0.002) * scale / 4.0
        acc, wsum = d.copy(), 1.0
        for k in range(1, 5):
            w = float(np.exp(-0.5 * k))
            acc += w * map_coordinates(d, [yy - k * step * gy / g, xx - k * step * gx / g], order=1, mode='nearest')
            wsum += w
        d += (acc / wsum - d) * P.get('amount', 0.5) * press
    else:
        raise ValueError(f'no graining tool named {tool!r}')
