"""Brush engine.

A stroke is a polyline plus a half-width. Every pixel under it is mapped to
(s, t): distance along the stroke and position across it. From those come the
width taper, ragged edges, bristle streaks, paint running out along the
stroke, dry-brush skipping over the canvas weave, and wet pick-up of the paint
already underneath. Each stroke also deposits thickness into a height map,
which `export` lights for the preview and bakes into the normal map.
"""
import numpy as np
from numba import njit

from noise import fbm

# stroke parameter columns: half-width, alpha, paint load, dryness, smear,
# thickness, edge roughness, bristle-table row, noise seed, end-cap softness,
# end-width ratio (taper), tool (0 brush, 1 palette knife, 2 a clean brush that only
# blends), pooling (how far a thin film settles into the hollows of the paint under
# it and wipes off the ridges), feather (0 a firm edge, 1 the paint thinning out
# from the middle of the stroke, as a soft brush barely pressed leaves it)
NPARAM = 14


@njit(cache=True)
def _hash(a, b):
    h = (a * 374761393 + b * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0


@njit(cache=True)
def _vnoise(seed, x):
    i = int(np.floor(x))
    f = x - i
    f = f * f * (3.0 - 2.0 * f)
    a = _hash(seed, i)
    return a + (_hash(seed, i + 1) - a) * f


@njit(cache=True)
def _ss(e0, e1, x):
    t = (x - e0) / (e1 - e0)
    if t < 0.0:
        return 0.0
    if t > 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, fastmath=True)
def render_strokes(canvas, height, weave, pts, offs, prm, cols, bprof, bcol, bthr, paint, bmix):
    """Lay the strokes in order. With no `paint` rows a stroke is flat colour blended in by its alpha.
    With them (pigments.Palette.load) the canvas is linear reflectance and a stroke is a film of
    pigment: alpha is its thickness, and what it hides or lets through follows from the pigment.
    `weave` is the tooth a dry brush or a knife catches on."""
    H = canvas.shape[0]
    W = canvas.shape[1]
    NR = bprof.shape[0]
    NB = bprof.shape[1]
    km = paint.shape[0] > 0
    for k in range(offs.shape[0] - 1):
        p0 = offs[k]
        m = offs[k + 1] - p0
        if m < 2:
            continue
        hw = prm[k, 0]
        alpha = prm[k, 1]
        load = prm[k, 2]
        dry = prm[k, 3]
        smear = prm[k, 4]
        thick = prm[k, 5]
        rough = prm[k, 6]
        row = int(prm[k, 7]) % NR
        seed = int(prm[k, 8])
        cap = prm[k, 9] * hw + 0.5
        taper = prm[k, 10]
        knife = prm[k, 11] > 0.5 and prm[k, 11] < 1.5
        blend = prm[k, 11] > 1.5
        pool = prm[k, 12]
        firm = 0.2 + 0.75 * prm[k, 13]

        cum = np.empty(m)
        cum[0] = 0.0
        minx = pts[p0, 0]
        maxx = minx
        miny = pts[p0, 1]
        maxy = miny
        for i in range(1, m):
            ex = pts[p0 + i, 0] - pts[p0 + i - 1, 0]
            ey = pts[p0 + i, 1] - pts[p0 + i - 1, 1]
            cum[i] = cum[i - 1] + np.sqrt(ex * ex + ey * ey)
            minx = min(minx, pts[p0 + i, 0])
            maxx = max(maxx, pts[p0 + i, 0])
            miny = min(miny, pts[p0 + i, 1])
            maxy = max(maxy, pts[p0 + i, 1])
        L = cum[m - 1]
        if L < 0.5:
            continue
        pad = hw * 1.25 + cap + 2.0
        x0 = max(int(minx - pad), 0)
        x1 = min(int(maxx + pad) + 1, W)
        y0 = max(int(miny - pad), 0)
        y1 = min(int(maxy + pad) + 1, H)
        if x0 >= x1 or y0 >= y1:
            continue
        src = canvas[y0:y1, x0:x1, :].copy()
        sw = x1 - x0
        sh = y1 - y0

        for y in range(y0, y1):
            py = y + 0.5
            for x in range(x0, x1):
                px = x + 0.5
                # nearest segment
                best = 1e30
                bi = 0
                for i in range(m - 1):
                    ax = pts[p0 + i, 0]
                    ay = pts[p0 + i, 1]
                    ex = pts[p0 + i + 1, 0] - ax
                    ey = pts[p0 + i + 1, 1] - ay
                    ll = ex * ex + ey * ey
                    u = 0.0
                    if ll > 1e-12:
                        u = ((px - ax) * ex + (py - ay) * ey) / ll
                        if u < 0.0:
                            u = 0.0
                        elif u > 1.0:
                            u = 1.0
                    qx = ax + u * ex - px
                    qy = ay + u * ey - py
                    d2 = qx * qx + qy * qy
                    if d2 < best:
                        best = d2
                        bi = i
                ax = pts[p0 + bi, 0]
                ay = pts[p0 + bi, 1]
                ex = pts[p0 + bi + 1, 0] - ax
                ey = pts[p0 + bi + 1, 1] - ay
                seg = np.sqrt(ex * ex + ey * ey)
                if seg < 1e-9:
                    continue
                tx = ex / seg
                ty = ey / seg
                rx = px - ax
                ry = py - ay
                along = rx * tx + ry * ty
                perp = ry * tx - rx * ty
                s = cum[bi] + along
                if bi > 0 and along < 0.0:
                    s = cum[bi]
                if bi < m - 2 and along > seg:
                    s = cum[bi + 1]
                # squarish ends with a soft cap
                if s < 0.0:
                    acap = _ss(-cap, 0.0, s)
                elif s > L:
                    acap = 1.0 - _ss(L, L + cap, s)
                else:
                    acap = 1.0
                if acap <= 0.0:
                    continue
                sn = min(max(s / L, 0.0), 1.0)
                # pressure: quick swell at the start, slight lift at the end
                wl = (hw * (0.8 + 0.2 * _ss(0.0, 0.2, sn)) * (1.0 - 0.2 * _ss(0.7, 1.0, sn))
                      * (1.0 + (taper - 1.0) * sn))
                tt = abs(perp) / wl
                if tt >= 1.0:
                    continue
                side = seed * 2 + (1 if perp > 0.0 else 0)
                if knife:
                    # a blade leaves a clean edge that wanders slowly
                    edge = 1.0 - rough * 0.25 * _vnoise(side, s / (2.5 * hw + 1.0))
                    aedge = 1.0 - _ss(edge - 0.06, edge, tt)
                else:
                    edge = 1.0 - rough * 0.4 * _vnoise(side, s / (0.8 * hw + 1.0))
                    aedge = 1.0 - _ss(edge - firm, edge, tt)
                if aedge <= 0.0:
                    continue
                # bristles: fixed lanes across the stroke, so streaks run along it
                bpos = (perp / wl * 0.5 + 0.5) * (NB - 1)
                if bpos < 0.0:
                    bpos = 0.0
                elif bpos > NB - 1.001:
                    bpos = NB - 1.001
                b0 = int(bpos)
                bf = bpos - b0
                bint = bprof[row, b0] + (bprof[row, b0 + 1] - bprof[row, b0]) * bf
                bth = bthr[row, b0] + (bthr[row, b0 + 1] - bthr[row, b0]) * bf
                rem = load * (1.0 - dry * sn)                   # paint left on the brush
                if knife:
                    # no bristles: one flat film, bridging the hollows more as the blade empties
                    bint = 1.0
                    a = alpha * acap * aedge * _ss(0.0, 0.1, rem - 0.8 * (1.0 - weave[y, x]) * (0.3 + 0.7 * sn))
                else:
                    brun = _ss(bth - 0.04, bth + 0.1, rem)           # this bristle ran dry?
                    cov = _ss(0.0, 0.25, rem - 0.45 * (1.0 - weave[y, x]))  # dry brush hits weave peaks first
                    a = alpha * acap * aedge * bint * brun * cov
                if a <= 0.003:
                    continue
                if blend:
                    # a clean soft brush (or a finger) through wet paint: it carries the paint along and
                    # levels it, and lays none of its own
                    if a > 1.0:
                        a = 1.0
                    reach = hw * (1.0 + 1.5 * _hash(seed, 7))
                    sx = min(max(int(px - tx * reach) - x0, 0), sw - 1)
                    sy = min(max(int(py - ty * reach) - y0, 0), sh - 1)
                    fx = min(max(int(px + tx * reach) - x0, 0), sw - 1)
                    fy = min(max(int(py + ty * reach) - y0, 0), sh - 1)
                    canvas[y, x, 0] += (0.5 * (src[sy, sx, 0] + src[fy, fx, 0]) - canvas[y, x, 0]) * a
                    canvas[y, x, 1] += (0.5 * (src[sy, sx, 1] + src[fy, fx, 1]) - canvas[y, x, 1]) * a
                    canvas[y, x, 2] += (0.5 * (src[sy, sx, 2] + src[fy, fx, 2]) - canvas[y, x, 2]) * a
                    height[y, x] *= 1.0 - 0.5 * a
                    continue
                if km:
                    film = a
                    if pool != 0.0:
                        film *= max(0.0, 1.0 + pool * (1.0 - 2.0 * weave[y, x]))
                    if a > 1.0:
                        a = 1.0
                    wmix = paint[k, 12] * (bmix[row, b0] + (bmix[row, b0 + 1] - bmix[row, b0]) * bf)
                    mk = 0.0
                    sx = 0
                    sy = 0
                    if smear > 0.0:
                        sx = int(px - tx * hw * 1.5) - x0
                        sy = int(py - ty * hw * 1.5) - y0
                        if sx >= 0 and sy >= 0 and sx < sw and sy < sh:
                            mk = smear * (0.3 + 0.7 * sn)
                    for c in range(3):
                        # the film's own reflectance and its transmittance, then what shows of the paint under it
                        kk = paint[k, c] + (paint[k, 6 + c] - paint[k, c]) * wmix
                        ks = paint[k, 3 + c] + (paint[k, 9 + c] - paint[k, 3 + c]) * wmix
                        if ks < 1e-4:
                            ks = 1e-4
                        if kk < 1e-6 * ks:
                            kk = 1e-6 * ks
                        ka = 1.0 + kk / ks
                        kb = np.sqrt(ka * ka - 1.0)
                        kx = kb * ks * film
                        if kx > 18.0:
                            kx = 18.0
                        ke = np.exp(kx)
                        ksh = 0.5 * (ke - 1.0 / ke)
                        den = ka * ksh + kb * 0.5 * (ke + 1.0 / ke)
                        fr = ksh / den
                        ft = kb / den
                        g = canvas[y, x, c]
                        r = fr + ft * ft * g / (1.0 - fr * g)
                        lo = 0.9 * min(g, cols[k, c])            # a glaze is wiped back before it goes darker than meant
                        if r < lo:
                            r = lo
                        if mk > 0.0:
                            r += (g + (src[sy, sx, c] - g) * a - r) * mk
                        canvas[y, x, c] = r
                else:
                    if a > 1.0:
                        a = 1.0
                    cr = cols[k, 0] + bcol[row, b0, 0]
                    cg = cols[k, 1] + bcol[row, b0, 1]
                    cb = cols[k, 2] + bcol[row, b0, 2]
                    if smear > 0.0:
                        # wet pick-up: drag in the paint from where the brush has just been
                        sx = int(px - tx * hw * 1.5) - x0
                        sy = int(py - ty * hw * 1.5) - y0
                        if sx >= 0 and sy >= 0 and sx < sw and sy < sh:
                            mk = smear * (0.3 + 0.7 * sn)
                            cr += (src[sy, sx, 0] - cr) * mk
                            cg += (src[sy, sx, 1] - cg) * mk
                            cb += (src[sy, sx, 2] - cb) * mk
                    canvas[y, x, 0] += (cr - canvas[y, x, 0]) * a
                    canvas[y, x, 1] += (cg - canvas[y, x, 1]) * a
                    canvas[y, x, 2] += (cb - canvas[y, x, 2]) * a
                if knife:
                    # scraped thin in the middle, squeezed out along both edges and into a lip where the blade lifts
                    ridge = 0.6 + 1.0 * _ss(0.72, 0.97, tt) + 0.6 * _ss(0.8, 1.0, sn)
                else:
                    ridge = 0.75 + 0.5 * _ss(0.5, 0.9, tt)          # paint piles up at the edges
                height[y, x] = (height[y, x] * (1.0 - 0.35 * a)
                                + thick * a * (0.3 + 0.7 * rem) * ridge * (0.55 + 0.45 * bint))


@njit(cache=True)
def clip_extent(region, cx, cy, dx, dy, half, over):
    """March from each stroke centre both ways until the region id changes; `over` is how far each
    stroke may run past that edge."""
    H = region.shape[0]
    W = region.shape[1]
    n = cx.shape[0]
    out = np.zeros((n, 2))
    for k in range(n):
        ix = int(cx[k])
        iy = int(cy[k])
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            continue
        rid = region[iy, ix]
        for side in range(2):
            sg = 1.0 if side == 0 else -1.0
            e = 0.0
            while e < half[k]:
                jx = int(np.floor(cx[k] + sg * dx[k] * (e + 1.0)))
                jy = int(np.floor(cy[k] + sg * dy[k] * (e + 1.0)))
                if jx < 0 or jy < 0 or jx >= W or jy >= H or region[jy, jx] != rid:
                    e += over[k]
                    break
                e += 1.0
            out[k, side] = min(e, half[k])
    return out


def make_bristles(rng, rows=96, nb=40):
    """Random brush heads: per-bristle density, colour offset and run-dry threshold."""
    prof = np.empty((rows, nb), np.float32)
    col = np.empty((rows, nb, 3), np.float32)
    thr = np.empty((rows, nb), np.float32)
    k3 = np.ones(3) / 3.0
    for r in range(rows):
        p = np.convolve(rng.random(nb), k3, mode='same')
        p = 0.5 + 0.5 * (p - p.min()) / (np.ptp(p) + 1e-6)
        gaps = rng.random(nb) < 0.07
        p[gaps] *= rng.uniform(0.1, 0.5, gaps.sum())
        prof[r] = p
        val = np.convolve(rng.normal(0, 1, nb), k3, mode='same') * 0.035
        temp = np.convolve(rng.normal(0, 1, nb), k3, mode='same') * 0.02
        col[r, :, 0] = val + temp
        col[r, :, 1] = val
        col[r, :, 2] = val - temp
        thr[r] = np.clip(np.convolve(rng.uniform(0.0, 0.35, nb), k3, mode='same'), 0.0, 0.45)
    return prof, col, thr


def make_streaks(rng, rows=96, nb=40):
    """Where, across each brush head, a half-stirred load shows its second colour: broad lanes in [0, 1]."""
    k7 = np.ones(7) / 7.0
    out = np.empty((rows, nb), np.float32)
    for r in range(rows):
        p = np.convolve(rng.random(nb + 12), k7, mode='same')[6:-6]
        t = np.clip(((p - p.min()) / (np.ptp(p) + 1e-6) - 0.3) / 0.4, 0.0, 1.0)
        out[r] = t * t * (3.0 - 2.0 * t)
    return out


def canvas_weave(H, W, period, rng):
    """Plain-weave canvas height in [0, 1]."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = (xx + 0.3 * period * fbm(H, W, 30, rng, 2)) / period
    b = (yy + 0.3 * period * fbm(H, W, 30, rng, 2)) / period
    warp = np.sin(np.pi * a) ** 2
    weft = np.sin(np.pi * b) ** 2
    over = ((np.floor(a) + np.floor(b)) % 2) > 0
    wv = np.where(over, warp * (0.55 + 0.45 * weft), weft * (0.55 + 0.45 * warp))
    wv = wv + 0.12 * fbm(H, W, 2.5, rng, 2)
    wv = (wv - wv.min()) / (np.ptp(wv) + 1e-6)
    return wv.astype(np.float32)


def gesso(H, W, rng):
    """Chalk-and-glue panel ground in [0, 1]: nearly smooth, faint scraper streaks."""
    g = 0.6 * fbm(H, W, (1.5, 60.0), rng, 3) + 0.4 * fbm(H, W, 5.0, rng, 3)
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    return g.astype(np.float32)
