"""Pigments: paint as a film that absorbs and scatters light (Kubelka-Munk), instead of a flat colour.

A pigment has an absorption K and a scattering S per colour channel, for the thickness of paint one
loaded brush stroke leaves. Mixing pigments adds their K and S by proportion, and a film of any
thickness over whatever is already on the canvas follows from two numbers per channel: what it
reflects by itself and what it lets through. That one rule covers the range of oil handling. Lead
white scatters hard, so a full stroke of it hides what is under it and a thin one is a scumble (cooler
than the white itself, because it scatters blue most). Lakes and earths barely scatter, so they are
glazes: they only darken and warm, and the ground shines up through them. Medium thins either.

A `Palette` is a painter's set of pigments and a table of mixtures of them. `load` picks, for each
stroke, the mixture whose film over the paint already there comes nearest the colour wanted. So the
picture can only hold colours those pigments mix, its darks come out transparent and its lights opaque.

Colours are display values (as the scenes and PNGs hold them) at the door, linear reflectance inside."""
import numpy as np
from numba import njit, prange

GAMMA = 2.2
NPAINT = 13     # a stroke's load: K and S of its paint, K and S of the streak in it, how streaked it is


def to_linear(c):
    return np.clip(np.asarray(c, np.float64), 0.0, 1.0) ** GAMMA


def to_display(c):
    return np.clip(np.asarray(c, np.float64), 0.0, 1.0) ** (1.0 / GAMMA)


def body(masstone, scatter):
    """A pigment with body, from the colour of a thick layer of it and how hard it scatters."""
    R = to_linear(masstone)
    S = np.broadcast_to(np.asarray(scatter, np.float64), (3,)).copy()
    return S * (1.0 - R) ** 2 / (2.0 * R), S


def lake(undertone, scatter):
    """A transparent pigment, from the colour of one film of it over white."""
    S = np.broadcast_to(np.asarray(scatter, np.float64), (3,)).copy()
    return -0.5 * np.log(to_linear(undertone)), S


PIGMENTS = dict(
    white=body((0.965, 0.955, 0.925), (8.5, 10.0, 12.0)),      # lead white
    chrome=body((0.96, 0.78, 0.10), (7.0, 7.0, 5.0)),          # chrome yellow
    ochre=body((0.76, 0.56, 0.22), (3.5, 3.5, 3.0)),           # yellow ochre
    vermilion=body((0.86, 0.22, 0.12), (6.0, 5.0, 5.0)),
    madder=lake((0.80, 0.16, 0.26), 0.25),                     # rose madder
    sienna=lake((0.62, 0.27, 0.10), 0.6),                      # burnt sienna
    umber=lake((0.26, 0.15, 0.09), 0.5),                       # burnt umber
    cobalt=body((0.10, 0.27, 0.66), (2.0, 2.2, 3.0)),
    ultramarine=lake((0.10, 0.16, 0.70), 0.35),
    prussian=lake((0.02, 0.17, 0.38), 0.12),
    viridian=lake((0.05, 0.50, 0.38), 0.3),
    indigo=lake((0.07, 0.10, 0.28), 0.02),                     # indigo and Vandyke brown: the grainer's glazes,
    vandyke=lake((0.24, 0.13, 0.07), 0.02),                    # clear as stained varnish, with no body at all
    black=body((0.045, 0.045, 0.05), 0.8),                     # ivory black
    # the faience painter's stains (faience.py): metal oxides fired into tin glaze, clear as coloured glass
    zaffre=lake((0.20, 0.33, 0.72), 0.02),                     # cobalt
    copper=lake((0.20, 0.62, 0.42), 0.02),
    antimony=lake((0.96, 0.80, 0.22), 0.04),                   # Naples yellow
    rust=lake((0.86, 0.42, 0.14), 0.03),                       # iron
    manganese=lake((0.42, 0.24, 0.34), 0.02),
    rose=lake((0.86, 0.40, 0.52), 0.02),                       # purple of Cassius: the pink that is made from gold
    medium=(np.array([0.004, 0.012, 0.035]), np.zeros(3)),     # megilp: thins a paint, and is a shade amber
)


def layer(K, S, d):
    """What a film of paint d thick reflects by itself (over black), and what it lets through."""
    S = np.maximum(S, 1e-4)
    K = np.maximum(K, 1e-6 * S)
    a = 1.0 + K / S
    b = np.sqrt(a * a - 1.0)
    x = np.minimum(b * S * d, 18.0)
    sh, ch = np.sinh(x), np.cosh(x)
    den = a * sh + b * ch
    return sh / den, b / den


def over(K, S, d, under):
    """The colour of that film laid over `under` (linear reflectance)."""
    R, T = layer(K, S, d)
    return R + T * T * under / (1.0 - R * under)


@njit(fastmath=True, inline='always')
def _film(K, S, d):
    if S < 1e-4:
        S = 1e-4
    if K < 1e-6 * S:
        K = 1e-6 * S
    a = 1.0 + K / S
    b = np.sqrt(a * a - 1.0)
    x = min(b * S * d, 18.0)
    e = np.exp(x)
    sh, ch = 0.5 * (e - 1.0 / e), 0.5 * (e + 1.0 / e)
    den = a * sh + b * ch
    return sh / den, b / den


@njit(fastmath=True)
def _miss(c, Kp, Sp, film, want, under, pg, cover):
    """How far a film of the mixture `c` (a share of each pigment) over `under` falls from `want`.
    Errors count for more in the darks, as they do to the eye."""
    e, h = 0.0, 0.0
    for ch in range(3):
        K, S = 0.0, 0.0
        for p in range(c.shape[0]):
            K += c[p] * Kp[p, ch]
            S += c[p] * Sp[p, ch]
        R, T = _film(K, S, film)
        u, t = under[ch], want[ch]
        r = R + T * T * u / (1.0 - R * u) - t
        e += r * r / (t + 0.03)
        h += S * film / (1.0 + S * film) / 3.0
    return e + pg * h + cover * (1.0 - h)


@njit(parallel=True, fastmath=True, cache=True)
def _choose(want, under, Rb, T2, hide, conc, Kp, Sp, film, glaze, cover):
    """For each stroke the mixture whose film over `under` comes nearest `want`: the best of the table,
    then stirred a little more or less of each pigment at a time until it stops getting nearer. A stroke
    that has to darken toward a dark is steered to paint that doesn't hide (`glaze`): darks are glazed,
    lights and half-tones have body. `cover` steers every stroke to paint that hides (or, negative, away)."""
    n, M, P = want.shape[0], Rb.shape[0], Kp.shape[0]
    out = np.zeros((n, P))
    for i in prange(n):
        t0, t1, t2 = want[i, 0], want[i, 1], want[i, 2]
        u0, u1, u2 = under[i, 0], under[i, 1], under[i, 2]
        w0, w1, w2 = 1.0 / (t0 + 0.03), 1.0 / (t1 + 0.03), 1.0 / (t2 + 0.03)
        lt = 0.3 * t0 + 0.6 * t1 + 0.1 * t2
        pg = glaze * max(0.0, 0.3 * u0 + 0.6 * u1 + 0.1 * u2 - lt) * max(0.0, 1.0 - lt / 0.22)
        best, k1 = 1e30, 0
        for k in range(M):
            r0 = Rb[k, 0] + T2[k, 0] * u0 / (1.0 - Rb[k, 0] * u0) - t0
            r1 = Rb[k, 1] + T2[k, 1] * u1 / (1.0 - Rb[k, 1] * u1) - t1
            r2 = Rb[k, 2] + T2[k, 2] * u2 / (1.0 - Rb[k, 2] * u2) - t2
            e = w0 * r0 * r0 + w1 * r1 * r1 + w2 * r2 * r2 + pg * hide[k] + cover * (1.0 - hide[k])
            if e < best:
                best, k1 = e, k
        c = conc[k1].copy()
        trial = np.empty(P)
        step = 0.6
        while step > 0.03:
            moved = False
            for a in range(P):
                for sign in range(2):
                    old = c[a]
                    new = old * (1.0 + step) + 0.004 * step if sign == 0 else max(old / (1.0 + step) - 0.004 * step, 0.0)
                    if new == old or new >= 0.999 or old >= 0.999:
                        continue
                    f = (1.0 - new) / (1.0 - old)
                    for p in range(P):
                        trial[p] = c[p] * f
                    trial[a] = new
                    e = _miss(trial, Kp, Sp, film, want[i], under[i], pg, cover)
                    if e < best:
                        best, moved = e, True
                        for p in range(P):
                            c[p] = trial[p]
            if not moved:
                step *= 0.5
        out[i] = c
    return out


class Palette:
    """A set of pigments (names from PIGMENTS, as one string) and a table of `n` mixtures of them:
    tints let down with white, glazes let down with medium, veils of a little white in medium, and
    colour straight from the tube."""

    def __init__(self, names, n=2500, seed=1):
        names = names.split() if isinstance(names, str) else list(names)
        names += [p for p in ('white', 'medium') if p not in names]
        self.names = names
        self.K = np.array([PIGMENTS[p][0] for p in names])
        self.S = np.array([PIGMENTS[p][1] for p in names])
        iw, im = names.index('white'), names.index('medium')
        tubes = [i for i in range(len(names)) if i not in (iw, im)]
        rng = np.random.default_rng(seed)
        conc = np.zeros((n, len(names)))
        for i in range(n):
            if i <= len(tubes):                 # white and each pigment unmixed
                conc[i, iw if i == len(tubes) else tubes[i]] = 1.0
                continue
            m = rng.choice([1, 2, 3], p=[0.25, 0.5, 0.25])
            js = rng.choice(tubes, m, replace=False)
            w = rng.dirichlet(np.ones(m))
            kind = rng.random()
            if kind < 0.5:
                c = 10.0 ** -rng.uniform(0.0, 2.8)
                conc[i, js], conc[i, iw] = c * w, 1.0 - c
            elif kind < 0.8:
                c = 10.0 ** -rng.uniform(0.0, 1.7)
                conc[i, js], conc[i, im] = c * w, 1.0 - c
            elif kind < 0.92:
                wh = 10.0 ** -rng.uniform(0.3, 1.3)
                c = wh * 10.0 ** -rng.uniform(0.0, 2.0)
                conc[i, js], conc[i, iw], conc[i, im] = c * w, wh, 1.0 - wh - c
            else:
                conc[i, js] = w
        self.conc = conc
        self.Kmix, self.Smix = conc @ self.K, conc @ self.S
        self._tables = {}

    def recipe(self, **parts):
        """K and S of a named mix, e.g. recipe(umber=2, madder=1, medium=3)."""
        c = np.zeros(len(self.names))
        for name, amount in parts.items():
            c[self.names.index(name)] = amount
        c /= c.sum()
        return c @ self.K, c @ self.S

    def _table(self, film):
        film = round(float(film), 2)
        if film not in self._tables:
            R, T = layer(self.Kmix, self.Smix, film)
            hide = (self.Smix * film / (1.0 + self.Smix * film)).mean(1)
            self._tables[film] = np.ascontiguousarray(R), np.ascontiguousarray(T * T), hide
        return self._tables[film]

    def load(self, want, under, film, rng, glaze=0.25, cover=0.0, streak=0.0, streak_to=None):
        """Brush loads for strokes that want the display colours `want` over the linear canvas colours
        `under`, laid `film` thick: an (n, NPAINT) array for brush.render_strokes. A streaked load is
        the mixture not quite stirred, so one of its own ingredients (or `streak_to`) shows in lanes."""
        want = np.ascontiguousarray(to_linear(want))
        under = np.ascontiguousarray(np.clip(under, 0.0, 0.98), np.float64)
        Rb, T2, hide = self._table(film)
        conc = _choose(want, under, Rb, T2, hide, self.conc, self.K, self.S, round(float(film), 2), glaze, cover)
        n = len(conc)
        out = np.zeros((n, NPAINT))
        out[:, 0:3], out[:, 3:6] = conc @ self.K, conc @ self.S
        if streak > 0:
            if streak_to is None:
                j = (rng.random(conc.shape) * (conc > 0.02)).argmax(1)
            else:
                j = np.full(n, self.names.index(streak_to))
            more = conc.copy()
            more[np.arange(n), j] = more[np.arange(n), j] * rng.uniform(1.3, 2.2, n) + (0.0 if streak_to is None else 0.15)
            more /= more.sum(1, keepdims=True)
            out[:, 6:9], out[:, 9:12] = more @ self.K, more @ self.S
            out[:, 12] = streak * rng.uniform(0.5, 1.0, n)
        else:
            out[:, 6:12] = out[:, 0:6]
        return out
