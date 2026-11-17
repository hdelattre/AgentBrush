"""Weathering: a picture made by rain.

A panel is given many flat coats of colour, one over another, each left to dry: the whole panel one colour,
then the whole panel the next. Nothing is drawn. Then it is stood out in the rain. Water lands on its face,
gathers, and runs down it, and where water runs the paint goes: a little water takes the top coat off, a
steady run cuts down through coat after coat, and a stream goes to the bare ground. The picture is the map of
where the water went, in the colours it found on the way down.

The painter has four things to say about it, each a field the scene supplies in `scene.masks`:
  wax     where the top coat is sealed. Water runs over wax and takes nothing
  ridge   dams: raised beads the water cannot cross. It runs along a dam to its end and pours off there, so
          a dam leaves a dry place under it and a deep bright cut below each end
  soft    where the paint was laid thin and washes out sooner (1 is ordinary paint)
  rain    how much rain lands where (1 is the open face)
and the coats themselves, top to bottom: a colour, a thickness, and how hard it dried.

`_rain` is the weather: drops land, run downhill (down the panel, off the dams, and into whatever channel is
already cut, which is how rills form), wear away a little paint wherever they go, and dry up after a while.
A style weathers a panel with one pass, `dict(kind='weather', coats=[(colour, thickness, hardness), ...], ...)`.
The relief is real: the cuts are in the normal map as deep as they are in the paint.
"""
import numpy as np
from numba import njit

from .noise import fbm, smoothstep
from .pigments import to_linear
from .raster import blur


@njit(fastmath=True, cache=True)
def _rain(depth, dam, wax, soft, rain, bounds, hard, drops, life, rate, splash, pull, inertia, wind, stray, seed):
    """Let `drops` drops fall. `depth` is how far the paint is worn at each point (in coats' thickness);
    `bounds[i]` is the depth at which coat i ends and `hard[i]` how hard it is; the last bound is the ground,
    which does not wear. `dam` is the dams' height, steep enough to turn water."""
    H, W = depth.shape
    np.random.seed(seed)
    ncoat = hard.size
    floor = bounds[ncoat - 1]
    for _ in range(drops):
        x, y = np.random.random() * (W - 2) + 1.0, np.random.random() * (H - 2) + 1.0
        if np.random.random() > rain[int(y), int(x)]:
            continue
        vx, vy = 0.0, 0.0
        # where it lands it knocks a little paint off, whether or not it runs anywhere
        ix, iy = int(x), int(y)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if wax[iy + dy, ix + dx] > 0.5:
                    continue
                d = depth[iy + dy, ix + dx]
                k = 0
                while k < ncoat - 1 and d >= bounds[k]:
                    k += 1
                share = 0.4 if (dx == 0 and dy == 0) else (0.1 if (dx == 0 or dy == 0) else 0.05)
                depth[iy + dy, ix + dx] = min(d + splash * soft[iy + dy, ix + dx] * share / hard[k], floor)
        for _ in range(int(life * (0.4 + 1.2 * np.random.random()))):
            ix, iy = int(x), int(y)
            if ix < 1 or iy < 1 or ix > W - 2 or iy > H - 2:
                break
            gx = 0.5 * (dam[iy, ix + 1] - dam[iy, ix - 1] - pull * (depth[iy, ix + 1] - depth[iy, ix - 1]))
            gy = 0.5 * (dam[iy + 1, ix] - dam[iy - 1, ix] - pull * (depth[iy + 1, ix] - depth[iy - 1, ix]))
            vx = vx * inertia + (wind - gx + stray * (np.random.random() - 0.5)) * (1.0 - inertia)
            vy = vy * inertia + (1.0 - gy) * (1.0 - inertia)
            speed = np.sqrt(vx * vx + vy * vy)
            if speed < 1e-4:
                break
            x, y = x + vx / speed, y + vy / speed
            ix, iy = int(x), int(y)
            if ix < 1 or iy < 1 or ix > W - 2 or iy > H - 2:
                break
            if wax[iy, ix] > 0.5:
                continue
            d = depth[iy, ix]
            k = 0
            while k < ncoat - 1 and d >= bounds[k]:
                k += 1
            wear = rate * soft[iy, ix] * min(speed, 1.5) / hard[k]
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if wax[iy + dy, ix + dx] > 0.5:
                        continue
                    share = 0.4 if (dx == 0 and dy == 0) else (0.1 if (dx == 0 or dy == 0) else 0.05)
                    depth[iy + dy, ix + dx] = min(depth[iy + dy, ix + dx] + wear * share, floor)


def weather(canvas, height, scene, P, scale, rng, linear=False):
    """Stand the panel in the rain and show what is left. Returns how many drops fell."""
    H, W = scene.H, scene.W
    s = scale / 2048.0
    coats = P['coats']
    bounds = np.cumsum([c[1] for c in coats]).astype(np.float64)
    hard = np.array([c[2] for c in coats], np.float64)
    wax = np.ascontiguousarray(scene.masks['wax'], np.float32)
    ridge = np.ascontiguousarray(scene.masks['ridge'], np.float32)
    dam = (P.get('dam', 30.0) * blur(ridge, 2.0 * max(s, 0.5))).astype(np.float32)
    depth = np.zeros((H, W), np.float32)
    drops = int(P.get('drops', 1.5) * H * W)
    life = P.get('life', 0.2) * scale
    # `splash` is how many coats' depth the rain knocks off just by landing; `wash` how much more the open face
    # loses to running water, far enough down that as much water dries up as lands
    rate = P.get('wash', 1.4) / (P.get('drops', 1.5) * life)
    _rain(depth, dam, wax, np.ascontiguousarray(scene.masks['soft'], np.float32), np.ascontiguousarray(scene.masks['rain'], np.float32),
          bounds, hard, drops, life, rate, P.get('splash', 1.9) / P.get('drops', 1.5), P.get('pull', 0.35), P.get('inertia', 0.6),
          P.get('wind', 0.1), P.get('stray', 0.5), int(rng.integers(1 << 30)))
    depth = blur(depth, P.get('settle', 1.4) * max(s, 0.5))       # wet paint slumps a little at every edge the water cut

    # the colours down through the paint. No coat was brushed on quite evenly, so where one ends wanders a little
    lut = np.zeros((4096, 3), np.float32)
    at = np.linspace(0.0, bounds[-1], 4096)
    top = 0.0
    for (colour, thick, _), end in zip(coats, bounds):
        lut += np.asarray(colour, np.float32) * (smoothstep(top - 0.02, top + 0.02, at) * (1.0 - smoothstep(end - 0.02, end + 0.02, at)))[:, None]
        top = end
    lut[at < 0.02] = coats[0][0]
    wander = 0.16 * fbm(H, W, 70.0 * s, rng, 3) + 0.05 * fbm(H, W, 9.0 * s, rng, 2)
    bare = depth >= bounds[-1] - 0.03                      # worn right through: the ground
    shown = np.clip(depth + wander * np.clip(depth / 0.4, 0.0, 1.0), 0.0, bounds[-1] - 1e-3)
    colour = lut[np.clip((shown / bounds[-1] * 4095.0).astype(np.int64), 0, 4095)]
    colour[bare] = P.get('ground', (0.97, 0.96, 0.92))
    colour *= (1.0 + 0.03 * fbm(H, W, (2.0, 50.0 * s), rng, 2))[..., None]                 # the brush that laid the coats
    amber = np.asarray(P.get('wax', (0.50, 0.36, 0.16)), np.float32)
    sealed = np.clip(0.05 * wax + 0.55 * np.clip(blur(ridge, 0.8 * max(s, 0.5)) * 1.6, 0.0, 1.0), 0.0, 1.0)
    colour = colour + (amber - colour) * sealed[..., None]
    canvas[:] = to_linear(colour) if linear else np.clip(colour, 0.0, 1.0)
    height += P.get('relief', 0.8) * (0.3 * wax + 1.6 * np.clip(blur(ridge, 1.2 * max(s, 0.5)) * 1.5, 0.0, 1.0) - depth)
    return drops
