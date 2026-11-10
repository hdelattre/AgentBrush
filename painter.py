"""Stroke planning: turn a Scene + a style into brush strokes, coarse to fine.

Each 'patches' pass follows Hertzmann (1998): blur the reference to the brush
scale, find where the canvas still disagrees with it, and put strokes there.
Strokes come in patches of parallel marks (Cezanne's 'constructive stroke'),
take their colour from the reference, and stop at region boundaries so
silhouettes stay legible.

A style with a `palette` (pigments.Palette) paints in pigment instead of flat
colour: each stroke is loaded with the mixture of the painter's pigments that
comes nearest what the picture wants there, and laid as a film over the paint
already down. Passes can then work as a painter's layers do: a pass may take
its own reference (`ref`) and keep to a `mask` from the scene, paint only
where the picture wants more light or less (`where`), drag dry over the ridges
of earlier paint (`tooth`), settle into its hollows (`pool`), be laid with a
knife, or lay nothing and only blend what is wet (`tool`), and its strokes can
thin out softly at the sides (`feather`).

A picture can also be taken out of a wet glaze instead of put on (graining.py):
a 'glaze' pass rubs one transparent paint over everything, the 'grain' passes
after it comb, wipe and mottle it along the scene's figures, and it dries
before the next pass of any other kind.
"""
import time
from dataclasses import dataclass

import numpy as np

import graining
from brush import render_strokes, clip_extent, make_bristles, make_streaks, canvas_weave, gesso, NPARAM
from noise import mix
from pigments import NPAINT, to_linear, to_display
from raster import blur
from scenes import Scene, LEAF

LUMA = np.array([0.3, 0.59, 0.11], np.float32)
TOOLS = dict(brush=0.0, knife=1.0, blend=2.0)    # blend: a clean brush drawn through the wet paint, laying none


@dataclass
class Painting:
    """Unlit paint layer plus what export needs to light it and bake game maps."""
    scene: Scene
    style: dict
    canvas: np.ndarray    # paint albedo (leaf is composited by export)
    height: np.ndarray    # paint thickness
    surface: np.ndarray   # ground texture: canvas weave or gesso, [0, 1]


def masked_blur(img, keep, sigma):
    """Blur only the paintable parts, so gilded areas never tint the colours brushes pick up."""
    if keep.min() >= 1.0:
        return blur(img, sigma)
    num = blur(img * keep[..., None], sigma)
    den = blur(keep, sigma)[..., None]
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), img).astype(np.float32)


def _trace(angle, x, y, dx, dy, length, follow, steps=3):
    """Walk `length` px from (x, y) along the direction field, starting along (dx, dy).
    The field is a line direction (a and a + pi agree), so each step keeps the heading it had;
    `follow` is how quickly the heading turns to the field."""
    H, W = angle.shape
    h = length / steps
    xs, ys = [], []
    for _ in range(steps):
        a = angle[np.clip(y.astype(np.int64), 0, H - 1), np.clip(x.astype(np.int64), 0, W - 1)]
        ux, uy = np.cos(a), np.sin(a)
        sgn = np.where(ux * dx + uy * dy < 0.0, -1.0, 1.0)
        dx, dy = dx + (sgn * ux - dx) * follow, dy + (sgn * uy - dy) * follow
        nrm = np.hypot(dx, dy) + 1e-9
        dx, dy = dx / nrm, dy / nrm
        x, y = x + dx * h, y + dy * h
        xs.append(x)
        ys.append(y)
    return np.stack(xs, 1), np.stack(ys, 1)


def plan_patches(canvas, ref, scene, hw, P, rng, mask=None):
    """Strokes for one pass: points, offsets, parameters, colours, and each stroke's centre (row, col).
    `mask` keeps the pass to part of the picture and thins the paint toward that part's edge."""
    H, W, _ = canvas.shape
    kmin, kmax = P['k']
    lmin, lmax = P['len']
    area = 0.5 * (kmin + kmax) * 2.0 * hw * 0.85 * 0.5 * (lmin + lmax) * hw
    cell = max(2, int(round(np.sqrt(area / P['coverage']))))
    ny, nx = -(-H // cell), -(-W // cell)

    if P['thr'] <= 0:
        jy, jx = np.nonzero(np.ones((ny, nx), bool))
        oy = rng.integers(0, cell, jy.size)
        ox = rng.integers(0, cell, jy.size)
    else:
        err = np.sqrt(((canvas - ref) ** 2).sum(2))
        if 'where' in P:
            # 'lighter': only where the picture wants more light than the canvas has; 'darker': less
            more = (ref - canvas) @ LUMA
            err = err * ((more > 0) if P['where'] == 'lighter' else (more < 0))
        if mask is not None:
            err = err * mask
        err[scene.region == LEAF] = 0.0     # leaf is laid, never painted
        thr = P['thr'] * (1.0 + P.get('detail_gain', 0.8) * (0.5 - scene.detail))
        e = np.pad(err - thr, ((0, ny * cell - H), (0, nx * cell - W)), constant_values=-1.0)
        e = e.reshape(ny, cell, nx, cell).transpose(0, 2, 1, 3).reshape(ny, nx, cell * cell)
        jy, jx = np.nonzero(e.mean(2) > 0)
        am = e[jy, jx].argmax(1)          # start where the canvas is most wrong
        oy, ox = am // cell, am % cell
    cy = jy * cell + oy + rng.random(jy.size)
    cx = jx * cell + ox + rng.random(jx.size)
    keep = (cy < H) & (cx < W)
    if mask is not None and P['thr'] <= 0:
        keep[keep] = rng.random(keep.sum()) < mask[cy[keep].astype(np.int64), cx[keep].astype(np.int64)]
    cy, cx = cy[keep], cx[keep]
    n = cy.size
    if n == 0:
        return None

    # patch-level choices: direction, stroke count, length, warm/cool + value shift
    iy = np.minimum(cy.astype(np.int64), H - 1)
    ix = np.minimum(cx.astype(np.int64), W - 1)
    ang = scene.angle[iy, ix]
    if P.get('edge_follow', 0.0) > 0:
        # Hertzmann's rule: on an edge, stroke along it (perpendicular to the colour gradient)
        gy, gx = np.gradient(ref @ np.array([0.3, 0.59, 0.11], np.float32))
        on_edge = np.hypot(gx[iy, ix], gy[iy, ix]) > P['edge_follow']
        ang = np.where(on_edge, np.arctan2(gx[iy, ix], -gy[iy, ix]), ang)
    th0 = ang + rng.normal(0.0, P['ang_jit'], n)
    k = rng.integers(kmin, kmax + 1, n)
    plen = hw * rng.uniform(lmin, lmax, n)
    shift = (rng.normal(0, 1, n)[:, None] * P['temp'] * np.array([1.0, 0.1, -1.0])
             + rng.normal(0, 1, n)[:, None] * P['val'])
    rank = rng.permutation(n)

    # expand patches into parallel strokes
    m = int(k.sum())
    pid = np.repeat(np.arange(n), k)
    j = np.arange(m) - np.repeat(np.cumsum(k) - k, k) - (k[pid] - 1) / 2.0
    th = th0[pid] + rng.normal(0.0, 0.035, m)
    dx, dy = np.cos(th), np.sin(th)
    slen = plen[pid] * rng.uniform(0.8, 1.15, m)
    pitch = 2.0 * hw * rng.uniform(*P.get('pitch', (0.72, 0.95)), m)
    along = rng.uniform(-0.2, 0.2, m) * slen
    sx = cx[pid] - dy * j * pitch + dx * along
    sy = cy[pid] + dx * j * pitch + dy * along

    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    ok[ok] = scene.region[sy[ok].astype(np.int64), sx[ok].astype(np.int64)] != LEAF
    pid, j, dx, dy, slen, sx, sy = pid[ok], j[ok], dx[ok], dy[ok], slen[ok], sx[ok], sy[ok]
    over = np.full(sx.size, P.get('overshoot', 0.4) * hw)
    if scene.soft is not None:
        # where the scene loses its edges, strokes run on across them
        over += P.get('soft_over', 3.0) * hw * scene.soft[sy.astype(np.int64), sx.astype(np.int64)]
    ext = clip_extent(scene.region, sx, sy, dx, dy, slen * 0.5, over)
    tot = ext[:, 0] + ext[:, 1]
    ok = tot > np.maximum(1.2 * hw, 0.3 * slen)
    pid, j, dx, dy, sx, sy, ext, tot = pid[ok], j[ok], dx[ok], dy[ok], sx[ok], sy[ok], ext[ok], tot[ok]
    m = pid.size
    if m == 0:
        return None

    ax, ay = sx - dx * ext[:, 1], sy - dy * ext[:, 1]
    bx, by = sx + dx * ext[:, 0], sy + dy * ext[:, 0]
    flip = rng.random(m) < 0.5
    ax, bx = np.where(flip, bx, ax), np.where(flip, ax, bx)
    ay, by = np.where(flip, by, ay), np.where(flip, ay, by)
    bow = rng.normal(0.0, 0.05, m) * tot
    if P.get('flow', 0.0) > 0:
        # curve each stroke along the direction field (Van Gogh's swirls): trace it both ways
        bx_, by_ = _trace(scene.angle, sx, sy, -dx, -dy, ext[:, 1], P['flow'])
        fx_, fy_ = _trace(scene.angle, sx, sy, dx, dy, ext[:, 0], P['flow'])
        px = np.concatenate([bx_[:, ::-1], sx[:, None], fx_], 1)
        py = np.concatenate([by_[:, ::-1], sy[:, None], fy_], 1)
        px = np.where(flip[:, None], px[:, ::-1], px)
        py = np.where(flip[:, None], py[:, ::-1], py)
    else:
        t = np.linspace(0.0, 1.0, 5)
        arc = 4.0 * t * (1.0 - t)
        px = ax[:, None] + (bx - ax)[:, None] * t + (-dy * bow)[:, None] * arc
        py = ay[:, None] + (by - ay)[:, None] * t + (dx * bow)[:, None] * arc

    ci = np.clip(sy.astype(np.int64), 0, H - 1)
    cj = np.clip(sx.astype(np.int64), 0, W - 1)
    cols = ref[ci, cj] + shift[pid] + rng.normal(0.0, P['jit'], (m, 3))
    if P.get('hue', 0.0) > 0:
        # divisionist touch: turn each patch's colour a little round the grey axis
        rot = rng.normal(0.0, P['hue'], n)[pid][:, None]
        grey = cols.mean(1, keepdims=True)
        d = cols - grey
        cols = grey + d * np.cos(rot) + np.cross(np.full(3, 3 ** -0.5), d) * np.sin(rot)

    prm = np.empty((m, NPARAM))
    prm[:, 0] = hw * rng.uniform(0.85, 1.12, m)
    prm[:, 1] = P['alpha'] * rng.uniform(0.85, 1.0, m)
    prm[:, 2] = P.get('load', 1.0) * rng.uniform(0.75, 1.0, m)
    prm[:, 3] = P['dry'] * rng.uniform(0.6, 1.3, m)
    prm[:, 4] = P['smear']
    prm[:, 5] = P['thick'] * rng.uniform(0.7, 1.2, m)
    impasto = getattr(scene, 'impasto', None)     # optional per-pixel thickness: heavier paint in the lights
    if impasto is not None:
        prm[:, 5] *= impasto[ci, cj]
    prm[:, 6] = P['rough']
    prm[:, 7] = rng.integers(0, 1 << 20, m)
    prm[:, 8] = rng.integers(0, 1 << 20, m)
    prm[:, 9] = P.get('cap', 0.35)
    prm[:, 10] = 1.0
    prm[:, 11] = TOOLS[P.get('tool', 'brush')]
    prm[:, 12] = P.get('pool', 0.0)
    prm[:, 13] = P.get('feather', 0.0)
    if mask is not None:
        prm[:, 1] *= mask[ci, cj]

    order = np.lexsort((j, rank[pid]))      # random patch order, strokes of a patch together
    pts = np.stack([px, py], 2)[order].reshape(-1, 2)
    npt = px.shape[1]
    offs = np.arange(0, npt * m + 1, npt, dtype=np.int64)
    return pts, offs, prm[order], cols[order].astype(np.float32), np.stack([ci, cj], 1)[order]


def plan_lines(lines, rng):
    """Explicit line work (contours, branches) split into overlapping or broken strokes."""
    polys, prm, cols = [], [], []
    for ln in lines:
        P = np.asarray(ln['pts'], np.float64)
        if len(P) < 2:
            continue
        seg = np.hypot(*(P[1:] - P[:-1]).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        hw0, hw1 = ln['hw'] if isinstance(ln['hw'], tuple) else (ln['hw'], ln['hw'])
        spans = []
        if 'gap' in ln:              # broken line: separate touches with gaps
            s = rng.uniform(0.0, ln['gap'][1])
            while s < total:
                e = min(total, s + rng.uniform(*ln['piece']))
                spans.append((s, e))
                s = e + rng.uniform(*ln['gap'])
        else:                        # continuous line: one tapered stroke
            spans.append((0.0, total))
        for s, e in spans:
            ss = np.linspace(s, e, max(2, int((e - s) / 3.0) + 1))
            q = np.stack([np.interp(ss, cum, P[:, 0]), np.interp(ss, cum, P[:, 1])], 1)
            wob = ln.get('wobble', 0.8)
            if wob > 0 and len(q) > 2:
                # a hand-drawn line drifts a little off the true contour
                tx, ty = np.gradient(q[:, 0]), np.gradient(q[:, 1])
                nrm = np.hypot(tx, ty) + 1e-9
                off = wob * np.sin(ss / rng.uniform(15, 40) + rng.uniform(0, 6.3)) + rng.normal(0, 0.4 * wob)
                q = q + np.stack([-ty / nrm, tx / nrm], 1) * off[:, None]
            polys.append(q)
            h0 = hw0 + (hw1 - hw0) * s / total
            h1 = hw0 + (hw1 - hw0) * e / total
            prm.append([h0 * rng.uniform(0.9, 1.1), ln['alpha'] * rng.uniform(0.85, 1.0), ln.get('load', 1.0) * rng.uniform(0.85, 1.0),
                        ln.get('dry', 0.4), ln.get('smear', 0.1), ln.get('thick', 0.5),
                        ln.get('rough', 0.4), rng.integers(1 << 20), rng.integers(1 << 20), ln.get('cap', 0.6),
                        h1 / max(h0, 1e-6), TOOLS[ln.get('tool', 'brush')], 0.0, ln.get('feather', 0.0)])
            cols.append(np.asarray(ln['color'], np.float32) + rng.normal(0.0, 0.02, 3))
    if not polys:
        return None
    offs = np.zeros(len(polys) + 1, np.int64)
    offs[1:] = np.cumsum([len(p) for p in polys])
    return (np.concatenate(polys), offs, np.asarray(prm, np.float64),
            np.asarray(cols, np.float32))


def _bled(scene, pad):
    """The scene grown by `pad` pixels on every side (edge values repeated), so strokes can run
    off the edge the way paint wraps round a stretcher, instead of thinning out at the border."""
    out = Scene(scene.W + 2 * pad, scene.H + 2 * pad)
    grow = lambda a: np.ascontiguousarray(np.pad(a, ((pad, pad), (pad, pad)) + ((0, 0),) * (a.ndim - 2), mode='edge'))
    for name in ('color', 'region', 'angle', 'detail', 'leaf', 'leaf_color', 'leaf_rough', 'tool', 'emit'):
        setattr(out, name, grow(getattr(scene, name)))
    out.lines = [dict(ln, pts=np.asarray(ln['pts'], np.float64) + pad) for ln in scene.lines]
    if getattr(scene, 'impasto', None) is not None:
        out.impasto = np.pad(scene.impasto, pad, mode='edge')
    out.refs = {name: grow(a) for name, a in scene.refs.items()}
    out.masks = {name: grow(a) for name, a in scene.masks.items()}
    out.figures = {name: tuple(grow(a) for a in fig) for name, fig in scene.figures.items()}
    if scene.soft is not None:
        out.soft = grow(scene.soft)
    return out


def _tooth(surface, height, k, W):
    """What a dry brush or a knife catches on once there is paint down: the ground's own texture and,
    by `k`, the paint's relief (0 in its hollows, 1 on its ridges)."""
    proud = height - blur(height, 3.0 * W / 1024.0)
    return (surface * (1.0 - k) + np.clip(0.5 + 2.0 * proud, 0.0, 1.0) * k).astype(np.float32)


def paint(scene, style, seed=0, verbose=True):
    pad = int(round(style.get('bleed', 0.0) * scene.W))
    work = _bled(scene, pad) if pad > 0 else scene
    H, W = work.H, work.W
    rng = np.random.default_rng(seed + 991)
    palette = style.get('palette')      # paint in pigment: the canvas holds linear reflectance until the end
    canvas = np.empty((H, W, 3), np.float32)
    canvas[:] = style['ground'] if palette is None else to_linear(style['ground'])
    height = np.zeros((H, W), np.float32)
    if style.get('surface', 'canvas') == 'gesso':
        surface = gesso(H, W, rng)
    else:
        surface = canvas_weave(H, W, style['weave_period'] * max(1.0, scene.W / 1400.0), rng)
    keep = (1.0 - work.leaf).astype(np.float32)
    bristles = make_bristles(rng)
    streaks = make_streaks(rng) if palette is not None else np.zeros((1, 2), np.float32)
    wet = None                          # a glaze still wet enough to work: laid by 'glaze', worked by 'grain'
    for i, P in enumerate(style['passes']):
        t0 = time.time()
        if wet is not None and P['kind'] != 'grain':
            wet.dry(canvas, height)
            wet = None
        mask = work.masks[P['mask']] if 'mask' in P else None
        tooth = _tooth(surface, height, P['tooth'], scene.W) if P.get('tooth', 0.0) > 0 else surface
        loads = np.zeros((0, NPAINT))
        batch = None
        if P['kind'] == 'patches':
            hw = P['hw'] * scene.W
            src = work.refs[P['ref']] if 'ref' in P else work.color
            sigma = max(0.6, hw * P['blur'])
            ref = masked_blur(src, keep, sigma)
            if work.soft is not None and P.get('soft', 0.0) > 0:
                # lost edges: where the scene says so, the brush sees a vaguer picture
                ref = mix(ref, masked_blur(src, keep, sigma + P['soft'] * scene.W), work.soft)
            shown = canvas if palette is None else to_display(canvas).astype(np.float32)
            batch = plan_patches(shown, ref, work, hw, P, rng, mask)
            if batch is not None:
                at, batch = batch[4], batch[:4]
                if palette is not None and P.get('tool') != 'blend':
                    under = blur(canvas, max(1.0, 0.5 * hw))[at[:, 0], at[:, 1]]
                    loads = palette.load(batch[3], under, 0.6 * P['alpha'], rng, P.get('glaze', 0.25),
                                         P.get('cover', 0.0), P.get('streak', 0.0), P.get('streak_to'))
        elif P['kind'] == 'print':
            # a flat block print: the reference laid down whole, the ink catching less in the
            # ground's low spots (paper grain, the baren's rubbing)
            ref = masked_blur(work.refs[P['ref']] if 'ref' in P else work.color, keep, P.get('blur', 0.6))
            if palette is not None:
                ref = to_linear(ref).astype(np.float32)
            ink = P.get('alpha', 1.0) * keep * (1.0 - P.get('grain', 0.0) * (1.0 - surface))
            canvas += (ref - canvas) * ink[..., None]
            height += P.get('thick', 0.0) * ink
        elif P['kind'] == 'glaze':
            # one transparent paint rubbed over the whole picture and left wet for the 'grain' passes
            wet = graining.lay(palette.recipe(**P['paint']), work, P, scene.W, rng)
        elif P['kind'] == 'grain':
            graining.work(wet, work, P, scene.W, rng)
        else:
            batch = plan_lines([l for l in work.lines if l['group'] == P['group']], rng)
            if batch is not None and palette is not None:
                mid = batch[0][(batch[1][:-1] + batch[1][1:]) // 2]
                under = canvas[np.clip(mid[:, 1].astype(np.int64), 0, H - 1), np.clip(mid[:, 0].astype(np.int64), 0, W - 1)]
                loads = palette.load(batch[3], under, 0.75 * float(np.median(batch[2][:, 1])), rng, P.get('glaze', 0.0),
                                     P.get('cover', 0.0), P.get('streak', 0.0), P.get('streak_to'))
        count = 0
        if batch is not None:
            count = batch[1].size - 1
            cols = batch[3] if palette is None else to_linear(batch[3]).astype(np.float32)
            render_strokes(canvas, height, tooth, batch[0], batch[1], batch[2], cols, *bristles, loads, streaks)
        if verbose:
            print(f'  pass {i} {P["kind"]:8s} {count:6d} strokes  {time.time() - t0:5.1f}s', flush=True)
    if wet is not None:
        wet.dry(canvas, height)
    if pad > 0:
        crop = (slice(pad, H - pad), slice(pad, W - pad))
        canvas, height, surface = (np.ascontiguousarray(a[crop]) for a in (canvas, height, surface))
    if palette is not None:
        canvas = to_display(canvas).astype(np.float32)
    for mask, col in scene.overlays:
        canvas = mix(canvas, col, mask)
    return Painting(scene, style, canvas, height, surface)
