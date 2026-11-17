"""Blur and anti-aliased shape rasterising for building scene references.

Shape helpers return (window, mask): a pair of slices into the full image and
the coverage mask for that window, so callers composite locally."""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, uniform_filter


def blur(img, sigma):
    """Gaussian blur over the first two axes; large radii use three box passes (O(n))."""
    img = np.asarray(img, np.float32)
    spatial = (sigma, sigma) + (0,) * (img.ndim - 2)
    if sigma < 3.0:
        return gaussian_filter(img, spatial)
    wl = int(np.floor(np.sqrt(4.0 * sigma * sigma + 1.0)))
    wl -= (wl % 2 == 0)
    mcount = int(round((12.0 * sigma * sigma - 3 * wl * wl - 12 * wl - 9) / (-4.0 * wl - 4.0)))
    out = img
    for i in range(3):
        s = wl if i < mcount else wl + 2
        out = uniform_filter(out, size=(s, s) + (1,) * (img.ndim - 2), mode='reflect')
    return out


def _window(x0, y0, x1, y1, H, W):
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(W, int(np.ceil(x1)) + 1), min(H, int(np.ceil(y1)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def ellipse(cx, cy, rx, ry, H, W, ang=0.0):
    r = max(rx, ry) + 2.0
    box = _window(cx - r, cy - r, cx + r, cy + r, H, W)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx + 0.5 - cx, yy + 0.5 - cy
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    ex = (dx * c + dy * s) / rx
    ey = (-dx * s + dy * c) / ry
    q = np.sqrt(ex * ex + ey * ey)
    m = np.clip((1.0 - q) * min(rx, ry) + 0.5, 0.0, 1.0)
    return (slice(y0, y1), slice(x0, x1)), m


def polygon(poly, H, W, ss=4):
    P = np.asarray(poly, np.float64)
    box = _window(P[:, 0].min() - 1, P[:, 1].min() - 1, P[:, 0].max() + 1, P[:, 1].max() + 1, H, W)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    im = Image.new('L', ((x1 - x0) * ss, (y1 - y0) * ss), 0)
    ImageDraw.Draw(im).polygon([((x - x0) * ss, (y - y0) * ss) for x, y in P], fill=255)
    m = np.asarray(im.reduce(ss), np.float32) / 255.0
    return (slice(y0, y1), slice(x0, x1)), m


def polyline(P, half_widths, H, W):
    """Thick tapered polyline: returns window, coverage, and the local segment angle."""
    P = np.asarray(P, np.float64)
    hw = np.asarray(half_widths, np.float64)
    pad = hw.max() + 2.0
    box = _window(P[:, 0].min() - pad, P[:, 1].min() - pad, P[:, 0].max() + pad, P[:, 1].max() + pad, H, W)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    px, py = xx + 0.5, yy + 0.5
    best = np.full(px.shape, 1e9)
    ang = np.zeros(px.shape, np.float32)
    for i in range(len(P) - 1):
        ax, ay = P[i]
        ex, ey = P[i + 1] - P[i]
        ll = ex * ex + ey * ey
        if ll < 1e-12:
            continue
        u = np.clip(((px - ax) * ex + (py - ay) * ey) / ll, 0.0, 1.0)
        d = np.hypot(ax + u * ex - px, ay + u * ey - py) - (hw[i] + (hw[i + 1] - hw[i]) * u)
        upd = d < best
        best = np.where(upd, d, best)
        ang = np.where(upd, np.float32(np.arctan2(ey, ex)), ang)
    m = np.clip(0.5 - best, 0.0, 1.0).astype(np.float32)
    return (slice(y0, y1), slice(x0, x1)), m, ang
