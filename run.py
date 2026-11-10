"""Render a Python scene builder with a painting style and export PNGs."""
import argparse
import importlib
import os
import pkgutil
import time

import numpy as np
from PIL import Image

import export
from painter import paint


def save(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def lookup(name):
    """Find a scene builder or style by name in any module here, so new work can live in new files."""
    here = os.path.dirname(os.path.abspath(__file__))
    for info in pkgutil.iter_modules([here]):
        if info.name != 'run':
            mod = importlib.import_module(info.name)
            if hasattr(mod, name):
                return getattr(mod, name)
    raise SystemExit(f'no scene or style named {name!r} in {here}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scene', help='scene builder function in a Python file beside run.py')
    ap.add_argument('style', help='style name, e.g. CEZANNE, TEMPERA, or BRUEGEL')
    ap.add_argument('--width', type=int, help='defaults to the scene builder\'s own size')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--pbr', action='store_true', help='also write game texture maps')
    ap.add_argument('--out', default='output', help='output directory (default: ./output)')
    ap.add_argument('--crop', type=float, nargs=2, default=(0.62, 0.55), help='centre of a 1:1 detail crop (u v)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t = time.time()
    builder = lookup(a.scene)
    kw = {k: v for k, v in (('W', a.width), ('seed', a.seed)) if v is not None}
    sc = builder(**kw)
    save(sc.color, os.path.join(a.out, f'{a.scene}_reference.png'))
    print(f'scene built in {time.time() - t:.1f}s')

    p = paint(sc, lookup(a.style), a.seed or 0)
    maps = export.bake(p)
    img = export.preview(p, maps)
    name = os.path.join(a.out, f'{a.scene}_{a.style.lower()}')
    save(img, name + '.png')
    cx, cy = int(a.crop[0] * sc.W), int(a.crop[1] * sc.H)
    save(img[max(0, cy - 300):cy + 300, max(0, cx - 400):cx + 400], name + '_detail.png')
    if a.pbr:
        export.save_maps(maps, name)
    print(f'done in {time.time() - t:.1f}s -> {name}.png')
