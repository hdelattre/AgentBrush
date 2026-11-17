"""Common arguments and render workflow for standalone and project scene libraries."""
import argparse
import importlib
import pkgutil
import time
from pathlib import Path

from . import export, styles
from .painter import paint


def parser(default_out='output'):
    ap = argparse.ArgumentParser(description='Render a Python scene with a painting style.')
    ap.add_argument('scene', help='scene builder function')
    ap.add_argument('style', help='style name, e.g. TEMPERA')
    ap.add_argument('--width', type=int, help="override the scene builder's default width")
    ap.add_argument('--seed', type=int)
    ap.add_argument('--pbr', action='store_true', help='write the full texture maps')
    ap.add_argument('--out', type=Path, default=Path(default_out))
    ap.add_argument('--crop', type=float, nargs=2, default=(0.62, 0.55), metavar=('U', 'V'))
    return ap


def lookup(name, directory):
    """The calling script's directory is also the import root for its scene modules."""
    for info in pkgutil.iter_modules([str(directory)]):
        if info.name != 'run' and not info.ispkg:
            module = importlib.import_module(info.name)
            if hasattr(module, name):
                return getattr(module, name)
    raise SystemExit(f'no scene or style named {name!r} in {directory}')


def render(args, directory):
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    builder = lookup(args.scene, directory)
    style = getattr(styles, args.style, None)
    if style is None:
        style = lookup(args.style, directory)
    kwargs = {k: v for k, v in (('W', args.width), ('seed', args.seed)) if v is not None}
    scene = builder(**kwargs)
    export.save_png(scene.color, args.out / f'{args.scene}_reference.png')
    painting = paint(scene, style, args.seed or 0)
    maps = export.bake(painting)
    preview = export.preview(painting, maps)
    base = str(args.out / f'{args.scene}_{args.style.lower()}')
    export.save_png(preview, base + '.png')
    cx, cy = int(args.crop[0] * scene.W), int(args.crop[1] * scene.H)
    export.save_png(preview[max(0, cy - 300):cy + 300, max(0, cx - 400):cx + 400], base + '_detail.png')
    if args.pbr:
        export.save_maps(maps, base)
    print(f'done in {time.perf_counter() - start:.1f}s -> {base}.png')
    return maps
