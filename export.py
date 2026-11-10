"""Bake painting textures and render a lit preview."""
import numpy as np
from PIL import Image

from noise import C, fbm, smoothstep, mix, voronoi
from raster import blur

BOLE = C(.55, .22, .15)       # red clay under gilding: shows where the leaf has rubbed through


def craquelure(H, W, cell, seed=12):
    """Age cracks: edges of a warped Voronoi network that fade in and out across the surface."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wx = xx + 0.25 * cell * fbm(H, W, 3 * cell, rng, 3)
    wy = yy + 0.25 * cell * fbm(H, W, 3 * cell, rng, 3)
    _, _, d1, d2 = voronoi((wx / cell).ravel(), (wy / (0.8 * cell)).ravel(), 77)
    crack = smoothstep(0.045, 0.0, (d2 - d1).reshape(H, W))
    return crack * smoothstep(-0.6, 0.4, fbm(H, W, 6 * cell, rng, 3))


def bake(p):
    """Albedo, normals, roughness, metalness and AO for a Painting."""
    sc, fx = p.scene, p.style['finish']
    H, W = sc.H, sc.W
    rng = np.random.default_rng(4242)
    leaf = sc.leaf
    wear = np.zeros((H, W), np.float32)
    if fx.get('wear', 0.0) > 0 and leaf.any():
        # gilding rubs through where hands hold the panel, and in the odd patch elsewhere
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        edge = np.minimum(np.minimum(xx, W - 1 - xx), np.minimum(yy, H - 1 - yy))
        rub = 0.6 * fbm(H, W, 30.0 * W / 1024, rng, 4) + 2.6 * smoothstep(0.05 * W, 0.0, edge) + fx['wear']
        wear = smoothstep(2.3, 2.8, rub) * leaf
    metal = leaf * (1.0 - wear)
    albedo = mix(p.canvas, BOLE, leaf * wear)
    albedo = mix(albedo, sc.leaf_color, metal)
    crack = np.zeros((H, W), np.float32)
    if fx.get('crackle', 0.0) > 0:
        crack = craquelure(H, W, fx.get('crackle_cell', 30.0) * W / 1800.0) * fx['crackle']
        albedo = albedo * (1.0 - crack)[..., None]
    height = (p.height * fx.get('relief', 1.0) + p.surface * fx.get('weave_relief', 0.3)
              + sc.tool * fx.get('tool_relief', 0.5) - 0.8 * crack)
    gy, gx = np.gradient(blur(height, 0.7))
    inv = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    normal = np.stack([-gx * inv, -gy * inv, inv], -1)       # x right, y down, z out of the panel
    rough = mix(fx.get('paint_rough', 0.70) + 0.04 * fbm(H, W, 8.0, rng, 3), sc.leaf_rough, metal)   # paint_rough: lacquered finishes
    rough = mix(rough, 0.8, leaf * wear)
    rough = np.clip(rough + 0.15 * crack, 0.0, 1.0)
    cavity = blur(height, 3.0 * W / 1024) - height
    ao = np.clip(1.0 - 2.5 * np.clip(cavity, 0.0, None), 0.5, 1.0)
    return dict(albedo=np.clip(albedo, 0, 1), normal=normal, rough=rough, metal=metal, ao=ao, emit=sc.emit)


def preview(p, maps):
    """How the panel reads under a raking gallery light: paint lit diffusely, leaf as a mirror."""
    fx = p.style['finish']
    H, W = p.scene.H, p.scene.W
    n = maps['normal']
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    l = np.asarray(fx.get('light', (-0.5, -0.7, 1.0)), np.float32)
    l = l / np.linalg.norm(l)
    diff = (nx * l[0] + ny * l[1] + nz * l[2]) / l[2]
    hv = l + np.array([0.0, 0.0, 1.0], np.float32)
    hv = hv / np.linalg.norm(hv)
    ndh = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0.0, 1.0)
    sheen = np.clip(fx.get('spec', 0.05) * (ndh ** 80 - hv[2] ** 80), 0.0, None)
    out = maps['albedo'] * np.clip(diff, 0.4, 1.6)[..., None] + sheen[..., None]
    metal = maps['metal']
    if metal.any():
        # Leaf is a mirror, so it shows the room. Seen from a camera at a finite distance the
        # reflection direction sweeps across the panel, which gives flat gold its broad bands of
        # light; tooled dents tip the normal and pick up sparkle from the softbox and lamp.
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        vx, vy, vz = -(xx - 0.5 * W), -(yy - 0.42 * H), np.full_like(xx, 2.4 * H)
        inv = 1.0 / np.sqrt(vx * vx + vy * vy + vz * vz)
        vx, vy, vz = vx * inv, vy * inv, vz * inv
        ndv = nx * vx + ny * vy + nz * vz
        rx, ry, rz = 2 * ndv * nx - vx, 2 * ndv * ny - vy, 2 * ndv * nz - vz
        gloss = 1.0 - maps['rough']
        broad = smoothstep(-0.6, 0.6, -(0.55 * rx + 0.83 * ry))              # the lit half of the room
        box = np.exp(-((rx + 0.10) ** 2 + (ry + 0.16) ** 2) / 0.07 ** 2)     # a softbox up and left
        lobe = np.clip(rx * l[0] + ry * l[1] + rz * l[2], 0.0, 1.0) ** (6.0 + 150.0 * gloss * gloss)
        shine = 0.42 + 0.55 * broad + 0.8 * box * gloss ** 2 + (0.3 + 1.0 * gloss) * lobe
        out = mix(out, maps['albedo'] * shine[..., None], metal)
    if maps['emit'].any():      # lit windows bloom a little, as they would at night in game
        lit = maps['albedo'] * maps['emit'][..., None]
        out = out + 0.15 * lit + 0.35 * blur(lit, 6.0 * W / 1024)
    out = out * np.asarray(fx.get('varnish', (1.0, 1.0, 1.0)), np.float32)
    if fx.get('vignette', 0.0):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) / 0.5
        out = out * (1.0 - fx['vignette'] * r2)[..., None]
    return np.clip(out, 0.0, 1.0)


def _save(img, path):
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def _normal_rgb(n):
    return np.stack([0.5 + 0.5 * n[..., 0], 0.5 - 0.5 * n[..., 1], 0.5 + 0.5 * n[..., 2]], -1)


def save_maps(maps, base):
    _save(maps['albedo'], base + '_color.png')
    _save(_normal_rgb(maps['normal']), base + '_normal.png')
    _save(maps['rough'], base + '_rough.png')
    _save(maps['metal'], base + '_metal.png')
    _save(maps['ao'], base + '_ao.png')
    if maps['emit'].any():
        _save(maps['emit'], base + '_selfillum.png')
