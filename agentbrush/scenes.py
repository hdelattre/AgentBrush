"""Scene data and drawing operations for the painting engine."""
import numpy as np

from .noise import C, mix

LEAF = -1                      # region id of gilded areas: brushes never start there and stop at their edge
GOLD = C(1.0, 0.80, 0.45)      # leaf base colours (a metal's albedo is its reflectance)
SILVER = C(0.93, 0.93, 0.95)


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.color = np.zeros((H, W, 3), np.float32)
        self.region = np.zeros((H, W), np.int32)
        self.angle = np.zeros((H, W), np.float32)
        self.detail = np.zeros((H, W), np.float32)
        self.lines = []
        # metal leaf laid on the ground: coverage, metal colour, roughness, tooled relief (punches < 0)
        self.leaf = np.zeros((H, W), np.float32)
        self.leaf_color = np.zeros((H, W, 3), np.float32)
        self.leaf_rough = np.full((H, W), 0.2, np.float32)
        self.tool = np.zeros((H, W), np.float32)
        self.emit = np.zeros((H, W), np.float32)   # self-illumination mask (lit windows)
        self.overlays = []             # (mask, colour) inked over the finished paint, e.g. lettering
        # for painters who work in layers: extra references and masks a pass can name, and where the
        # picture loses its edges (0 crisp, 1 lost), so strokes there run across them
        self.refs = {}
        self.masks = {}
        self.soft = None
        # for graining: the figures a wet glaze is combed along, name -> (u, v), v constant along each line
        self.figures = {}
        self.marks = {}     # named drawing instructions for specialist passes
        self.tilt = None    # optional (H, W, 2) slopes added to the baked relief

    def put(self, mask, color, region=None, angle=None, detail=None):
        m = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
        self.color = np.ascontiguousarray(mix(self.color, color, m))
        self.leaf *= 1.0 - m           # paint laid over leaf hides it
        self.tool *= 1.0 - m
        self.emit *= 1.0 - m
        hard = m > 0.5
        for arr, val in ((self.region, region), (self.angle, angle), (self.detail, detail)):
            if val is not None:
                arr[hard] = val[hard] if np.ndim(val) == 2 else val

    def put_local(self, shape, color, region=None, angle=None, detail=None, alpha=1.0):
        """Composite a (window, mask[, angle]) shape from `raster` into the scene."""
        if shape is None:
            return
        win, m = shape[0], shape[1] * alpha
        sub = self.color[win]
        sub += (np.asarray(color, np.float32) - sub) * m[..., None]
        self.leaf[win] *= 1.0 - m
        self.tool[win] *= 1.0 - m
        self.emit[win] *= 1.0 - m
        hard = m > 0.5
        for arr, val in ((self.region, region), (self.angle, angle), (self.detail, detail)):
            if val is not None:
                arr[win][hard] = val[hard] if np.ndim(val) == 2 else val

    def gild(self, mask, metal=GOLD, rough=0.2):
        """Lay metal leaf. Brushes skip it; paint put afterwards covers it."""
        m = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
        self.leaf = np.maximum(self.leaf, m)
        self.leaf_color = np.ascontiguousarray(mix(self.leaf_color, metal, m))
        self.leaf_rough = mix(self.leaf_rough, rough, m)
        self.tool *= 1.0 - m           # fresh leaf, fresh surface
        self.region[m > 0.5] = LEAF

    def gild_local(self, shape, metal=GOLD, rough=0.2):
        if shape is None:
            return
        win, m = shape[0], shape[1]
        np.maximum(self.leaf[win], m, out=self.leaf[win])
        sub = self.leaf_color[win]
        sub += (np.asarray(metal, np.float32) - sub) * m[..., None]
        r = self.leaf_rough[win]
        r += (rough - r) * m
        self.tool[win] *= 1.0 - m
        self.region[win][m > 0.5] = LEAF

    def glow_local(self, shape):
        """Mark a raster shape as self-lit (baked into the _selfillum map)."""
        if shape is not None:
            np.maximum(self.emit[shape[0]], shape[1], out=self.emit[shape[0]])

    def overlay(self, mask, color):
        self.overlays.append((np.asarray(mask, np.float32), np.asarray(color, np.float32)))

    def line(self, pts, group, **kw):
        self.lines.append(dict(pts=np.asarray(pts, np.float64), group=group, **kw))
