"""Painter profiles. A style is data: ground colour, brush passes coarse to fine, finish.

Pass fields: hw = brush half-width as a fraction of canvas width; blur = reference
blur relative to hw; thr = colour error that triggers repainting (0 = everywhere);
k = strokes per patch; len = stroke length in half-widths; temp/val = per-patch
warm-cool and light-dark modulation; dry = how fast the brush runs out of paint.
"""

CEZANNE = dict(
    ground=(0.91, 0.88, 0.80),
    weave_period=3.0,
    passes=[
        # thin lay-in: establishes colour, a little canvas still breathes through
        dict(kind='patches', hw=0.0135, blur=1.0, thr=0.0, k=(1, 3), len=(2.6, 4.2), coverage=1.0,
             alpha=0.75, load=0.9, dry=0.5, smear=0.08, thick=0.2, rough=0.8, ang_jit=0.22,
             temp=0.02, val=0.02, jit=0.01, overshoot=0.8),
        # the constructive stroke: patches of parallel diagonal hatching
        dict(kind='patches', hw=0.0075, blur=0.8, thr=0.03, k=(3, 6), len=(3.4, 6.0), coverage=1.1,
             alpha=0.92, dry=0.35, smear=0.18, thick=0.9, rough=0.6, ang_jit=0.12,
             temp=0.045, val=0.03, jit=0.008),
        dict(kind='patches', hw=0.0048, blur=0.7, thr=0.055, k=(2, 5), len=(3.0, 5.5), coverage=0.9,
             alpha=0.92, dry=0.3, smear=0.15, thick=0.8, rough=0.55, ang_jit=0.12,
             temp=0.035, val=0.025, jit=0.008),
        dict(kind='lines', group='structure'),
        dict(kind='patches', hw=0.003, blur=0.6, thr=0.07, k=(2, 4), len=(2.8, 5.0), coverage=0.9,
             alpha=0.95, dry=0.2, smear=0.08, thick=0.7, rough=0.5, ang_jit=0.12,
             temp=0.02, val=0.015, jit=0.008),
        dict(kind='patches', hw=0.0018, blur=0.5, thr=0.085, k=(1, 2), len=(2.5, 4.5), coverage=0.8,
             alpha=0.97, dry=0.15, smear=0.05, thick=0.6, rough=0.45, ang_jit=0.1,
             temp=0.01, val=0.008, jit=0.006),
        # broken blue contours drawn last
        dict(kind='lines', group='contour'),
    ],
    finish=dict(relief=1.0, weave_relief=0.3, light=(-0.5, -0.7, 1.0), spec=0.05,
                varnish=(1.0, 0.985, 0.95), vignette=0.05),
)

# Egg tempera on a gesso panel, the technique of gold-ground altarpieces and the
# Visconti tarocchi: flat colour laid in, then modelled with fine parallel hatching
# (tratteggio) that follows the form. Tempera dries at once, so strokes never blend
# (no smear) and sit thin (little relief). Gold is laid as leaf, not painted.
TEMPERA = dict(
    ground=(0.94, 0.92, 0.86),
    surface='gesso',
    passes=[
        dict(kind='patches', hw=0.011, blur=1.0, thr=0.0, k=(1, 2), len=(2.5, 4.0), coverage=1.35,
             alpha=0.92, load=1.0, dry=0.05, smear=0.0, thick=0.05, rough=0.25, ang_jit=0.1,
             temp=0.0, val=0.004, jit=0.003, overshoot=0.3),
        dict(kind='patches', hw=0.0055, blur=0.8, thr=0.02, k=(3, 6), len=(3.0, 5.0), coverage=1.1,
             alpha=0.72, dry=0.05, smear=0.0, thick=0.06, rough=0.3, ang_jit=0.06, pitch=(0.95, 1.3),
             temp=0.004, val=0.018, jit=0.004, edge_follow=0.012, overshoot=0.3),
        dict(kind='patches', hw=0.0032, blur=0.6, thr=0.03, k=(3, 7), len=(3.0, 5.0), coverage=1.0,
             alpha=0.72, dry=0.05, smear=0.0, thick=0.06, rough=0.3, ang_jit=0.06, pitch=(0.95, 1.3),
             temp=0.004, val=0.015, jit=0.004, edge_follow=0.012, overshoot=0.3),
        dict(kind='patches', hw=0.0019, blur=0.5, thr=0.04, k=(2, 5), len=(3.0, 5.0), coverage=0.9,
             alpha=0.8, dry=0.05, smear=0.0, thick=0.05, rough=0.25, ang_jit=0.06, pitch=(0.95, 1.3),
             temp=0.003, val=0.01, jit=0.003, edge_follow=0.015, overshoot=0.25),
        dict(kind='patches', hw=0.0011, blur=0.4, thr=0.055, k=(1, 3), len=(2.5, 4.5), coverage=0.8,
             alpha=0.9, dry=0.04, smear=0.0, thick=0.04, rough=0.2, ang_jit=0.05,
             temp=0.0, val=0.0, jit=0.003, edge_follow=0.02, overshoot=0.2),
        dict(kind='lines', group='contour'),
    ],
    finish=dict(relief=0.35, weave_relief=0.05, tool_relief=0.5, light=(-0.5, -0.7, 1.0), spec=0.03,
                varnish=(1.0, 0.975, 0.93), vignette=0.0, crackle=0.06, crackle_cell=24.0, wear=0.4),
)

# Oil on an oak panel: thin, smooth, blended paint in small brushes, strokes that
# follow the form, near-zero colour modulation, crisp silhouettes, and an old
# varnish with craquelure.
BRUEGEL = dict(
    ground=(0.90, 0.87, 0.78),
    weave_period=2.2,
    passes=[
        dict(kind='patches', hw=0.011, blur=1.0, thr=0.0, k=(1, 2), len=(2.5, 4.0), coverage=1.25,
             alpha=0.9, load=1.0, dry=0.15, smear=0.3, thick=0.12, rough=0.5, ang_jit=0.12,
             temp=0.006, val=0.006, jit=0.004, overshoot=0.8),
        dict(kind='patches', hw=0.0055, blur=0.8, thr=0.025, k=(1, 3), len=(2.5, 4.5), coverage=1.15,
             alpha=0.95, dry=0.12, smear=0.3, thick=0.15, rough=0.4, ang_jit=0.1,
             temp=0.008, val=0.008, jit=0.004),
        dict(kind='patches', hw=0.0032, blur=0.6, thr=0.035, k=(1, 3), len=(2.5, 4.0), coverage=1.0,
             alpha=0.95, dry=0.1, smear=0.2, thick=0.15, rough=0.35, ang_jit=0.1,
             temp=0.006, val=0.006, jit=0.004),
        dict(kind='patches', hw=0.0019, blur=0.5, thr=0.045, k=(1, 2), len=(2.2, 3.8), coverage=0.9,
             alpha=0.97, dry=0.08, smear=0.12, thick=0.12, rough=0.3, ang_jit=0.1,
             temp=0.004, val=0.004, jit=0.003),
        dict(kind='patches', hw=0.0011, blur=0.4, thr=0.06, k=(1, 2), len=(2.0, 3.5), coverage=0.8,
             alpha=0.98, dry=0.05, smear=0.05, thick=0.1, rough=0.25, ang_jit=0.1,
             temp=0.0, val=0.0, jit=0.003),
        # trees, spears, legs, birds: line work laid over the paint
        dict(kind='lines', group='contour'),
    ],
    finish=dict(relief=0.6, weave_relief=0.1, light=(-0.5, -0.7, 1.0), spec=0.02,
                varnish=(1.0, 0.97, 0.89), vignette=0.14, crackle=0.09),
)
