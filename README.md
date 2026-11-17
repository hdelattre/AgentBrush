# Painting Tools

A procedural painting renderer for coding agents. An agent can write a scene in Python, render it from the command line, inspect the PNG, and revise the scene.

The renderer uses the scene’s colors, region boundaries, and brush directions to build up strokes on a textured ground. It records the paint thickness and lights the result. The three included style presets are `CEZANNE`, `TEMPERA`, and `BRUEGEL`.

## First render

Install the dependencies from the repository directory.

```sh
python -m pip install -r requirements.txt
```

Create `my_scene.py` beside `run.py`. This swatch uses two regions with different brush directions. Colors are RGB values from 0 to 1, and angles are in radians.

```python
import numpy as np
from agentbrush import Scene


def swatch(W=512):
    scene = Scene(W, W)
    scene.color[:] = (0.18, 0.31, 0.37)
    y, x = np.ogrid[:W, :W]
    disk = (x - W / 2) ** 2 + (y - W / 2) ** 2 < (W / 3) ** 2
    scene.put(disk, (0.88, 0.52, 0.23), region=1, angle=np.pi / 4)
    return scene
```

Run the builder by its function name.

```sh
python run.py swatch CEZANNE --pbr
```

Open `output/swatch_cezanne.png` for the lit painting. The renderer also saves the flat reference and a full-resolution detail crop. With `--pbr` it writes albedo, normal, roughness, metalness, and ambient occlusion maps, plus an emission mask for scenes with lit regions.

Use `--out DIRECTORY` to choose where to write the files. The optional `--width` and `--seed` flags pass `W` and `seed` keyword arguments to your builder, so include those parameters if you use the corresponding flags. The seed also controls stroke placement.

## Using AgentBrush from another project

Put the checkout on your Python import path. Your scene files and custom styles can live in your own project. To render a scene in memory, call `agentbrush.painter.paint(scene, style, seed)`, then pass the painting to `agentbrush.export.bake` for its texture maps and `agentbrush.export.preview` for a lit view.

The package also includes faience stains on fired glaze, kintsugi joins, marbling on laid paper, weathered paint layers, and field-line drawings. Each technique module describes its scene inputs.
