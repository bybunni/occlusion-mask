# occlusion-mask

Python utilities for modeling a pitch/roll stabilized sensor under an aircraft and checking whether the aircraft body occludes a point inside the sensor scan volume.

## What It Does

- Transforms world/NED points into a level-stabilized sensor frame
- Applies azimuth, elevation, and range scan-volume gating
- Applies a 5-point piecewise-linear body occlusion mask defined in the aircraft center plane
- Produces interactive 3D Plotly visualizations of the sensor frame, body frame, scan frustum, occlusion ribbon, and sample points

## Conventions

- World frame: NED
- Body and sensor axes: `x` forward, `y` right, `z` down
- Sensor yaw is relative to the aircraft heading after pitch/roll stabilization
- Positive elevation is up, even though `z` is down
- The occlusion mask defaults to `z <= boundary(x)` in body coordinates, which is the usual "aircraft is above the boundary" interpretation for an under-mounted sensor

## Quick Start

```bash
uv sync --group dev
uv run pytest
uv run python examples/visualize_occlusion.py
```

The example script launches a local Dash app with interactive platform `yaw`, `pitch`, and `roll` sliders.

## Visual Demo

The repository includes:

- `examples/visualize_occlusion.py` for a quick Plotly scene
- `notebooks/occlusion_demo.ipynb` for a guided visual walkthrough
