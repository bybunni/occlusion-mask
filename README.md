# occlusion-mask

Python utilities for modeling a pitch/roll stabilized sensor under an aircraft and checking whether the aircraft body occludes a point inside the sensor scan volume.

## What It Does

- Transforms world/NED points into a level-stabilized sensor frame
- Applies azimuth, elevation, and range scan-volume gating
- Applies a 5-point body-attached occlusion mask authored as sensor `(azimuth, elevation, range)` boundary points
- Produces interactive 3D Plotly visualizations of the sensor frame, body frame, scan frustum, occlusion ribbon, and sample points

## Conventions

- World frame: NED
- Body and sensor axes: `x` forward, `y` right, `z` down
- Sensor yaw is relative to the aircraft heading after pitch/roll stabilization
- Positive elevation is up, even though `z` is down
- The occlusion mask is authored as five sensor-style `(azimuth, elevation, range)` points and internally converted into body-attached 3D points
- The default occlusion rule is `elevation >= boundary_elevation(azimuth)` and `range >= boundary_range(azimuth)`

## Quick Start

```bash
uv sync --group dev
uv run pytest
uv run python examples/visualize_occlusion.py
```

The example script launches a local Dash app with interactive platform `yaw`, `pitch`, and `roll` sliders plus editable `(azimuth, elevation, range)` inputs for the five occlusion boundary points.

## Visual Demo

The repository includes:

- `examples/visualize_occlusion.py` for a quick Plotly scene
- `notebooks/occlusion_demo.ipynb` for a guided visual walkthrough
