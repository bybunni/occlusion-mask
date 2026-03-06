from __future__ import annotations

import numpy as np

from occlusion_mask import OcclusionProfile, PlatformState, ScanVolume, make_visibility_figure


def sample_points() -> np.ndarray:
    x_values = np.linspace(0.5, 8.0, 12)
    y_values = np.linspace(-3.0, 3.0, 9)
    z_values = np.linspace(-1.5, 2.5, 10)
    xx, yy, zz = np.meshgrid(x_values, y_values, z_values, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def main() -> None:
    state = PlatformState.from_degrees(pitch_deg=-10.0, roll_deg=5.0, sensor_yaw_deg=0.0)
    scan = ScanVolume.from_degrees(
        az_min_deg=-35.0,
        az_max_deg=35.0,
        el_min_deg=-25.0,
        el_max_deg=30.0,
        range_min_m=0.5,
        range_max_m=9.0,
    )
    profile = OcclusionProfile(
        points_body=np.array(
            [
                (0.0, 0.0, -0.8),
                (1.0, 0.0, -0.7),
                (2.0, 0.0, -0.5),
                (3.0, 0.0, -0.25),
                (4.0, 0.0, 0.0),
            ],
            dtype=float,
        ),
        occluded_if="z_le_boundary",
    )

    figure = make_visibility_figure(sample_points(), state, scan, profile)
    figure.show()


if __name__ == "__main__":
    main()
