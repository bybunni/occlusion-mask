"""Plotly helpers for visualizing the sensor frame, frustum, and occlusion mask."""

from __future__ import annotations

from math import degrees

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from .geometry import (
    OcclusionProfile,
    PlatformState,
    ScanVolume,
    evaluate_visibility,
    ray_from_sensor_angles,
)

Vector = NDArray[np.float64]


def _frame_axis_traces(rotation_sensor_from_frame: Vector, *, label: str, length: float) -> list[go.Scatter3d]:
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    axis_labels = ("x", "y", "z")
    traces: list[go.Scatter3d] = []
    for index, basis in enumerate(np.eye(3)):
        tip = rotation_sensor_from_frame @ (basis * length)
        traces.append(
            go.Scatter3d(
                x=[0.0, float(tip[0])],
                y=[0.0, float(tip[1])],
                z=[0.0, float(tip[2])],
                mode="lines",
                line={"color": colors[index], "width": 7},
                name=f"{label} {axis_labels[index]}",
                hovertemplate=f"{label} {axis_labels[index]}<extra></extra>",
                showlegend=True,
            )
        )
    return traces


def _frustum_wireframe_trace(scan_volume: ScanVolume) -> go.Scatter3d:
    def corners(range_m: float) -> list[Vector]:
        return [
            ray_from_sensor_angles(scan_volume.az_min_rad, scan_volume.el_min_rad, range_m),
            ray_from_sensor_angles(scan_volume.az_max_rad, scan_volume.el_min_rad, range_m),
            ray_from_sensor_angles(scan_volume.az_max_rad, scan_volume.el_max_rad, range_m),
            ray_from_sensor_angles(scan_volume.az_min_rad, scan_volume.el_max_rad, range_m),
        ]

    near_corners = corners(scan_volume.range_min_m)
    far_corners = corners(scan_volume.range_max_m)

    segments: list[Vector | None] = []
    for ring in (near_corners, far_corners):
        for start, end in zip(ring, ring[1:] + ring[:1]):
            segments.extend([start, end, None])
    for start, end in zip(near_corners, far_corners):
        segments.extend([start, end, None])

    x_values = [None if point is None else float(point[0]) for point in segments]
    y_values = [None if point is None else float(point[1]) for point in segments]
    z_values = [None if point is None else float(point[2]) for point in segments]

    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="lines",
        line={"color": "#4c78a8", "width": 4},
        name="scan volume",
        hoverinfo="skip",
    )


def _occlusion_ribbon_mesh(profile: OcclusionProfile, state: PlatformState, *, half_width: float) -> go.Mesh3d:
    left_body = profile.points_body.copy()
    right_body = profile.points_body.copy()
    left_body[:, 1] = -half_width
    right_body[:, 1] = half_width

    left_sensor = (state.sensor_from_body @ left_body.T).T
    right_sensor = (state.sensor_from_body @ right_body.T).T

    vertices = np.vstack([left_sensor, right_sensor])
    count = profile.points_body.shape[0]

    i_values: list[int] = []
    j_values: list[int] = []
    k_values: list[int] = []
    for index in range(count - 1):
        i_values.extend([index, index + 1])
        j_values.extend([index + 1, count + index + 1])
        k_values.extend([count + index, count + index])

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i_values,
        j=j_values,
        k=k_values,
        color="#e45756",
        opacity=0.35,
        name="occlusion mask",
        hoverinfo="skip",
        showscale=False,
    )


def make_visibility_figure(
    points_ned: NDArray[np.float64],
    state: PlatformState,
    scan_volume: ScanVolume,
    profile: OcclusionProfile,
    *,
    axis_length: float = 1.5,
    ribbon_half_width: float = 2.5,
) -> go.Figure:
    """Render the scene in sensor coordinates with classified sample points."""

    points = np.asarray(points_ned, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_ned must have shape (N, 3), got {points.shape!r}")

    results = [evaluate_visibility(point, state, scan_volume, profile) for point in points]

    traces: list[go.BaseTraceType] = []
    traces.extend(_frame_axis_traces(np.eye(3), label="sensor", length=axis_length))
    traces.extend(_frame_axis_traces(state.sensor_from_body, label="body", length=axis_length))
    traces.extend(_frame_axis_traces(state.sensor_from_world, label="world", length=axis_length))
    traces.append(_frustum_wireframe_trace(scan_volume))
    traces.append(_occlusion_ribbon_mesh(profile, state, half_width=ribbon_half_width))

    groups = {
        "visible": {"color": "#54a24b", "points": [], "text": []},
        "occluded": {"color": "#e45756", "points": [], "text": []},
        "out of scan": {"color": "#9d9da1", "points": [], "text": []},
    }

    for result in results:
        if result.visible:
            key = "visible"
        elif result.occluded:
            key = "occluded"
        else:
            key = "out of scan"

        groups[key]["points"].append(result.point_sensor)
        groups[key]["text"].append(
            (
                f"az={degrees(result.azimuth_rad):.1f} deg"
                f"<br>el={degrees(result.elevation_rad):.1f} deg"
                f"<br>range={result.range_m:.2f} m"
                f"<br>in_scan={result.in_scan}"
                f"<br>occluded={result.occluded}"
            )
        )

    for name, payload in groups.items():
        if not payload["points"]:
            continue
        points_sensor = np.asarray(payload["points"], dtype=float)
        traces.append(
            go.Scatter3d(
                x=points_sensor[:, 0],
                y=points_sensor[:, 1],
                z=points_sensor[:, 2],
                mode="markers",
                marker={"size": 4, "color": payload["color"]},
                name=name,
                text=payload["text"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    figure = go.Figure(traces)
    figure.update_layout(
        title="Sensor occlusion view in sensor frame",
        scene={
            "xaxis_title": "x forward",
            "yaxis_title": "y right",
            "zaxis_title": "z down",
            "aspectmode": "data",
        },
        legend={"orientation": "h"},
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    return figure
