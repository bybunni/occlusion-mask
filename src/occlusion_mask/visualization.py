"""Plotly helpers for visualizing the sensor frame, frustum, and occlusion mask."""

from __future__ import annotations

from math import degrees

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from .geometry import AzElMask2D, OcclusionProfile, PlatformState, ScanVolume, evaluate_visibility, ray_from_sensor_angles

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


def _occlusion_profile_traces(profile: OcclusionProfile, state: PlatformState) -> list[go.BaseTraceType]:
    boundary_points = profile.sensor_boundary_points(state)
    vertices = np.vstack([np.zeros((1, 3), dtype=float), boundary_points])
    count = boundary_points.shape[0]

    fan = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=[0] * (count - 1),
        j=list(range(1, count)),
        k=list(range(2, count + 1)),
        color="#e45756",
        opacity=0.18,
        name="occlusion fan",
        hoverinfo="skip",
        showscale=False,
    )
    boundary = go.Scatter3d(
        x=boundary_points[:, 0],
        y=boundary_points[:, 1],
        z=boundary_points[:, 2],
        mode="lines+markers",
        line={"color": "#c73b2f", "width": 7},
        marker={"size": 5, "color": "#c73b2f"},
        name="occlusion boundary",
        hoverinfo="skip",
    )
    return [fan, boundary]


def make_visibility_figure(
    points_ned: NDArray[np.float64],
    state: PlatformState,
    scan_volume: ScanVolume,
    profile: OcclusionProfile,
    *,
    axis_length: float = 1.5,
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
    traces.extend(_occlusion_profile_traces(profile, state))

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
            "zaxis": {"autorange": "reversed"},
            "aspectmode": "data",
        },
        legend={"orientation": "h"},
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    return figure


def _az_el_mask_fill_trace(
    points_az_el_deg: NDArray[np.float64],
    *,
    occluded_if: str,
    el_limits_deg: tuple[float, float],
) -> go.Scatter:
    if occluded_if == "el_ge_boundary":
        polygon_x = [float(points_az_el_deg[0, 0]), float(points_az_el_deg[-1, 0]), *points_az_el_deg[::-1, 0].tolist()]
        polygon_y = [el_limits_deg[1], el_limits_deg[1], *points_az_el_deg[::-1, 1].tolist()]
    else:
        polygon_x = [float(points_az_el_deg[0, 0]), float(points_az_el_deg[-1, 0]), *points_az_el_deg[::-1, 0].tolist()]
        polygon_y = [el_limits_deg[0], el_limits_deg[0], *points_az_el_deg[::-1, 1].tolist()]

    return go.Scatter(
        x=polygon_x,
        y=polygon_y,
        mode="lines",
        line={"width": 0},
        fill="toself",
        fillcolor="rgba(228, 87, 86, 0.18)",
        name="occluded region",
        hoverinfo="skip",
    )


def make_az_el_mask_figure(
    mask: AzElMask2D,
    *,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    query_point_deg: tuple[float, float] | None = None,
    az_limits_deg: tuple[float, float] = (-50.0, 50.0),
    el_limits_deg: tuple[float, float] = (-25.0, 25.0),
) -> go.Figure:
    """Render a 2D azimuth/elevation view of the simplified mask."""

    nominal_points = mask.points_az_el_deg
    transformed_points = mask.transformed_points_deg(pitch_deg=pitch_deg, roll_deg=roll_deg)

    traces: list[go.BaseTraceType] = []
    traces.append(
        go.Scatter(
            x=nominal_points[:, 0],
            y=nominal_points[:, 1],
            mode="lines+markers",
            line={"color": "#99785b", "width": 2, "dash": "dash"},
            marker={"size": 6, "color": "#99785b"},
            name="nominal mask",
            hoverinfo="skip",
        )
    )
    traces.append(
        _az_el_mask_fill_trace(
            transformed_points,
            occluded_if=mask.occluded_if,
            el_limits_deg=el_limits_deg,
        )
    )
    traces.append(
        go.Scatter(
            x=transformed_points[:, 0],
            y=transformed_points[:, 1],
            mode="lines+markers",
            line={"color": "#c73b2f", "width": 4},
            marker={"size": 8, "color": "#c73b2f"},
            name="transformed mask",
            hoverinfo="skip",
        )
    )

    if query_point_deg is not None:
        query_azimuth_deg, query_elevation_deg = query_point_deg
        occluded = mask.is_occluded_deg(
            query_azimuth_deg,
            query_elevation_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
        boundary_elevation_deg = mask.boundary_elevation_deg(
            query_azimuth_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
        query_color = "#e45756" if occluded else "#54a24b"
        if boundary_elevation_deg is None:
            hover_text = (
                f"az={query_azimuth_deg:+.1f} deg"
                f"<br>el={query_elevation_deg:+.1f} deg"
                "<br>boundary=outside mask span"
                f"<br>occluded={occluded}"
            )
        else:
            hover_text = (
                f"az={query_azimuth_deg:+.1f} deg"
                f"<br>el={query_elevation_deg:+.1f} deg"
                f"<br>boundary={boundary_elevation_deg:+.1f} deg"
                f"<br>occluded={occluded}"
            )
        traces.append(
            go.Scatter(
                x=[query_azimuth_deg],
                y=[query_elevation_deg],
                mode="markers",
                marker={"size": 12, "color": query_color, "line": {"color": "#13212b", "width": 1.5}},
                name="query point",
                text=[hover_text],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    figure = go.Figure(traces)
    figure.update_layout(
        title="2D sensor azimuth/elevation mask",
        xaxis={
            "title": "azimuth (deg)",
            "range": list(az_limits_deg),
            "gridcolor": "#d9cdb8",
            "zerolinecolor": "#6d7b84",
        },
        yaxis={
            "title": "elevation (deg)",
            "range": list(el_limits_deg),
            "gridcolor": "#d9cdb8",
            "zerolinecolor": "#6d7b84",
        },
        legend={"orientation": "h"},
        plot_bgcolor="#fcf8f1",
        paper_bgcolor="#f5f0e6",
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    return figure
