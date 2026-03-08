from __future__ import annotations

from dash import ALL, Dash, Input, Output, dcc, html
import plotly.graph_objects as go

from occlusion_mask import AzElMask2D, make_az_el_mask_figure

POINT_LABELS = ("A", "B", "C", "D", "E")
DEFAULT_MASK_POINTS = [
    (-100.0, 43.0),
    (-55.0, 41.0),
    (0.0, 6.5),
    (55.0, 41.0),
    (100.0, 43.0),
]

INITIAL_PITCH_DEG = 2.0
INITIAL_ROLL_DEG = 0.0
INITIAL_QUERY_AZ_DEG = 5.0
INITIAL_QUERY_EL_DEG = 9.0
DEFAULT_SENSOR_VOLUME_DEG = (-35.0, 35.0, -10.0, 20.0)


def build_mask(mask_points: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None) -> AzElMask2D:
    active_mask_points = DEFAULT_MASK_POINTS if mask_points is None else list(mask_points)
    return AzElMask2D.from_degrees(active_mask_points, occluded_if="el_ge_boundary")


MASK_2D = build_mask()


def transformed_summary(mask: AzElMask2D, *, pitch_deg: float, roll_deg: float) -> html.Div:
    transformed_points = mask.transformed_points_deg(
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        sort_by_azimuth=False,
    )
    return html.Div(
        [
            html.Div(
                "Transformed A-E Mask Points",
                style={"fontSize": "0.95rem", "fontWeight": "700", "letterSpacing": "0.04em"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                f"Point {point_label}",
                                style={"fontSize": "0.82rem", "fontWeight": "700", "letterSpacing": "0.04em"},
                            ),
                            html.Div(
                                f"az {azimuth_deg:+.1f} deg | el {elevation_deg:+.1f} deg",
                                style={"fontSize": "0.88rem", "color": "#22313a"},
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gap": "0.2rem",
                            "padding": "0.8rem 0.9rem",
                            "border": "1px solid #d6c8b0",
                            "borderRadius": "14px",
                            "background": "#fffdf8",
                        },
                    )
                    for point_label, (azimuth_deg, elevation_deg) in zip(POINT_LABELS, transformed_points, strict=True)
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "0.75rem",
                },
            ),
        ],
        style={"display": "grid", "gap": "0.65rem"},
    )


def current_mask_summary(mask_points: list[tuple[float, float]]) -> html.Div:
    return html.Div(
        [
            html.Div(
                "Current A-E Settings",
                style={"fontSize": "0.95rem", "fontWeight": "700", "letterSpacing": "0.04em"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                f"Point {point_label}",
                                style={"fontSize": "0.82rem", "fontWeight": "700", "letterSpacing": "0.04em"},
                            ),
                            html.Div(
                                f"az {azimuth_deg:+.1f} deg | el {elevation_deg:+.1f} deg",
                                style={"fontSize": "0.88rem", "color": "#22313a"},
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gap": "0.2rem",
                            "padding": "0.8rem 0.9rem",
                            "border": "1px solid #d6c8b0",
                            "borderRadius": "14px",
                            "background": "#fffdf8",
                        },
                    )
                    for point_label, (azimuth_deg, elevation_deg) in zip(POINT_LABELS, mask_points, strict=True)
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "0.75rem",
                },
            ),
        ],
        style={"display": "grid", "gap": "0.65rem"},
    )


def build_figure(
    *,
    pitch_deg: float,
    roll_deg: float,
    query_az_deg: float,
    query_el_deg: float,
    sensor_volume_deg: tuple[float, float, float, float] = DEFAULT_SENSOR_VOLUME_DEG,
    mask_points: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None,
):
    active_mask_points = DEFAULT_MASK_POINTS if mask_points is None else list(mask_points)
    mask = MASK_2D if mask_points is None else build_mask(active_mask_points)
    figure = make_az_el_mask_figure(
        mask,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        query_point_deg=(query_az_deg, query_el_deg),
        sensor_volume_deg=sensor_volume_deg,
        az_limits_deg=(-55.0, 55.0),
        el_limits_deg=(-25.0, 30.0),
    )
    figure.update_layout(
        title=(
            "2D Sensor Mask Explorer"
            f" | pitch {pitch_deg:+.1f} deg"
            f" | roll {roll_deg:+.1f} deg"
            f" | query az {query_az_deg:+.1f} deg"
            f" | query el {query_el_deg:+.1f} deg"
        ),
        font={"family": '"Avenir Next", "Helvetica Neue", sans-serif', "color": "#13212b"},
        uirevision="sensor-mask-2d-demo",
    )
    labeled_points = mask.transformed_points_deg(
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        sort_by_azimuth=False,
    )
    for trace in figure.data:
        if trace.name == "transformed mask":
            trace.mode = "lines+markers"
            break
    figure.add_trace(
        go.Scatter(
            x=labeled_points[:, 0],
            y=labeled_points[:, 1],
            mode="markers+text",
            marker={"size": 9, "color": "#c73b2f", "line": {"width": 1, "color": "#6e2016"}},
            text=list(POINT_LABELS),
            textposition="top center",
            textfont={"size": 12, "color": "#6e2016"},
            hovertext=[
                f"Point {point_label}<br>az {azimuth_deg:+.1f} deg<br>el {elevation_deg:+.1f} deg"
                for point_label, (azimuth_deg, elevation_deg) in zip(POINT_LABELS, labeled_points, strict=True)
            ],
            hovertemplate="%{hovertext}<extra></extra>",
            name="mask points",
            showlegend=False,
        )
    )
    return figure


def slider_marks(min_deg: int, max_deg: int, step_deg: int) -> dict[int, str]:
    return {value: f"{value:+d}" for value in range(min_deg, max_deg + 1, step_deg)}


def slider_block(
    *,
    slider_id: str,
    label: str,
    min_deg: int,
    max_deg: int,
    value: float,
    marks_step_deg: int = 15,
) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "0.9rem", "fontWeight": "700", "letterSpacing": "0.04em"}),
            dcc.Slider(
                id=slider_id,
                min=min_deg,
                max=max_deg,
                step=1,
                value=value,
                marks=slider_marks(min_deg, max_deg, marks_step_deg),
                tooltip={"always_visible": True, "placement": "bottom"},
                updatemode="drag",
            ),
        ],
        style={"display": "grid", "gap": "0.6rem"},
    )


def mask_points_from_lists(azimuth_values: list[float | None], elevation_values: list[float | None]) -> list[tuple[float, float]]:
    if len(azimuth_values) != len(POINT_LABELS) or len(elevation_values) != len(POINT_LABELS):
        raise ValueError("Every mask point needs azimuth and elevation values")

    points: list[tuple[float, float]] = []
    for azimuth_deg, elevation_deg in zip(azimuth_values, elevation_values, strict=True):
        if azimuth_deg is None or elevation_deg is None:
            raise ValueError("Every mask point needs azimuth and elevation values")
        points.append((float(azimuth_deg), float(elevation_deg)))

    if any(next_azimuth <= current_azimuth for (current_azimuth, _), (next_azimuth, _) in zip(points, points[1:])):
        raise ValueError("Mask points must remain ordered left to right by increasing azimuth")
    return points


def point_input(*, point_label: str, field_label: str, field_key: str, value: float, step: float) -> html.Div:
    return html.Div(
        [
            html.Label(
                field_label,
                htmlFor=f"{point_label}-{field_key}",
                style={"fontSize": "0.78rem", "fontWeight": "700", "letterSpacing": "0.03em"},
            ),
            dcc.Input(
                id={"type": f"mask-{field_key}", "index": point_label},
                type="number",
                value=value,
                step=step,
                debounce=False,
                inputMode="numeric",
                style={
                    "width": "100%",
                    "padding": "0.65rem 0.75rem",
                    "border": "1px solid #c8b9a2",
                    "borderRadius": "12px",
                    "backgroundColor": "#fffdf8",
                    "fontSize": "0.95rem",
                    "color": "#13212b",
                },
            ),
        ],
        style={"display": "grid", "gap": "0.35rem"},
    )


def volume_input(*, input_id: str, label: str, value: float, step: float) -> html.Div:
    return html.Div(
        [
            html.Label(
                label,
                htmlFor=input_id,
                style={"fontSize": "0.78rem", "fontWeight": "700", "letterSpacing": "0.03em"},
            ),
            dcc.Input(
                id=input_id,
                type="number",
                value=value,
                step=step,
                debounce=False,
                inputMode="numeric",
                style={
                    "width": "100%",
                    "padding": "0.65rem 0.75rem",
                    "border": "1px solid #b9d4bc",
                    "borderRadius": "12px",
                    "backgroundColor": "#f8fff7",
                    "fontSize": "0.95rem",
                    "color": "#13212b",
                },
            ),
        ],
        style={"display": "grid", "gap": "0.35rem"},
    )


def sensor_volume_from_values(
    az_min_deg: float | None,
    az_max_deg: float | None,
    el_min_deg: float | None,
    el_max_deg: float | None,
) -> tuple[float, float, float, float]:
    if az_min_deg is None or az_max_deg is None or el_min_deg is None or el_max_deg is None:
        raise ValueError("Sensor volume needs all four azimuth/elevation limits")

    normalized = (
        float(az_min_deg),
        float(az_max_deg),
        float(el_min_deg),
        float(el_max_deg),
    )
    if normalized[0] >= normalized[1]:
        raise ValueError("Sensor volume azimuth min must be less than azimuth max")
    if normalized[2] >= normalized[3]:
        raise ValueError("Sensor volume elevation min must be less than elevation max")
    return normalized


def sensor_volume_editor() -> html.Div:
    az_min_deg, az_max_deg, el_min_deg, el_max_deg = DEFAULT_SENSOR_VOLUME_DEG
    return html.Div(
        [
            html.Div(
                [
                    html.H2(
                        "Sensor Volume Box",
                        style={"margin": "0", "fontSize": "1.2rem", "fontWeight": "700"},
                    ),
                    html.P(
                        "Edit the green sensor-volume rectangle directly in azimuth/elevation space using min and max bounds.",
                        style={"margin": "0", "lineHeight": "1.5", "color": "#33434d"},
                    ),
                ],
                style={"display": "grid", "gap": "0.35rem"},
            ),
            html.Div(
                [
                    volume_input(input_id="sensor-volume-az-min", label="Azimuth Min (deg)", value=az_min_deg, step=1.0),
                    volume_input(input_id="sensor-volume-az-max", label="Azimuth Max (deg)", value=az_max_deg, step=1.0),
                    volume_input(input_id="sensor-volume-el-min", label="Elevation Min (deg)", value=el_min_deg, step=0.5),
                    volume_input(input_id="sensor-volume-el-max", label="Elevation Max (deg)", value=el_max_deg, step=0.5),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "1rem",
                },
            ),
            html.Div(
                id="sensor-volume-editor-status",
                children="Sensor volume accepted. Edit the limits to reshape the green rectangle.",
                style={
                    "minHeight": "1.5rem",
                    "fontSize": "0.92rem",
                    "fontWeight": "600",
                    "color": "#2a5e37",
                },
            ),
        ],
        style={
            "display": "grid",
            "gap": "1rem",
            "padding": "1.25rem",
            "border": "1px solid #b9d4bc",
            "borderRadius": "18px",
            "background": "linear-gradient(180deg, #f7fff5 0%, #e2f0df 100%)",
            "boxShadow": "0 18px 40px rgba(19, 33, 43, 0.08)",
        },
    )


def mask_point_card(point_label: str, azimuth_deg: float, elevation_deg: float) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        f"Point {point_label}",
                        style={"fontSize": "0.92rem", "fontWeight": "700", "letterSpacing": "0.05em"},
                    ),
                    html.Span(
                        "left to right in azimuth",
                        style={"fontSize": "0.76rem", "color": "#55626b"},
                    ),
                ],
                style={"display": "grid", "gap": "0.15rem"},
            ),
            html.Div(
                [
                    point_input(
                        point_label=point_label,
                        field_label="Azimuth (deg)",
                        field_key="az",
                        value=azimuth_deg,
                        step=1.0,
                    ),
                    point_input(
                        point_label=point_label,
                        field_label="Elevation (deg)",
                        field_key="el",
                        value=elevation_deg,
                        step=0.5,
                    ),
                ],
                style={"display": "grid", "gap": "0.75rem"},
            ),
        ],
        style={
            "display": "grid",
            "gap": "0.9rem",
            "padding": "1rem",
            "border": "1px solid #d9cdb8",
            "borderRadius": "16px",
            "background": "#fffaf2",
            "boxShadow": "0 10px 24px rgba(19, 33, 43, 0.06)",
        },
    )


def mask_editor() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H2(
                        "Mask Boundary Points",
                        style={"margin": "0", "fontSize": "1.2rem", "fontWeight": "700"},
                    ),
                    html.P(
                        "Edit the five piecewise mask points directly in sensor azimuth and elevation. Keep them ordered from left to right by azimuth.",
                        style={"margin": "0", "lineHeight": "1.5", "color": "#33434d"},
                    ),
                ],
                style={"display": "grid", "gap": "0.35rem"},
            ),
            html.Div(
                [
                    mask_point_card(point_label, azimuth_deg, elevation_deg)
                    for point_label, (azimuth_deg, elevation_deg) in zip(POINT_LABELS, DEFAULT_MASK_POINTS, strict=True)
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "1rem",
                },
            ),
            html.Div(
                id="mask-2d-editor-status",
                children="Mask accepted. Edit A through E to reshape the polygon.",
                style={
                    "minHeight": "1.5rem",
                    "fontSize": "0.92rem",
                    "fontWeight": "600",
                    "color": "#8a2f1d",
                },
            ),
            html.Div(
                id="mask-2d-current-values",
                children=current_mask_summary(DEFAULT_MASK_POINTS),
                style={"display": "grid", "gap": "0.8rem"},
            ),
        ],
        style={
            "display": "grid",
            "gap": "1rem",
            "padding": "1.25rem",
            "border": "1px solid #d9cdb8",
            "borderRadius": "18px",
            "background": "linear-gradient(180deg, #fff8ec 0%, #f4e6cf 100%)",
            "boxShadow": "0 18px 40px rgba(19, 33, 43, 0.08)",
        },
    )


def status_panel(mask: AzElMask2D, *, pitch_deg: float, roll_deg: float, query_az_deg: float, query_el_deg: float) -> html.Div:
    occluded = mask.is_occluded_deg(
        query_az_deg,
        query_el_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )
    summary = (
        f"The query point at az {query_az_deg:+.1f} deg, el {query_el_deg:+.1f} deg is "
        f"{'inside' if occluded else 'outside'} the transformed A-B-C-D-E-A polygon."
    )
    return html.Div(
        [
            html.Div(
                "Occlusion Test",
                style={"fontSize": "0.95rem", "fontWeight": "700", "letterSpacing": "0.04em"},
            ),
            html.Div(summary, style={"fontSize": "0.94rem", "lineHeight": "1.5", "color": "#22313a"}),
        ],
        style={
            "display": "grid",
            "gap": "0.65rem",
            "padding": "1rem 1.1rem",
            "border": "1px solid #d6c8b0",
            "borderRadius": "16px",
            "background": "#fffdf8",
        },
    )


def make_app() -> Dash:
    app = Dash(__name__)
    app.title = "2D Sensor Mask Explorer"
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "2D Sensor Mask Explorer",
                        style={"margin": "0", "fontSize": "2rem", "fontWeight": "700"},
                    ),
                    html.P(
                        "This simplified sensor-frame approximation keeps the occlusion mask as a closed A-B-C-D-E-A polygon in azimuth/elevation space. Positive platform roll rotates the polygon clockwise, platform pitch moves it along the rolled local axis, and the green rectangle shows the sensor volume box.",
                        style={"margin": "0", "maxWidth": "72ch", "lineHeight": "1.5"},
                    ),
                ],
                style={"display": "grid", "gap": "0.5rem"},
            ),
            dcc.Graph(
                id="mask-2d-graph",
                figure=build_figure(
                    pitch_deg=INITIAL_PITCH_DEG,
                    roll_deg=INITIAL_ROLL_DEG,
                    query_az_deg=INITIAL_QUERY_AZ_DEG,
                    query_el_deg=INITIAL_QUERY_EL_DEG,
                ),
                style={"height": "70vh"},
                config={"displaylogo": False, "responsive": True},
            ),
            html.Div(
                [
                    slider_block(
                        slider_id="pitch-slider",
                        label="Platform Pitch (deg)",
                        min_deg=-90,
                        max_deg=90,
                        value=INITIAL_PITCH_DEG,
                        marks_step_deg=30,
                    ),
                    slider_block(
                        slider_id="roll-slider",
                        label="Platform Roll (deg)",
                        min_deg=-180,
                        max_deg=180,
                        value=INITIAL_ROLL_DEG,
                        marks_step_deg=60,
                    ),
                    slider_block(
                        slider_id="query-az-slider",
                        label="Query Azimuth (deg)",
                        min_deg=-50,
                        max_deg=50,
                        value=INITIAL_QUERY_AZ_DEG,
                    ),
                    slider_block(
                        slider_id="query-el-slider",
                        label="Query Elevation (deg)",
                        min_deg=-20,
                        max_deg=25,
                        value=INITIAL_QUERY_EL_DEG,
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
                    "gap": "1.25rem",
                    "padding": "1.25rem",
                    "border": "1px solid #d9cdb8",
                    "borderRadius": "18px",
                    "background": "linear-gradient(180deg, #fcf8f1 0%, #f0e4d2 100%)",
                    "boxShadow": "0 18px 40px rgba(19, 33, 43, 0.08)",
                },
            ),
            sensor_volume_editor(),
            mask_editor(),
            html.Div(
                [
                    html.Div(
                        id="mask-2d-status",
                        children=status_panel(
                            MASK_2D,
                            pitch_deg=INITIAL_PITCH_DEG,
                            roll_deg=INITIAL_ROLL_DEG,
                            query_az_deg=INITIAL_QUERY_AZ_DEG,
                            query_el_deg=INITIAL_QUERY_EL_DEG,
                        ),
                    ),
                    html.Div(
                        id="mask-2d-summary",
                        children=transformed_summary(
                            MASK_2D,
                            pitch_deg=INITIAL_PITCH_DEG,
                            roll_deg=INITIAL_ROLL_DEG,
                        ),
                    ),
                ],
                style={"display": "grid", "gap": "1rem"},
            ),
        ],
        style={
            "minHeight": "100vh",
            "padding": "2rem",
            "background": "linear-gradient(180deg, #efe6d6 0%, #f7f3ea 100%)",
            "color": "#13212b",
            "fontFamily": '"Avenir Next", "Helvetica Neue", sans-serif',
            "display": "grid",
            "gap": "1.5rem",
        },
    )

    @app.callback(
        Output("mask-2d-graph", "figure"),
        Output("mask-2d-status", "children"),
        Output("mask-2d-summary", "children"),
        Output("mask-2d-editor-status", "children"),
        Output("mask-2d-current-values", "children"),
        Output("sensor-volume-editor-status", "children"),
        Input("pitch-slider", "value"),
        Input("roll-slider", "value"),
        Input("query-az-slider", "value"),
        Input("query-el-slider", "value"),
        Input("sensor-volume-az-min", "value"),
        Input("sensor-volume-az-max", "value"),
        Input("sensor-volume-el-min", "value"),
        Input("sensor-volume-el-max", "value"),
        Input({"type": "mask-az", "index": ALL}, "value"),
        Input({"type": "mask-el", "index": ALL}, "value"),
    )
    def update_figure(
        pitch_deg: float,
        roll_deg: float,
        query_az_deg: float,
        query_el_deg: float,
        sensor_volume_az_min_deg: float | None,
        sensor_volume_az_max_deg: float | None,
        sensor_volume_el_min_deg: float | None,
        sensor_volume_el_max_deg: float | None,
        azimuth_values: list[float | None],
        elevation_values: list[float | None],
    ):
        mask_points = [
            (
                None if azimuth_deg is None else float(azimuth_deg),
                None if elevation_deg is None else float(elevation_deg),
            )
            for azimuth_deg, elevation_deg in zip(azimuth_values, elevation_values, strict=True)
        ]
        summary_points = [
            (
                DEFAULT_MASK_POINTS[index][0] if azimuth_deg is None else azimuth_deg,
                DEFAULT_MASK_POINTS[index][1] if elevation_deg is None else elevation_deg,
            )
            for index, (azimuth_deg, elevation_deg) in enumerate(mask_points)
        ]
        try:
            normalized_sensor_volume = sensor_volume_from_values(
                sensor_volume_az_min_deg,
                sensor_volume_az_max_deg,
                sensor_volume_el_min_deg,
                sensor_volume_el_max_deg,
            )
            sensor_volume_status = "Sensor volume accepted. The green rectangle uses the edited limits."
        except ValueError as exc:
            normalized_sensor_volume = DEFAULT_SENSOR_VOLUME_DEG
            sensor_volume_status = f"Sensor volume update skipped: {exc}"

        try:
            normalized_mask_points = mask_points_from_lists(azimuth_values, elevation_values)
            mask = build_mask(normalized_mask_points)
            figure = build_figure(
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                query_az_deg=query_az_deg,
                query_el_deg=query_el_deg,
                sensor_volume_deg=normalized_sensor_volume,
                mask_points=normalized_mask_points,
            )
        except ValueError as exc:
            return (
                build_figure(
                    pitch_deg=pitch_deg,
                    roll_deg=roll_deg,
                    query_az_deg=query_az_deg,
                    query_el_deg=query_el_deg,
                    sensor_volume_deg=normalized_sensor_volume,
                ),
                status_panel(
                    MASK_2D,
                    pitch_deg=pitch_deg,
                    roll_deg=roll_deg,
                    query_az_deg=query_az_deg,
                    query_el_deg=query_el_deg,
                ),
                transformed_summary(
                    MASK_2D,
                    pitch_deg=pitch_deg,
                    roll_deg=roll_deg,
                ),
                f"Mask update skipped: {exc}",
                current_mask_summary(summary_points),
                sensor_volume_status,
            )

        return (
            figure,
            status_panel(
                mask,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                query_az_deg=query_az_deg,
                query_el_deg=query_el_deg,
            ),
            transformed_summary(
                mask,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
            ),
            "Mask accepted. Positive platform roll still rotates the edited polygon clockwise.",
            current_mask_summary(normalized_mask_points),
            sensor_volume_status,
        )

    return app


def main() -> None:
    app = make_app()
    app.run(debug=False)


if __name__ == "__main__":
    main()
