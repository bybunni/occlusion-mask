from __future__ import annotations

import numpy as np
from dash import ALL, Dash, Input, Output, dcc, html

from occlusion_mask import OcclusionProfile, PlatformState, ScanVolume, make_visibility_figure

POINT_LABELS = ("A", "B", "C", "D", "E")
DEFAULT_BOUNDARY_POINTS = [
    (-40.0, 14.0, 3.0),
    (-20.0, 12.0, 2.8),
    (0.0, 8.0, 2.5),
    (20.0, 12.0, 2.8),
    (40.0, 14.0, 3.0),
]


def sample_points() -> np.ndarray:
    x_values = np.linspace(0.5, 8.0, 12)
    y_values = np.linspace(-3.0, 3.0, 9)
    z_values = np.linspace(-1.5, 2.5, 10)
    xx, yy, zz = np.meshgrid(x_values, y_values, z_values, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def make_scan() -> ScanVolume:
    return ScanVolume.from_degrees(
        az_min_deg=-35.0,
        az_max_deg=35.0,
        el_min_deg=-25.0,
        el_max_deg=30.0,
        range_min_m=0.5,
        range_max_m=9.0,
    )


def build_profile(boundary_points: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...]) -> OcclusionProfile:
    return OcclusionProfile.from_sensor_az_el_range_degrees(
        boundary_points,
        occluded_if="el_ge_boundary",
    )


def make_profile() -> OcclusionProfile:
    return build_profile(DEFAULT_BOUNDARY_POINTS)


SAMPLE_POINTS = sample_points()
SCAN_VOLUME = make_scan()
OCCLUSION_PROFILE = make_profile()

INITIAL_YAW_DEG = 0.0
INITIAL_PITCH_DEG = -10.0
INITIAL_ROLL_DEG = 5.0


def boundary_points_from_lists(
    azimuth_values: list[float],
    elevation_values: list[float],
    range_values: list[float],
) -> list[tuple[float, float, float]]:
    return [
        (float(azimuth_deg), float(elevation_deg), float(range_m))
        for azimuth_deg, elevation_deg, range_m in zip(azimuth_values, elevation_values, range_values, strict=True)
    ]


def build_figure(
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    boundary_points: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...] | None = None,
):
    state = PlatformState.from_degrees(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        sensor_yaw_deg=0.0,
    )
    profile = OCCLUSION_PROFILE if boundary_points is None else build_profile(boundary_points)
    figure = make_visibility_figure(
        SAMPLE_POINTS,
        state,
        SCAN_VOLUME,
        profile,
    )
    figure.update_layout(
        title=(
            "Aircraft Occlusion Explorer"
            f" | yaw {yaw_deg:+.1f} deg"
            f" | pitch {pitch_deg:+.1f} deg"
            f" | roll {roll_deg:+.1f} deg"
        ),
        paper_bgcolor="#f5f0e6",
        plot_bgcolor="#f5f0e6",
        font={"family": '"Avenir Next", "Helvetica Neue", sans-serif', "color": "#13212b"},
        uirevision="aircraft-occlusion-demo",
    )
    figure.update_scenes(
        bgcolor="#fcf8f1",
        xaxis={"backgroundcolor": "#fcf8f1", "gridcolor": "#d9cdb8", "zerolinecolor": "#6d7b84"},
        yaxis={"backgroundcolor": "#fcf8f1", "gridcolor": "#d9cdb8", "zerolinecolor": "#6d7b84"},
        zaxis={"backgroundcolor": "#fcf8f1", "gridcolor": "#d9cdb8", "zerolinecolor": "#6d7b84"},
    )
    return figure


def slider_marks(min_deg: int, max_deg: int, step_deg: int) -> dict[int, str]:
    return {value: f"{value:+d}" for value in range(min_deg, max_deg + 1, step_deg)}


def slider_block(*, slider_id: str, label: str, min_deg: int, max_deg: int, value: float) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "0.9rem", "fontWeight": "700", "letterSpacing": "0.04em"}),
            dcc.Slider(
                id=slider_id,
                min=min_deg,
                max=max_deg,
                step=1,
                value=value,
                marks=slider_marks(min_deg, max_deg, 15),
                tooltip={"always_visible": True, "placement": "bottom"},
                updatemode="drag",
            ),
        ],
        style={"display": "grid", "gap": "0.6rem"},
    )


def point_input(*, point_label: str, field_label: str, field_key: str, value: float, step: float) -> html.Div:
    return html.Div(
        [
            html.Label(
                field_label,
                htmlFor=f"{point_label}-{field_key}",
                style={"fontSize": "0.78rem", "fontWeight": "700", "letterSpacing": "0.03em"},
            ),
            dcc.Input(
                id={"type": f"boundary-{field_key}", "index": point_label},
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


def boundary_point_card(point_label: str, azimuth_deg: float, elevation_deg: float, range_m: float) -> html.Div:
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
                    point_input(
                        point_label=point_label,
                        field_label="Range (m)",
                        field_key="range",
                        value=range_m,
                        step=0.1,
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


def boundary_editor() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H2(
                        "Occlusion Boundary Points",
                        style={"margin": "0", "fontSize": "1.2rem", "fontWeight": "700"},
                    ),
                    html.P(
                        "Edit the five body-attached boundary points in sensor azimuth, elevation, and range. Keep them ordered from left to right by azimuth.",
                        style={"margin": "0", "lineHeight": "1.5", "color": "#33434d"},
                    ),
                ],
                style={"display": "grid", "gap": "0.35rem"},
            ),
            html.Div(
                [
                    boundary_point_card(point_label, azimuth_deg, elevation_deg, range_m)
                    for point_label, (azimuth_deg, elevation_deg, range_m) in zip(POINT_LABELS, DEFAULT_BOUNDARY_POINTS, strict=True)
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "1rem",
                },
            ),
            html.Div(
                id="boundary-status",
                style={
                    "minHeight": "1.5rem",
                    "fontSize": "0.92rem",
                    "fontWeight": "600",
                    "color": "#8a2f1d",
                },
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


def make_app() -> Dash:
    app = Dash(__name__)
    app.title = "Aircraft Occlusion Explorer"
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "Aircraft Occlusion Explorer",
                        style={"margin": "0", "fontSize": "2rem", "fontWeight": "700"},
                    ),
                    html.P(
                        "Move the platform yaw, pitch, and roll sliders and edit the five boundary points to see a body-attached occlusion profile, authored in sensor azimuth/elevation/range space, rotate around the level-stabilized sensor.",
                        style={"margin": "0", "maxWidth": "68ch", "lineHeight": "1.5"},
                    ),
                ],
                style={"display": "grid", "gap": "0.5rem"},
            ),
            dcc.Graph(
                id="occlusion-graph",
                figure=build_figure(
                    yaw_deg=INITIAL_YAW_DEG,
                    pitch_deg=INITIAL_PITCH_DEG,
                    roll_deg=INITIAL_ROLL_DEG,
                ),
                style={"height": "72vh"},
                config={"displaylogo": False, "responsive": True},
            ),
            html.Div(
                [
                    slider_block(
                        slider_id="yaw-slider",
                        label="Platform Yaw (deg)",
                        min_deg=-45,
                        max_deg=45,
                        value=INITIAL_YAW_DEG,
                    ),
                    slider_block(
                        slider_id="pitch-slider",
                        label="Platform Pitch (deg)",
                        min_deg=-25,
                        max_deg=25,
                        value=INITIAL_PITCH_DEG,
                    ),
                    slider_block(
                        slider_id="roll-slider",
                        label="Platform Roll (deg)",
                        min_deg=-25,
                        max_deg=25,
                        value=INITIAL_ROLL_DEG,
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
            boundary_editor(),
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
        Output("occlusion-graph", "figure"),
        Output("boundary-status", "children"),
        Input("yaw-slider", "value"),
        Input("pitch-slider", "value"),
        Input("roll-slider", "value"),
        Input({"type": "boundary-az", "index": ALL}, "value"),
        Input({"type": "boundary-el", "index": ALL}, "value"),
        Input({"type": "boundary-range", "index": ALL}, "value"),
    )
    def update_figure(
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        azimuth_values: list[float],
        elevation_values: list[float],
        range_values: list[float],
    ):
        boundary_points = boundary_points_from_lists(azimuth_values, elevation_values, range_values)
        try:
            figure = build_figure(
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                boundary_points=boundary_points,
            )
        except ValueError as exc:
            fallback_figure = build_figure(
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                boundary_points=DEFAULT_BOUNDARY_POINTS,
            )
            return fallback_figure, f"Boundary update skipped: {exc}"

        return figure, "Boundary accepted. The profile is defined from Point A through Point E."

    return app


def main() -> None:
    app = make_app()
    app.run(debug=False)


if __name__ == "__main__":
    main()
