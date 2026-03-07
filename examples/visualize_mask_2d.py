from __future__ import annotations

from dash import Dash, Input, Output, dcc, html

from occlusion_mask import AzElMask2D, make_az_el_mask_figure

POINT_LABELS = ("A", "B", "C", "D", "E")
DEFAULT_MASK_POINTS = [
    (-40.0, 16.0),
    (-20.0, 13.0),
    (0.0, 10.0),
    (20.0, 13.0),
    (40.0, 16.0),
]

INITIAL_PITCH_DEG = -6.0
INITIAL_ROLL_DEG = 8.0
INITIAL_QUERY_AZ_DEG = 5.0
INITIAL_QUERY_EL_DEG = 9.0


def build_mask() -> AzElMask2D:
    return AzElMask2D.from_degrees(DEFAULT_MASK_POINTS, occluded_if="el_ge_boundary")


MASK_2D = build_mask()


def transformed_summary(mask: AzElMask2D, *, pitch_deg: float, roll_deg: float) -> html.Div:
    transformed_points = mask.transformed_points_deg(pitch_deg=pitch_deg, roll_deg=roll_deg)
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


def build_figure(*, pitch_deg: float, roll_deg: float, query_az_deg: float, query_el_deg: float):
    figure = make_az_el_mask_figure(
        MASK_2D,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        query_point_deg=(query_az_deg, query_el_deg),
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
    for trace in figure.data:
        if trace.name == "transformed mask":
            trace.mode = "lines+markers+text"
            trace.text = list(POINT_LABELS)
            trace.textposition = "top center"
            trace.textfont = {"size": 12, "color": "#6e2016"}
            trace.hovertext = [
                f"Point {point_label}<br>az {azimuth_deg:+.1f} deg<br>el {elevation_deg:+.1f} deg"
                for point_label, (azimuth_deg, elevation_deg) in zip(
                    POINT_LABELS,
                    MASK_2D.transformed_points_deg(pitch_deg=pitch_deg, roll_deg=roll_deg),
                    strict=True,
                )
            ]
            trace.hovertemplate = "%{hovertext}<extra></extra>"
            break
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


def status_panel(*, pitch_deg: float, roll_deg: float, query_az_deg: float, query_el_deg: float) -> html.Div:
    boundary_elevation_deg = MASK_2D.boundary_elevation_deg(
        query_az_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )
    occluded = MASK_2D.is_occluded_deg(
        query_az_deg,
        query_el_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )
    if boundary_elevation_deg is None:
        summary = "The query azimuth is outside the mask span, so the 2D mask does not occlude it."
    else:
        summary = (
            f"The transformed mask boundary at az {query_az_deg:+.1f} deg is el {boundary_elevation_deg:+.1f} deg. "
            f"The query point is {'occluded' if occluded else 'clear'}."
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
                        "This simplified sensor-frame approximation keeps the occlusion mask piecewise-linear in azimuth/elevation space. Positive platform roll rotates the mask clockwise, and platform pitch translates it vertically.",
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
                        min_deg=-25,
                        max_deg=25,
                        value=INITIAL_PITCH_DEG,
                    ),
                    slider_block(
                        slider_id="roll-slider",
                        label="Platform Roll (deg)",
                        min_deg=-35,
                        max_deg=35,
                        value=INITIAL_ROLL_DEG,
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
            html.Div(
                [
                    html.Div(id="mask-2d-status"),
                    html.Div(id="mask-2d-summary"),
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
        Input("pitch-slider", "value"),
        Input("roll-slider", "value"),
        Input("query-az-slider", "value"),
        Input("query-el-slider", "value"),
    )
    def update_figure(pitch_deg: float, roll_deg: float, query_az_deg: float, query_el_deg: float):
        return (
            build_figure(
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                query_az_deg=query_az_deg,
                query_el_deg=query_el_deg,
            ),
            status_panel(
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                query_az_deg=query_az_deg,
                query_el_deg=query_el_deg,
            ),
            transformed_summary(MASK_2D, pitch_deg=pitch_deg, roll_deg=roll_deg),
        )

    return app


def main() -> None:
    app = make_app()
    app.run(debug=False)


if __name__ == "__main__":
    main()
