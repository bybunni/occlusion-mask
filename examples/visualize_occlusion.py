from __future__ import annotations

import numpy as np
from dash import Dash, Input, Output, dcc, html

from occlusion_mask import OcclusionProfile, PlatformState, ScanVolume, make_visibility_figure


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


def make_profile() -> OcclusionProfile:
    return OcclusionProfile(
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


SAMPLE_POINTS = sample_points()
SCAN_VOLUME = make_scan()
OCCLUSION_PROFILE = make_profile()

INITIAL_YAW_DEG = 0.0
INITIAL_PITCH_DEG = -10.0
INITIAL_ROLL_DEG = 5.0


def build_figure(*, yaw_deg: float, pitch_deg: float, roll_deg: float):
    state = PlatformState.from_degrees(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        sensor_yaw_deg=0.0,
    )
    figure = make_visibility_figure(
        SAMPLE_POINTS,
        state,
        SCAN_VOLUME,
        OCCLUSION_PROFILE,
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
                        "Move the platform yaw, pitch, and roll sliders to see how the body frame rotates around the level-stabilized sensor.",
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
        Input("yaw-slider", "value"),
        Input("pitch-slider", "value"),
        Input("roll-slider", "value"),
    )
    def update_figure(yaw_deg: float, pitch_deg: float, roll_deg: float):
        return build_figure(yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg)

    return app


def main() -> None:
    app = make_app()
    app.run(debug=False)


if __name__ == "__main__":
    main()
