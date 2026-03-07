import importlib.util
from pathlib import Path

from dash.development.base_component import Component
from plotly.graph_objects import Figure
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "visualize_mask_2d.py"
SPEC = importlib.util.spec_from_file_location("visualize_mask_2d", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _collect_ids(component: Component) -> list[object]:
    ids: list[object] = []
    component_id = getattr(component, "id", None)
    if component_id is not None:
        ids.append(component_id)

    children = getattr(component, "children", None)
    if isinstance(children, Component):
        ids.extend(_collect_ids(children))
    elif isinstance(children, (list, tuple)):
        for child in children:
            if isinstance(child, Component):
                ids.extend(_collect_ids(child))
    return ids


def test_build_figure_reflects_platform_pitch_roll_and_query() -> None:
    figure = MODULE.build_figure(
        pitch_deg=-4.0,
        roll_deg=11.0,
        query_az_deg=7.0,
        query_el_deg=9.0,
    )

    assert isinstance(figure, Figure)
    assert "pitch -4.0 deg" in figure.layout.title.text
    assert "roll +11.0 deg" in figure.layout.title.text
    assert "query az +7.0 deg" in figure.layout.title.text
    assert "query el +9.0 deg" in figure.layout.title.text


def test_build_figure_labels_original_points_even_when_sorted_boundary_reorders() -> None:
    mask_points = [
        (-40.0, 0.0),
        (-20.0, 0.0),
        (0.0, 0.0),
        (20.0, 50.0),
        (40.0, 0.0),
    ]
    figure = MODULE.build_figure(
        pitch_deg=0.0,
        roll_deg=60.0,
        query_az_deg=0.0,
        query_el_deg=0.0,
        mask_points=mask_points,
    )
    label_trace = next(trace for trace in figure.data if trace.name == "mask points")
    transformed = MODULE.build_mask(mask_points).transformed_points_deg(
        pitch_deg=0.0,
        roll_deg=60.0,
        sort_by_azimuth=False,
    )

    assert list(label_trace.text) == list(MODULE.POINT_LABELS)
    assert label_trace.x[3] == pytest.approx(transformed[3, 0])
    assert label_trace.y[3] == pytest.approx(transformed[3, 1])


def test_make_app_sets_expected_title() -> None:
    app = MODULE.make_app()

    assert app.title == "2D Sensor Mask Explorer"
    graph = app.layout.children[1]
    assert graph.figure.layout.title.text is not None
    assert f"pitch {MODULE.INITIAL_PITCH_DEG:+.1f} deg" in graph.figure.layout.title.text
    assert f"roll {MODULE.INITIAL_ROLL_DEG:+.1f} deg" in graph.figure.layout.title.text


def test_make_app_uses_expanded_pitch_and_roll_slider_ranges() -> None:
    app = MODULE.make_app()

    slider_panel = app.layout.children[2]
    sliders_by_id = {block.children[1].id: block.children[1] for block in slider_panel.children}

    assert sliders_by_id["pitch-slider"].min == -90
    assert sliders_by_id["pitch-slider"].max == 90
    assert sliders_by_id["roll-slider"].min == -180
    assert sliders_by_id["roll-slider"].max == 180


def test_make_app_includes_mask_point_inputs() -> None:
    app = MODULE.make_app()
    all_ids = _collect_ids(app.layout)

    for point_label in MODULE.POINT_LABELS:
        assert {"type": "mask-az", "index": point_label} in all_ids
        assert {"type": "mask-el", "index": point_label} in all_ids


def test_mask_points_from_lists_accepts_ordered_points() -> None:
    points = MODULE.mask_points_from_lists(
        [-45.0, -15.0, 0.0, 18.0, 42.0],
        [18.0, 14.0, 11.0, 15.0, 19.0],
    )

    assert points[0] == (-45.0, 18.0)
    assert points[-1] == (42.0, 19.0)


def test_mask_points_from_lists_rejects_missing_or_unsorted_points() -> None:
    with pytest.raises(ValueError, match="Every mask point needs azimuth and elevation values"):
        MODULE.mask_points_from_lists([-40.0, None, 0.0, 20.0, 40.0], [16.0, 13.0, 10.0, 13.0, 16.0])

    with pytest.raises(ValueError, match="ordered left to right"):
        MODULE.mask_points_from_lists([-40.0, 5.0, 0.0, 20.0, 40.0], [16.0, 13.0, 10.0, 13.0, 16.0])
