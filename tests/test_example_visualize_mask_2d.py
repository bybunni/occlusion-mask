import importlib.util
from pathlib import Path

from plotly.graph_objects import Figure

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "visualize_mask_2d.py"
SPEC = importlib.util.spec_from_file_location("visualize_mask_2d", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


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


def test_make_app_sets_expected_title() -> None:
    app = MODULE.make_app()

    assert app.title == "2D Sensor Mask Explorer"
    graph = app.layout.children[1]
    assert graph.figure.layout.title.text is not None
    assert f"pitch {MODULE.INITIAL_PITCH_DEG:+.1f} deg" in graph.figure.layout.title.text
    assert f"roll {MODULE.INITIAL_ROLL_DEG:+.1f} deg" in graph.figure.layout.title.text
