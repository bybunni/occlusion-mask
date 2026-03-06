import importlib.util
from pathlib import Path

from plotly.graph_objects import Figure

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "visualize_occlusion.py"
SPEC = importlib.util.spec_from_file_location("visualize_occlusion", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_figure_reflects_platform_angles_in_title() -> None:
    figure = MODULE.build_figure(yaw_deg=12.0, pitch_deg=-4.0, roll_deg=7.0)

    assert isinstance(figure, Figure)
    assert "yaw +12.0 deg" in figure.layout.title.text
    assert "pitch -4.0 deg" in figure.layout.title.text
    assert "roll +7.0 deg" in figure.layout.title.text


def test_make_app_sets_expected_title() -> None:
    app = MODULE.make_app()

    assert app.title == "Aircraft Occlusion Explorer"
    graph = app.layout.children[1]
    assert graph.figure.layout.title.text is not None
    assert f"yaw {MODULE.INITIAL_YAW_DEG:+.1f} deg" in graph.figure.layout.title.text
    assert f"pitch {MODULE.INITIAL_PITCH_DEG:+.1f} deg" in graph.figure.layout.title.text
    assert f"roll {MODULE.INITIAL_ROLL_DEG:+.1f} deg" in graph.figure.layout.title.text
