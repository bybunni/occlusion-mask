from math import isclose

import pytest
import torch

from occlusion_mask import AzElMask2D, TorchAzElMask2D


def make_numpy_mask(occluded_if: str = "el_ge_boundary") -> AzElMask2D:
    return AzElMask2D.from_degrees(
        [
            (-40.0, 16.0),
            (-20.0, 13.0),
            (0.0, 10.0),
            (20.0, 13.0),
            (40.0, 16.0),
        ],
        occluded_if=occluded_if,
    )


def make_torch_mask(occluded_if: str = "el_ge_boundary") -> TorchAzElMask2D:
    return TorchAzElMask2D.from_degrees(
        [
            (-40.0, 16.0),
            (-20.0, 13.0),
            (0.0, 10.0),
            (20.0, 13.0),
            (40.0, 16.0),
        ],
        occluded_if=occluded_if,
    )


def column(values: list[float], dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype).unsqueeze(1)


def test_torch_mask_rejects_unsorted_points() -> None:
    with pytest.raises(AssertionError, match="increasing azimuth"):
        TorchAzElMask2D.from_degrees(
            [
                (-20.0, 13.0),
                (-40.0, 16.0),
                (0.0, 10.0),
                (20.0, 13.0),
                (40.0, 16.0),
            ]
        )


def test_torch_mask_defaults_to_float32_storage() -> None:
    mask = make_torch_mask()
    assert mask.points_az_el_rad.dtype == torch.float32


def test_torch_mask_requires_column_vectors() -> None:
    mask = make_torch_mask()
    with pytest.raises(AssertionError, match="shape \\(n, 1\\)"):
        mask.is_occluded_deg(
            torch.tensor([0.0, 5.0]),
            column([12.0, 9.0]),
            column([0.0, 0.0]),
            column([0.0, 0.0]),
        )


def test_torch_mask_requires_matching_shapes() -> None:
    mask = make_torch_mask()
    with pytest.raises(AssertionError, match="same shape"):
        mask.is_occluded_deg(
            column([0.0, 5.0]),
            column([12.0, 9.0]),
            column([0.0]),
            column([0.0, 0.0]),
        )


def test_torch_mask_pitch_translates_polygon_points() -> None:
    mask = make_torch_mask()
    transformed = mask.transformed_points_deg(
        pitch_deg=column([5.0]),
        roll_deg=column([0.0]),
        sort_by_azimuth=False,
    )

    assert transformed.shape == (1, 5, 2)
    assert isclose(float(transformed[0, 2, 1]), 15.0, rel_tol=1e-6, abs_tol=1e-6)


def test_torch_mask_positive_roll_rotates_right_side_down() -> None:
    flat_mask = TorchAzElMask2D.from_degrees(
        [
            (-40.0, 0.0),
            (-20.0, 0.0),
            (0.0, 0.0),
            (20.0, 0.0),
            (40.0, 0.0),
        ]
    )
    transformed = flat_mask.transformed_points_deg(
        pitch_deg=column([0.0]),
        roll_deg=column([10.0]),
    )

    assert transformed.shape == (1, 5, 2)
    assert float(transformed[0, 0, 1]) > 0.0
    assert float(transformed[0, -1, 1]) < 0.0


def test_torch_polygon_points_close_back_to_a() -> None:
    mask = make_torch_mask()
    polygon = mask.polygon_points_deg(
        pitch_deg=column([0.0]),
        roll_deg=column([0.0]),
    )

    assert polygon.shape == (1, 6, 2)
    assert torch.allclose(polygon[:, 0, :], polygon[:, -1, :])


def test_torch_mask_returns_bool_column_tensor() -> None:
    mask = make_torch_mask()
    occluded = mask.is_occluded_deg(
        column([0.0, 0.0]),
        column([12.0, 8.0]),
        column([0.0, 0.0]),
        column([0.0, 0.0]),
    )

    assert occluded.shape == (2, 1)
    assert occluded.dtype == torch.bool
    assert occluded.tolist() == [[True], [False]]


def test_torch_mask_point_on_edge_is_occluded() -> None:
    mask = make_torch_mask()
    occluded = mask.is_occluded_deg(
        column([-10.0]),
        column([11.5]),
        column([0.0]),
        column([0.0]),
    )

    assert occluded.tolist() == [[True]]


def test_torch_mask_point_on_vertex_is_occluded() -> None:
    mask = make_torch_mask()
    occluded = mask.is_occluded_deg(
        column([0.0]),
        column([10.0]),
        column([0.0]),
        column([0.0]),
    )

    assert occluded.tolist() == [[True]]


def test_torch_mask_occluded_if_is_ignored_for_polygon_mode() -> None:
    ge_mask = make_torch_mask(occluded_if="el_ge_boundary")
    le_mask = make_torch_mask(occluded_if="el_le_boundary")

    ge_result = ge_mask.is_occluded_deg(column([0.0, 0.0]), column([12.0, 8.0]), column([0.0, 0.0]), column([0.0, 0.0]))
    le_result = le_mask.is_occluded_deg(column([0.0, 0.0]), column([12.0, 8.0]), column([0.0, 0.0]), column([0.0, 0.0]))

    assert torch.equal(ge_result, le_result)


def test_torch_mask_matches_numpy_row_by_row() -> None:
    numpy_mask = make_numpy_mask()
    torch_mask = make_torch_mask()

    azimuth = [0.0, 5.0, -15.0, 55.0]
    elevation = [12.0, 9.0, 14.0, 20.0]
    pitch = [0.0, -6.0, 3.0, 0.0]
    roll = [0.0, 8.0, -4.0, 0.0]

    torch_result = torch_mask.is_occluded_deg(
        column(azimuth),
        column(elevation),
        column(pitch),
        column(roll),
    )

    expected = [
        [numpy_mask.is_occluded_deg(azimuth_deg=az, elevation_deg=el, pitch_deg=pt, roll_deg=rl)]
        for az, el, pt, rl in zip(azimuth, elevation, pitch, roll)
    ]
    assert torch_result.tolist() == expected


def test_torch_mask_requires_floating_query_tensors() -> None:
    mask = make_torch_mask()
    with pytest.raises(AssertionError, match="floating point"):
        mask.is_occluded_deg(
            torch.tensor([[0]], dtype=torch.int64),
            column([12.0]),
            column([0.0]),
            column([0.0]),
        )


def test_torch_mask_does_not_expose_boundary_helpers() -> None:
    mask = make_torch_mask()
    assert not hasattr(mask, "boundary_elevation_deg")
    assert not hasattr(mask, "boundary_elevation_rad")


def test_torch_runtime_smoke() -> None:
    assert isinstance(torch.__version__, str)
    mask = make_torch_mask()
    result = mask.is_occluded_deg(
        column([0.0]),
        column([12.0]),
        column([0.0]),
        column([0.0]),
    )
    assert result.tolist() == [[True]]


def test_torch_mask_ascii_render_contains_labels_and_footer() -> None:
    mask = make_torch_mask()
    rendered = mask.render_ascii_deg(
        pitch_deg=5.0,
        roll_deg=0.0,
        width=33,
        height=11,
    )

    assert "AzEl mask debug  pitch=5.0 deg  roll=0.0 deg" in rendered
    assert "A=(" in rendered
    assert "E=(" in rendered
    for label in "ABCDE":
        assert label in rendered

    lines = rendered.splitlines()
    grid_lines = lines[2:13]
    assert len(grid_lines) == 11
    assert all(len(line) == 33 for line in grid_lines)


def test_torch_mask_ascii_render_footer_uses_original_point_labels() -> None:
    mask = TorchAzElMask2D.from_degrees(
        [
            (-40.0, 0.0),
            (-20.0, 0.0),
            (0.0, 0.0),
            (20.0, 50.0),
            (40.0, 0.0),
        ]
    )
    transformed = mask.transformed_points_deg(
        pitch_deg=column([0.0]),
        roll_deg=column([60.0]),
        sort_by_azimuth=False,
    )[0]
    rendered = mask.render_ascii_deg(pitch_deg=0.0, roll_deg=60.0, width=41, height=13)

    expected_d = f"D=({float(transformed[3, 0].item()):6.2f} az, {float(transformed[3, 1].item()):6.2f} el)"
    expected_e = f"E=({float(transformed[4, 0].item()):6.2f} az, {float(transformed[4, 1].item()):6.2f} el)"

    assert expected_d in rendered
    assert expected_e in rendered


def test_torch_mask_ascii_render_requires_single_state() -> None:
    mask = make_torch_mask()
    with pytest.raises(AssertionError, match="single-value tensor"):
        mask.render_ascii_deg(
            pitch_deg=column([0.0, 1.0]),
            roll_deg=0.0,
        )
