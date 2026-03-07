from __future__ import annotations

from math import isclose

import pytest
import torch

from occlusion_mask import AzElMask2D, TorchAzElMask2D


def make_numpy_mask() -> AzElMask2D:
    return AzElMask2D.from_degrees(
        [
            (-40.0, 16.0),
            (-20.0, 13.0),
            (0.0, 10.0),
            (20.0, 13.0),
            (40.0, 16.0),
        ]
    )


def make_torch_mask() -> TorchAzElMask2D:
    return TorchAzElMask2D.from_degrees(
        [
            (-40.0, 16.0),
            (-20.0, 13.0),
            (0.0, 10.0),
            (20.0, 13.0),
            (40.0, 16.0),
        ]
    )


def column(values: list[float], *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype).unsqueeze(1)


def test_torch_mask_rejects_unsorted_points() -> None:
    with pytest.raises(ValueError, match="increasing azimuth"):
        TorchAzElMask2D.from_degrees(
            [
                (-20.0, 13.0),
                (-40.0, 16.0),
                (0.0, 10.0),
                (20.0, 13.0),
                (40.0, 16.0),
            ]
        )


def test_torch_mask_requires_column_vectors() -> None:
    mask = make_torch_mask()
    with pytest.raises(ValueError, match="shape \\(n, 1\\)"):
        mask.is_occluded_deg(
            torch.tensor([0.0, 5.0]),
            column([12.0, 9.0]),
            column([0.0, 0.0]),
            column([0.0, 0.0]),
        )


def test_torch_mask_requires_matching_shapes() -> None:
    mask = make_torch_mask()
    with pytest.raises(ValueError, match="same shape"):
        mask.is_occluded_deg(
            column([0.0, 5.0]),
            column([12.0, 9.0]),
            column([0.0]),
            column([0.0, 0.0]),
        )


def test_torch_mask_pitch_translates_boundary() -> None:
    mask = make_torch_mask()
    boundary = mask.boundary_elevation_deg(
        column([0.0]),
        pitch_deg=column([5.0]),
        roll_deg=column([0.0]),
    )
    assert boundary.shape == (1, 1)
    assert isclose(float(boundary.item()), 15.0)


def test_torch_mask_positive_roll_rotates_right_side_up() -> None:
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
    assert float(transformed[0, 0, 1]) < 0.0
    assert float(transformed[0, -1, 1]) > 0.0


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


def test_torch_mask_outside_span_returns_false_and_nan_boundary() -> None:
    mask = make_torch_mask()
    boundary = mask.boundary_elevation_deg(
        column([55.0]),
        pitch_deg=column([0.0]),
        roll_deg=column([0.0]),
    )
    occluded = mask.is_occluded_deg(
        column([55.0]),
        column([20.0]),
        column([0.0]),
        column([0.0]),
    )

    assert torch.isnan(boundary).item()
    assert occluded.tolist() == [[False]]


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
        for az, el, pt, rl in zip(azimuth, elevation, pitch, roll, strict=True)
    ]
    assert torch_result.tolist() == expected


def test_torch_boundary_matches_numpy_row_by_row() -> None:
    numpy_mask = make_numpy_mask()
    torch_mask = make_torch_mask()

    azimuth = [0.0, 5.0, -15.0, 55.0]
    pitch = [0.0, -6.0, 3.0, 0.0]
    roll = [0.0, 8.0, -4.0, 0.0]

    torch_boundary = torch_mask.boundary_elevation_deg(
        column(azimuth),
        pitch_deg=column(pitch),
        roll_deg=column(roll),
    )

    expected = [
        numpy_mask.boundary_elevation_deg(azimuth_deg=az, pitch_deg=pt, roll_deg=rl)
        for az, pt, rl in zip(azimuth, pitch, roll, strict=True)
    ]

    for actual, target in zip(torch_boundary.squeeze(1).tolist(), expected, strict=True):
        if target is None:
            assert torch.isnan(torch.tensor(actual)).item()
        else:
            assert isclose(actual, target, rel_tol=1e-9, abs_tol=1e-9)


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
