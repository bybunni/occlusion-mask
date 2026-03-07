from __future__ import annotations

from math import isclose

import numpy as np
import pytest

from occlusion_mask import (
    AzElMask2D,
    OcclusionProfile,
    PlatformState,
    ScanVolume,
    cartesian_to_sensor_angles,
    evaluate_visibility,
    ray_from_sensor_angles,
    transform_world_to_sensor,
)


def make_profile() -> OcclusionProfile:
    return OcclusionProfile.from_sensor_az_el_range_degrees(
        [
            (-40.0, 14.0, 3.0),
            (-20.0, 12.0, 2.8),
            (0.0, 8.0, 2.5),
            (20.0, 12.0, 2.8),
            (40.0, 14.0, 3.0),
        ],
        occluded_if="el_ge_boundary",
    )


def make_scan() -> ScanVolume:
    return ScanVolume.from_degrees(
        az_min_deg=-40.0,
        az_max_deg=40.0,
        el_min_deg=-30.0,
        el_max_deg=30.0,
        range_min_m=0.5,
        range_max_m=100.0,
    )


def make_mask_2d(*, occluded_if: str = "el_ge_boundary") -> AzElMask2D:
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


def test_world_to_sensor_is_identity_for_zero_attitude() -> None:
    state = PlatformState.from_degrees()
    point_sensor = transform_world_to_sensor((10.0, 2.0, 1.0), state)
    assert np.allclose(point_sensor, np.array([10.0, 2.0, 1.0]))


def test_sensor_is_pitch_roll_stabilized() -> None:
    state = PlatformState.from_degrees(pitch_deg=20.0, roll_deg=-15.0)
    point_sensor = transform_world_to_sensor((10.0, 0.0, 0.0), state)
    assert np.allclose(point_sensor, np.array([10.0, 0.0, 0.0]), atol=1e-9)


def test_cartesian_to_sensor_angles() -> None:
    azimuth_rad, elevation_rad, range_m = cartesian_to_sensor_angles((10.0, 10.0, -10.0))
    assert isclose(azimuth_rad, np.deg2rad(45.0))
    assert isclose(elevation_rad, np.deg2rad(35.26438968), rel_tol=1e-6)
    assert isclose(range_m, np.sqrt(300.0))


def test_boundary_interpolation_and_occlusion() -> None:
    profile = make_profile()
    state = PlatformState.from_degrees()

    boundary = profile.boundary_at_azimuth(np.deg2rad(10.0), state)

    assert boundary is not None
    boundary_elevation_rad, boundary_range_m = boundary
    assert isclose(np.rad2deg(boundary_elevation_rad), 10.0)
    assert isclose(boundary_range_m, 2.65)
    assert profile.is_occluded_sensor_point(ray_from_sensor_angles(np.deg2rad(10.0), np.deg2rad(12.0), 5.0), state)
    assert not profile.is_occluded_sensor_point(ray_from_sensor_angles(np.deg2rad(10.0), np.deg2rad(8.0), 5.0), state)


def test_pitch_down_increases_occlusion() -> None:
    point_ned = tuple(ray_from_sensor_angles(0.0, np.deg2rad(0.0), 5.0))
    scan = make_scan()
    profile = make_profile()

    pitch_down = PlatformState.from_degrees(pitch_deg=-15.0)
    level = PlatformState.from_degrees()
    pitch_up = PlatformState.from_degrees(pitch_deg=15.0)

    assert evaluate_visibility(point_ned, pitch_down, scan, profile).occluded
    assert not evaluate_visibility(point_ned, level, scan, profile).occluded
    assert not evaluate_visibility(point_ned, pitch_up, scan, profile).occluded


def test_visible_requires_in_scan_and_not_occluded() -> None:
    result = evaluate_visibility(tuple(ray_from_sensor_angles(0.0, np.deg2rad(0.0), 5.0)), PlatformState.from_degrees(), make_scan(), make_profile())
    assert result.in_scan
    assert not result.occluded
    assert result.visible


def test_scan_volume_rejects_large_azimuth() -> None:
    result = evaluate_visibility((2.0, 10.0, 0.0), PlatformState.from_degrees(), make_scan(), make_profile())
    assert not result.in_scan
    assert not result.visible


def test_2d_mask_pitch_translates_polygon_points() -> None:
    mask = make_mask_2d()

    nominal = mask.transformed_points_deg(sort_by_azimuth=False)
    shifted = mask.transformed_points_deg(pitch_deg=5.0, sort_by_azimuth=False)

    assert isclose(nominal[2, 1], 10.0)
    assert isclose(shifted[2, 1], 15.0)


def test_2d_mask_positive_roll_rotates_right_side_down() -> None:
    flat_mask = AzElMask2D.from_degrees(
        [
            (-40.0, 0.0),
            (-20.0, 0.0),
            (0.0, 0.0),
            (20.0, 0.0),
            (40.0, 0.0),
        ]
    )

    transformed = flat_mask.transformed_points_deg(roll_deg=10.0)

    assert transformed[0, 1] > 0.0
    assert transformed[-1, 1] < 0.0


def test_2d_mask_polygon_points_close_back_to_a() -> None:
    mask = make_mask_2d()
    polygon = mask.polygon_points_deg()

    assert polygon.shape == (6, 2)
    assert np.allclose(polygon[0], polygon[-1])


def test_2d_mask_point_inside_polygon_is_occluded() -> None:
    mask = make_mask_2d()

    assert mask.is_occluded_deg(0.0, 12.0)


def test_2d_mask_point_outside_polygon_is_clear() -> None:
    mask = make_mask_2d()

    assert not mask.is_occluded_deg(0.0, 8.0)
    assert not mask.is_occluded_deg(55.0, 20.0)


def test_2d_mask_point_on_edge_is_occluded() -> None:
    mask = make_mask_2d()

    assert mask.is_occluded_deg(-10.0, 11.5)


def test_2d_mask_point_on_vertex_is_occluded() -> None:
    mask = make_mask_2d()

    assert mask.is_occluded_deg(0.0, 10.0)


def test_2d_mask_occluded_if_is_ignored_for_polygon_mode() -> None:
    ge_mask = make_mask_2d(occluded_if="el_ge_boundary")
    le_mask = make_mask_2d(occluded_if="el_le_boundary")

    assert ge_mask.is_occluded_deg(0.0, 12.0) == le_mask.is_occluded_deg(0.0, 12.0)
    assert ge_mask.is_occluded_deg(0.0, 8.0) == le_mask.is_occluded_deg(0.0, 8.0)


def test_2d_mask_boundary_helpers_raise_for_polygon_mode() -> None:
    mask = make_mask_2d()

    with pytest.raises(NotImplementedError, match="single boundary elevation"):
        mask.boundary_elevation_deg(0.0)
