from __future__ import annotations

from math import isclose

import numpy as np

from occlusion_mask import (
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
