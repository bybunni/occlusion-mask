"""Core geometry and visibility routines for the occlusion-mask package."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, hypot, radians, sin

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
ArrayLike3 = Vector | tuple[float, float, float] | list[float]


def _as_vector(value: ArrayLike3, *, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape!r}")
    return vector


def _rotation_x(angle_rad: float) -> Vector:
    c_value = cos(angle_rad)
    s_value = sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c_value, s_value],
            [0.0, -s_value, c_value],
        ],
        dtype=float,
    )


def _rotation_y(angle_rad: float) -> Vector:
    c_value = cos(angle_rad)
    s_value = sin(angle_rad)
    return np.array(
        [
            [c_value, 0.0, -s_value],
            [0.0, 1.0, 0.0],
            [s_value, 0.0, c_value],
        ],
        dtype=float,
    )


def _rotation_z(angle_rad: float) -> Vector:
    c_value = cos(angle_rad)
    s_value = sin(angle_rad)
    return np.array(
        [
            [c_value, s_value, 0.0],
            [-s_value, c_value, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def body_from_world_rotation(yaw_rad: float, pitch_rad: float, roll_rad: float) -> Vector:
    """Return the NED world-to-body DCM using aerospace 3-2-1 yaw-pitch-roll."""

    return _rotation_x(roll_rad) @ _rotation_y(pitch_rad) @ _rotation_z(yaw_rad)


def sensor_from_world_rotation(platform_yaw_rad: float, sensor_yaw_rad: float) -> Vector:
    """Return the NED world-to-sensor DCM for a level-stabilized sensor."""

    return _rotation_z(platform_yaw_rad + sensor_yaw_rad)


def body_from_wind_rotation(alpha_rad: float, beta_rad: float) -> Vector:
    """Return the optional wind-to-body rotation."""

    return _rotation_y(alpha_rad) @ _rotation_z(beta_rad)


@dataclass(frozen=True)
class PlatformState:
    """Aircraft and sensor state with coincident body and sensor origins."""

    position_ned: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    sensor_yaw_rad: float = 0.0
    alpha_rad: float | None = None
    beta_rad: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_ned", _as_vector(self.position_ned, name="position_ned"))

    @classmethod
    def from_degrees(
        cls,
        *,
        position_ned: ArrayLike3 = (0.0, 0.0, 0.0),
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        sensor_yaw_deg: float = 0.0,
        alpha_deg: float | None = None,
        beta_deg: float | None = None,
    ) -> "PlatformState":
        return cls(
            position_ned=_as_vector(position_ned, name="position_ned"),
            yaw_rad=radians(yaw_deg),
            pitch_rad=radians(pitch_deg),
            roll_rad=radians(roll_deg),
            sensor_yaw_rad=radians(sensor_yaw_deg),
            alpha_rad=None if alpha_deg is None else radians(alpha_deg),
            beta_rad=None if beta_deg is None else radians(beta_deg),
        )

    @property
    def body_from_world(self) -> Vector:
        return body_from_world_rotation(self.yaw_rad, self.pitch_rad, self.roll_rad)

    @property
    def sensor_from_world(self) -> Vector:
        return sensor_from_world_rotation(self.yaw_rad, self.sensor_yaw_rad)

    @property
    def sensor_from_body(self) -> Vector:
        return self.sensor_from_world @ self.body_from_world.T

    @property
    def body_from_sensor(self) -> Vector:
        return self.body_from_world @ self.sensor_from_world.T


@dataclass(frozen=True)
class ScanVolume:
    """Scan limits expressed in sensor azimuth, elevation, and range."""

    az_min_rad: float
    az_max_rad: float
    el_min_rad: float
    el_max_rad: float
    range_min_m: float
    range_max_m: float

    @classmethod
    def from_degrees(
        cls,
        *,
        az_min_deg: float,
        az_max_deg: float,
        el_min_deg: float,
        el_max_deg: float,
        range_min_m: float,
        range_max_m: float,
    ) -> "ScanVolume":
        return cls(
            az_min_rad=radians(az_min_deg),
            az_max_rad=radians(az_max_deg),
            el_min_rad=radians(el_min_deg),
            el_max_rad=radians(el_max_deg),
            range_min_m=range_min_m,
            range_max_m=range_max_m,
        )

    def contains(self, azimuth_rad: float, elevation_rad: float, range_m: float) -> bool:
        return (
            self.az_min_rad <= azimuth_rad <= self.az_max_rad
            and self.el_min_rad <= elevation_rad <= self.el_max_rad
            and self.range_min_m <= range_m <= self.range_max_m
        )


@dataclass(frozen=True)
class OcclusionProfile:
    """Five-point piecewise-linear occlusion boundary in the body x-z plane."""

    points_body: NDArray[np.float64]
    occluded_if: str = "z_le_boundary"

    def __post_init__(self) -> None:
        points = np.asarray(self.points_body, dtype=float)
        if points.shape != (5, 3):
            raise ValueError(f"points_body must have shape (5, 3), got {points.shape!r}")
        if not np.allclose(points[:, 1], 0.0):
            raise ValueError("All occlusion profile points must lie in the body y=0 plane")
        order = np.argsort(points[:, 0])
        points = points[order]
        if np.any(np.diff(points[:, 0]) <= 0.0):
            raise ValueError("Occlusion profile x coordinates must be strictly increasing")
        if self.occluded_if not in {"z_le_boundary", "z_ge_boundary"}:
            raise ValueError("occluded_if must be 'z_le_boundary' or 'z_ge_boundary'")
        object.__setattr__(self, "points_body", points)

    def boundary_z(self, x_value: float) -> float | None:
        x_min = float(self.points_body[0, 0])
        x_max = float(self.points_body[-1, 0])
        if not (x_min <= x_value <= x_max):
            return None
        return float(np.interp(x_value, self.points_body[:, 0], self.points_body[:, 2]))

    def is_occluded_body_point(self, point_body: ArrayLike3, *, tolerance: float = 1e-9) -> bool:
        point = _as_vector(point_body, name="point_body")
        boundary_z = self.boundary_z(float(point[0]))
        if boundary_z is None:
            return False
        if self.occluded_if == "z_le_boundary":
            return float(point[2]) <= boundary_z + tolerance
        return float(point[2]) >= boundary_z - tolerance


@dataclass(frozen=True)
class VisibilityResult:
    """Visibility assessment for a single point."""

    point_sensor: Vector
    point_body: Vector
    azimuth_rad: float
    elevation_rad: float
    range_m: float
    in_scan: bool
    occluded: bool
    visible: bool


def transform_world_to_sensor(point_ned: ArrayLike3, state: PlatformState) -> Vector:
    """Transform a world/NED point into the stabilized sensor frame."""

    point_world = _as_vector(point_ned, name="point_ned")
    return state.sensor_from_world @ (point_world - state.position_ned)


def transform_sensor_to_body(point_sensor: ArrayLike3, state: PlatformState) -> Vector:
    """Transform a sensor-frame point into body coordinates."""

    return state.body_from_sensor @ _as_vector(point_sensor, name="point_sensor")


def transform_body_to_sensor(point_body: ArrayLike3, state: PlatformState) -> Vector:
    """Transform a body-frame point into sensor coordinates."""

    return state.sensor_from_body @ _as_vector(point_body, name="point_body")


def cartesian_to_sensor_angles(point_sensor: ArrayLike3) -> tuple[float, float, float]:
    """Return azimuth, elevation, and range in the sensor frame."""

    x_value, y_value, z_value = _as_vector(point_sensor, name="point_sensor")
    range_m = float(np.linalg.norm([x_value, y_value, z_value]))
    azimuth_rad = atan2(y_value, x_value)
    elevation_rad = atan2(-z_value, hypot(x_value, y_value))
    return azimuth_rad, elevation_rad, range_m


def ray_from_sensor_angles(azimuth_rad: float, elevation_rad: float, range_m: float) -> Vector:
    """Convert sensor azimuth, elevation, and range back to sensor-frame Cartesian."""

    cos_el = cos(elevation_rad)
    return np.array(
        [
            range_m * cos_el * cos(azimuth_rad),
            range_m * cos_el * sin(azimuth_rad),
            -range_m * sin(elevation_rad),
        ],
        dtype=float,
    )


def evaluate_visibility(
    point_ned: ArrayLike3,
    state: PlatformState,
    scan_volume: ScanVolume,
    profile: OcclusionProfile,
) -> VisibilityResult:
    """Evaluate scan gating and body occlusion for a world-frame point."""

    point_sensor = transform_world_to_sensor(point_ned, state)
    point_body = transform_sensor_to_body(point_sensor, state)
    azimuth_rad, elevation_rad, range_m = cartesian_to_sensor_angles(point_sensor)
    in_scan = scan_volume.contains(azimuth_rad, elevation_rad, range_m)
    occluded = profile.is_occluded_body_point(point_body)
    return VisibilityResult(
        point_sensor=point_sensor,
        point_body=point_body,
        azimuth_rad=azimuth_rad,
        elevation_rad=elevation_rad,
        range_m=range_m,
        in_scan=in_scan,
        occluded=occluded,
        visible=in_scan and not occluded,
    )
