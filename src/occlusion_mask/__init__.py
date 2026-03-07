"""Occlusion-mask modeling for a level-stabilized aircraft sensor."""

from .geometry import (
    AzElMask2D,
    OcclusionProfile,
    PlatformState,
    ScanVolume,
    VisibilityResult,
    body_from_world_rotation,
    body_from_wind_rotation,
    cartesian_to_sensor_angles,
    evaluate_visibility,
    ray_from_sensor_angles,
    sensor_from_world_rotation,
    transform_body_to_sensor,
    transform_sensor_to_body,
    transform_world_to_sensor,
)
from .visualization import make_az_el_mask_figure, make_visibility_figure

__all__ = [
    "AzElMask2D",
    "OcclusionProfile",
    "PlatformState",
    "ScanVolume",
    "VisibilityResult",
    "body_from_world_rotation",
    "body_from_wind_rotation",
    "cartesian_to_sensor_angles",
    "evaluate_visibility",
    "make_az_el_mask_figure",
    "make_visibility_figure",
    "ray_from_sensor_angles",
    "sensor_from_world_rotation",
    "transform_body_to_sensor",
    "transform_sensor_to_body",
    "transform_world_to_sensor",
]
