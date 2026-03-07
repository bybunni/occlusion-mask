"""Torch-native geometry helpers for batched azimuth/elevation masking."""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


def _validate_points(points_az_el_rad: Tensor, *, name: str) -> Tensor:
    points = torch.as_tensor(points_az_el_rad)
    if points.ndim != 2 or points.shape != (5, 2):
        raise ValueError(f"{name} must have shape (5, 2), got {tuple(points.shape)!r}")
    if not torch.is_floating_point(points):
        points = points.to(torch.get_default_dtype())
    if torch.any(points[1:, 0] <= points[:-1, 0]):
        raise ValueError("2D mask points must be ordered left-to-right by increasing azimuth")
    return points


def _validate_column_tensor(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape (n, 1), got {tuple(value.shape)!r}")
    return value


def _coerce_query_tensors(*, azimuth: Tensor, elevation: Tensor, pitch: Tensor, roll: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    azimuth = _validate_column_tensor(azimuth, name="azimuth")
    elevation = _validate_column_tensor(elevation, name="elevation")
    pitch = _validate_column_tensor(pitch, name="pitch")
    roll = _validate_column_tensor(roll, name="roll")

    shape = azimuth.shape
    if elevation.shape != shape or pitch.shape != shape or roll.shape != shape:
        raise ValueError("azimuth, elevation, pitch, and roll must all have the same shape (n, 1)")

    devices = {tensor.device for tensor in (azimuth, elevation, pitch, roll)}
    if len(devices) != 1:
        raise ValueError("azimuth, elevation, pitch, and roll must all be on the same device")

    dtype = azimuth.dtype
    for tensor in (elevation, pitch, roll):
        dtype = torch.promote_types(dtype, tensor.dtype)
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.get_default_dtype()

    return tuple(tensor.to(dtype=dtype) for tensor in (azimuth, elevation, pitch, roll))


def _coerce_pitch_roll(*, pitch: Tensor, roll: Tensor) -> tuple[Tensor, Tensor]:
    pitch = _validate_column_tensor(pitch, name="pitch")
    roll = _validate_column_tensor(roll, name="roll")
    if pitch.shape != roll.shape:
        raise ValueError("pitch and roll must have the same shape (n, 1)")
    if pitch.device != roll.device:
        raise ValueError("pitch and roll must be on the same device")

    dtype = torch.promote_types(pitch.dtype, roll.dtype)
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.get_default_dtype()

    return pitch.to(dtype=dtype), roll.to(dtype=dtype)


@dataclass(frozen=True)
class TorchAzElMask2D:
    """Torch-native 2D occlusion mask in sensor azimuth/elevation space."""

    points_az_el_rad: Tensor
    occluded_if: str = "el_ge_boundary"

    def __post_init__(self) -> None:
        points = _validate_points(self.points_az_el_rad, name="points_az_el_rad")
        if self.occluded_if not in {"el_ge_boundary", "el_le_boundary"}:
            raise ValueError("occluded_if must be 'el_ge_boundary' or 'el_le_boundary'")
        object.__setattr__(self, "points_az_el_rad", points)

    @classmethod
    def from_degrees(
        cls,
        points_az_el_deg: Tensor | list[tuple[float, float]] | tuple[tuple[float, float], ...],
        *,
        occluded_if: str = "el_ge_boundary",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "TorchAzElMask2D":
        target_dtype = dtype
        if target_dtype is None and not isinstance(points_az_el_deg, torch.Tensor):
            target_dtype = torch.float64

        points = torch.as_tensor(points_az_el_deg, device=device, dtype=target_dtype)
        if dtype is not None:
            points = points.to(dtype=dtype)
        elif not torch.is_floating_point(points):
            points = points.to(torch.float64)

        points = _validate_points(points, name="points_az_el_deg")
        return cls(
            points_az_el_rad=torch.deg2rad(points),
            occluded_if=occluded_if,
        )

    @property
    def points_az_el_deg(self) -> Tensor:
        return torch.rad2deg(self.points_az_el_rad)

    def _points_like(self, reference: Tensor) -> Tensor:
        return self.points_az_el_rad.to(device=reference.device, dtype=reference.dtype)

    def transformed_points_rad(
        self,
        *,
        pitch_rad: Tensor,
        roll_rad: Tensor,
    ) -> Tensor:
        pitch_rad, roll_rad = _coerce_pitch_roll(pitch=pitch_rad, roll=roll_rad)
        mask_points = self._points_like(pitch_rad)
        mask_az = mask_points[:, 0].unsqueeze(0)
        mask_el = mask_points[:, 1].unsqueeze(0)

        cos_roll = torch.cos(roll_rad)
        sin_roll = torch.sin(roll_rad)

        transformed_az = cos_roll * mask_az - sin_roll * mask_el
        transformed_el = sin_roll * mask_az + cos_roll * mask_el + pitch_rad

        order = transformed_az.argsort(dim=1)
        transformed_az = transformed_az.gather(1, order)
        transformed_el = transformed_el.gather(1, order)
        return torch.stack((transformed_az, transformed_el), dim=-1)

    def transformed_points_deg(
        self,
        *,
        pitch_deg: Tensor,
        roll_deg: Tensor,
    ) -> Tensor:
        pitch_deg, roll_deg = _coerce_pitch_roll(pitch=pitch_deg, roll=roll_deg)
        return torch.rad2deg(
            self.transformed_points_rad(
                pitch_rad=torch.deg2rad(pitch_deg),
                roll_rad=torch.deg2rad(roll_deg),
            )
        )

    def boundary_elevation_rad(
        self,
        azimuth_rad: Tensor,
        *,
        pitch_rad: Tensor,
        roll_rad: Tensor,
    ) -> Tensor:
        azimuth_rad = _validate_column_tensor(azimuth_rad, name="azimuth")
        pitch_rad, roll_rad = _coerce_pitch_roll(pitch=pitch_rad, roll=roll_rad)
        if azimuth_rad.shape != pitch_rad.shape:
            raise ValueError("azimuth, pitch, and roll must all have the same shape (n, 1)")
        if azimuth_rad.device != pitch_rad.device:
            raise ValueError("azimuth, pitch, and roll must all be on the same device")

        dtype = torch.promote_types(azimuth_rad.dtype, pitch_rad.dtype)
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            dtype = torch.get_default_dtype()

        azimuth_rad = azimuth_rad.to(dtype=dtype)
        pitch_rad = pitch_rad.to(dtype=dtype)
        roll_rad = roll_rad.to(dtype=dtype)

        transformed_points = self.transformed_points_rad(pitch_rad=pitch_rad, roll_rad=roll_rad)
        point_az = transformed_points[..., 0]
        point_el = transformed_points[..., 1]

        outside = (azimuth_rad < point_az[:, :1]) | (azimuth_rad > point_az[:, -1:])

        indices = torch.searchsorted(point_az.contiguous(), azimuth_rad.contiguous(), right=False)
        indices = indices.clamp(min=1, max=point_az.shape[1] - 1)

        left_indices = indices - 1
        right_indices = indices

        az0 = point_az.gather(1, left_indices)
        az1 = point_az.gather(1, right_indices)
        el0 = point_el.gather(1, left_indices)
        el1 = point_el.gather(1, right_indices)

        delta = az1 - az0
        weight = torch.where(delta.abs() > 0.0, (azimuth_rad - az0) / delta, torch.zeros_like(azimuth_rad))
        boundary = el0 + weight * (el1 - el0)

        nan_value = torch.full_like(boundary, torch.nan)
        return torch.where(outside, nan_value, boundary)

    def boundary_elevation_deg(
        self,
        azimuth_deg: Tensor,
        *,
        pitch_deg: Tensor,
        roll_deg: Tensor,
    ) -> Tensor:
        return torch.rad2deg(
            self.boundary_elevation_rad(
                torch.deg2rad(_validate_column_tensor(azimuth_deg, name="azimuth")),
                pitch_rad=torch.deg2rad(_validate_column_tensor(pitch_deg, name="pitch")),
                roll_rad=torch.deg2rad(_validate_column_tensor(roll_deg, name="roll")),
            )
        )

    def is_occluded_deg(
        self,
        azimuth_deg: Tensor,
        elevation_deg: Tensor,
        pitch_deg: Tensor,
        roll_deg: Tensor,
        *,
        tolerance_deg: float = 1e-7,
    ) -> Tensor:
        azimuth_deg, elevation_deg, pitch_deg, roll_deg = _coerce_query_tensors(
            azimuth=azimuth_deg,
            elevation=elevation_deg,
            pitch=pitch_deg,
            roll=roll_deg,
        )
        boundary = self.boundary_elevation_deg(
            azimuth_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
        inside_span = ~torch.isnan(boundary)
        tolerance = torch.full_like(boundary, tolerance_deg)
        if self.occluded_if == "el_ge_boundary":
            return inside_span & (elevation_deg >= boundary - tolerance)
        return inside_span & (elevation_deg <= boundary + tolerance)
