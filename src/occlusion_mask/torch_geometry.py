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


def _coerce_single_value(value: float | Tensor, *, name: str, like: Tensor) -> Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a float or a single-value tensor")
        tensor = value.reshape(1, 1).to(device=like.device)
    else:
        tensor = torch.tensor([[float(value)]], device=like.device)

    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=like.dtype)
    else:
        tensor = tensor.to(dtype=torch.promote_types(tensor.dtype, like.dtype))

    return tensor


@dataclass(frozen=True)
class TorchAzElMask2D:
    """Torch-native 2D occlusion mask as a closed polygon in sensor az/el space."""

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
            target_dtype = torch.float32

        points = torch.as_tensor(points_az_el_deg, device=device, dtype=target_dtype)
        if dtype is not None:
            points = points.to(dtype=dtype)
        elif not torch.is_floating_point(points):
            points = points.to(torch.float32)

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
        sort_by_azimuth: bool = True,
    ) -> Tensor:
        pitch_rad, roll_rad = _coerce_pitch_roll(pitch=pitch_rad, roll=roll_rad)
        mask_points = self._points_like(pitch_rad)
        mask_az = mask_points[:, 0].unsqueeze(0)
        mask_el = mask_points[:, 1].unsqueeze(0)

        cos_roll = torch.cos(roll_rad)
        sin_roll = torch.sin(roll_rad)

        transformed_az = cos_roll * mask_az + sin_roll * mask_el
        transformed_el = -sin_roll * mask_az + cos_roll * mask_el + pitch_rad

        if sort_by_azimuth:
            order = transformed_az.argsort(dim=1)
            transformed_az = transformed_az.gather(1, order)
            transformed_el = transformed_el.gather(1, order)
        return torch.stack((transformed_az, transformed_el), dim=-1)

    def transformed_points_deg(
        self,
        *,
        pitch_deg: Tensor,
        roll_deg: Tensor,
        sort_by_azimuth: bool = True,
    ) -> Tensor:
        pitch_deg, roll_deg = _coerce_pitch_roll(pitch=pitch_deg, roll=roll_deg)
        return torch.rad2deg(
            self.transformed_points_rad(
                pitch_rad=torch.deg2rad(pitch_deg),
                roll_rad=torch.deg2rad(roll_deg),
                sort_by_azimuth=sort_by_azimuth,
            )
        )

    def polygon_points_rad(
        self,
        *,
        pitch_rad: Tensor,
        roll_rad: Tensor,
    ) -> Tensor:
        points = self.transformed_points_rad(
            pitch_rad=pitch_rad,
            roll_rad=roll_rad,
            sort_by_azimuth=False,
        )
        return torch.cat([points, points[:, :1, :]], dim=1)

    def polygon_points_deg(
        self,
        *,
        pitch_deg: Tensor,
        roll_deg: Tensor,
    ) -> Tensor:
        pitch_deg, roll_deg = _coerce_pitch_roll(pitch=pitch_deg, roll=roll_deg)
        return torch.rad2deg(
            self.polygon_points_rad(
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
        raise NotImplementedError(
            "Closed polygon 2D masks do not define a single boundary elevation. "
            "Use is_occluded_deg or polygon_points_* instead."
        )

    def boundary_elevation_deg(
        self,
        azimuth_deg: Tensor,
        *,
        pitch_deg: Tensor,
        roll_deg: Tensor,
    ) -> Tensor:
        raise NotImplementedError(
            "Closed polygon 2D masks do not define a single boundary elevation. "
            "Use is_occluded_deg or polygon_points_* instead."
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
        polygon = self.polygon_points_deg(
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
        start_points = polygon[:, :-1, :]
        end_points = polygon[:, 1:, :]

        x1 = start_points[..., 0]
        y1 = start_points[..., 1]
        x2 = end_points[..., 0]
        y2 = end_points[..., 1]

        query_azimuth = azimuth_deg.expand(-1, x1.shape[1])
        query_elevation = elevation_deg.expand(-1, y1.shape[1])
        tolerance = torch.full_like(query_azimuth, tolerance_deg)

        cross = (query_azimuth - x1) * (y2 - y1) - (query_elevation - y1) * (x2 - x1)
        on_segment = (
            cross.abs() <= tolerance
        ) & (
            query_azimuth >= torch.minimum(x1, x2) - tolerance
        ) & (
            query_azimuth <= torch.maximum(x1, x2) + tolerance
        ) & (
            query_elevation >= torch.minimum(y1, y2) - tolerance
        ) & (
            query_elevation <= torch.maximum(y1, y2) + tolerance
        )

        straddles = (y1 > query_elevation) != (y2 > query_elevation)
        denominator = y2 - y1
        x_intersection = torch.where(
            denominator.abs() > 0.0,
            x1 + (query_elevation - y1) * (x2 - x1) / denominator,
            torch.full_like(x1, torch.inf),
        )
        crossings = straddles & (x_intersection >= query_azimuth - tolerance)
        inside = (crossings.sum(dim=1, keepdim=True) % 2) == 1
        return on_segment.any(dim=1, keepdim=True) | inside

    def render_ascii_deg(
        self,
        *,
        pitch_deg: float | Tensor = 0.0,
        roll_deg: float | Tensor = 0.0,
        width: int = 61,
        height: int = 21,
        azimuth_limits_deg: tuple[float, float] | None = None,
        elevation_limits_deg: tuple[float, float] | None = None,
    ) -> str:
        """Render one transformed mask state on an az/el plane as ASCII text."""

        if width < 9:
            raise ValueError("width must be at least 9")
        if height < 7:
            raise ValueError("height must be at least 7")

        like = self.points_az_el_rad
        pitch_tensor = _coerce_single_value(pitch_deg, name="pitch_deg", like=like)
        roll_tensor = _coerce_single_value(roll_deg, name="roll_deg", like=like)
        points = self.transformed_points_deg(
            pitch_deg=pitch_tensor,
            roll_deg=roll_tensor,
            sort_by_azimuth=False,
        )[0]

        az_points = points[:, 0]
        el_points = points[:, 1]

        if azimuth_limits_deg is None:
            az_span = float((az_points.max() - az_points.min()).item())
            az_margin = max(5.0, 0.08 * az_span)
            az_min = float(az_points.min().item()) - az_margin
            az_max = float(az_points.max().item()) + az_margin
        else:
            az_min, az_max = azimuth_limits_deg

        if elevation_limits_deg is None:
            el_span = float((el_points.max() - el_points.min()).item())
            el_margin = max(3.0, 0.25 * max(el_span, 1.0))
            el_min = float(el_points.min().item()) - el_margin
            el_max = float(el_points.max().item()) + el_margin
        else:
            el_min, el_max = elevation_limits_deg

        if not az_min < az_max:
            raise ValueError("azimuth_limits_deg must satisfy min < max")
        if not el_min < el_max:
            raise ValueError("elevation_limits_deg must satisfy min < max")

        grid = [[" " for _ in range(width)] for _ in range(height)]

        def to_col(azimuth_value: float) -> int:
            normalized = (azimuth_value - az_min) / (az_max - az_min)
            return max(0, min(width - 1, int(round(normalized * (width - 1)))))

        def to_row(elevation_value: float) -> int:
            normalized = (el_max - elevation_value) / (el_max - el_min)
            return max(0, min(height - 1, int(round(normalized * (height - 1)))))

        if az_min <= 0.0 <= az_max:
            axis_col = to_col(0.0)
            for row in range(height):
                grid[row][axis_col] = "|"

        if el_min <= 0.0 <= el_max:
            axis_row = to_row(0.0)
            for col in range(width):
                grid[axis_row][col] = "-"

        if az_min <= 0.0 <= az_max and el_min <= 0.0 <= el_max:
            grid[to_row(0.0)][to_col(0.0)] = "+"

        polygon_points = torch.cat([points, points[:1]], dim=0)
        for start, end in zip(polygon_points[:-1], polygon_points[1:], strict=True):
            start_col = to_col(float(start[0].item()))
            start_row = to_row(float(start[1].item()))
            end_col = to_col(float(end[0].item()))
            end_row = to_row(float(end[1].item()))
            steps = max(abs(end_col - start_col), abs(end_row - start_row), 1)

            for step in range(steps + 1):
                alpha = step / steps
                col = int(round(start_col + alpha * (end_col - start_col)))
                row = int(round(start_row + alpha * (end_row - start_row)))
                grid[row][col] = "#"

        for label, point in zip("ABCDE", points, strict=True):
            col = to_col(float(point[0].item()))
            row = to_row(float(point[1].item()))
            grid[row][col] = label

        header = [
            (
                f"AzEl mask debug  pitch={float(pitch_tensor.item()):.1f} deg  "
                f"roll={float(roll_tensor.item()):.1f} deg"
            ),
            f"az=[{az_min:.1f}, {az_max:.1f}] deg  el=[{el_min:.1f}, {el_max:.1f}] deg",
        ]

        body = ["".join(row) for row in grid]
        footer = [
            (
                f"{label}=({float(point[0].item()):6.2f} az, "
                f"{float(point[1].item()):6.2f} el)"
            )
            for label, point in zip("ABCDE", points, strict=True)
        ]
        return "\n".join([*header, *body, *footer])
