"""Torch-native geometry helpers for batched azimuth/elevation masking."""

from typing import List, Tuple, Union

import torch


class TorchAzElMask2D:
    """Torch-native 2D occlusion mask as a closed polygon in sensor az/el space."""

    def __init__(
        self,
        points_az_el_rad: torch.Tensor,
        occluded_if: str = "el_ge_boundary",
    ) -> None:
        if not isinstance(points_az_el_rad, torch.Tensor):
            raise AssertionError("points_az_el_rad must be a torch.Tensor")
        if points_az_el_rad.shape != (5, 2):
            raise AssertionError("points_az_el_rad must have shape (5, 2)")
        if points_az_el_rad.dtype != torch.float32:
            raise AssertionError("points_az_el_rad must be torch.float32")
        if torch.any(points_az_el_rad[1:, 0] <= points_az_el_rad[:-1, 0]):
            raise AssertionError("2D mask points must be ordered left-to-right by increasing azimuth")
        if occluded_if not in {"el_ge_boundary", "el_le_boundary"}:
            raise AssertionError("occluded_if must be 'el_ge_boundary' or 'el_le_boundary'")

        self.points_az_el_rad = points_az_el_rad
        self.occluded_if = occluded_if

    @classmethod
    def from_degrees(
        cls,
        points_az_el_deg: Union[torch.Tensor, List[Tuple[float, float]], Tuple[Tuple[float, float], ...]],
        occluded_if: str = "el_ge_boundary",
        device: Union[torch.device, str, None] = None,
    ) -> "TorchAzElMask2D":
        if isinstance(points_az_el_deg, torch.Tensor):
            if points_az_el_deg.dtype != torch.float32:
                raise AssertionError("points_az_el_deg must be torch.float32")
            points = points_az_el_deg if device is None else points_az_el_deg.to(device=device)
        else:
            points = torch.tensor(points_az_el_deg, dtype=torch.float32, device=device)

        return cls(
            points_az_el_rad=torch.deg2rad(points),
            occluded_if=occluded_if,
        )

    @property
    def points_az_el_deg(self) -> torch.Tensor:
        return torch.rad2deg(self.points_az_el_rad)

    def transformed_points_rad(
        self,
        pitch_rad: torch.Tensor,
        roll_rad: torch.Tensor,
        sort_by_azimuth: bool = True,
    ) -> torch.Tensor:
        if not isinstance(pitch_rad, torch.Tensor) or not isinstance(roll_rad, torch.Tensor):
            raise AssertionError("pitch_rad and roll_rad must be torch.Tensor column vectors")
        if pitch_rad.ndim != 2 or pitch_rad.shape[1] != 1:
            raise AssertionError("pitch_rad must have shape (n, 1)")
        if roll_rad.ndim != 2 or roll_rad.shape[1] != 1:
            raise AssertionError("roll_rad must have shape (n, 1)")
        if pitch_rad.shape != roll_rad.shape:
            raise AssertionError("pitch_rad and roll_rad must have the same shape (n, 1)")
        if pitch_rad.device != roll_rad.device:
            raise AssertionError("pitch_rad and roll_rad must be on the same device")
        if pitch_rad.dtype != torch.float32 or roll_rad.dtype != torch.float32:
            raise AssertionError("pitch_rad and roll_rad must be torch.float32")

        mask_points = self.points_az_el_rad.to(device=pitch_rad.device)

        mask_az = mask_points[:, 0].unsqueeze(0)
        mask_el = mask_points[:, 1].unsqueeze(0) + pitch_rad

        cos_roll = torch.cos(roll_rad)
        sin_roll = torch.sin(roll_rad)

        transformed_az = cos_roll * mask_az + sin_roll * mask_el
        transformed_el = -sin_roll * mask_az + cos_roll * mask_el

        if sort_by_azimuth:
            order = transformed_az.argsort(dim=1)
            transformed_az = transformed_az.gather(1, order)
            transformed_el = transformed_el.gather(1, order)
        return torch.stack((transformed_az, transformed_el), dim=-1)

    def transformed_points_deg(
        self,
        pitch_deg: torch.Tensor,
        roll_deg: torch.Tensor,
        sort_by_azimuth: bool = True,
    ) -> torch.Tensor:
        if not isinstance(pitch_deg, torch.Tensor) or not isinstance(roll_deg, torch.Tensor):
            raise AssertionError("pitch_deg and roll_deg must be torch.Tensor column vectors")
        if pitch_deg.ndim != 2 or pitch_deg.shape[1] != 1:
            raise AssertionError("pitch_deg must have shape (n, 1)")
        if roll_deg.ndim != 2 or roll_deg.shape[1] != 1:
            raise AssertionError("roll_deg must have shape (n, 1)")
        if pitch_deg.shape != roll_deg.shape:
            raise AssertionError("pitch_deg and roll_deg must have the same shape (n, 1)")
        if pitch_deg.device != roll_deg.device:
            raise AssertionError("pitch_deg and roll_deg must be on the same device")
        if pitch_deg.dtype != torch.float32 or roll_deg.dtype != torch.float32:
            raise AssertionError("pitch_deg and roll_deg must be torch.float32")

        return torch.rad2deg(
            self.transformed_points_rad(
                pitch_rad=torch.deg2rad(pitch_deg),
                roll_rad=torch.deg2rad(roll_deg),
                sort_by_azimuth=sort_by_azimuth,
            )
        )

    def polygon_points_rad(
        self,
        pitch_rad: torch.Tensor,
        roll_rad: torch.Tensor,
    ) -> torch.Tensor:
        points = self.transformed_points_rad(
            pitch_rad=pitch_rad,
            roll_rad=roll_rad,
            sort_by_azimuth=False,
        )
        return torch.cat([points, points[:, :1, :]], dim=1)

    def polygon_points_deg(
        self,
        pitch_deg: torch.Tensor,
        roll_deg: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(pitch_deg, torch.Tensor) or not isinstance(roll_deg, torch.Tensor):
            raise AssertionError("pitch_deg and roll_deg must be torch.Tensor column vectors")
        if pitch_deg.ndim != 2 or pitch_deg.shape[1] != 1:
            raise AssertionError("pitch_deg must have shape (n, 1)")
        if roll_deg.ndim != 2 or roll_deg.shape[1] != 1:
            raise AssertionError("roll_deg must have shape (n, 1)")
        if pitch_deg.shape != roll_deg.shape:
            raise AssertionError("pitch_deg and roll_deg must have the same shape (n, 1)")
        if pitch_deg.device != roll_deg.device:
            raise AssertionError("pitch_deg and roll_deg must be on the same device")
        if pitch_deg.dtype != torch.float32 or roll_deg.dtype != torch.float32:
            raise AssertionError("pitch_deg and roll_deg must be torch.float32")

        return torch.rad2deg(
            self.polygon_points_rad(
                pitch_rad=torch.deg2rad(pitch_deg),
                roll_rad=torch.deg2rad(roll_deg),
            )
        )

    def is_occluded_deg(
        self,
        azimuth_deg: torch.Tensor,
        elevation_deg: torch.Tensor,
        pitch_deg: torch.Tensor,
        roll_deg: torch.Tensor,
        tolerance_deg: float = 1e-7,
    ) -> torch.Tensor:
        if not isinstance(azimuth_deg, torch.Tensor):
            raise AssertionError("azimuth_deg must be a torch.Tensor")
        if not isinstance(elevation_deg, torch.Tensor):
            raise AssertionError("elevation_deg must be a torch.Tensor")
        if not isinstance(pitch_deg, torch.Tensor):
            raise AssertionError("pitch_deg must be a torch.Tensor")
        if not isinstance(roll_deg, torch.Tensor):
            raise AssertionError("roll_deg must be a torch.Tensor")
        if azimuth_deg.ndim != 2 or azimuth_deg.shape[1] != 1:
            raise AssertionError("azimuth_deg must have shape (n, 1)")
        if elevation_deg.ndim != 2 or elevation_deg.shape[1] != 1:
            raise AssertionError("elevation_deg must have shape (n, 1)")
        if pitch_deg.ndim != 2 or pitch_deg.shape[1] != 1:
            raise AssertionError("pitch_deg must have shape (n, 1)")
        if roll_deg.ndim != 2 or roll_deg.shape[1] != 1:
            raise AssertionError("roll_deg must have shape (n, 1)")
        if elevation_deg.shape != azimuth_deg.shape or pitch_deg.shape != azimuth_deg.shape or roll_deg.shape != azimuth_deg.shape:
            raise AssertionError("azimuth_deg, elevation_deg, pitch_deg, and roll_deg must all have the same shape (n, 1)")
        if elevation_deg.device != azimuth_deg.device or pitch_deg.device != azimuth_deg.device or roll_deg.device != azimuth_deg.device:
            raise AssertionError("azimuth_deg, elevation_deg, pitch_deg, and roll_deg must all be on the same device")
        if (
            azimuth_deg.dtype != torch.float32
            or elevation_deg.dtype != torch.float32
            or pitch_deg.dtype != torch.float32
            or roll_deg.dtype != torch.float32
        ):
            raise AssertionError("azimuth_deg, elevation_deg, pitch_deg, and roll_deg must all be torch.float32")
        if tolerance_deg < 0.0:
            raise AssertionError("tolerance_deg must be non-negative")

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
        pitch_deg: Union[float, torch.Tensor] = 0.0,
        roll_deg: Union[float, torch.Tensor] = 0.0,
        width: int = 61,
        height: int = 21,
        azimuth_limits_deg: Union[Tuple[float, float], None] = None,
        elevation_limits_deg: Union[Tuple[float, float], None] = None,
    ) -> str:
        """Render one transformed mask state on an az/el plane as ASCII text."""

        if width < 9:
            raise AssertionError("width must be at least 9")
        if height < 7:
            raise AssertionError("height must be at least 7")

        if isinstance(pitch_deg, torch.Tensor):
            if pitch_deg.numel() != 1:
                raise AssertionError("pitch_deg must be a float or a single-value tensor")
            if pitch_deg.dtype != torch.float32:
                raise AssertionError("pitch_deg tensor must be torch.float32")
            pitch_tensor = pitch_deg.reshape(1, 1).to(device=self.points_az_el_rad.device)
        else:
            pitch_tensor = torch.tensor([[float(pitch_deg)]], dtype=torch.float32, device=self.points_az_el_rad.device)

        if isinstance(roll_deg, torch.Tensor):
            if roll_deg.numel() != 1:
                raise AssertionError("roll_deg must be a float or a single-value tensor")
            if roll_deg.dtype != torch.float32:
                raise AssertionError("roll_deg tensor must be torch.float32")
            roll_tensor = roll_deg.reshape(1, 1).to(device=self.points_az_el_rad.device)
        else:
            roll_tensor = torch.tensor([[float(roll_deg)]], dtype=torch.float32, device=self.points_az_el_rad.device)

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
            raise AssertionError("azimuth_limits_deg must satisfy min < max")
        if not el_min < el_max:
            raise AssertionError("elevation_limits_deg must satisfy min < max")

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
        for start, end in zip(polygon_points[:-1], polygon_points[1:]):
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

        for label, point in zip("ABCDE", points):
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
            for label, point in zip("ABCDE", points)
        ]
        return "\n".join([*header, *body, *footer])
