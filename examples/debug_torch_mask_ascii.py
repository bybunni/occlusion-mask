from __future__ import annotations

import torch

from occlusion_mask import TorchAzElMask2D


MASK_POINTS = [
    (-40.0, 16.0),
    (-20.0, 13.0),
    (0.0, 10.0),
    (20.0, 13.0),
    (40.0, 16.0),
]

SAMPLE_STATES = [
    ("level", 0.0, 0.0),
    ("pitch up", 6.0, 0.0),
    ("pitch down", -6.0, 0.0),
    ("right roll", 0.0, 10.0),
    ("combined", -4.0, 8.0),
]

QUERY_AZ_DEG = 5.0
QUERY_EL_DEG = 12.0


def column(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).unsqueeze(1)


def print_vectorized_example(mask: TorchAzElMask2D) -> None:
    azimuth_deg = column([0.0, 5.0, -15.0, 25.0])
    elevation_deg = column([12.0, 9.0, 14.0, 8.0])
    pitch_deg = column([0.0, -6.0, 3.0, 0.0])
    roll_deg = column([0.0, 8.0, -4.0, 10.0])

    occluded = mask.is_occluded_deg(
        azimuth_deg,
        elevation_deg,
        pitch_deg,
        roll_deg,
    )

    print()
    print("=" * 80)
    print("vectorized batch example")
    print("input tensors have shape (n, 1), and each row is evaluated in parallel")
    print(f"azimuth_deg.shape={tuple(azimuth_deg.shape)}")
    print(f"elevation_deg.shape={tuple(elevation_deg.shape)}")
    print(f"pitch_deg.shape={tuple(pitch_deg.shape)}")
    print(f"roll_deg.shape={tuple(roll_deg.shape)}")
    print(f"occluded.shape={tuple(occluded.shape)}")

    for index, (azimuth_value, elevation_value, pitch_value, roll_value, occluded_value) in enumerate(
        zip(
            azimuth_deg.squeeze(1).tolist(),
            elevation_deg.squeeze(1).tolist(),
            pitch_deg.squeeze(1).tolist(),
            roll_deg.squeeze(1).tolist(),
            occluded.squeeze(1).tolist(),
            strict=True,
        ),
        start=1,
    ):
        print(
            f"row {index}: az={azimuth_value:+5.1f} deg | "
            f"el={elevation_value:+5.1f} deg | "
            f"pitch={pitch_value:+5.1f} deg | "
            f"roll={roll_value:+5.1f} deg | "
            f"occluded={bool(occluded_value)}"
        )


def main() -> None:
    mask = TorchAzElMask2D.from_degrees(MASK_POINTS)

    print("TorchAzElMask2D ASCII debug demo")
    print(f"sample query: az={QUERY_AZ_DEG:.1f} deg, el={QUERY_EL_DEG:.1f} deg")

    for label, pitch_deg, roll_deg in SAMPLE_STATES:
        occluded = mask.is_occluded_deg(
            torch.tensor([[QUERY_AZ_DEG]], dtype=torch.float32),
            torch.tensor([[QUERY_EL_DEG]], dtype=torch.float32),
            torch.tensor([[pitch_deg]], dtype=torch.float32),
            torch.tensor([[roll_deg]], dtype=torch.float32),
        )

        print()
        print("=" * 80)
        print(
            f"{label}: pitch={pitch_deg:.1f} deg, roll={roll_deg:.1f} deg, "
            f"query_occluded={bool(occluded.item())}"
        )
        print(mask.render_ascii_deg(pitch_deg=pitch_deg, roll_deg=roll_deg, width=61, height=21))

    print_vectorized_example(mask)


if __name__ == "__main__":
    main()
