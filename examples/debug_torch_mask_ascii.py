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


if __name__ == "__main__":
    main()
