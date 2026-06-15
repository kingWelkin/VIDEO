from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import make_scathach_wallpaper_two_source_rife_interpolated_v2 as gen


class TwoSourceRifeInterpolatedV2Tests(unittest.TestCase):
    def test_v2_targets_shorter_240_frame_loop_without_replacing_v1(self) -> None:
        self.assertEqual(gen.TARGET_FRAMES, 240)
        self.assertEqual(gen.SEGMENT_FRAMES, 120)
        self.assertEqual(gen.OUTPUT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v2.mp4")
        self.assertEqual(gen.PROJECT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v2.mp4")

    def test_rife_segment_command_requests_double_minus_two_frames(self) -> None:
        command = gen.build_rife_command(Path("keys"), Path("rife"), frames=gen.SEGMENT_RIFE_FRAMES)

        self.assertEqual(command[command.index("-n") + 1], "238")
        self.assertEqual(command[command.index("-i") + 1], "keys")
        self.assertEqual(command[command.index("-o") + 1], "rife")

    def test_copy_motion_half_renumbers_first_half_of_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            for index in range(1, 239):
                Image.new("RGB", (8, 8), (index % 255, 0, 0)).save(source / f"{index:08d}.png")

            copied = gen.copy_motion_half(source, target, start_index=10, segment_frames=120)

            self.assertEqual(len(copied), 120)
            self.assertEqual(copied[0].name, "00000010.png")
            self.assertEqual(copied[-1].name, "00000129.png")
            self.assertEqual(Image.open(copied[0]).getpixel((0, 0)), (1, 0, 0))
            self.assertEqual(Image.open(copied[-1]).getpixel((0, 0)), (120, 0, 0))

    def test_temporal_smoothing_is_mask_limited(self) -> None:
        previous = np.zeros((12, 12, 3), dtype=np.float32)
        current = np.zeros_like(previous)
        following = np.zeros_like(previous)
        current[4:8, 4:8, :] = 120.0
        mask = np.zeros((12, 12), dtype=np.float32)
        mask[4:8, 4:8] = 1.0

        smoothed = gen.temporal_smooth_region(previous, current, following, mask, strength=0.5)

        self.assertLess(float(smoothed[5, 5, 0]), 120.0)
        self.assertEqual(float(smoothed[1, 1, 0]), 0.0)

    def test_detail_restore_is_mask_limited_and_increases_local_contrast(self) -> None:
        frame = np.full((64, 64, 3), 80.0, dtype=np.float32)
        reference = frame.copy()
        reference[:, ::2, :] = 120.0
        reference[:, 1::2, :] = 40.0
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[:, 32:] = 1.0

        restored = gen.restore_local_detail(frame, reference, mask, amount=0.6)

        self.assertGreater(float(restored[:, 32:, :].std()), float(frame[:, 32:, :].std()))
        self.assertAlmostEqual(float(restored[:, :20, :].std()), 0.0, places=5)

    def test_static_tail_metric_detects_repeated_frames(self) -> None:
        frames: list[np.ndarray] = []
        for index in range(12):
            frame = np.full((16, 16, 3), index, dtype=np.float32)
            frames.append(frame)
        frames.extend([frames[-1].copy() for _ in range(5)])

        metric = gen.static_tail_count(frames, threshold=0.01)

        self.assertEqual(metric, 5)

    def test_endpoint_settle_only_acts_near_loop_end(self) -> None:
        self.assertEqual(gen.endpoint_settle_weight(gen.TARGET_FRAMES - 30), 0.0)
        self.assertGreater(gen.endpoint_settle_weight(gen.TARGET_FRAMES - 8), 0.60)
        self.assertAlmostEqual(gen.endpoint_settle_weight(gen.TARGET_FRAMES - 1), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
