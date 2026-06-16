from __future__ import annotations

import unittest

import numpy as np

import make_scathach_wallpaper_two_source_rife_interpolated_v8 as gen


class TwoSourceRifeInterpolatedV8Tests(unittest.TestCase):
    def test_v8_uses_source_locked_output_name(self) -> None:
        self.assertEqual(gen.OUTPUT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v8_source_locked.mp4")
        self.assertEqual(gen.PROJECT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v8_source_locked.mp4")

    def test_locked_reference_keeps_source_a_dominant_through_cycle(self) -> None:
        source_a = np.full((8, 8, 3), 40.0, dtype=np.float32)
        source_b = np.full_like(source_a, 200.0)

        reference = gen.locked_reference(source_a, source_b, alpha=1.0)

        self.assertLess(float(reference[4, 4, 0]), 80.0)

    def test_source_lock_replaces_rife_ghosts_inside_mask(self) -> None:
        frame = np.full((32, 32, 3), 120.0, dtype=np.float32)
        source_a = np.full_like(frame, 120.0)
        source_b = np.full_like(frame, 120.0)
        frame[:, 9:11, :] = 64.0
        frame[:, 20:22, :] = 72.0
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[:, 6:24] = 1.0

        cleaned = gen.apply_source_locked_region(frame, source_a, source_b, mask, alpha=0.55, strength=1.0)

        self.assertGreater(float(cleaned[16, 10, 0]), 116.0)
        self.assertGreater(float(cleaned[16, 21, 0]), 116.0)

    def test_source_lock_is_mask_limited(self) -> None:
        frame = np.full((20, 20, 3), 100.0, dtype=np.float32)
        source_a = np.full_like(frame, 150.0)
        source_b = np.full_like(frame, 180.0)
        mask = np.zeros((20, 20), dtype=np.float32)
        mask[10:, :] = 1.0

        cleaned = gen.apply_source_locked_region(frame, source_a, source_b, mask, alpha=0.5, strength=0.97)

        self.assertAlmostEqual(float(cleaned[4, 4, 0]), 100.0, delta=0.5)
        self.assertGreater(float(cleaned[15, 4, 0]), 142.0)

    def test_source_lock_mask_targets_torso_and_weapon_without_face(self) -> None:
        mask = gen.build_source_lock_mask((3840, 2160))

        self.assertGreater(float(mask[1180, 1920]), 0.80)
        self.assertGreater(float(mask[1430, 1080]), 0.72)
        self.assertGreater(float(mask[1370, 2200]), 0.65)
        self.assertLess(float(mask[340, 1920]), 0.05)
        self.assertLess(float(mask[1700, 3350]), 0.15)


if __name__ == "__main__":
    unittest.main()
