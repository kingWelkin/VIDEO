from __future__ import annotations

import unittest

import numpy as np

import make_scathach_wallpaper_two_source_rife_interpolated_v6 as gen


class TwoSourceRifeInterpolatedV6Tests(unittest.TestCase):
    def test_v6_uses_highpass_deghost_output_name(self) -> None:
        self.assertEqual(gen.OUTPUT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v6_highpass_deghost.mp4")
        self.assertEqual(gen.PROJECT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v6_highpass_deghost.mp4")

    def test_highpass_deghost_suppresses_opposite_source_line(self) -> None:
        frame = np.full((32, 32, 3), 120.0, dtype=np.float32)
        source_a = np.full_like(frame, 120.0)
        source_b = np.full_like(frame, 120.0)
        source_a[:, 8, :] = 20.0
        source_b[:, 23, :] = 20.0
        frame[:, 8, :] = 70.0
        frame[:, 23, :] = 70.0
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[:, 6:26] = 1.0

        cleaned_a = gen.apply_highpass_deghost(frame, source_a, source_b, mask, alpha=0.25, strength=1.0)
        cleaned_b = gen.apply_highpass_deghost(frame, source_a, source_b, mask, alpha=0.75, strength=1.0)

        self.assertLess(float(cleaned_a[16, 8, 0]), 62.0)
        self.assertGreater(float(cleaned_a[16, 23, 0]), 93.0)
        self.assertGreater(float(cleaned_b[16, 8, 0]), 93.0)
        self.assertLess(float(cleaned_b[16, 23, 0]), 62.0)

    def test_highpass_deghost_preserves_masked_low_frequency_color(self) -> None:
        frame = np.full((24, 24, 3), 118.0, dtype=np.float32)
        source_a = np.full_like(frame, 42.0)
        source_b = np.full_like(frame, 210.0)
        mask = np.ones((24, 24), dtype=np.float32)

        cleaned_a = gen.apply_highpass_deghost(frame, source_a, source_b, mask, alpha=0.20, strength=1.0)
        cleaned_b = gen.apply_highpass_deghost(frame, source_a, source_b, mask, alpha=0.80, strength=1.0)

        self.assertAlmostEqual(float(cleaned_a[12, 12, 0]), 118.0, delta=1.0)
        self.assertAlmostEqual(float(cleaned_b[12, 12, 0]), 118.0, delta=1.0)

    def test_highpass_cleanup_mask_finds_lines_inside_risk_area_only(self) -> None:
        source_a = np.full((80, 120, 3), 120.0, dtype=np.float32)
        source_b = source_a.copy()
        source_a[35:60, 62:64, :] = 10.0
        source_b[35:60, 88:90, :] = 10.0
        risk = np.zeros((80, 120), dtype=np.float32)
        risk[20:70, 40:105] = 1.0

        mask = gen.build_highpass_cleanup_mask_from_risk(source_a, source_b, risk)

        self.assertGreater(float(mask[45, 63]), 0.40)
        self.assertGreater(float(mask[45, 89]), 0.40)
        self.assertLess(float(mask[10, 63]), 0.05)
        self.assertLess(float(mask[45, 20]), 0.05)


if __name__ == "__main__":
    unittest.main()
