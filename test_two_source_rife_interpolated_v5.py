from __future__ import annotations

import unittest

import numpy as np

import make_scathach_wallpaper_two_source_rife_interpolated_v5 as gen


class TwoSourceRifeInterpolatedV5Tests(unittest.TestCase):
    def test_v5_uses_deghost_output_name(self) -> None:
        self.assertEqual(gen.OUTPUT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v5_deghost.mp4")
        self.assertEqual(gen.PROJECT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v5_deghost.mp4")

    def test_phase_source_cleanup_suppresses_opposite_source_line(self) -> None:
        frame = np.full((24, 24, 3), 120.0, dtype=np.float32)
        source_a = np.full_like(frame, 120.0)
        source_b = np.full_like(frame, 120.0)
        source_a[:, 7, :] = 20.0
        source_b[:, 16, :] = 20.0
        frame[:, 7, :] = 70.0
        frame[:, 16, :] = 70.0
        mask = np.zeros((24, 24), dtype=np.float32)
        mask[:, 6:18] = 1.0

        cleaned_a = gen.apply_phase_source_cleanup(frame, source_a, source_b, mask, alpha=0.25, strength=0.55)
        cleaned_b = gen.apply_phase_source_cleanup(frame, source_a, source_b, mask, alpha=0.75, strength=0.55)

        self.assertLess(float(cleaned_a[12, 7, 0]), 55.0)
        self.assertGreater(float(cleaned_a[12, 16, 0]), 88.0)
        self.assertGreater(float(cleaned_b[12, 7, 0]), 88.0)
        self.assertLess(float(cleaned_b[12, 16, 0]), 55.0)

    def test_phase_source_cleanup_is_mask_limited(self) -> None:
        frame = np.full((16, 16, 3), 100.0, dtype=np.float32)
        source_a = np.zeros_like(frame)
        source_b = np.zeros_like(frame)
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[8:, :] = 1.0

        cleaned = gen.apply_phase_source_cleanup(frame, source_a, source_b, mask, alpha=0.1, strength=0.5)

        self.assertAlmostEqual(float(cleaned[2, 2, 0]), 100.0, places=5)
        self.assertLess(float(cleaned[12, 2, 0]), 55.0)

    def test_ghost_cleanup_mask_finds_edges_inside_risk_area_only(self) -> None:
        source_a = np.full((80, 120, 3), 120.0, dtype=np.float32)
        source_b = source_a.copy()
        source_a[35:60, 62:64, :] = 10.0
        source_b[35:60, 88:90, :] = 10.0
        risk = np.zeros((80, 120), dtype=np.float32)
        risk[20:70, 40:105] = 1.0

        mask = gen.build_ghost_cleanup_mask_from_risk(source_a, source_b, risk)

        self.assertGreater(float(mask[45, 63]), 0.45)
        self.assertGreater(float(mask[45, 89]), 0.45)
        self.assertLess(float(mask[10, 63]), 0.05)

    def test_ghost_cleanup_mask_excludes_gold_highlights(self) -> None:
        source_a = np.full((80, 120, 3), 120.0, dtype=np.float32)
        source_b = source_a.copy()
        source_a[35:60, 62:64, :] = np.array([230.0, 170.0, 72.0], dtype=np.float32)
        risk = np.ones((80, 120), dtype=np.float32)

        mask = gen.build_ghost_cleanup_mask_from_risk(source_a, source_b, risk)

        self.assertLess(float(mask[45, 63]), 0.20)


if __name__ == "__main__":
    unittest.main()
